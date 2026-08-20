//! `POST /v1/chat/completions` — the ecosystem-compatibility protocol adapter.
//!
//! Chat Completions is *not* the internal architecture: this module only
//! translates between the OpenAI request/response shapes and
//! `ohmygpu_inference` types. Supported subset (documented in the README):
//!
//! * `model`, `messages` (system/developer/user/assistant/tool; string or text-part content)
//! * `tools` (function), `tool_choice` (auto/none/required/named)
//! * `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `stop`,
//!   `seed`, `presence_penalty`, `frequency_penalty`
//! * `stream`, `stream_options.include_usage`
//!
//! `n > 1`, non-text content parts and non-text `response_format` are rejected
//! with 400; unknown fields are ignored.

use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{Stream, StreamExt};
use ohmygpu_inference::{
    ContentPart, FinishReason, GenerationOptions, InferenceRequest, InferenceResponse, InputItem,
    OutputItem, Role, StreamEvent, ToolCall, ToolChoice, ToolDefinition, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::images;
use crate::api::{new_id, now_secs, ApiJson};
use crate::error::ApiError;
use crate::state::SharedState;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallSpec>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPartSpec>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPartSpec {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    /// `{"url": "...", "detail": ...}` per the spec; a bare string is accepted too.
    #[serde(default)]
    pub image_url: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallSpec {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: FunctionCallSpec,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FunctionCallSpec {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

/// Append text, merging with a preceding text part (adjacent text parts are
/// one piece of text as far as the model is concerned).
fn push_text(out: &mut Vec<ContentPart>, text: &str) {
    if let Some(ContentPart::Text { text: last }) = out.last_mut() {
        last.push_str(text);
    } else {
        out.push(ContentPart::text(text));
    }
}

/// OpenAI content (string or parts) → internal parts. Images (`image_url`)
/// are only accepted in user messages.
fn content_parts(content: Option<Content>, role: &str) -> Result<Vec<ContentPart>, ApiError> {
    match content {
        None => Ok(Vec::new()),
        Some(Content::Text(t)) => Ok(vec![ContentPart::text(t)]),
        Some(Content::Parts(parts)) => {
            let mut out = Vec::with_capacity(parts.len());
            for p in parts {
                match p.kind.as_str() {
                    "text" | "input_text" | "output_text" => {
                        push_text(&mut out, p.text.as_deref().unwrap_or(""))
                    }
                    "image_url" => {
                        if role != "user" {
                            return Err(ApiError::invalid(
                                "images are only supported in user messages",
                            )
                            .with_param("messages"));
                        }
                        let url = match p.image_url {
                            Some(Value::String(u)) => u,
                            Some(Value::Object(o)) => o
                                .get("url")
                                .and_then(|u| u.as_str())
                                .map(str::to_string)
                                .unwrap_or_default(),
                            _ => String::new(),
                        };
                        if url.is_empty() {
                            return Err(ApiError::invalid(
                                "image_url parts require `image_url.url`",
                            )
                            .with_param("messages"));
                        }
                        out.push(ContentPart::image(url));
                    }
                    "input_audio" => {
                        return Err(ApiError::unsupported("audio input (input_audio)")
                            .with_param("messages"))
                    }
                    other => {
                        return Err(ApiError::unsupported(format!(
                            "content part type '{other}' (in a {role} message)"
                        ))
                        .with_param("messages"))
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Text-only roles: flatten to one string; images here are an error.
fn content_text(content: Option<Content>, role: &str) -> Result<String, ApiError> {
    let mut out = String::new();
    for p in content_parts(content, role)? {
        if let ContentPart::Text { text } = p {
            out.push_str(&text);
        }
    }
    Ok(out)
}

/// Arguments may arrive as a JSON string (spec) or an object (lenient clients).
fn arguments_string(v: Option<Value>) -> String {
    match v {
        None => "{}".to_string(),
        Some(Value::String(s)) => s,
        Some(other) => other.to_string(),
    }
}

/// The whole point of this module: OpenAI request → one internal request.
pub fn to_inference_request(req: ChatCompletionRequest) -> Result<InferenceRequest, ApiError> {
    if req.model.trim().is_empty() {
        return Err(ApiError::invalid("`model` is required").with_param("model"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid("`messages` must not be empty").with_param("messages"));
    }
    if let Some(n) = req.n {
        if n != 1 {
            return Err(ApiError::unsupported("n > 1").with_param("n"));
        }
    }
    if let Some(rf) = &req.response_format {
        let kind = rf.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        if kind != "text" {
            return Err(
                ApiError::unsupported(format!("response_format type '{kind}'"))
                    .with_param("response_format"),
            );
        }
    }

    let mut input = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        match m.role.as_str() {
            "system" | "developer" => {
                input.push(InputItem::system(content_text(m.content, "system")?))
            }
            "user" => input.push(InputItem::Message {
                role: Role::User,
                content: content_parts(m.content, "user")?,
            }),
            "assistant" => {
                let text = content_text(m.content, "assistant")?;
                if !text.is_empty() {
                    input.push(InputItem::assistant(text));
                }
                for tc in m.tool_calls.unwrap_or_default() {
                    input.push(InputItem::ToolCall(ToolCall {
                        id: tc.id.unwrap_or_else(|| new_id("call_")),
                        name: tc.function.name,
                        arguments: arguments_string(tc.function.arguments),
                    }));
                }
            }
            "tool" => {
                let call_id = m.tool_call_id.ok_or_else(|| {
                    ApiError::invalid("tool messages require `tool_call_id`").with_param("messages")
                })?;
                input.push(InputItem::ToolResult {
                    call_id,
                    output: content_text(m.content, "tool")?,
                });
            }
            other => {
                return Err(ApiError::invalid(format!("unknown message role '{other}'"))
                    .with_param("messages"));
            }
        }
    }

    let mut tools = Vec::new();
    for t in req.tools {
        match (t.kind.as_deref().unwrap_or("function"), t.function) {
            ("function", Some(f)) => tools.push(ToolDefinition {
                name: f.name,
                description: f.description,
                parameters: f
                    .parameters
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            }),
            ("function", None) => {
                return Err(
                    ApiError::invalid("tool of type 'function' requires `function`")
                        .with_param("tools"),
                )
            }
            (other, _) => {
                return Err(
                    ApiError::unsupported(format!("tool type '{other}'")).with_param("tools")
                )
            }
        }
    }

    let tool_choice = match req.tool_choice {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(ApiError::invalid(format!("unknown tool_choice '{other}'"))
                    .with_param("tool_choice"))
            }
        },
        Some(v) => {
            let name = v
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .ok_or_else(|| ApiError::invalid("tool_choice object must be {\"type\":\"function\",\"function\":{\"name\":...}}").with_param("tool_choice"))?;
            ToolChoice::Named {
                name: name.to_string(),
            }
        }
    };

    let stop = match req.stop {
        None => vec![],
        Some(StringOrVec::One(s)) => vec![s],
        Some(StringOrVec::Many(v)) => v,
    };

    let ireq = InferenceRequest {
        model: req.model,
        input,
        tools,
        tool_choice,
        options: GenerationOptions {
            max_output_tokens: req.max_completion_tokens.or(req.max_tokens),
            temperature: req.temperature,
            top_p: req.top_p,
            stop,
            seed: req.seed,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
        },
    };
    ireq.validate().map_err(ApiError::from)?;
    Ok(ireq)
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: UsageObject,
    pub system_fingerprint: String,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
    pub logprobs: Option<()>,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallObject>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallObject {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionObject,
}

#[derive(Debug, Serialize, Clone)]
pub struct FunctionObject {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct UsageObject {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<Usage> for UsageObject {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens(),
        }
    }
}

fn fingerprint() -> String {
    format!("ohmygpu-{}", ohmygpu_core::VERSION)
}

/// Internal response → OpenAI chat completion object.
pub fn to_chat_completion(id: String, created: u64, resp: InferenceResponse) -> ChatCompletion {
    let text = resp.text();
    let tool_calls: Vec<ToolCallObject> = resp
        .output
        .iter()
        .filter_map(|o| match o {
            OutputItem::ToolCall(c) => Some(ToolCallObject {
                id: c.id.clone(),
                kind: "function",
                function: FunctionObject {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                },
            }),
            _ => None,
        })
        .collect();
    ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: if text.is_empty() && !tool_calls.is_empty() {
                    None
                } else {
                    Some(text)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: resp.finish_reason.as_str(),
            logprobs: None,
        }],
        usage: resp.usage.into(),
        system_fingerprint: fingerprint(),
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub async fn create(
    State(state): State<SharedState>,
    ApiJson(req): ApiJson<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let stream = req.stream.unwrap_or(false);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let mut ireq = to_inference_request(req)?;
    let instance = state.manager.instance_for(&ireq.model).await?;
    if ireq.has_images() {
        images::require_vision(
            state.manager.capabilities_of(&ireq.model),
            &ireq.model,
            "messages",
        )?;
        images::resolve_images(&mut ireq, &state.http, "messages").await?;
    }
    let id = new_id("chatcmpl-");
    let created = now_secs();

    if !stream {
        let resp = instance.infer(ireq).await?;
        return Ok(Json(to_chat_completion(id, created, resp)).into_response());
    }

    let model = ireq.model.clone();
    let events = instance.infer_stream(ireq).await?;
    Ok(
        Sse::new(stream_chunks(id, created, model, include_usage, events))
            .keep_alive(KeepAlive::default())
            .into_response(),
    )
}

/// One `chat.completion.chunk`.
fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
    usage: Option<Value>,
) -> Value {
    let mut v = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint(),
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": null }],
    });
    if let Some(u) = usage {
        v["usage"] = u;
    }
    v
}

fn stream_chunks(
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    mut events: ohmygpu_inference::InferenceStream,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let data = |v: Value| Ok::<Event, Infallible>(Event::default().data(v.to_string()));
        yield data(chunk(&id, created, &model, json!({"role": "assistant", "content": ""}), None, None));
        let mut finished = false;
        while let Some(item) = events.next().await {
            match item {
                Ok(StreamEvent::TextDelta { text }) => {
                    yield data(chunk(&id, created, &model, json!({"content": text}), None, None));
                }
                Ok(StreamEvent::ToolCallStart { index, id: call_id, name }) => {
                    yield data(chunk(&id, created, &model, json!({"tool_calls": [{
                        "index": index, "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": ""}
                    }]}), None, None));
                }
                Ok(StreamEvent::ToolCallArgumentsDelta { index, delta }) => {
                    yield data(chunk(&id, created, &model, json!({"tool_calls": [{
                        "index": index, "function": {"arguments": delta}
                    }]}), None, None));
                }
                Ok(StreamEvent::Completed { finish_reason, usage }) => {
                    finished = true;
                    yield data(chunk(&id, created, &model, json!({}), Some(finish_reason.as_str()), None));
                    if include_usage {
                        let u: UsageObject = usage.into();
                        let mut v = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "system_fingerprint": fingerprint(), "choices": [],
                        });
                        v["usage"] = serde_json::to_value(u).unwrap();
                        yield data(v);
                    }
                }
                Err(e) => {
                    let api: ApiError = e.into();
                    yield data(serde_json::to_value(api.envelope()).unwrap());
                    finished = true;
                    break;
                }
            }
        }
        if !finished {
            yield data(chunk(&id, created, &model, json!({}), Some(FinishReason::Stop.as_str()), None));
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> ChatCompletionRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn maps_roles_tools_and_options() {
        let req = parse(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "42"},
                {"role": "developer", "content": "dev"}
            ],
            "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
            "temperature": 0.5, "top_p": 0.9, "max_completion_tokens": 77, "stop": "END", "seed": 7,
            "presence_penalty": 0.1, "frequency_penalty": 0.2, "stream": true
        }));
        let ireq = to_inference_request(req).unwrap();
        assert_eq!(ireq.model, "m");
        assert_eq!(ireq.input.len(), 5);
        assert_eq!(ireq.input[0], InputItem::system("be brief"));
        assert_eq!(ireq.input[1], InputItem::user("hi there"));
        assert_eq!(
            ireq.input[2],
            InputItem::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "f".into(),
                arguments: "{\"a\":1}".into()
            })
        );
        assert_eq!(
            ireq.input[3],
            InputItem::ToolResult {
                call_id: "call_1".into(),
                output: "42".into()
            }
        );
        assert_eq!(ireq.input[4], InputItem::system("dev"));
        assert_eq!(ireq.tools.len(), 1);
        assert_eq!(ireq.tool_choice, ToolChoice::Named { name: "f".into() });
        assert_eq!(ireq.options.max_output_tokens, Some(77));
        assert_eq!(ireq.options.stop, vec!["END".to_string()]);
        assert_eq!(ireq.options.seed, Some(7));
        assert_eq!(ireq.options.presence_penalty, Some(0.1));
    }

    #[test]
    fn rejects_unsupported_features_with_400() {
        let e = to_inference_request(parse(
            json!({"model": "m", "messages": [{"role":"user","content":"x"}], "n": 2}),
        ))
        .unwrap_err();
        assert_eq!(e.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(e.body.code, "unsupported");
        let e = to_inference_request(parse(json!({"model": "m", "messages": [{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"x","format":"wav"}}]}]}))).unwrap_err();
        assert_eq!(e.body.code, "unsupported");
        // images parse into image parts (the vision gate runs later, in the handler)
        let r = to_inference_request(parse(json!({"model": "m", "messages": [{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}))).unwrap();
        assert_eq!(r.input[0].image_count(), 1);
        assert_eq!(r.input[0].text().as_deref(), Some("what is this?"));
        // …but only in user messages
        let e = to_inference_request(parse(json!({"model": "m", "messages": [{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}))).unwrap_err();
        assert_eq!(e.body.code, "invalid_request");
        let e = to_inference_request(parse(json!({"model": "m", "messages": [{"role":"user","content":"x"}], "response_format": {"type":"json_object"}}))).unwrap_err();
        assert_eq!(e.body.code, "unsupported");
        let e = to_inference_request(parse(json!({"model": "m", "messages": []}))).unwrap_err();
        assert_eq!(e.body.param.as_deref(), Some("messages"));
        let e = to_inference_request(parse(
            json!({"model": "m", "messages": [{"role":"tool","content":"x"}]}),
        ))
        .unwrap_err();
        assert!(e.body.message.contains("tool_call_id"));
    }

    #[test]
    fn serializes_completion_with_tool_calls() {
        let resp = InferenceResponse {
            model: "m".into(),
            output: vec![OutputItem::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "f".into(),
                arguments: "{}".into(),
            })],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let v = serde_json::to_value(to_chat_completion("chatcmpl-x".into(), 1, resp)).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "f"
        );
        assert_eq!(v["usage"]["total_tokens"], 15);
    }
}
