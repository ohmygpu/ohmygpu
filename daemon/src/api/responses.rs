//! `POST /v1/responses` — the canonical, forward-looking protocol adapter
//! (OpenAI Responses API–compatible subset).
//!
//! Supported subset (documented in the README):
//!
//! * `model`, `input` (string, or items: `message` with text content,
//!   `function_call`, `function_call_output`), `instructions`
//! * `tools` (`function`), `tool_choice` (auto/none/required/named)
//! * `temperature`, `top_p`, `max_output_tokens`, `metadata` (echoed)
//! * `stream` with Responses-style events (`response.created`,
//!   `response.output_item.added`, `response.output_text.delta`, …,
//!   `response.completed`)
//!
//! Not supported (400): `previous_response_id`, `background`, hosted tools,
//! non-text input parts, non-text `text.format`. Responses are never stored.

use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{Stream, StreamExt};
use ohmygpu_inference::{
    FinishReason, GenerationOptions, InferenceRequest, InferenceResponse, InputItem, OutputItem,
    Role, StreamEvent, ToolCall, ToolChoice, ToolDefinition, Usage,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::api::{new_id, now_secs, ApiJson};
use crate::error::ApiError;
use crate::state::SharedState;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}

/// Request fields echoed back on the response object.
#[derive(Debug, Clone)]
pub struct Echo {
    pub instructions: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub metadata: Value,
    pub parallel_tool_calls: bool,
}

fn text_from_content(content: &Value, what: &str) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                let kind = p.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match kind {
                    "input_text" | "output_text" | "text" => {
                        out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    other => {
                        return Err(ApiError::unsupported(format!(
                            "content part type '{other}' (in {what})"
                        ))
                        .with_param("input"))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::invalid(format!(
            "{what} content must be a string or an array of text parts"
        ))
        .with_param("input")),
    }
}

fn parse_role(role: &str) -> Result<Role, ApiError> {
    Ok(match role {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" | "developer" => Role::System,
        other => {
            return Err(
                ApiError::invalid(format!("unknown message role '{other}'")).with_param("input")
            )
        }
    })
}

fn parse_input_items(input: Option<Value>) -> Result<Vec<InputItem>, ApiError> {
    let mut items = Vec::new();
    match input {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => items.push(InputItem::user(s)),
        Some(Value::Array(arr)) => {
            for item in arr {
                let obj = item.as_object().ok_or_else(|| {
                    ApiError::invalid("input items must be objects").with_param("input")
                })?;
                let kind = obj.get("type").and_then(|t| t.as_str());
                let role = obj.get("role").and_then(|r| r.as_str());
                match (kind, role) {
                    (Some("message"), Some(role)) | (None, Some(role)) => {
                        let role = parse_role(role)?;
                        let content = text_from_content(
                            obj.get("content").unwrap_or(&Value::Null),
                            "message",
                        )?;
                        items.push(InputItem::Message { role, content });
                    }
                    (Some("function_call"), _) => {
                        let call_id =
                            obj.get("call_id").and_then(|v| v.as_str()).ok_or_else(|| {
                                ApiError::invalid("function_call items require `call_id`")
                                    .with_param("input")
                            })?;
                        let name = obj.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                            ApiError::invalid("function_call items require `name`")
                                .with_param("input")
                        })?;
                        let arguments = match obj.get("arguments") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) if !other.is_null() => other.to_string(),
                            _ => "{}".to_string(),
                        };
                        items.push(InputItem::ToolCall(ToolCall {
                            id: call_id.to_string(),
                            name: name.to_string(),
                            arguments,
                        }));
                    }
                    (Some("function_call_output"), _) => {
                        let call_id =
                            obj.get("call_id").and_then(|v| v.as_str()).ok_or_else(|| {
                                ApiError::invalid("function_call_output items require `call_id`")
                                    .with_param("input")
                            })?;
                        let output = match obj.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v @ Value::Array(_)) => {
                                text_from_content(v, "function_call_output")?
                            }
                            Some(other) if !other.is_null() => other.to_string(),
                            _ => String::new(),
                        };
                        items.push(InputItem::ToolResult {
                            call_id: call_id.to_string(),
                            output,
                        });
                    }
                    // Clients often echo whole previous outputs; reasoning items carry nothing we can use.
                    (Some("reasoning"), _) => {}
                    (Some(other), _) => {
                        return Err(ApiError::unsupported(format!("input item type '{other}'"))
                            .with_param("input"))
                    }
                    (None, None) => {
                        return Err(ApiError::invalid("input item needs a `type` or a `role`")
                            .with_param("input"))
                    }
                }
            }
        }
        Some(_) => {
            return Err(
                ApiError::invalid("`input` must be a string or an array of items")
                    .with_param("input"),
            )
        }
    }
    Ok(items)
}

/// Responses request → internal request (+ the fields to echo back).
pub fn to_inference_request(req: ResponsesRequest) -> Result<(InferenceRequest, Echo), ApiError> {
    if req.model.trim().is_empty() {
        return Err(ApiError::invalid("`model` is required").with_param("model"));
    }
    if req.previous_response_id.is_some() {
        return Err(
            ApiError::unsupported("previous_response_id (responses are not stored)")
                .with_param("previous_response_id"),
        );
    }
    if req.background == Some(true) {
        return Err(ApiError::unsupported("background responses").with_param("background"));
    }
    if let Some(text) = &req.text {
        let kind = text
            .get("format")
            .and_then(|f| f.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("text");
        if kind != "text" {
            return Err(
                ApiError::unsupported(format!("text.format type '{kind}'")).with_param("text")
            );
        }
    }
    if let Some(m) = &req.metadata {
        if !m.is_object() && !m.is_null() {
            return Err(ApiError::invalid("`metadata` must be an object").with_param("metadata"));
        }
    }

    let mut input = Vec::new();
    if let Some(instr) = &req.instructions {
        if !instr.is_empty() {
            input.push(InputItem::system(instr.clone()));
        }
    }
    input.extend(parse_input_items(req.input)?);
    if input.is_empty() {
        return Err(ApiError::invalid("`input` must not be empty").with_param("input"));
    }

    let mut tools = Vec::new();
    let mut echoed_tools = Vec::new();
    for t in req.tools {
        let kind = t.get("type").and_then(|k| k.as_str()).unwrap_or("");
        if kind != "function" {
            return Err(ApiError::unsupported(format!("tool type '{kind}'")).with_param("tools"));
        }
        let name = t
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| ApiError::invalid("function tools require `name`").with_param("tools"))?
            .to_string();
        let description = t
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());
        let parameters = t
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        echoed_tools.push(json!({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
            "strict": t.get("strict").and_then(|s| s.as_bool()).unwrap_or(false),
        }));
        tools.push(ToolDefinition {
            name,
            description,
            parameters,
        });
    }

    let (tool_choice, echoed_choice) = match req.tool_choice {
        None => (ToolChoice::Auto, json!("auto")),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => (ToolChoice::Auto, json!("auto")),
            "none" => (ToolChoice::None, json!("none")),
            "required" => (ToolChoice::Required, json!("required")),
            other => {
                return Err(ApiError::invalid(format!("unknown tool_choice '{other}'"))
                    .with_param("tool_choice"))
            }
        },
        Some(v) => {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .or_else(|| {
                    v.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                })
                .ok_or_else(|| {
                    ApiError::invalid(
                        "tool_choice object must be {\"type\":\"function\",\"name\":...}",
                    )
                    .with_param("tool_choice")
                })?
                .to_string();
            (
                ToolChoice::Named { name: name.clone() },
                json!({"type": "function", "name": name}),
            )
        }
    };

    let ireq = InferenceRequest {
        model: req.model,
        input,
        tools,
        tool_choice,
        options: GenerationOptions {
            max_output_tokens: req.max_output_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            ..Default::default()
        },
    };
    ireq.validate().map_err(ApiError::from)?;
    let echo = Echo {
        instructions: req.instructions,
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_output_tokens,
        tools: echoed_tools,
        tool_choice: echoed_choice,
        metadata: req.metadata.unwrap_or_else(|| Value::Object(Map::new())),
        parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
    };
    Ok((ireq, echo))
}

// ---------------------------------------------------------------------------
// Response object
// ---------------------------------------------------------------------------

fn usage_json(u: Usage) -> Value {
    json!({
        "input_tokens": u.input_tokens,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens": u.output_tokens,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": u.total_tokens(),
    })
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "status": status,
        "role": "assistant",
        "content": [ { "type": "output_text", "text": text, "annotations": [], "logprobs": [] } ],
    })
}

fn function_call_item(id: &str, call: &ToolCall, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "status": status,
    })
}

/// The `response` object shared by non-streaming and streaming paths.
#[allow(clippy::too_many_arguments)]
pub fn response_object(
    id: &str,
    created_at: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Option<Usage>,
    incomplete_reason: Option<&str>,
    error: Option<Value>,
    echo: &Echo,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": false,
        "error": error,
        "incomplete_details": incomplete_reason.map(|r| json!({ "reason": r })),
        "instructions": echo.instructions,
        "max_output_tokens": echo.max_output_tokens,
        "max_tool_calls": null,
        "model": model,
        "output": output,
        "parallel_tool_calls": echo.parallel_tool_calls,
        "previous_response_id": null,
        "prompt_cache_key": null,
        "reasoning": { "effort": null, "summary": null },
        "safety_identifier": null,
        "service_tier": "default",
        "store": false,
        "temperature": echo.temperature,
        "text": { "format": { "type": "text" }, "verbosity": null },
        "tool_choice": echo.tool_choice,
        "tools": echo.tools,
        "top_logprobs": 0,
        "top_p": echo.top_p,
        "truncation": "disabled",
        "usage": usage.map(usage_json),
        "user": null,
        "metadata": echo.metadata,
    })
}

/// Internal response → completed Responses object.
pub fn to_response(id: &str, created_at: u64, resp: &InferenceResponse, echo: &Echo) -> Value {
    let mut output = Vec::new();
    let text = resp.text();
    if !text.is_empty() {
        output.push(message_item(&new_id("msg_"), &text, "completed"));
    }
    for item in &resp.output {
        if let OutputItem::ToolCall(call) = item {
            output.push(function_call_item(&new_id("fc_"), call, "completed"));
        }
    }
    let (status, incomplete) = match resp.finish_reason {
        FinishReason::Length => ("incomplete", Some("max_output_tokens")),
        _ => ("completed", None),
    };
    response_object(
        id,
        created_at,
        &resp.model,
        status,
        output,
        Some(resp.usage),
        incomplete,
        None,
        echo,
    )
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub async fn create(
    State(state): State<SharedState>,
    ApiJson(req): ApiJson<ResponsesRequest>,
) -> Result<Response, ApiError> {
    let stream = req.stream.unwrap_or(false);
    let (ireq, echo) = to_inference_request(req)?;
    let instance = state.manager.instance_for(&ireq.model).await?;
    let id = new_id("resp_");
    let created_at = now_secs();

    if !stream {
        let resp = instance.infer(ireq).await?;
        return Ok(Json(to_response(&id, created_at, &resp, &echo)).into_response());
    }

    let model = ireq.model.clone();
    let events = instance.infer_stream(ireq).await?;
    let streamer = ResponseStreamer::new(id, created_at, model, echo);
    Ok(Sse::new(stream_events(streamer, events))
        .keep_alive(KeepAlive::default())
        .into_response())
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

struct OpenText {
    item_id: String,
    output_index: usize,
    text: String,
}

struct OpenCall {
    item_id: String,
    output_index: usize,
    call: ToolCall,
}

/// Turns internal `StreamEvent`s into Responses SSE events, tracking open output
/// items and sequence numbers.
pub struct ResponseStreamer {
    id: String,
    created_at: u64,
    model: String,
    echo: Echo,
    seq: u64,
    output: Vec<Value>,
    next_output_index: usize,
    text: Option<OpenText>,
    calls: Vec<(u32, OpenCall)>,
}

impl ResponseStreamer {
    pub fn new(id: String, created_at: u64, model: String, echo: Echo) -> Self {
        Self {
            id,
            created_at,
            model,
            echo,
            seq: 0,
            output: Vec::new(),
            next_output_index: 0,
            text: None,
            calls: Vec::new(),
        }
    }

    fn event(&mut self, kind: &str, mut body: Value) -> (String, Value) {
        body["type"] = json!(kind);
        body["sequence_number"] = json!(self.seq);
        self.seq += 1;
        (kind.to_string(), body)
    }

    fn snapshot(
        &self,
        status: &str,
        incomplete: Option<&str>,
        usage: Option<Usage>,
        error: Option<Value>,
    ) -> Value {
        response_object(
            &self.id,
            self.created_at,
            &self.model,
            status,
            self.output.clone(),
            usage,
            incomplete,
            error,
            &self.echo,
        )
    }

    pub fn start(&mut self) -> Vec<(String, Value)> {
        let created = self.snapshot("in_progress", None, None, None);
        let e1 = self.event("response.created", json!({ "response": created }));
        let in_progress = self.snapshot("in_progress", None, None, None);
        let e2 = self.event("response.in_progress", json!({ "response": in_progress }));
        vec![e1, e2]
    }

    fn close_text(&mut self, out: &mut Vec<(String, Value)>) {
        if let Some(t) = self.text.take() {
            out.push(self.event(
                "response.output_text.done",
                json!({ "item_id": t.item_id, "output_index": t.output_index, "content_index": 0, "text": t.text, "logprobs": [] }),
            ));
            out.push(self.event(
                "response.content_part.done",
                json!({ "item_id": t.item_id, "output_index": t.output_index, "content_index": 0,
                        "part": { "type": "output_text", "text": t.text, "annotations": [], "logprobs": [] } }),
            ));
            let item = message_item(&t.item_id, &t.text, "completed");
            out.push(self.event(
                "response.output_item.done",
                json!({ "output_index": t.output_index, "item": item.clone() }),
            ));
            self.output.push(item);
        }
    }

    pub fn on_event(&mut self, ev: StreamEvent) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        match ev {
            StreamEvent::TextDelta { text } => {
                if self.text.is_none() {
                    let item_id = new_id("msg_");
                    let output_index = self.next_output_index;
                    self.next_output_index += 1;
                    out.push(self.event(
                        "response.output_item.added",
                        json!({ "output_index": output_index, "item": { "type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": [] } }),
                    ));
                    out.push(self.event(
                        "response.content_part.added",
                        json!({ "item_id": item_id, "output_index": output_index, "content_index": 0,
                                "part": { "type": "output_text", "text": "", "annotations": [], "logprobs": [] } }),
                    ));
                    self.text = Some(OpenText {
                        item_id,
                        output_index,
                        text: String::new(),
                    });
                }
                let (item_id, output_index) = {
                    let t = self.text.as_mut().unwrap();
                    t.text.push_str(&text);
                    (t.item_id.clone(), t.output_index)
                };
                out.push(self.event(
                    "response.output_text.delta",
                    json!({ "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": text, "logprobs": [] }),
                ));
            }
            StreamEvent::ToolCallStart { index, id, name } => {
                self.close_text(&mut out);
                let item_id = new_id("fc_");
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let call = ToolCall {
                    id,
                    name,
                    arguments: String::new(),
                };
                out.push(self.event(
                    "response.output_item.added",
                    json!({ "output_index": output_index, "item": function_call_item(&item_id, &call, "in_progress") }),
                ));
                self.calls.push((
                    index,
                    OpenCall {
                        item_id,
                        output_index,
                        call,
                    },
                ));
            }
            StreamEvent::ToolCallArgumentsDelta { index, delta } => {
                if let Some((_, c)) = self.calls.iter_mut().find(|(i, _)| *i == index) {
                    c.call.arguments.push_str(&delta);
                    let (item_id, output_index) = (c.item_id.clone(), c.output_index);
                    out.push(self.event(
                        "response.function_call_arguments.delta",
                        json!({ "item_id": item_id, "output_index": output_index, "delta": delta }),
                    ));
                }
            }
            StreamEvent::Completed {
                finish_reason,
                usage,
            } => {
                self.close_text(&mut out);
                let calls = std::mem::take(&mut self.calls);
                for (_, c) in calls {
                    out.push(self.event(
                        "response.function_call_arguments.done",
                        json!({ "item_id": c.item_id, "output_index": c.output_index, "arguments": c.call.arguments }),
                    ));
                    let item = function_call_item(&c.item_id, &c.call, "completed");
                    out.push(self.event(
                        "response.output_item.done",
                        json!({ "output_index": c.output_index, "item": item.clone() }),
                    ));
                    self.output.push(item);
                }
                let (kind, status, incomplete) = match finish_reason {
                    FinishReason::Length => (
                        "response.incomplete",
                        "incomplete",
                        Some("max_output_tokens"),
                    ),
                    _ => ("response.completed", "completed", None),
                };
                let resp = self.snapshot(status, incomplete, Some(usage), None);
                out.push(self.event(kind, json!({ "response": resp })));
            }
        }
        out
    }

    pub fn on_error(&mut self, err: &ApiError) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        self.close_text(&mut out);
        let error = json!({ "code": err.body.code, "message": err.body.message });
        let resp = self.snapshot("failed", None, None, Some(error));
        out.push(self.event("response.failed", json!({ "response": resp })));
        out.push(self.event(
            "error",
            json!({ "code": err.body.code, "message": err.body.message, "param": err.body.param }),
        ));
        out
    }
}

fn stream_events(
    mut streamer: ResponseStreamer,
    mut events: ohmygpu_inference::InferenceStream,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        for (kind, body) in streamer.start() {
            yield Ok::<Event, Infallible>(Event::default().event(kind).data(body.to_string()));
        }
        let mut done = false;
        while let Some(item) = events.next().await {
            match item {
                Ok(ev) => {
                    let terminal = matches!(ev, StreamEvent::Completed { .. });
                    for (kind, body) in streamer.on_event(ev) {
                        yield Ok(Event::default().event(kind).data(body.to_string()));
                    }
                    if terminal { done = true; break; }
                }
                Err(e) => {
                    let api: ApiError = e.into();
                    for (kind, body) in streamer.on_error(&api) {
                        yield Ok(Event::default().event(kind).data(body.to_string()));
                    }
                    done = true;
                    break;
                }
            }
        }
        if !done {
            for (kind, body) in streamer.on_event(StreamEvent::Completed { finish_reason: FinishReason::Stop, usage: Usage::default() }) {
                yield Ok(Event::default().event(kind).data(body.to_string()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> ResponsesRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn string_input_and_instructions_map_to_messages() {
        let (ireq, echo) = to_inference_request(parse(json!({
            "model": "m", "instructions": "be nice", "input": "hello", "temperature": 0.3, "max_output_tokens": 12,
            "metadata": {"k": "v"}
        })))
        .unwrap();
        assert_eq!(
            ireq.input,
            vec![InputItem::system("be nice"), InputItem::user("hello")]
        );
        assert_eq!(ireq.options.temperature, Some(0.3));
        assert_eq!(ireq.options.max_output_tokens, Some(12));
        assert_eq!(echo.metadata["k"], "v");
    }

    #[test]
    fn item_input_with_tools_and_tool_results() {
        let (ireq, echo) = to_inference_request(parse(json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "weather?"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                {"type": "reasoning", "summary": []}
            ],
            "tools": [{"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "get_weather"}
        })))
        .unwrap();
        assert_eq!(ireq.input.len(), 4);
        assert_eq!(ireq.input[1], InputItem::assistant("checking"));
        assert_eq!(
            ireq.input[2],
            InputItem::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into()
            })
        );
        assert_eq!(
            ireq.input[3],
            InputItem::ToolResult {
                call_id: "call_1".into(),
                output: "sunny".into()
            }
        );
        assert_eq!(
            ireq.tool_choice,
            ToolChoice::Named {
                name: "get_weather".into()
            }
        );
        assert_eq!(echo.tools[0]["name"], "get_weather");
    }

    #[test]
    fn rejects_unsupported_with_400() {
        let e = to_inference_request(parse(
            json!({"model": "m", "input": "x", "previous_response_id": "resp_1"}),
        ))
        .unwrap_err();
        assert_eq!(e.body.code, "unsupported");
        let e = to_inference_request(parse(
            json!({"model": "m", "input": [{"type": "input_image", "image_url": "x"}]}),
        ))
        .unwrap_err();
        assert_eq!(e.body.code, "unsupported");
        let e = to_inference_request(parse(
            json!({"model": "m", "input": "x", "tools": [{"type": "web_search_preview"}]}),
        ))
        .unwrap_err();
        assert_eq!(e.body.code, "unsupported");
        let e = to_inference_request(parse(json!({"model": "m"}))).unwrap_err();
        assert_eq!(e.body.param.as_deref(), Some("input"));
    }

    #[test]
    fn response_object_shape() {
        let (_, echo) = to_inference_request(parse(json!({"model": "m", "input": "x"}))).unwrap();
        let resp = InferenceResponse {
            model: "m".into(),
            output: vec![
                OutputItem::Text { text: "hi".into() },
                OutputItem::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "f".into(),
                    arguments: "{}".into(),
                }),
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 4,
            },
        };
        let v = to_response("resp_1", 42, &resp, &echo);
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["content"][0]["text"], "hi");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "call_1");
        assert_eq!(v["usage"]["total_tokens"], 7);
        assert_eq!(v["store"], false);
        assert!(v["error"].is_null());
    }

    #[test]
    fn streamer_emits_responses_events_in_order() {
        let (_, echo) = to_inference_request(parse(json!({"model": "m", "input": "x"}))).unwrap();
        let mut s = ResponseStreamer::new("resp_1".into(), 1, "m".into(), echo);
        let mut kinds: Vec<String> = s.start().into_iter().map(|(k, _)| k).collect();
        kinds.extend(
            s.on_event(StreamEvent::TextDelta { text: "Hel".into() })
                .into_iter()
                .map(|(k, _)| k),
        );
        kinds.extend(
            s.on_event(StreamEvent::TextDelta { text: "lo".into() })
                .into_iter()
                .map(|(k, _)| k),
        );
        kinds.extend(
            s.on_event(StreamEvent::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "f".into(),
            })
            .into_iter()
            .map(|(k, _)| k),
        );
        kinds.extend(
            s.on_event(StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: "{}".into(),
            })
            .into_iter()
            .map(|(k, _)| k),
        );
        let last = s.on_event(StreamEvent::Completed {
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
            },
        });
        kinds.extend(last.iter().map(|(k, _)| k.clone()));
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let (_, completed) = last.last().unwrap();
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
        assert_eq!(completed["response"]["output"][1]["arguments"], "{}");
        assert_eq!(completed["response"]["usage"]["total_tokens"], 3);
        // sequence numbers are contiguous
        assert_eq!(completed["sequence_number"], (kinds.len() - 1) as u64);
    }
}
