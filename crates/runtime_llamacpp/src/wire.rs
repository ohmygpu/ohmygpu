//! Translation between OhMyGPU's internal inference model and llama-server's
//! wire format (its OpenAI-style `/v1/chat/completions` endpoint).
//!
//! This is the *only* place that knows llama-server's JSON. Note that this is a
//! backend wire format, not OhMyGPU's public API: the public protocol adapters
//! live in the daemon and only ever see `ohmygpu_inference` types.

use ohmygpu_inference::{
    ContentPart, FinishReason, InferenceError, InferenceRequest, InputItem, StreamEvent,
    ToolChoice, Usage,
};
use serde::Deserialize;
use serde_json::{json, Value};

/// Text-only messages go over as a plain string (what every llama-server build
/// accepts); messages carrying images use the content-parts array that
/// llama-server's multimodal path (`--mmproj`) understands.
fn wire_content(parts: &[ContentPart]) -> Value {
    if parts.iter().all(|p| !p.is_image()) {
        let text: String = parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        return Value::String(text);
    }
    Value::Array(
        parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => json!({ "type": "text", "text": text }),
                ContentPart::Image { url } => {
                    json!({ "type": "image_url", "image_url": { "url": url } })
                }
            })
            .collect(),
    )
}

/// Build the JSON body for `POST /v1/chat/completions` (streaming).
pub fn build_request(req: &InferenceRequest) -> Value {
    let mut messages: Vec<Value> = Vec::with_capacity(req.input.len());

    for item in &req.input {
        match item {
            InputItem::Message { role, content } => {
                messages.push(json!({ "role": role.as_str(), "content": wire_content(content) }));
            }
            InputItem::ToolCall(call) => {
                let call_json = json!({
                    "id": call.id,
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments }
                });
                // Merge consecutive tool calls (optionally following assistant text)
                // into one assistant message, as chat templates expect.
                let merge = messages
                    .last()
                    .map(|m| m["role"] == "assistant" && m.get("_open").is_some())
                    .unwrap_or(false);
                if merge {
                    let last = messages.last_mut().unwrap();
                    last["tool_calls"].as_array_mut().unwrap().push(call_json);
                } else {
                    // If the previous message is plain assistant text, attach calls to it.
                    let attach = messages
                        .last()
                        .map(|m| m["role"] == "assistant" && m.get("tool_calls").is_none())
                        .unwrap_or(false);
                    if attach {
                        let last = messages.last_mut().unwrap();
                        last["tool_calls"] = json!([call_json]);
                        last["_open"] = json!(true);
                    } else {
                        messages.push(json!({
                            "role": "assistant",
                            "content": Value::Null,
                            "tool_calls": [call_json],
                            "_open": true
                        }));
                    }
                }
            }
            InputItem::ToolResult { call_id, output } => {
                // Close any open assistant tool-call message.
                if let Some(last) = messages.last_mut() {
                    if let Some(obj) = last.as_object_mut() {
                        obj.remove("_open");
                    }
                }
                messages
                    .push(json!({ "role": "tool", "tool_call_id": call_id, "content": output }));
            }
        }
    }
    for m in messages.iter_mut() {
        if let Some(obj) = m.as_object_mut() {
            obj.remove("_open");
        }
    }

    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });

    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description.clone().unwrap_or_default(),
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect(),
        );
        body["tool_choice"] = match &req.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Named { name } => {
                json!({ "type": "function", "function": { "name": name } })
            }
        };
    }

    let o = &req.options;
    if let Some(v) = o.max_output_tokens {
        body["max_tokens"] = json!(v);
    }
    if let Some(v) = o.temperature {
        body["temperature"] = json!(v);
    }
    if let Some(v) = o.top_p {
        body["top_p"] = json!(v);
    }
    if !o.stop.is_empty() {
        body["stop"] = json!(o.stop);
    }
    if let Some(v) = o.seed {
        body["seed"] = json!(v);
    }
    if let Some(v) = o.presence_penalty {
        body["presence_penalty"] = json!(v);
    }
    if let Some(v) = o.frequency_penalty {
        body["frequency_penalty"] = json!(v);
    }
    body
}

// ---------------------------------------------------------------------------
// Streaming response parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize, Default)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Incrementally converts llama-server SSE `data:` payloads into [`StreamEvent`]s.
///
/// llama-server streams tool calls as OpenAI-style deltas keyed by `index`; the
/// first delta for an index carries `id` and `function.name`, later ones carry
/// `function.arguments` fragments. We emit `ToolCallStart` once per index and
/// forward argument fragments verbatim.
#[derive(Debug, Default)]
pub struct StreamParser {
    started: Vec<u32>,
    next_index: u32,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    saw_tool_call: bool,
    completed: bool,
}

impl StreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one `data:` payload. `[DONE]` yields the terminal `Completed` event.
    pub fn feed(&mut self, data: &str) -> Result<Vec<StreamEvent>, InferenceError> {
        let data = data.trim();
        if data.is_empty() {
            return Ok(vec![]);
        }
        if data == "[DONE]" {
            return Ok(self.finish());
        }
        let chunk: Chunk = serde_json::from_str(data).map_err(|e| {
            InferenceError::Backend(format!(
                "invalid stream chunk from llama-server: {e}: {data}"
            ))
        })?;
        if let Some(err) = chunk.error {
            return Err(InferenceError::Backend(error_message(&err)));
        }
        let mut events = Vec::new();
        for choice in chunk.choices {
            if let Some(text) = choice.delta.content {
                if !text.is_empty() {
                    events.push(StreamEvent::TextDelta { text });
                }
            }
            for tc in choice.delta.tool_calls {
                let index = tc.index.unwrap_or(self.next_index);
                let func = tc.function.unwrap_or_default();
                if !self.started.contains(&index) {
                    self.started.push(index);
                    self.next_index = self.next_index.max(index + 1);
                    self.saw_tool_call = true;
                    let id = tc
                        .id
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| format!("call_{}", short_id()));
                    events.push(StreamEvent::ToolCallStart {
                        index,
                        id,
                        name: func.name.clone().unwrap_or_default(),
                    });
                }
                if let Some(args) = func.arguments {
                    if !args.is_empty() {
                        events.push(StreamEvent::ToolCallArgumentsDelta { index, delta: args });
                    }
                }
            }
            if let Some(fr) = choice.finish_reason {
                self.finish_reason = Some(match fr.as_str() {
                    "length" => FinishReason::Length,
                    "tool_calls" | "function_call" => FinishReason::ToolCalls,
                    _ => FinishReason::Stop,
                });
            }
        }
        if let Some(u) = chunk.usage {
            self.usage = Some(Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            });
        }
        Ok(events)
    }

    /// Terminal event (idempotent). Also used when the stream ends without `[DONE]`.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        if self.completed {
            return vec![];
        }
        self.completed = true;
        let finish_reason = self.finish_reason.unwrap_or(if self.saw_tool_call {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        });
        // A response that ended with tool calls is reported as such even if the
        // backend said "stop".
        let finish_reason = if self.saw_tool_call && finish_reason == FinishReason::Stop {
            FinishReason::ToolCalls
        } else {
            finish_reason
        };
        vec![StreamEvent::Completed {
            finish_reason,
            usage: self.usage.unwrap_or_default(),
        }]
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

/// llama-server error bodies: `{"error":{"code":..,"message":..,"type":..}}`.
pub fn error_message(err: &Value) -> String {
    err.get("message")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| err.to_string())
}

fn short_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ohmygpu_inference::{GenerationOptions, ToolCall, ToolDefinition};

    #[test]
    fn builds_messages_with_merged_tool_calls() {
        let req = InferenceRequest {
            model: "m".into(),
            input: vec![
                InputItem::system("sys"),
                InputItem::user("what's the weather?"),
                InputItem::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"Paris\"}".into(),
                }),
                InputItem::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: "get_time".into(),
                    arguments: "{}".into(),
                }),
                InputItem::ToolResult {
                    call_id: "c1".into(),
                    output: "sunny".into(),
                },
                InputItem::ToolResult {
                    call_id: "c2".into(),
                    output: "noon".into(),
                },
                InputItem::user("thanks"),
            ],
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: json!({"type":"object"}),
            }],
            tool_choice: ToolChoice::Auto,
            options: GenerationOptions {
                max_output_tokens: Some(50),
                temperature: Some(0.2),
                ..Default::default()
            },
        };
        let body = build_request(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert!(msgs[2].get("_open").is_none());
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "c1");
        assert_eq!(msgs[5]["content"], "thanks");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 50);
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn assistant_text_followed_by_tool_call_becomes_one_message() {
        let req = InferenceRequest::new(
            "m",
            vec![
                InputItem::user("hi"),
                InputItem::assistant("let me check"),
                InputItem::ToolCall(ToolCall {
                    id: "c".into(),
                    name: "f".into(),
                    arguments: "{}".into(),
                }),
                InputItem::ToolResult {
                    call_id: "c".into(),
                    output: "ok".into(),
                },
            ],
        );
        let body = build_request(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["content"], "let me check");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "c");
    }

    #[test]
    fn parses_text_stream() {
        let mut p = StreamParser::new();
        let mut ev = vec![];
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"content":"lo"}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#)
                .unwrap(),
        );
        ev.extend(p.feed("[DONE]").unwrap());
        assert_eq!(
            ev,
            vec![
                StreamEvent::TextDelta { text: "Hel".into() },
                StreamEvent::TextDelta { text: "lo".into() },
                StreamEvent::Completed {
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        input_tokens: 5,
                        output_tokens: 2
                    }
                },
            ]
        );
        assert!(p.finish().is_empty(), "finish is idempotent");
    }

    #[test]
    fn parses_tool_call_stream() {
        let mut p = StreamParser::new();
        let mut ev = vec![];
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}"#)
                .unwrap(),
        );
        ev.extend(
            p.feed(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#)
                .unwrap(),
        );
        ev.extend(p.feed("[DONE]").unwrap());
        assert_eq!(
            ev[0],
            StreamEvent::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "get_weather".into()
            }
        );
        assert_eq!(
            ev[1],
            StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: "{\"city\":".into()
            }
        );
        assert_eq!(
            ev[2],
            StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: "\"Paris\"}".into()
            }
        );
        assert!(matches!(
            ev[3],
            StreamEvent::Completed {
                finish_reason: FinishReason::ToolCalls,
                ..
            }
        ));
    }

    #[test]
    fn missing_tool_call_id_gets_generated() {
        let mut p = StreamParser::new();
        let ev = p.feed(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]}}]}"#).unwrap();
        match &ev[0] {
            StreamEvent::ToolCallStart { id, name, .. } => {
                assert!(id.starts_with("call_"));
                assert_eq!(name, "f");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stream_error_chunk_is_reported() {
        let mut p = StreamParser::new();
        let err = p.feed(r#"{"error":{"code":500,"message":"context shift is disabled","type":"server_error"}}"#).unwrap_err();
        assert!(matches!(err, InferenceError::Backend(m) if m.contains("context shift")));
    }
}

#[cfg(test)]
mod content_tests {
    use super::*;
    use ohmygpu_inference::Role;

    #[test]
    fn text_only_messages_stay_plain_strings_and_images_become_parts() {
        let req = InferenceRequest::new(
            "m",
            vec![
                InputItem::system("be brief"),
                InputItem::message(
                    Role::User,
                    vec![
                        ContentPart::text("what colour?"),
                        ContentPart::image("data:image/png;base64,AAAA"),
                    ],
                ),
            ],
        );
        let body = build_request(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "be brief");
        let parts = msgs[1]["content"].as_array().unwrap();
        assert_eq!(parts[0], json!({"type": "text", "text": "what colour?"}));
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }
}
