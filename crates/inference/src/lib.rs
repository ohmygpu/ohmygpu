//! ohmygpu_inference — the protocol-independent inference model.
//!
//! Every external API (`/v1/responses`, `/v1/chat/completions`) is translated by a
//! *protocol adapter* into [`InferenceRequest`], executed by a runtime adapter, and
//! the resulting [`InferenceResponse`] / [`StreamEvent`]s are translated back.
//! Runtime adapters never see OpenAI schemas; protocol adapters never see backend
//! wire formats. This crate is the only thing both sides share.
//!
//! The model deliberately covers what makes sense for local inference and nothing
//! more: text in/out, tool definitions + tool calls (executed by the *application*,
//! never by OhMyGPU), and a small set of generation options.

use std::pin::Pin;

use futures_util::Stream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One item of conversation input, in order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// A plain text message.
    Message { role: Role, content: String },
    /// A tool call previously emitted by the assistant (echoed back so the model
    /// sees its own call before the result).
    ToolCall(ToolCall),
    /// The application's result for a previous tool call.
    ToolResult { call_id: String, output: String },
}

impl InputItem {
    pub fn system(content: impl Into<String>) -> Self {
        InputItem::Message {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        InputItem::Message {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        InputItem::Message {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A function tool the *application* offers to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments object.
    #[serde(default = "default_parameters")]
    pub parameters: serde_json::Value,
}

fn default_parameters() -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": {} })
}

/// How the model may use tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides.
    #[default]
    Auto,
    /// Never call tools.
    None,
    /// Must call at least one tool.
    Required,
    /// Must call this specific tool.
    Named { name: String },
}

/// Sampling / length controls. `None` means "backend default".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
}

/// The single internal request type. Both public APIs produce exactly this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub input: Vec<InputItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)]
    pub options: GenerationOptions,
}

impl InferenceRequest {
    pub fn new(model: impl Into<String>, input: Vec<InputItem>) -> Self {
        Self {
            model: model.into(),
            input,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            options: GenerationOptions::default(),
        }
    }

    /// Validate what every backend needs regardless of protocol.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.model.trim().is_empty() {
            return Err(InferenceError::InvalidRequest("`model` is required".into()));
        }
        if self.input.is_empty() {
            return Err(InferenceError::InvalidRequest(
                "input must not be empty".into(),
            ));
        }
        if let Some(t) = self.options.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(InferenceError::InvalidRequest(
                    "temperature must be between 0 and 2".into(),
                ));
            }
        }
        if let Some(p) = self.options.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(InferenceError::InvalidRequest(
                    "top_p must be between 0 and 1".into(),
                ));
            }
        }
        if let Some(0) = self.options.max_output_tokens {
            return Err(InferenceError::InvalidRequest(
                "max output tokens must be > 0".into(),
            ));
        }
        if let ToolChoice::Named { name } = &self.tool_choice {
            if !self.tools.iter().any(|t| &t.name == name) {
                return Err(InferenceError::InvalidRequest(format!(
                    "tool_choice names unknown tool '{name}'"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Response side
// ---------------------------------------------------------------------------

/// A tool call the model wants the application to execute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments object (kept as a string, exactly as produced).
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Text { text: String },
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of generation (or a stop sequence).
    Stop,
    /// `max_output_tokens` reached.
    Length,
    /// Generation ended with tool call(s) for the application to run.
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model: String,
    pub output: Vec<OutputItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl InferenceResponse {
    /// All text output concatenated (convenience for adapters/tests).
    pub fn text(&self) -> String {
        self.output
            .iter()
            .filter_map(|o| match o {
                OutputItem::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.output
            .iter()
            .filter_map(|o| match o {
                OutputItem::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Incremental events produced by a runtime adapter while generating.
///
/// Tool calls are streamed as `ToolCallStart` (index, id, name) followed by zero or
/// more `ToolCallArgumentsDelta` for the same index. A `Completed` event always
/// terminates a successful stream; errors terminate it via `Err(InferenceError)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    ToolCallStart {
        index: u32,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: u32,
        delta: String,
    },
    Completed {
        finish_reason: FinishReason,
        usage: Usage,
    },
}

pub type InferenceStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, InferenceError>> + Send + 'static>>;

/// Folds a sequence of stream events into a full [`InferenceResponse`].
///
/// This is how non-streaming inference is derived from streaming, so there is only
/// one code path through every backend.
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    text: String,
    tool_calls: Vec<(u32, ToolCall)>,
    finish: Option<(FinishReason, Usage)>,
}

impl ResponseAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta { text } => self.text.push_str(text),
            StreamEvent::ToolCallStart { index, id, name } => {
                self.tool_calls.push((
                    *index,
                    ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    },
                ));
            }
            StreamEvent::ToolCallArgumentsDelta { index, delta } => {
                if let Some((_, call)) = self.tool_calls.iter_mut().find(|(i, _)| i == index) {
                    call.arguments.push_str(delta);
                }
            }
            StreamEvent::Completed {
                finish_reason,
                usage,
            } => {
                self.finish = Some((*finish_reason, *usage));
            }
        }
    }

    pub fn finish(self, model: impl Into<String>) -> InferenceResponse {
        let mut output = Vec::new();
        if !self.text.is_empty() {
            output.push(OutputItem::Text { text: self.text });
        }
        let has_calls = !self.tool_calls.is_empty();
        let mut calls = self.tool_calls;
        calls.sort_by_key(|(i, _)| *i);
        output.extend(calls.into_iter().map(|(_, c)| OutputItem::ToolCall(c)));
        let (finish_reason, usage) = self.finish.unwrap_or((
            if has_calls {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            },
            Usage::default(),
        ));
        InferenceResponse {
            model: model.into(),
            output,
            finish_reason,
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that cross the boundary between adapters and backends.
///
/// Protocol adapters map these onto HTTP status codes and the OpenAI error
/// object; nothing here is protocol specific.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum InferenceError {
    /// The model id is unknown to this runtime (not installed, not in catalog).
    #[error("model '{0}' not found")]
    ModelNotFound(String),
    /// The model exists but is not running; `state` is the lifecycle state name.
    #[error("model '{model}' is not running (state: {state})")]
    ModelNotRunning { model: String, state: String },
    /// The request is malformed or asks for something unsupported.
    #[error("{0}")]
    InvalidRequest(String),
    /// The backend failed while serving the request.
    #[error("backend error: {0}")]
    Backend(String),
    /// The backend went away mid-request.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

impl InferenceError {
    /// A short machine-readable code, stable across versions.
    pub fn code(&self) -> &'static str {
        match self {
            InferenceError::ModelNotFound(_) => "model_not_found",
            InferenceError::ModelNotRunning { .. } => "model_not_running",
            InferenceError::InvalidRequest(_) => "invalid_request",
            InferenceError::Backend(_) => "backend_error",
            InferenceError::Unavailable(_) => "backend_unavailable",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_builds_text_and_tool_calls_in_index_order() {
        let mut acc = ResponseAccumulator::new();
        acc.push(&StreamEvent::TextDelta { text: "Hel".into() });
        acc.push(&StreamEvent::TextDelta { text: "lo".into() });
        acc.push(&StreamEvent::ToolCallStart {
            index: 1,
            id: "b".into(),
            name: "second".into(),
        });
        acc.push(&StreamEvent::ToolCallStart {
            index: 0,
            id: "a".into(),
            name: "first".into(),
        });
        acc.push(&StreamEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"x\":".into(),
        });
        acc.push(&StreamEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "1}".into(),
        });
        acc.push(&StreamEvent::Completed {
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 4,
            },
        });
        let resp = acc.finish("m");
        assert_eq!(resp.text(), "Hello");
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "first");
        assert_eq!(calls[0].arguments, "{\"x\":1}");
        assert_eq!(calls[1].name, "second");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
        assert_eq!(resp.usage.total_tokens(), 7);
    }

    #[test]
    fn accumulator_without_completed_event_infers_finish_reason() {
        let mut acc = ResponseAccumulator::new();
        acc.push(&StreamEvent::TextDelta { text: "x".into() });
        assert_eq!(acc.finish("m").finish_reason, FinishReason::Stop);

        let mut acc = ResponseAccumulator::new();
        acc.push(&StreamEvent::ToolCallStart {
            index: 0,
            id: "a".into(),
            name: "f".into(),
        });
        assert_eq!(acc.finish("m").finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn validate_rejects_bad_requests() {
        let mut req = InferenceRequest::new("m", vec![InputItem::user("hi")]);
        assert!(req.validate().is_ok());
        req.options.temperature = Some(3.0);
        assert!(matches!(
            req.validate(),
            Err(InferenceError::InvalidRequest(_))
        ));
        req.options.temperature = None;
        req.tool_choice = ToolChoice::Named {
            name: "nope".into(),
        };
        assert!(req.validate().is_err());
        assert!(InferenceRequest::new("", vec![InputItem::user("hi")])
            .validate()
            .is_err());
        assert!(InferenceRequest::new("m", vec![]).validate().is_err());
    }

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(
            InferenceError::ModelNotFound("x".into()).code(),
            "model_not_found"
        );
        assert_eq!(
            InferenceError::ModelNotRunning {
                model: "x".into(),
                state: "installed".into()
            }
            .code(),
            "model_not_running"
        );
        assert_eq!(
            InferenceError::InvalidRequest("x".into()).code(),
            "invalid_request"
        );
    }

    #[test]
    fn request_round_trips_through_json() {
        let mut req =
            InferenceRequest::new("m", vec![InputItem::system("s"), InputItem::user("u")]);
        req.tools.push(ToolDefinition {
            name: "get_weather".into(),
            description: Some("Weather".into()),
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
        });
        req.tool_choice = ToolChoice::Required;
        let json = serde_json::to_string(&req).unwrap();
        let back: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }
}
