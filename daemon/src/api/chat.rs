use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};

use crate::state::AppState;
use ohmygpu_runtime_api::{ChatMessage, ChatRequest, Runtime};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessageInput>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Deserialize)]
pub struct ChatMessageInput {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOutput,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatMessageOutput {
    pub role: &'static str,
    pub content: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Serialize)]
pub struct ChatChoiceDelta {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Auto-load model if not loaded
    if let Err(e) = state.load_model(&request.model).await {
        tracing::error!("Failed to load model {}: {}", request.model, e);
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to load model '{}': {}", request.model, e),
                    r#type: "invalid_request_error",
                },
            }),
        )
            .into_response();
    }

    if request.stream {
        chat_completions_stream(state, request).await.into_response()
    } else {
        chat_completions_non_stream(state, request)
            .await
            .into_response()
    }
}

async fn chat_completions_non_stream(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let runtime = state.runtime.read().await;

    let chat_request = ChatRequest {
        messages: request
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        stream: false,
    };

    match runtime.chat(chat_request).await {
        Ok(response) => {
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;
            Ok(Json(ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid_simple()),
                object: "chat.completion",
                created,
                model: request.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessageOutput {
                        role: "assistant",
                        content: response.content,
                    },
                    finish_reason: response.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: response.tokens_used,
                    total_tokens: response.tokens_used,
                },
            }))
        }
        Err(e) => {
            tracing::error!("Chat error: {}", e);
            Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Generation error: {}", e),
                        r#type: "server_error",
                    },
                }),
            ))
        }
    }
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-{}", uuid_simple());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let model = request.model.clone();

    let runtime = state.runtime.clone();

    let stream = async_stream::stream! {
        // Send initial chunk with role
        let initial_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChoiceDelta {
                index: 0,
                delta: Delta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&initial_chunk).unwrap()));

        let chat_request = ChatRequest {
            messages: request
                .messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: true,
        };

        let runtime_guard = runtime.read().await;
        match runtime_guard.chat_stream(chat_request).await {
            Ok(mut rx) => {
                while let Some(token) = rx.recv().await {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChatChoiceDelta {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: if token.content.is_empty() { None } else { Some(token.content) },
                            },
                            finish_reason: token.finish_reason,
                        }],
                    };
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                }
            }
            Err(e) => {
                tracing::error!("Stream error: {}", e);
            }
        }

        // Send [DONE] marker
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", nanos)
}
