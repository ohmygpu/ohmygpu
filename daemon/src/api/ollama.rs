//! Ollama-compatible API endpoints for drop-in replacement compatibility.
//!
//! Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

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

// ============================================================================
// POST /api/chat - Ollama chat endpoint
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub num_predict: Option<u32>,
}

#[derive(Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaChatMessageOutput,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

#[derive(Serialize)]
pub struct OllamaChatMessageOutput {
    pub role: String,
    pub content: String,
}

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OllamaChatRequest>,
) -> Response {
    // Auto-load model if not loaded
    if let Err(e) = state.load_model(&request.model).await {
        tracing::error!("Failed to load model {}: {}", request.model, e);
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Failed to load model '{}': {}", request.model, e)
            })),
        )
            .into_response();
    }

    let stream = request.stream.unwrap_or(true); // Ollama defaults to streaming

    if stream {
        chat_stream(state, request).await.into_response()
    } else {
        chat_non_stream(state, request).await.into_response()
    }
}

async fn chat_non_stream(
    state: Arc<AppState>,
    request: OllamaChatRequest,
) -> Json<OllamaChatResponse> {
    let runtime = state.runtime.read().await;

    let options = request.options.unwrap_or_default();
    let chat_request = ChatRequest {
        messages: request
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect(),
        max_tokens: options.num_predict.unwrap_or(2048),
        temperature: options.temperature.unwrap_or(0.7),
        stream: false,
    };

    match runtime.chat(chat_request).await {
        Ok(response) => Json(OllamaChatResponse {
            model: request.model,
            created_at: chrono::Utc::now().to_rfc3339(),
            message: OllamaChatMessageOutput {
                role: "assistant".to_string(),
                content: response.content,
            },
            done: true,
            total_duration: None,
            eval_count: Some(response.tokens_used),
        }),
        Err(e) => {
            tracing::error!("Chat error: {}", e);
            Json(OllamaChatResponse {
                model: request.model,
                created_at: chrono::Utc::now().to_rfc3339(),
                message: OllamaChatMessageOutput {
                    role: "assistant".to_string(),
                    content: format!("Error: {}", e),
                },
                done: true,
                total_duration: None,
                eval_count: None,
            })
        }
    }
}

async fn chat_stream(
    state: Arc<AppState>,
    request: OllamaChatRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let model = request.model.clone();
    let runtime = state.runtime.clone();

    let stream = async_stream::stream! {
        let options = request.options.unwrap_or_default();
        let chat_request = ChatRequest {
            messages: request
                .messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect(),
            max_tokens: options.num_predict.unwrap_or(2048),
            temperature: options.temperature.unwrap_or(0.7),
            stream: true,
        };

        let runtime_guard = runtime.read().await;
        match runtime_guard.chat_stream(chat_request).await {
            Ok(mut rx) => {
                while let Some(token) = rx.recv().await {
                    let chunk = OllamaChatResponse {
                        model: model.clone(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                        message: OllamaChatMessageOutput {
                            role: "assistant".to_string(),
                            content: token.content,
                        },
                        done: token.finish_reason.is_some(),
                        total_duration: None,
                        eval_count: None,
                    };
                    // Ollama uses newline-delimited JSON, not SSE
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                }
            }
            Err(e) => {
                tracing::error!("Stream error: {}", e);
                let error_chunk = OllamaChatResponse {
                    model: model.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    message: OllamaChatMessageOutput {
                        role: "assistant".to_string(),
                        content: format!("Error: {}", e),
                    },
                    done: true,
                    total_duration: None,
                    eval_count: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&error_chunk).unwrap()));
            }
        }
    };

    Sse::new(stream)
}

// ============================================================================
// POST /api/generate - Ollama generate endpoint (non-chat completion)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OllamaGenerateRequest>,
) -> Response {
    // Convert generate request to chat request (single user message)
    let chat_request = OllamaChatRequest {
        model: request.model,
        messages: vec![OllamaChatMessage {
            role: "user".to_string(),
            content: request.prompt,
        }],
        stream: request.stream,
        options: request.options,
    };

    chat(State(state), Json(chat_request)).await
}

// ============================================================================
// GET /api/tags - List local models (Ollama format)
// ============================================================================

#[derive(Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Serialize)]
pub struct OllamaModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

pub async fn tags(State(state): State<Arc<AppState>>) -> Json<OllamaTagsResponse> {
    let registry = state.registry.read().await;
    let models: Vec<OllamaModelInfo> = registry
        .list()
        .iter()
        .map(|m| OllamaModelInfo {
            name: m.name.clone(),
            modified_at: chrono::Utc::now().to_rfc3339(), // TODO: actual modified time
            size: 0,                                       // TODO: actual size
            digest: "sha256:unknown".to_string(),
            details: OllamaModelDetails {
                format: "safetensors".to_string(),
                family: "unknown".to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: "unknown".to_string(),
            },
        })
        .collect();

    Json(OllamaTagsResponse { models })
}

// ============================================================================
// GET /api/version - Version info
// ============================================================================

#[derive(Serialize)]
pub struct OllamaVersionResponse {
    pub version: String,
}

pub async fn version() -> Json<OllamaVersionResponse> {
    Json(OllamaVersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// ============================================================================
// POST /api/show - Show model information
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OllamaShowRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: OllamaModelDetails,
}

pub async fn show(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OllamaShowRequest>,
) -> Response {
    let registry = state.registry.read().await;

    if let Some(_model) = registry.list().iter().find(|m| m.name == request.name) {
        Json(OllamaShowResponse {
            modelfile: format!("FROM {}", request.name),
            parameters: "".to_string(),
            template: "{{ .Prompt }}".to_string(),
            details: OllamaModelDetails {
                format: "safetensors".to_string(),
                family: "unknown".to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: "unknown".to_string(),
            },
        })
        .into_response()
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("model '{}' not found", request.name)
            })),
        )
            .into_response()
    }
}
