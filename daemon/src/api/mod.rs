pub mod chat;
pub mod models;
pub mod ollama;

use axum::{routing::get, routing::post, Router};

use crate::state::AppState;
use std::sync::Arc;

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Health check
        .route("/health", get(health))
        // OpenAI-compatible API
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        // Ollama-compatible API (drop-in replacement)
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/tags", get(ollama::tags))
        .route("/api/version", get(ollama::version))
        .route("/api/show", post(ollama::show))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}
