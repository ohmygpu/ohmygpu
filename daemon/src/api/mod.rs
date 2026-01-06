pub mod chat;
pub mod models;

use axum::{routing::get, Router};

use crate::state::AppState;
use std::sync::Arc;

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", axum::routing::post(chat::chat_completions))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}
