//! HTTP surface of the runtime.
//!
//! ```text
//! /v1/*          inference compatibility (OpenAI-style)   → protocol adapters
//! /ohmygpu/v1/*  local runtime management                 → ModelManager
//! ```

pub mod chat_completions;
pub mod images;
pub mod management;
pub mod models;
pub mod responses;

use axum::async_trait;
use axum::extract::{DefaultBodyLimit, FromRequest, Request};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::error::ApiError;
use crate::state::SharedState;

/// Build the complete router for a daemon state.
pub fn router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    Router::new()
        // legacy/simple health check
        .route("/health", get(management::health))
        // --- inference (OpenAI-compatible subset) ---
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/:id", get(models::get_model))
        .route("/v1/chat/completions", post(chat_completions::create))
        .route("/v1/responses", post(responses::create))
        // --- management ---
        .route("/ohmygpu/v1/health", get(management::health))
        .route("/ohmygpu/v1/status", get(management::status))
        .route("/ohmygpu/v1/hardware", get(management::hardware))
        .route("/ohmygpu/v1/backend", get(management::backend))
        .route(
            "/ohmygpu/v1/backend/install",
            post(management::backend_install),
        )
        .route("/ohmygpu/v1/catalog", get(management::catalog))
        .route("/ohmygpu/v1/models", get(management::list_models))
        .route("/ohmygpu/v1/models/pull", post(management::pull_model))
        .route("/ohmygpu/v1/models/:id", get(management::get_model))
        .route("/ohmygpu/v1/models/:id", delete(management::delete_model))
        .route(
            "/ohmygpu/v1/models/:id/start",
            post(management::start_model),
        )
        .route("/ohmygpu/v1/models/:id/stop", post(management::stop_model))
        .route("/ohmygpu/v1/shutdown", post(management::shutdown))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// `Json<T>` whose rejection is our OpenAI-style error envelope.
pub struct ApiJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(rej) => Err(ApiError::from(rej)),
        }
    }
}

/// Seconds since the Unix epoch.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `prefix_<24 hex>` identifiers (chatcmpl-…, resp_…, msg_…, fc_…, call_…).
pub fn new_id(prefix: &str) -> String {
    let u = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}{}", &u[..24])
}
