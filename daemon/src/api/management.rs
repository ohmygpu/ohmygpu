//! `/ohmygpu/v1/*` — local runtime management (what third-party applications
//! and the CLI use to control the runtime). Separate namespace from `/v1/*`.

use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ohmygpu_core::catalog::{self, CatalogEntry};
use ohmygpu_core::lifecycle::ModelState;
use ohmygpu_runtime_api::BackendAvailability;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::ApiJson;
use crate::error::ApiError;
use crate::manager::{ModelView, StartOptions};
use crate::state::SharedState;

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": ohmygpu_core::VERSION }))
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub uptime_seconds: u64,
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub data_dir: String,
    pub backend: BackendStatus,
    pub models: ModelsSummary,
    pub hardware_backend: String,
}

#[derive(Debug, Serialize)]
pub struct BackendStatus {
    pub id: &'static str,
    #[serde(flatten)]
    pub availability: BackendAvailability,
}

#[derive(Debug, Serialize)]
pub struct ModelsSummary {
    pub installed: usize,
    pub running: Vec<String>,
    pub downloading: Vec<String>,
}

pub async fn status(State(state): State<SharedState>) -> Json<StatusResponse> {
    let availability = state.manager.backend_availability().await;
    Json(StatusResponse {
        status: "ok",
        version: ohmygpu_core::VERSION,
        uptime_seconds: state.uptime_seconds(),
        pid: std::process::id(),
        host: state.host.clone(),
        port: state.port,
        data_dir: state.paths.base_dir().display().to_string(),
        backend: BackendStatus {
            id: state.manager.backend().id(),
            availability,
        },
        models: ModelsSummary {
            installed: state.manager.installed_count(),
            running: state.manager.running_ids(),
            downloading: state.manager.downloading_ids(),
        },
        hardware_backend: state.hardware.backend.clone(),
    })
}

pub async fn hardware(
    State(state): State<SharedState>,
) -> Json<ohmygpu_core::hardware::HardwareInfo> {
    Json(state.hardware.clone())
}

pub async fn backend(State(state): State<SharedState>) -> Json<Value> {
    let availability = state.manager.backend_availability().await;
    Json(json!({ "id": state.manager.backend().id(), "availability": availability }))
}

/// Install/verify the backend binary now (otherwise it happens on first start).
pub async fn backend_install(State(state): State<SharedState>) -> Result<Json<Value>, ApiError> {
    let availability = state.manager.backend().prepare(None).await.map_err(|e| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "server_error",
            "backend_install_failed",
            e.to_string(),
        )
    })?;
    Ok(Json(
        json!({ "id": state.manager.backend().id(), "availability": availability }),
    ))
}

#[derive(Debug, Serialize)]
pub struct CatalogModel {
    #[serde(flatten)]
    pub entry: &'static CatalogEntry,
    pub installed: bool,
    pub state: String,
}

pub async fn catalog(State(state): State<SharedState>) -> Json<Value> {
    let models: Vec<CatalogModel> = catalog::CATALOG
        .iter()
        .map(|entry| {
            let view = state.manager.get(entry.id);
            CatalogModel {
                entry,
                installed: view.as_ref().map(|v| v.installed).unwrap_or(false),
                state: view
                    .map(|v| v.state)
                    .unwrap_or_else(|| ModelState::NotInstalled.name().to_string()),
            }
        })
        .collect();
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize, Default)]
pub struct ListQuery {
    #[serde(default)]
    pub installed: Option<bool>,
}

pub async fn list_models(
    State(state): State<SharedState>,
    Query(q): Query<ListQuery>,
) -> Json<Value> {
    let mut models = state.manager.list();
    if q.installed == Some(true) {
        models.retain(|m| m.installed);
    }
    Json(json!({ "models": models }))
}

pub async fn get_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<ModelView>, ApiError> {
    state
        .manager
        .get(&id)
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("model '{id}' not found")))
}

#[derive(Debug, Deserialize)]
pub struct PullRequest {
    /// Catalog id, `hf:owner/repo/file.gguf`, or a huggingface.co URL.
    pub model: String,
    /// Optional id to install a non-catalog model under.
    #[serde(default)]
    pub id: Option<String>,
    /// Multimodal projector for a non-catalog vision model: a `.gguf` file name in
    /// the same Hugging Face repo, or a full URL. Makes the model accept images.
    #[serde(default)]
    pub mmproj: Option<String>,
}

/// 202 while downloading, 200 if it was already installed.
pub async fn pull_model(
    State(state): State<SharedState>,
    ApiJson(req): ApiJson<PullRequest>,
) -> Result<Response, ApiError> {
    let view = state
        .manager
        .pull(&req.model, req.id.as_deref(), req.mmproj.as_deref())?;
    let code = if view.state == "downloading" {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    Ok((code, Json(json!({ "model": view }))).into_response())
}

pub async fn delete_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let view = state.manager.delete(&id).await?;
    Ok(Json(json!({ "id": id, "deleted": true, "model": view })))
}

#[derive(Debug, Deserialize, Default)]
pub struct StartQuery {
    /// Block until the model is running (or failed) instead of returning 202.
    #[serde(default)]
    pub wait: Option<bool>,
    /// Max seconds to wait when `wait=true` (default: backend startup timeout).
    #[serde(default)]
    pub timeout: Option<u64>,
}

/// 202 `starting` (default) or, with `?wait=true`, 200 `running` / 502 error.
pub async fn start_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(q): Query<StartQuery>,
    body: Option<ApiJson<StartOptions>>,
) -> Result<Response, ApiError> {
    let opts = body.map(|ApiJson(o)| o).unwrap_or_default();
    let view = state.manager.start(&id, opts)?;
    if !q.wait.unwrap_or(false) {
        let code = if view.state == "running" {
            StatusCode::OK
        } else {
            StatusCode::ACCEPTED
        };
        return Ok((code, Json(json!({ "model": view }))).into_response());
    }
    let timeout = Duration::from_secs(
        q.timeout
            .unwrap_or(state.config.backend.llamacpp.startup_timeout_secs)
            .max(1),
    );
    let final_state = state.manager.wait_started(&id, timeout).await;
    let view = state
        .manager
        .get(&id)
        .ok_or_else(|| ApiError::not_found(format!("model '{id}' not found")))?;
    match final_state {
        Some(ModelState::Running) => {
            Ok((StatusCode::OK, Json(json!({ "model": view }))).into_response())
        }
        Some(ModelState::Error { message }) => Err(ApiError::new(
            StatusCode::BAD_GATEWAY,
            "server_error",
            "model_start_failed",
            message,
        )),
        Some(ModelState::Starting { .. }) => {
            Ok((StatusCode::ACCEPTED, Json(json!({ "model": view }))).into_response())
        }
        Some(other) => Err(ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "invalid_state",
            format!("model '{id}' ended up {} instead of running", other.name()),
        )),
        None => Err(ApiError::not_found(format!("model '{id}' not found"))),
    }
}

pub async fn stop_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let view = state.manager.stop(&id).await?;
    Ok(Json(json!({ "model": view })))
}

/// Graceful shutdown: stop all models, close the server.
pub async fn shutdown(State(state): State<SharedState>) -> (StatusCode, Json<Value>) {
    tracing::info!("shutdown requested via API");
    state.request_shutdown();
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "shutting_down" })),
    )
}
