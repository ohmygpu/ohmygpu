//! `GET /v1/models` and `GET /v1/models/{id}` — OpenAI-shaped list of the
//! models that can actually be used here (installed models), plus what an
//! application needs to pick one: `kind`, `state`, `capabilities`,
//! `modalities`, `context_length`. The runtime never picks a model for a
//! request — `model` is always explicit — so this is where that choice is
//! informed.

use axum::extract::{Path, State};
use axum::Json;
use ohmygpu_core::registry::{Modalities, ModelCapabilities};
use serde::Serialize;

use crate::api::now_secs;
use crate::error::ApiError;
use crate::manager::ModelView;
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// OhMyGPU extension: lifecycle state, so OpenAI clients can tell whether a
    /// model is running without a second call.
    pub state: String,
    /// OhMyGPU extension: `llm` or `whisper`.
    pub kind: String,
    /// OhMyGPU extension: `tools` (native tool calling), `vision` (image input).
    pub capabilities: ModelCapabilities,
    /// OhMyGPU extension: what goes in and what comes out, e.g.
    /// `{"input": ["text", "image"], "output": ["text"]}`.
    pub modalities: Modalities,
    /// OhMyGPU extension: the model's native context window in tokens (from the
    /// model file's metadata); absent when unknown. The window a running model
    /// actually serves is `runtime.context_length` in the Management API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

fn to_object(m: ModelView) -> ModelObject {
    ModelObject {
        id: m.id,
        object: "model",
        created: m
            .installed_at
            .map(|t| t.timestamp().max(0) as u64)
            .unwrap_or_else(now_secs),
        owned_by: "ohmygpu",
        state: m.state,
        kind: m.kind.as_str().to_string(),
        capabilities: m.capabilities,
        modalities: m.modalities,
        context_length: m.context_length,
    }
}

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let data = state
        .manager
        .installed()
        .into_iter()
        .map(to_object)
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}

pub async fn get_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    match state.manager.get(&id) {
        Some(m) if m.installed => Ok(Json(to_object(m))),
        _ => Err(ApiError::from(
            ohmygpu_inference::InferenceError::ModelNotFound(id),
        )),
    }
}
