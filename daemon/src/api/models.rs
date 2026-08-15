//! `GET /v1/models` and `GET /v1/models/{id}` — OpenAI-shaped list of the
//! models that can actually be used here (installed models).

use axum::extract::{Path, State};
use axum::Json;
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
