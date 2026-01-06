use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::state::AppState;

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let registry = state.registry.read().await;
    let models: Vec<ModelObject> = registry
        .list()
        .iter()
        .map(|m| ModelObject {
            id: m.name.clone(),
            object: "model",
            owned_by: "user",
        })
        .collect();

    Json(ModelsResponse {
        object: "list",
        data: models,
    })
}
