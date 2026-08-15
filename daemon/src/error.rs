//! One error shape for every endpoint, in the OpenAI error envelope:
//!
//! ```json
//! { "error": { "message": "...", "type": "invalid_request_error", "code": "model_not_found", "param": null } }
//! ```
//!
//! HTTP status mapping:
//! * 400 `invalid_request_error` — malformed / unsupported request
//! * 404 `not_found_error`       — unknown model (`model_not_found`)
//! * 409 `invalid_request_error` — model exists but is not running (`model_not_running`) / bad lifecycle state
//! * 502 `server_error`          — the backend failed (`backend_error`)
//! * 503 `server_error`          — the backend is unavailable (`backend_unavailable`)
//! * 500 `server_error`          — anything else

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ohmygpu_inference::InferenceError;
use serde::Serialize;

use crate::manager::ManagerError;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub code: String,
    pub param: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ErrorBody,
}

impl ApiError {
    pub fn new(
        status: StatusCode,
        kind: &'static str,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            body: ErrorBody {
                message: message.into(),
                kind,
                code: code.into(),
                param: None,
            },
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_request",
            message,
        )
    }

    pub fn unsupported(what: impl std::fmt::Display) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unsupported",
            format!("{what} is not supported by OhMyGPU v0.1"),
        )
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "not_found",
            message,
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
            message,
        )
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.body.param = Some(param.into());
        self
    }

    pub fn envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: self.body.clone(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}

impl From<InferenceError> for ApiError {
    fn from(e: InferenceError) -> Self {
        let code = e.code();
        match e {
            InferenceError::ModelNotFound(m) => ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found_error",
                code,
                format!("model '{m}' is not installed. Pull it first (POST /ohmygpu/v1/models/pull or `omg model pull {m}`)."),
            )
            .with_param("model"),
            InferenceError::ModelNotRunning { model, state } => ApiError::new(
                StatusCode::CONFLICT,
                "invalid_request_error",
                code,
                format!(
                    "model '{model}' is not running (state: {state}). Start it first (POST /ohmygpu/v1/models/{model}/start or `omg run {model}`)."
                ),
            )
            .with_param("model"),
            InferenceError::InvalidRequest(m) => ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", code, m),
            InferenceError::Backend(m) => ApiError::new(StatusCode::BAD_GATEWAY, "server_error", code, m),
            InferenceError::Unavailable(m) => ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", code, m),
        }
    }
}

impl From<ManagerError> for ApiError {
    fn from(e: ManagerError) -> Self {
        match e {
            ManagerError::NotFound(m) => ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found_error",
                "model_not_found",
                format!("model '{m}' not found"),
            ),
            ManagerError::NotInstalled(m) => ApiError::new(
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_not_installed",
                format!("model '{m}' is not installed"),
            ),
            ManagerError::InvalidReference(m) => ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_model_reference",
                m,
            ),
            ManagerError::InvalidState { .. } => ApiError::new(
                StatusCode::CONFLICT,
                "invalid_request_error",
                "invalid_state",
                e.to_string(),
            ),
            ManagerError::Backend(m) => {
                ApiError::new(StatusCode::BAD_GATEWAY, "server_error", "backend_error", m)
            }
            ManagerError::Io(m) => ApiError::internal(m),
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(r: JsonRejection) -> Self {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_json",
            r.body_text(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_errors_map_to_status_and_code() {
        let e: ApiError = InferenceError::ModelNotFound("x".into()).into();
        assert_eq!(e.status, StatusCode::NOT_FOUND);
        assert_eq!(e.body.code, "model_not_found");
        let e: ApiError = InferenceError::ModelNotRunning {
            model: "x".into(),
            state: "installed".into(),
        }
        .into();
        assert_eq!(e.status, StatusCode::CONFLICT);
        assert!(e.body.message.contains("omg run x"));
        let e: ApiError = InferenceError::InvalidRequest("bad".into()).into();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let e: ApiError = InferenceError::Backend("boom".into()).into();
        assert_eq!(e.status, StatusCode::BAD_GATEWAY);
        let json = serde_json::to_value(e.envelope()).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["code"], "backend_error");
        assert!(json["error"]["param"].is_null());
    }
}
