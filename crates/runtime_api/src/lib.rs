//! ohmygpu_runtime_api — the contract every inference backend implements.
//!
//! OhMyGPU *orchestrates* proven runtimes rather than implementing inference
//! itself. A backend (llama.cpp today; MLX / vLLM / whisper.cpp later) is
//! described by two small traits:
//!
//! * [`RuntimeBackend`] — process-independent: is the backend available on this
//!   machine (`available`), make it available (`prepare`, e.g. download the
//!   binary), and `start` a model, yielding a…
//! * [`ModelInstance`] — one running model: `status`, `infer` / `infer_stream`,
//!   `wait` for unexpected exit, and `stop`.
//!
//! Both operate purely on `ohmygpu_inference` types. A backend never knows which
//! public API a request came from.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use ohmygpu_inference::{
    InferenceError, InferenceRequest, InferenceResponse, InferenceStream, ModelKind,
    ResponseAccumulator, TranscriptionRequest, TranscriptionResponse,
};
use serde::{Deserialize, Serialize};

/// Whether the backend can be used right now, and where it lives.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendAvailability {
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Human-readable detail (why unavailable, or how it was found).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Everything a backend needs to start one model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StartSpec {
    /// OhMyGPU model id (used for logging / aliasing).
    pub model_id: String,
    /// What kind of model this is (decides which backend starts it).
    #[serde(default)]
    pub kind: ModelKind,
    /// Path to the model file (GGUF for llama.cpp).
    pub model_path: PathBuf,
    /// Multimodal projector (vision) that belongs to this model, if it can see.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<PathBuf>,
    /// Context window to allocate. `None` = backend default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Layers to offload to the GPU. `None` = backend default (all that fit).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    /// CPU threads for generation. `None` = backend default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
}

/// Coarse state of a running instance as seen by the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum InstanceStatus {
    Starting,
    Running,
    Exited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<i32>,
        message: String,
    },
}

/// Details the orchestrator exposes to clients about a running instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstanceInfo {
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_version: Option<String>,
}

/// Errors from backend lifecycle operations (not inference — see `InferenceError`).
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("backend not available: {0}")]
    NotAvailable(String),
    #[error("failed to install backend: {0}")]
    Install(String),
    #[error("failed to start model: {0}")]
    Start(String),
    #[error("backend process error: {0}")]
    Process(String),
    #[error("{0}")]
    Other(String),
}

/// Progress callback used by long-running preparation steps (binary download).
pub type ProgressFn = Arc<dyn Fn(ProgressUpdate) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Short human-readable phase, e.g. "downloading llama.cpp b1234".
    pub message: String,
    pub done_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

/// A backend that can run models. One backend can back many instances.
#[async_trait]
pub trait RuntimeBackend: Send + Sync {
    /// Stable identifier, e.g. `"llamacpp"`.
    fn id(&self) -> &'static str;

    /// Is the backend usable on this machine right now?
    async fn available(&self) -> BackendAvailability;

    /// Make the backend usable (download/verify binaries). Idempotent.
    async fn prepare(
        &self,
        progress: Option<ProgressFn>,
    ) -> Result<BackendAvailability, RuntimeError>;

    /// Start a model and return once it is ready to serve inference (or fail).
    async fn start(&self, spec: StartSpec) -> Result<Arc<dyn ModelInstance>, RuntimeError>;
}

/// One running model.
#[async_trait]
pub trait ModelInstance: Send + Sync {
    fn model_id(&self) -> &str;

    fn info(&self) -> InstanceInfo;

    async fn status(&self) -> InstanceStatus;

    /// Streaming inference — the primary path every backend must implement.
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, InferenceError>;

    /// Non-streaming inference. Default: collect the stream. Backends may
    /// override, but must produce identical results either way.
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let model = request.model.clone();
        let mut stream = self.infer_stream(request).await?;
        let mut acc = ResponseAccumulator::new();
        while let Some(item) = stream.next().await {
            acc.push(&item?);
        }
        Ok(acc.finish(model))
    }

    /// Speech to text. Only speech models implement this; everything else
    /// answers `Unsupported`.
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, InferenceError> {
        Err(InferenceError::Unsupported(format!(
            "model '{}' does not transcribe audio (not a speech-to-text model)",
            request.model
        )))
    }

    /// Resolves when the instance exits for any reason (crash or `stop`).
    async fn wait(&self) -> InstanceStatus;

    /// Stop gracefully; force-kill after a timeout. Idempotent.
    async fn stop(&self) -> Result<(), RuntimeError>;
}
