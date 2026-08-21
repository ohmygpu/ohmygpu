//! ohmygpu_runtime_whisper — the speech-to-text backend: a supervised
//! `whisper-server` (whisper.cpp) per running whisper model.
//!
//! * `install.rs` — find or install the binary (official Linux/Windows assets;
//!   our own macOS builds from the OhMyGPU release)
//! * `wire.rs`    — internal `TranscriptionRequest` ⇄ `/inference` multipart + JSON
//! * this file    — [`WhisperBackend`] / [`WhisperInstance`] behind the common
//!   `RuntimeBackend` / `ModelInstance` contract. Chat/Responses calls on a
//!   whisper model answer `Unsupported`.

pub mod install;
pub mod wire;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use ohmygpu_core::config::WhisperConfig;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_inference::{
    InferenceError, InferenceRequest, InferenceStream, ModelKind, TranscriptionRequest,
    TranscriptionResponse,
};
use ohmygpu_runtime_api::{
    BackendAvailability, InstanceInfo, InstanceStatus, ModelInstance, ProgressFn, RuntimeBackend,
    RuntimeError, StartSpec,
};
use ohmygpu_runtime_common::process::{free_port, ServerProcess};
use tokio::sync::Mutex;

use install::{Installer, LocatedBinary, Locator};

pub const BACKEND_ID: &str = "whisper";

pub struct WhisperBackend {
    config: WhisperConfig,
    hardware: HardwareInfo,
    managed_root: PathBuf,
    located: Mutex<Option<LocatedBinary>>,
    http: reqwest::Client,
}

impl WhisperBackend {
    pub fn new(
        config: WhisperConfig,
        runtimes_dir: &std::path::Path,
        hardware: HardwareInfo,
    ) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("reqwest client");
        Self {
            config,
            hardware,
            managed_root: Locator::managed_root(runtimes_dir),
            located: Mutex::new(None),
            http,
        }
    }

    fn locator(&self) -> Locator {
        Locator {
            explicit: self.config.server_path.clone(),
            managed_root: self.managed_root.clone(),
        }
    }

    async fn locate(&self) -> Option<LocatedBinary> {
        let mut cache = self.located.lock().await;
        if let Some(l) = cache.as_ref() {
            if l.path.is_file() {
                return Some(l.clone());
            }
            *cache = None;
        }
        let found = self.locator().locate().await;
        *cache = found.clone();
        found
    }

    fn availability(
        located: Option<&LocatedBinary>,
        message: Option<String>,
    ) -> BackendAvailability {
        match located {
            Some(l) => BackendAvailability {
                available: true,
                version: l.version.clone(),
                path: Some(l.path.clone()),
                message: Some(format!("whisper-server ({:?})", l.source).to_lowercase()),
            },
            None => BackendAvailability {
                available: false,
                version: None,
                path: None,
                message,
            },
        }
    }

    /// whisper-server loads the model before it listens, so "answers HTTP at
    /// all" means ready.
    async fn wait_ready(
        &self,
        proc: &ServerProcess,
        timeout: Duration,
    ) -> Result<(), RuntimeError> {
        let url = format!("http://127.0.0.1:{}/", proc.port);
        let start = Instant::now();
        loop {
            if let Some(exit) = proc.has_exited() {
                let msg = match exit {
                    InstanceStatus::Exited { message, .. } => message,
                    _ => "exited".into(),
                };
                return Err(RuntimeError::Start(msg));
            }
            if start.elapsed() > timeout {
                proc.stop(Duration::from_secs(5)).await;
                return Err(RuntimeError::Start(format!(
                    "whisper-server did not become ready within {}s{}",
                    timeout.as_secs(),
                    proc.log_tail()
                )));
            }
            if self
                .http
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .is_ok()
            {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}

#[async_trait]
impl RuntimeBackend for WhisperBackend {
    fn id(&self) -> &'static str {
        BACKEND_ID
    }

    async fn available(&self) -> BackendAvailability {
        let located = self.locate().await;
        let msg = if located.is_none() {
            Some(if self.config.auto_install {
                "whisper-server not found; it will be downloaded on first speech model start"
                    .to_string()
            } else {
                "whisper-server not found and auto_install is off; set backend.whisper.server_path"
                    .to_string()
            })
        } else {
            None
        };
        Self::availability(located.as_ref(), msg)
    }

    async fn prepare(
        &self,
        progress: Option<ProgressFn>,
    ) -> Result<BackendAvailability, RuntimeError> {
        if let Some(l) = self.locate().await {
            return Ok(Self::availability(Some(&l), None));
        }
        if !self.config.auto_install {
            return Err(RuntimeError::NotAvailable(
                "whisper-server not found and backend.whisper.auto_install is false".into(),
            ));
        }
        let installer = Installer {
            managed_root: self.managed_root.clone(),
            hardware: self.hardware.clone(),
            release: self.config.release.clone(),
        };
        let located = installer.install(progress).await?;
        *self.located.lock().await = Some(located.clone());
        Ok(Self::availability(Some(&located), None))
    }

    async fn start(&self, spec: StartSpec) -> Result<Arc<dyn ModelInstance>, RuntimeError> {
        if spec.kind != ModelKind::Whisper {
            return Err(RuntimeError::Start(format!(
                "the whisper backend only runs speech models (got a {} model)",
                spec.kind.as_str()
            )));
        }
        let located = match self.locate().await {
            Some(l) => l,
            None => {
                self.prepare(None).await?;
                self.locate()
                    .await
                    .ok_or_else(|| RuntimeError::NotAvailable("whisper-server not found".into()))?
            }
        };
        if !spec.model_path.is_file() {
            return Err(RuntimeError::Start(format!(
                "model file not found: {}",
                spec.model_path.display()
            )));
        }
        let threads = spec.threads.or(self.config.threads);
        let port = free_port().await?;
        let mut args: Vec<String> = vec![
            "-m".into(),
            spec.model_path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "-l".into(),
            "auto".into(),
        ];
        if let Some(t) = threads {
            args.push("-t".into());
            args.push(t.to_string());
        }
        let proc = ServerProcess::spawn(
            "whisper-server",
            &spec.model_id,
            &located.path,
            &args,
            port,
            |model, line| tracing::debug!(target: "whisper", model = %model, "{line}"),
        )?;
        let timeout = Duration::from_secs(self.config.startup_timeout_secs.max(5));
        self.wait_ready(&proc, timeout).await?;
        tracing::info!(model = %spec.model_id, port, pid = ?proc.pid, "speech model ready");
        Ok(Arc::new(WhisperInstance {
            model_id: spec.model_id,
            proc,
            http: self.http.clone(),
            backend_version: located.version.clone(),
            stopping: AtomicBool::new(false),
        }))
    }
}

pub struct WhisperInstance {
    model_id: String,
    proc: ServerProcess,
    http: reqwest::Client,
    backend_version: Option<String>,
    stopping: AtomicBool,
}

#[async_trait]
impl ModelInstance for WhisperInstance {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn info(&self) -> InstanceInfo {
        InstanceInfo {
            backend: BACKEND_ID.to_string(),
            pid: self.proc.pid,
            port: Some(self.proc.port),
            backend_version: self.backend_version.clone(),
        }
    }

    async fn status(&self) -> InstanceStatus {
        self.proc.has_exited().unwrap_or(InstanceStatus::Running)
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, InferenceError> {
        Err(InferenceError::Unsupported(format!(
            "model '{}' is a speech-to-text model; use POST /v1/audio/transcriptions",
            request.model
        )))
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, InferenceError> {
        request.validate()?;
        if let Some(InstanceStatus::Exited { message, .. }) = self.proc.has_exited() {
            return Err(InferenceError::Unavailable(message));
        }
        let url = format!("http://127.0.0.1:{}/inference", self.proc.port);
        let form = wire::build_form(&request);
        let resp = self
            .http
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                InferenceError::Unavailable(format!("cannot reach whisper-server: {e}"))
            })?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            let message = wire::error_message(&text);
            return Err(if status.as_u16() == 400 {
                InferenceError::InvalidRequest(message)
            } else {
                InferenceError::Backend(format!("whisper-server returned {status}: {message}"))
            });
        }
        let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            InferenceError::Backend(format!("whisper-server returned non-JSON: {e}"))
        })?;
        wire::parse_response(&request.model, &request.audio, &body)
    }

    async fn wait(&self) -> InstanceStatus {
        self.proc.wait().await
    }

    async fn stop(&self) -> Result<(), RuntimeError> {
        if self.stopping.swap(true, Ordering::SeqCst) {
            self.proc.wait().await;
            return Ok(());
        }
        self.proc.stop(Duration::from_secs(10)).await;
        Ok(())
    }
}
