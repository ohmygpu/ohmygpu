//! ohmygpu_runtime_llamacpp — the v0.1 inference backend.
//!
//! OhMyGPU does not link llama.cpp. It supervises the upstream `llama-server`
//! binary as a child process (one per running model, bound to
//! `127.0.0.1:<ephemeral port>`) and talks to it over HTTP. That gives process
//! isolation, upstream chat templates + tool-call parsing, and prebuilt
//! Metal/CUDA/Vulkan binaries — everything an application developer should
//! never have to think about.
//!
//! * [`install`] — find or download the binary
//! * [`process`] — spawn / supervise / stop
//! * [`wire`]    — internal model ⇄ llama-server JSON

pub mod install;
pub mod process;
pub mod wire;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt;
use ohmygpu_core::config::LlamaCppConfig;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_inference::{InferenceError, InferenceRequest, InferenceStream};
use ohmygpu_runtime_api::{
    BackendAvailability, InstanceInfo, InstanceStatus, ModelInstance, ProgressFn, RuntimeBackend,
    RuntimeError, StartSpec,
};
use tokio::sync::Mutex;

use install::{Installer, LocatedBinary, Locator};
use process::ServerProcess;

pub const BACKEND_ID: &str = "llamacpp";

pub struct LlamaCppBackend {
    config: LlamaCppConfig,
    hardware: HardwareInfo,
    managed_root: PathBuf,
    /// Cached binary location (cleared if the file disappears).
    located: Mutex<Option<LocatedBinary>>,
    http: reqwest::Client,
}

impl LlamaCppBackend {
    pub fn new(
        config: LlamaCppConfig,
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
                message: Some(format!("llama-server ({:?})", l.source).to_lowercase()),
            },
            None => BackendAvailability {
                available: false,
                version: None,
                path: None,
                message,
            },
        }
    }

    /// Wait until `GET /health` answers 200, the process exits, or we time out.
    async fn wait_ready(
        &self,
        proc: &ServerProcess,
        timeout: Duration,
    ) -> Result<(), RuntimeError> {
        let url = format!("http://127.0.0.1:{}/health", proc.port);
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
                    "model did not become ready within {}s{}",
                    timeout.as_secs(),
                    proc.log_tail()
                )));
            }
            if let Ok(resp) = self
                .http
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}

#[async_trait]
impl RuntimeBackend for LlamaCppBackend {
    fn id(&self) -> &'static str {
        BACKEND_ID
    }

    async fn available(&self) -> BackendAvailability {
        let located = self.locate().await;
        let msg = if located.is_none() {
            Some(if self.config.auto_install {
                "llama-server not found; it will be downloaded on first model start".to_string()
            } else {
                "llama-server not found and auto_install is off; set backend.llamacpp.server_path"
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
                "llama-server not found and backend.llamacpp.auto_install is false".into(),
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
        let located = match self.locate().await {
            Some(l) => l,
            None => {
                self.prepare(None).await?;
                self.locate()
                    .await
                    .ok_or_else(|| RuntimeError::NotAvailable("llama-server not found".into()))?
            }
        };
        if !spec.model_path.is_file() {
            return Err(RuntimeError::Start(format!(
                "model file not found: {}",
                spec.model_path.display()
            )));
        }
        let mut spec = spec;
        if spec.context_length.is_none() && self.config.context_length > 0 {
            spec.context_length = Some(self.config.context_length);
        }
        if spec.gpu_layers.is_none() {
            spec.gpu_layers = self.config.gpu_layers;
        }
        if spec.threads.is_none() {
            spec.threads = self.config.threads;
        }

        let port = process::free_port().await?;
        let proc = ServerProcess::spawn(&located.path, &spec, port)?;
        let timeout = Duration::from_secs(self.config.startup_timeout_secs.max(5));
        self.wait_ready(&proc, timeout).await?;
        tracing::info!(model = %spec.model_id, port, pid = ?proc.pid, "model ready");

        Ok(Arc::new(LlamaCppInstance {
            model_id: spec.model_id,
            proc,
            http: self.http.clone(),
            backend_version: located.version.clone(),
            stopping: AtomicBool::new(false),
        }))
    }
}

pub struct LlamaCppInstance {
    model_id: String,
    proc: ServerProcess,
    http: reqwest::Client,
    backend_version: Option<String>,
    stopping: AtomicBool,
}

impl LlamaCppInstance {
    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.proc.port)
    }
}

#[async_trait]
impl ModelInstance for LlamaCppInstance {
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
        if let Some(InstanceStatus::Exited { message, .. }) = self.proc.has_exited() {
            return Err(InferenceError::Unavailable(message));
        }
        let body = wire::build_request(&request);
        let url = format!("{}/v1/chat/completions", self.base_url());
        let resp =
            self.http.post(&url).json(&body).send().await.map_err(|e| {
                InferenceError::Unavailable(format!("cannot reach llama-server: {e}"))
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("error").map(wire::error_message))
                .unwrap_or(text);
            return Err(if status.as_u16() == 400 {
                InferenceError::InvalidRequest(message)
            } else {
                InferenceError::Backend(format!("llama-server returned {status}: {message}"))
            });
        }

        let mut bytes = resp.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut parser = wire::StreamParser::new();
            let mut buf = String::new();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| InferenceError::Unavailable(format!("stream interrupted: {e}")))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));
                // Process complete lines; keep the remainder.
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim_end_matches('\r').to_string();
                    buf.drain(..=pos);
                    if let Some(data) = line.strip_prefix("data:") {
                        for ev in parser.feed(data)? {
                            yield ev;
                        }
                    }
                }
                if parser.is_completed() { break; }
            }
            if !parser.is_completed() {
                if let Some(data) = buf.trim().strip_prefix("data:") {
                    for ev in parser.feed(data)? { yield ev; }
                }
                for ev in parser.finish() { yield ev; }
            }
        };
        Ok(Box::pin(stream))
    }

    async fn wait(&self) -> InstanceStatus {
        self.proc.wait().await
    }

    async fn stop(&self) -> Result<(), RuntimeError> {
        if self.stopping.swap(true, Ordering::SeqCst) {
            // Another stop in flight; just wait for it.
            let _ = tokio::time::timeout(Duration::from_secs(15), self.proc.wait()).await;
            return Ok(());
        }
        self.proc.stop(Duration::from_secs(10)).await;
        Ok(())
    }
}
