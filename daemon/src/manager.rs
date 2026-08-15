//! `ModelManager` — the runtime orchestrator.
//!
//! Owns the explicit per-model lifecycle (see `ohmygpu_core::lifecycle`), the
//! installed-model registry, background downloads, and the running
//! `ModelInstance`s produced by the `RuntimeBackend`. Everything the Management
//! API exposes and everything the inference adapters need ("give me the running
//! instance for model X") goes through here.
//!
//! Concurrency model: one `std::sync::Mutex` around the in-memory state, never
//! held across an `.await`; long operations (download, backend start, stop) run
//! in spawned tasks or after releasing the lock, and re-check a per-model
//! `generation` counter before applying their result so that a stop/delete that
//! raced with them wins.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use ohmygpu_core::catalog::{self, ModelRef};
use ohmygpu_core::config::Config;
use ohmygpu_core::download::{DownloadError, Downloader};
use ohmygpu_core::lifecycle::{DownloadProgress, ModelState};
use ohmygpu_core::paths::Paths;
use ohmygpu_core::registry::{InstalledModel, ModelCapabilities, ModelRegistry, ModelSource};
use ohmygpu_inference::InferenceError;
use ohmygpu_runtime_api::{
    BackendAvailability, InstanceInfo, InstanceStatus, ModelInstance, ProgressUpdate,
    RuntimeBackend, StartSpec,
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio::task::JoinHandle;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-start overrides accepted by `POST /ohmygpu/v1/models/{id}/start`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct StartOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
}

/// What clients see for a model (Management API + `/v1/models`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelView {
    pub id: String,
    pub display_name: String,
    /// Lifecycle state name: `not_installed` | `downloading` | `installed` | …
    pub state: String,
    pub installed: bool,
    pub curated: bool,
    pub capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes_approx: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_at: Option<DateTime<Utc>>,
    /// Present while `state == downloading`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download: Option<DownloadView>,
    /// Present while `state == starting` (what the runtime is doing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Present when `state == error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorView>,
    /// Present while `state == running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownloadView {
    pub downloaded_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorView {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeView {
    #[serde(flatten)]
    pub instance: InstanceInfo,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("model '{0}' not found")]
    NotFound(String),
    #[error("{0}")]
    InvalidReference(String),
    #[error("cannot {action} model '{model}' while it is {state}")]
    InvalidState {
        model: String,
        state: String,
        action: &'static str,
    },
    #[error("model '{0}' is not installed")]
    NotInstalled(String),
    #[error("{0}")]
    Backend(String),
    #[error("{0}")]
    Io(String),
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct Record {
    state: ModelState,
    instance: Option<Arc<dyn ModelInstance>>,
    started_at: Option<DateTime<Utc>>,
    /// Bumped on every start/stop/pull/delete; background tasks compare before
    /// applying results.
    generation: u64,
    task: Option<JoinHandle<()>>,
    cancel_download: Option<Arc<AtomicBool>>,
    /// Reference being downloaded (for the view while not yet in the registry).
    pending_ref: Option<ModelRef>,
}

impl Record {
    fn new(state: ModelState) -> Self {
        Self {
            state,
            instance: None,
            started_at: None,
            generation: 0,
            task: None,
            cancel_download: None,
            pending_ref: None,
        }
    }
}

struct Inner {
    models: BTreeMap<String, Record>,
}

pub struct ModelManager {
    paths: Paths,
    config: Config,
    backend: Arc<dyn RuntimeBackend>,
    downloader: Downloader,
    registry: Mutex<ModelRegistry>,
    inner: Mutex<Inner>,
    /// Bumped on any state change; used by `wait_for`.
    changed: watch::Sender<u64>,
}

impl ModelManager {
    pub fn new(
        paths: Paths,
        config: Config,
        backend: Arc<dyn RuntimeBackend>,
    ) -> anyhow::Result<Arc<Self>> {
        paths.ensure_dirs()?;
        let mut registry = ModelRegistry::load(paths.registry_path())?;
        let pruned = registry.prune_missing()?;
        for id in &pruned {
            tracing::warn!("model '{id}' was in the registry but its file is missing; removed");
        }
        let mut models = BTreeMap::new();
        for m in registry.list() {
            models.insert(m.id.clone(), Record::new(ModelState::Installed));
        }
        for e in catalog::CATALOG {
            models
                .entry(e.id.to_string())
                .or_insert_with(|| Record::new(ModelState::NotInstalled));
        }
        let (changed, _) = watch::channel(0);
        Ok(Arc::new(Self {
            downloader: Downloader::new(config.models.hf_token.clone()),
            paths,
            config,
            backend,
            registry: Mutex::new(registry),
            inner: Mutex::new(Inner { models }),
            changed,
        }))
    }

    pub fn backend(&self) -> &Arc<dyn RuntimeBackend> {
        &self.backend
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn paths(&self) -> &Paths {
        &self.paths
    }

    fn notify(&self) {
        self.changed.send_modify(|v| *v = v.wrapping_add(1));
    }

    // -----------------------------------------------------------------------
    // Views
    // -----------------------------------------------------------------------

    fn view_locked(&self, id: &str, rec: &Record, registry: &ModelRegistry) -> ModelView {
        let installed = registry.get(id);
        let cat = catalog::find(id);
        let pending = rec.pending_ref.as_ref();
        let display_name = installed
            .map(|m| m.display_name.clone())
            .or_else(|| cat.map(|c| c.display_name.to_string()))
            .or_else(|| pending.map(|p| p.display_name.clone()))
            .unwrap_or_else(|| id.to_string());
        let curated = installed.map(|m| m.curated).unwrap_or(cat.is_some());
        let capabilities = installed
            .map(|m| m.capabilities)
            .or_else(|| cat.map(|c| ModelCapabilities { tools: c.tools }))
            .or_else(|| pending.map(|p| ModelCapabilities { tools: p.tools }))
            .unwrap_or_default();
        let source = installed.map(|m| m.source.clone()).or_else(|| {
            cat.map(|c| ModelSource::HuggingFace {
                repo: c.repo.to_string(),
                file: c.file.to_string(),
            })
            .or_else(|| pending.map(|p| p.source.clone()))
        });
        let download = match &rec.state {
            ModelState::Downloading { progress } => Some(DownloadView {
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes,
                percent: progress.percent(),
            }),
            _ => None,
        };
        let message = match &rec.state {
            ModelState::Starting { message } => message.clone(),
            _ => None,
        };
        let error = rec.state.error_message().map(|m| ErrorView {
            message: m.to_string(),
        });
        let runtime = match (&rec.state, &rec.instance, rec.started_at) {
            (ModelState::Running, Some(inst), Some(started_at)) => Some(RuntimeView {
                instance: inst.info(),
                started_at,
            }),
            _ => None,
        };
        ModelView {
            id: id.to_string(),
            display_name,
            state: rec.state.name().to_string(),
            installed: installed.is_some(),
            curated,
            capabilities,
            source,
            format: installed.map(|m| m.format.clone()),
            size_bytes: installed.map(|m| m.size_bytes),
            size_bytes_approx: if installed.is_none() {
                cat.map(|c| c.size_bytes_approx)
            } else {
                None
            },
            path: installed.map(|m| m.path.clone()),
            installed_at: installed.map(|m| m.installed_at),
            download,
            message,
            error,
            runtime,
        }
    }

    pub fn list(&self) -> Vec<ModelView> {
        let inner = self.inner.lock().unwrap();
        let registry = self.registry.lock().unwrap();
        inner
            .models
            .iter()
            .map(|(id, rec)| self.view_locked(id, rec, &registry))
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<ModelView> {
        let inner = self.inner.lock().unwrap();
        let registry = self.registry.lock().unwrap();
        inner
            .models
            .get(id)
            .map(|rec| self.view_locked(id, rec, &registry))
    }

    pub fn state_of(&self, id: &str) -> Option<ModelState> {
        self.inner
            .lock()
            .unwrap()
            .models
            .get(id)
            .map(|r| r.state.clone())
    }

    /// Installed models only (what `/v1/models` lists).
    pub fn installed(&self) -> Vec<ModelView> {
        self.list().into_iter().filter(|m| m.installed).collect()
    }

    pub fn running_ids(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap();
        inner
            .models
            .iter()
            .filter(|(_, r)| r.state.is_running())
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn downloading_ids(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap();
        inner
            .models
            .iter()
            .filter(|(_, r)| matches!(r.state, ModelState::Downloading { .. }))
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn installed_count(&self) -> usize {
        self.registry.lock().unwrap().list().len()
    }

    // -----------------------------------------------------------------------
    // Pull / delete
    // -----------------------------------------------------------------------

    /// Start (or resume) downloading a model. Returns immediately with the
    /// current view; progress is visible through `get()`.
    pub fn pull(
        self: &Arc<Self>,
        reference: &str,
        id_override: Option<&str>,
    ) -> Result<ModelView, ManagerError> {
        let mref =
            ModelRef::parse(reference, id_override).map_err(ManagerError::InvalidReference)?;
        let id = mref.id.clone();

        let (gen, cancel) = {
            let mut inner = self.inner.lock().unwrap();
            let registry = self.registry.lock().unwrap();
            let rec = inner
                .models
                .entry(id.clone())
                .or_insert_with(|| Record::new(ModelState::NotInstalled));
            match &rec.state {
                ModelState::Downloading { .. } => return Ok(self.view_locked(&id, rec, &registry)),
                s if registry.contains(&id) && s.is_installed() => {
                    return Ok(self.view_locked(&id, rec, &registry))
                }
                ModelState::NotInstalled | ModelState::Error { .. } => {}
                other => {
                    return Err(ManagerError::InvalidState {
                        model: id,
                        state: other.name().to_string(),
                        action: "pull",
                    })
                }
            }
            rec.generation += 1;
            let cancel = Arc::new(AtomicBool::new(false));
            rec.cancel_download = Some(cancel.clone());
            rec.pending_ref = Some(mref.clone());
            rec.state = ModelState::Downloading {
                progress: DownloadProgress {
                    downloaded_bytes: 0,
                    total_bytes: mref.size_bytes_approx,
                },
            };
            (rec.generation, cancel)
        };
        self.notify();

        let this = self.clone();
        let task_ref = mref.clone();
        let task = tokio::spawn(async move {
            this.run_download(task_ref, gen, cancel).await;
        });
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(rec) = inner.models.get_mut(&id) {
                if rec.generation == gen {
                    rec.task = Some(task);
                }
            }
        }
        Ok(self.get(&id).expect("record exists"))
    }

    async fn run_download(self: Arc<Self>, mref: ModelRef, gen: u64, cancel: Arc<AtomicBool>) {
        let id = mref.id.clone();
        let url = mref.url.clone();
        let file_name = mref
            .file
            .rsplit('/')
            .next()
            .unwrap_or(&mref.file)
            .to_string();
        let dest = self
            .config
            .models_dir(&self.paths)
            .join(&id)
            .join(&file_name);
        tracing::info!(model = %id, "downloading {url} → {}", dest.display());

        let progress_self = self.clone();
        let progress_id = id.clone();
        let last = Arc::new(Mutex::new(0u64));
        let progress: ohmygpu_core::download::ProgressCallback =
            Arc::new(move |p: DownloadProgress| {
                // Throttle lock traffic: update on every 512 KiB or on completion.
                let mut l = last.lock().unwrap();
                let done = p
                    .total_bytes
                    .map(|t| p.downloaded_bytes >= t)
                    .unwrap_or(false);
                if !done && p.downloaded_bytes.saturating_sub(*l) < 512 * 1024 {
                    return;
                }
                *l = p.downloaded_bytes;
                drop(l);
                let mut inner = progress_self.inner.lock().unwrap();
                if let Some(rec) = inner.models.get_mut(&progress_id) {
                    if rec.generation == gen {
                        if let ModelState::Downloading { progress } = &mut rec.state {
                            *progress = p;
                        }
                    }
                }
                drop(inner);
                progress_self.notify();
            });

        let result = self
            .downloader
            .download(&url, &dest, Some(progress), Some(cancel))
            .await;
        match result {
            Ok(size) => {
                let installed = InstalledModel {
                    id: id.clone(),
                    display_name: mref.display_name.clone(),
                    source: mref.source.clone(),
                    format: "gguf".into(),
                    path: dest.clone(),
                    size_bytes: size,
                    installed_at: Utc::now(),
                    capabilities: ModelCapabilities { tools: mref.tools },
                    curated: mref.curated,
                };
                let mut inner = self.inner.lock().unwrap();
                let Some(rec) = inner.models.get_mut(&id) else {
                    return;
                };
                if rec.generation != gen {
                    // A delete raced with us; leave the file for delete to clean up.
                    return;
                }
                if let Err(e) = self.registry.lock().unwrap().add(installed) {
                    rec.state = ModelState::Error {
                        message: format!("failed to save registry: {e}"),
                    };
                } else {
                    rec.state = ModelState::Installed;
                }
                rec.pending_ref = None;
                rec.cancel_download = None;
                rec.task = None;
                drop(inner);
                tracing::info!(model = %id, "installed ({size} bytes)");
                self.notify();
            }
            Err(DownloadError::Cancelled) => {
                tracing::info!(model = %id, "download cancelled");
            }
            Err(e) => {
                tracing::warn!(model = %id, "download failed: {e}");
                let mut inner = self.inner.lock().unwrap();
                if let Some(rec) = inner.models.get_mut(&id) {
                    if rec.generation == gen {
                        rec.state = ModelState::Error {
                            message: format!("download failed: {e}"),
                        };
                        rec.cancel_download = None;
                        rec.task = None;
                    }
                }
                drop(inner);
                self.notify();
            }
        }
    }

    /// Stop (if needed), delete files, and forget the model.
    pub async fn delete(self: &Arc<Self>, id: &str) -> Result<ModelView, ManagerError> {
        let state = self
            .state_of(id)
            .ok_or_else(|| ManagerError::NotFound(id.to_string()))?;
        match state {
            ModelState::Running | ModelState::Starting { .. } | ModelState::Stopping => {
                self.stop(id).await?;
            }
            _ => {}
        }
        let (task, cancel) = {
            let mut inner = self.inner.lock().unwrap();
            let rec = inner
                .models
                .get_mut(id)
                .ok_or_else(|| ManagerError::NotFound(id.to_string()))?;
            rec.generation += 1;
            (rec.task.take(), rec.cancel_download.take())
        };
        if let Some(c) = cancel {
            c.store(true, Ordering::Relaxed);
        }
        if let Some(t) = task {
            t.abort();
            let _ = t.await;
        }
        let removed = self
            .registry
            .lock()
            .unwrap()
            .remove(id)
            .map_err(|e| ManagerError::Io(e.to_string()))?;
        let dir = self.config.models_dir(&self.paths).join(id);
        if dir.exists() {
            tokio::fs::remove_dir_all(&dir)
                .await
                .map_err(|e| ManagerError::Io(format!("removing {}: {e}", dir.display())))?;
        }
        if let Some(m) = &removed {
            // The file may live outside the model dir (custom storage layouts).
            if m.path.exists() {
                tokio::fs::remove_file(&m.path).await.ok();
            }
        }
        let view = {
            let mut inner = self.inner.lock().unwrap();
            let registry = self.registry.lock().unwrap();
            if catalog::find(id).is_some() {
                let rec = inner.models.get_mut(id).unwrap();
                *rec = Record::new(ModelState::NotInstalled);
                self.view_locked(id, rec, &registry)
            } else {
                let rec = inner.models.remove(id).unwrap();
                let mut v = self.view_locked(id, &rec, &registry);
                v.state = ModelState::NotInstalled.name().to_string();
                v.installed = false;
                v
            }
        };
        self.notify();
        tracing::info!(model = %id, "deleted");
        Ok(view)
    }

    // -----------------------------------------------------------------------
    // Start / stop
    // -----------------------------------------------------------------------

    /// Begin starting a model. Returns immediately (state `starting`); use
    /// `wait_for` / polling to observe `running` or `error`.
    pub fn start(
        self: &Arc<Self>,
        id: &str,
        opts: StartOptions,
    ) -> Result<ModelView, ManagerError> {
        let (gen, model_path) = {
            let mut inner = self.inner.lock().unwrap();
            let registry = self.registry.lock().unwrap();
            let rec = inner
                .models
                .get_mut(id)
                .ok_or_else(|| ManagerError::NotFound(id.to_string()))?;
            let installed = registry
                .get(id)
                .ok_or_else(|| ManagerError::NotInstalled(id.to_string()))?;
            match &rec.state {
                ModelState::Running | ModelState::Starting { .. } => {
                    return Ok(self.view_locked(id, rec, &registry));
                }
                s if s.can_start() => {}
                other => {
                    return Err(ManagerError::InvalidState {
                        model: id.to_string(),
                        state: other.name().to_string(),
                        action: "start",
                    })
                }
            }
            rec.generation += 1;
            rec.state = ModelState::Starting { message: None };
            rec.instance = None;
            rec.started_at = None;
            (rec.generation, installed.path.clone())
        };
        self.notify();

        let spec = StartSpec {
            model_id: id.to_string(),
            model_path,
            context_length: opts.context_length,
            gpu_layers: opts.gpu_layers,
            threads: opts.threads,
        };
        let this = self.clone();
        let task_id = id.to_string();
        let task = tokio::spawn(async move {
            this.run_start(task_id, gen, spec).await;
        });
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(rec) = inner.models.get_mut(id) {
                if rec.generation == gen {
                    rec.task = Some(task);
                }
            }
        }
        Ok(self.get(id).expect("record exists"))
    }

    fn set_starting_message(&self, id: &str, gen: u64, message: impl Into<String>) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(rec) = inner.models.get_mut(id) {
            if rec.generation == gen {
                if let ModelState::Starting { message: m } = &mut rec.state {
                    *m = Some(message.into());
                }
            }
        }
        drop(inner);
        self.notify();
    }

    fn set_error_if_current(&self, id: &str, gen: u64, message: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(rec) = inner.models.get_mut(id) {
            if rec.generation == gen {
                rec.state = ModelState::Error { message };
                rec.instance = None;
                rec.started_at = None;
                rec.task = None;
            }
        }
        drop(inner);
        self.notify();
    }

    async fn run_start(self: Arc<Self>, id: String, gen: u64, spec: StartSpec) {
        // 1. Make sure the backend binary exists (may download it).
        let progress_self = self.clone();
        let progress_id = id.clone();
        let progress: ohmygpu_runtime_api::ProgressFn = Arc::new(move |u: ProgressUpdate| {
            let msg = match (u.done_bytes, u.total_bytes) {
                (Some(d), Some(t)) if t > 0 => {
                    format!("{} ({:.0}%)", u.message, d as f64 / t as f64 * 100.0)
                }
                _ => u.message.clone(),
            };
            progress_self.set_starting_message(&progress_id, gen, msg);
        });
        self.set_starting_message(&id, gen, "preparing backend");
        if let Err(e) = self.backend.prepare(Some(progress)).await {
            self.set_error_if_current(&id, gen, format!("backend not available: {e}"));
            return;
        }

        // 2. Start the model.
        self.set_starting_message(&id, gen, "loading model");
        let instance = match self.backend.start(spec).await {
            Ok(inst) => inst,
            Err(e) => {
                self.set_error_if_current(&id, gen, e.to_string());
                return;
            }
        };

        // 3. Publish, unless a stop/delete raced with us.
        let accepted = {
            let mut inner = self.inner.lock().unwrap();
            match inner.models.get_mut(&id) {
                Some(rec)
                    if rec.generation == gen
                        && matches!(rec.state, ModelState::Starting { .. }) =>
                {
                    rec.state = ModelState::Running;
                    rec.instance = Some(instance.clone());
                    rec.started_at = Some(Utc::now());
                    true
                }
                _ => false,
            }
        };
        self.notify();
        if !accepted {
            tracing::info!(model = %id, "start superseded; stopping instance");
            let _ = instance.stop().await;
            return;
        }
        tracing::info!(model = %id, "running");

        // 4. Watch for unexpected exit.
        let status = instance.wait().await;
        let mut inner = self.inner.lock().unwrap();
        if let Some(rec) = inner.models.get_mut(&id) {
            if rec.generation == gen && rec.state.is_running() {
                let message = match status {
                    InstanceStatus::Exited { message, .. } => message,
                    _ => "backend process exited".to_string(),
                };
                tracing::error!(model = %id, "stopped unexpectedly: {message}");
                rec.state = ModelState::Error {
                    message: format!("stopped unexpectedly: {message}"),
                };
                rec.instance = None;
                rec.started_at = None;
                rec.task = None;
            }
        }
        drop(inner);
        self.notify();
    }

    /// Stop a running (or starting) model.
    pub async fn stop(self: &Arc<Self>, id: &str) -> Result<ModelView, ManagerError> {
        let (gen, instance, task, was_starting) = {
            let mut inner = self.inner.lock().unwrap();
            let rec = inner
                .models
                .get_mut(id)
                .ok_or_else(|| ManagerError::NotFound(id.to_string()))?;
            match &rec.state {
                ModelState::Running | ModelState::Starting { .. } => {}
                ModelState::Stopped | ModelState::Installed => {
                    let registry = self.registry.lock().unwrap();
                    return Ok(self.view_locked(id, rec, &registry));
                }
                other => {
                    return Err(ManagerError::InvalidState {
                        model: id.to_string(),
                        state: other.name().to_string(),
                        action: "stop",
                    })
                }
            }
            let was_starting = matches!(rec.state, ModelState::Starting { .. });
            rec.generation += 1;
            rec.state = ModelState::Stopping;
            (
                rec.generation,
                rec.instance.take(),
                rec.task.take(),
                was_starting,
            )
        };
        self.notify();

        if let Some(t) = task {
            if was_starting {
                // Dropping the in-flight start kills any half-started process.
                t.abort();
                let _ = t.await;
            }
        }
        if let Some(inst) = instance {
            if let Err(e) = inst.stop().await {
                tracing::warn!(model = %id, "stop reported: {e}");
            }
        }
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(rec) = inner.models.get_mut(id) {
                if rec.generation == gen {
                    rec.state = ModelState::Stopped;
                    rec.instance = None;
                    rec.started_at = None;
                }
            }
        }
        self.notify();
        tracing::info!(model = %id, "stopped");
        Ok(self.get(id).expect("record exists"))
    }

    /// Stop every running/starting model (daemon shutdown).
    pub async fn stop_all(self: &Arc<Self>) {
        let ids: Vec<String> = {
            let inner = self.inner.lock().unwrap();
            inner
                .models
                .iter()
                .filter(|(_, r)| {
                    matches!(r.state, ModelState::Running | ModelState::Starting { .. })
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        for id in ids {
            let _ = self.stop(&id).await;
        }
    }

    // -----------------------------------------------------------------------
    // Waiting
    // -----------------------------------------------------------------------

    /// Wait until `pred(state)` holds (or the model disappears), with a timeout.
    /// Returns the final state (`None` if the model no longer exists).
    pub async fn wait_for(
        &self,
        id: &str,
        timeout: Duration,
        pred: impl Fn(&ModelState) -> bool,
    ) -> Result<Option<ModelState>, tokio::time::error::Elapsed> {
        let mut rx = self.changed.subscribe();
        tokio::time::timeout(timeout, async {
            loop {
                let state = self.state_of(id);
                match &state {
                    None => return None,
                    Some(s) if pred(s) => return state,
                    _ => {}
                }
                if rx.changed().await.is_err() {
                    return self.state_of(id);
                }
            }
        })
        .await
    }

    /// Wait for a start to settle: `running`, `error`, or a stop.
    pub async fn wait_started(&self, id: &str, timeout: Duration) -> Option<ModelState> {
        self.wait_for(id, timeout, |s| !matches!(s, ModelState::Starting { .. }))
            .await
            .unwrap_or_else(|_| self.state_of(id))
    }

    // -----------------------------------------------------------------------
    // Inference entry point
    // -----------------------------------------------------------------------

    /// The running instance for `model`, or the right `InferenceError`.
    /// Honors `inference.auto_start`.
    pub async fn instance_for(
        self: &Arc<Self>,
        model: &str,
    ) -> Result<Arc<dyn ModelInstance>, InferenceError> {
        for attempt in 0..2 {
            let (state, instance, installed) = {
                let inner = self.inner.lock().unwrap();
                let registry = self.registry.lock().unwrap();
                match inner.models.get(model) {
                    None => return Err(InferenceError::ModelNotFound(model.to_string())),
                    Some(rec) => (
                        rec.state.clone(),
                        rec.instance.clone(),
                        registry.contains(model),
                    ),
                }
            };
            if let (ModelState::Running, Some(inst)) = (&state, instance) {
                return Ok(inst);
            }
            if !installed {
                return Err(InferenceError::ModelNotFound(model.to_string()));
            }
            let can_auto = self.config.inference.auto_start && attempt == 0;
            match &state {
                ModelState::Starting { .. } if can_auto => {}
                s if s.can_start() && can_auto => {
                    self.start(model, StartOptions::default())
                        .map_err(|e| InferenceError::Backend(e.to_string()))?;
                }
                _ => {
                    return Err(InferenceError::ModelNotRunning {
                        model: model.to_string(),
                        state: state.name().to_string(),
                    })
                }
            }
            let timeout =
                Duration::from_secs(self.config.backend.llamacpp.startup_timeout_secs.max(5));
            match self.wait_started(model, timeout).await {
                Some(ModelState::Running) => continue,
                Some(ModelState::Error { message }) => {
                    return Err(InferenceError::Backend(message))
                }
                Some(other) => {
                    return Err(InferenceError::ModelNotRunning {
                        model: model.to_string(),
                        state: other.name().to_string(),
                    })
                }
                None => return Err(InferenceError::ModelNotFound(model.to_string())),
            }
        }
        Err(InferenceError::ModelNotRunning {
            model: model.to_string(),
            state: "starting".into(),
        })
    }

    pub async fn backend_availability(&self) -> BackendAvailability {
        self.backend.available().await
    }
}
