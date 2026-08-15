//! Daemon bootstrap: build the runtime, bind, serve, shut down cleanly.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ohmygpu_core::config::Config;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_core::paths::Paths;
use ohmygpu_runtime_api::RuntimeBackend;
use ohmygpu_runtime_llamacpp::LlamaCppBackend;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::api::router;
use crate::manager::ModelManager;
use crate::state::{AppState, SharedState};

/// What a running daemon writes to `<data-dir>/daemon.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRecord {
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub version: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl DaemonRecord {
    pub fn read(paths: &Paths) -> Option<DaemonRecord> {
        let text = std::fs::read_to_string(paths.daemon_state_path()).ok()?;
        serde_json::from_str(&text).ok()
    }
}

pub struct ServeOptions {
    pub paths: Paths,
    pub config: Config,
}

/// Build the shared state with an explicit backend (tests use a mock).
pub fn build_state(
    paths: Paths,
    config: Config,
    hardware: HardwareInfo,
    backend: Arc<dyn RuntimeBackend>,
    host: String,
    port: u16,
) -> Result<(SharedState, watch::Receiver<bool>)> {
    let manager = ModelManager::new(paths.clone(), config.clone(), backend)?;
    Ok(AppState::new(config, paths, hardware, manager, host, port))
}

/// Run the daemon until Ctrl-C / SIGTERM / `POST /ohmygpu/v1/shutdown`.
pub async fn serve(opts: ServeOptions) -> Result<()> {
    let ServeOptions { paths, config } = opts;
    paths
        .ensure_dirs()
        .with_context(|| format!("creating {}", paths.base_dir().display()))?;

    let hardware = HardwareInfo::detect();
    tracing::info!(
        "hardware: {} / {} — {} ({} cores), {} RAM, backend: {}{}",
        hardware.platform,
        hardware.architecture,
        hardware.cpu.name,
        hardware.cpu.cores,
        hardware
            .system_memory_bytes
            .map(human_bytes)
            .unwrap_or_else(|| "?".into()),
        hardware.backend,
        hardware
            .gpu
            .as_ref()
            .map(|g| format!(" ({})", g.name))
            .unwrap_or_default()
    );

    let backend: Arc<dyn RuntimeBackend> = Arc::new(LlamaCppBackend::new(
        config.backend.llamacpp.clone(),
        &paths.runtimes_dir(),
        hardware.clone(),
    ));

    let bind_host = config.daemon.host.clone();
    let listener = TcpListener::bind((bind_host.as_str(), config.daemon.port))
        .await
        .with_context(|| format!("binding {}:{}", bind_host, config.daemon.port))?;
    let addr: SocketAddr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        tracing::warn!(
            "listening on a non-loopback address ({addr}); the API has no authentication"
        );
    }

    let (state, shutdown_rx) = build_state(
        paths.clone(),
        config,
        hardware,
        backend,
        addr.ip().to_string(),
        addr.port(),
    )?;

    let record = DaemonRecord {
        pid: std::process::id(),
        host: addr.ip().to_string(),
        port: addr.port(),
        version: ohmygpu_core::VERSION.to_string(),
        started_at: chrono::Utc::now(),
    };
    std::fs::write(
        paths.daemon_state_path(),
        serde_json::to_string_pretty(&record)?,
    )?;

    let availability = state.manager.backend_availability().await;
    tracing::info!(
        "backend llamacpp: {}",
        if availability.available {
            format!(
                "{} ({})",
                availability.version.as_deref().unwrap_or("?"),
                availability
                    .path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            )
        } else {
            availability
                .message
                .clone()
                .unwrap_or_else(|| "not available".into())
        }
    );
    tracing::info!(
        "OhMyGPU Runtime v{} listening on http://{addr}",
        ohmygpu_core::VERSION
    );
    tracing::info!(
        "  inference:  POST http://{addr}/v1/responses | POST http://{addr}/v1/chat/completions"
    );
    tracing::info!("  management: GET  http://{addr}/ohmygpu/v1/status");
    tracing::info!("  data dir:   {}", paths.base_dir().display());

    let app = router(state.clone());
    let graceful = shutdown_signal(shutdown_rx.clone());
    let server = axum::serve(listener, app).with_graceful_shutdown(graceful);

    // Give in-flight requests a moment, but never hang on a stuck stream.
    let force = async {
        shutdown_signal(shutdown_rx).await;
        tokio::time::sleep(Duration::from_secs(10)).await;
        tracing::warn!("connections still open after 10s; forcing shutdown");
    };
    tokio::select! {
        r = server => { r.context("server error")?; }
        _ = force => {}
    }

    tracing::info!("stopping models…");
    state.manager.stop_all().await;
    let _ = std::fs::remove_file(paths.daemon_state_path());
    tracing::info!("bye");
    Ok(())
}

async fn shutdown_signal(mut rx: watch::Receiver<bool>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    let api = async {
        loop {
            if *rx.borrow() {
                return;
            }
            if rx.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    };
    tokio::select! {
        _ = ctrl_c => tracing::info!("received Ctrl-C"),
        _ = terminate => tracing::info!("received SIGTERM"),
        _ = api => {},
    }
}

fn human_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    format!("{:.0} GiB", b as f64 / GB)
}
