//! Shared per-daemon state handed to every handler.

use std::sync::Arc;
use std::time::Instant;

use ohmygpu_core::config::Config;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_core::paths::Paths;
use tokio::sync::watch;

use crate::manager::ModelManager;

pub struct AppState {
    pub config: Config,
    pub paths: Paths,
    pub hardware: HardwareInfo,
    pub manager: Arc<ModelManager>,
    pub started_at: Instant,
    /// Actual bound address (host/port after binding).
    pub host: String,
    pub port: u16,
    shutdown_tx: watch::Sender<bool>,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    pub fn new(
        config: Config,
        paths: Paths,
        hardware: HardwareInfo,
        manager: Arc<ModelManager>,
        host: String,
        port: u16,
    ) -> (Arc<Self>, watch::Receiver<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let state = Arc::new(Self {
            config,
            paths,
            hardware,
            manager,
            started_at: Instant::now(),
            host,
            port,
            shutdown_tx,
        });
        (state, shutdown_rx)
    }

    /// Ask the server to shut down gracefully (used by `POST /ohmygpu/v1/shutdown`).
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}
