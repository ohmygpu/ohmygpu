use ohmygpu_core::ModelRegistry;
use ohmygpu_runtime_api::{Runtime, RuntimeStatus};
use ohmygpu_runtime_candle::CandleRuntime;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct AppState {
    pub registry: Arc<RwLock<ModelRegistry>>,
    pub runtime: Arc<RwLock<CandleRuntime>>,
    pub current_model: Arc<RwLock<Option<String>>>,
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            registry: Arc::new(RwLock::new(ModelRegistry::load()?)),
            runtime: Arc::new(RwLock::new(CandleRuntime::new())),
            current_model: Arc::new(RwLock::new(None)),
        })
    }

    pub async fn is_model_loaded(&self) -> bool {
        let runtime = self.runtime.read().await;
        runtime.status() == RuntimeStatus::Ready
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model.read().await.clone()
    }
}
