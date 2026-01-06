use anyhow::Result;
use ohmygpu_core::ModelRegistry;
use ohmygpu_runtime_api::{Runtime, RuntimeConfig, RuntimeStatus};
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

    /// Load a model by name. Returns Ok if model is already loaded or loads successfully.
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        // Check if already loaded
        {
            let current = self.current_model.read().await;
            if current.as_deref() == Some(model_name) {
                let runtime = self.runtime.read().await;
                if runtime.status() == RuntimeStatus::Ready {
                    tracing::info!("Model {} is already loaded", model_name);
                    return Ok(());
                }
            }
        }

        // Find model in registry
        let model_path = {
            let registry = self.registry.read().await;
            let model_info = registry
                .get(model_name)
                .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", model_name))?;
            model_info.path.clone()
        };

        tracing::info!("Loading model {} from {:?}", model_name, model_path);

        // Unload current model if any
        {
            let mut runtime = self.runtime.write().await;
            if runtime.status() == RuntimeStatus::Ready {
                runtime.unload().await?;
            }
        }

        // Load the new model
        {
            let mut runtime = self.runtime.write().await;
            let config = RuntimeConfig {
                model_path,
                gpu_id: Some(0),
                vram_budget_mb: None,
                cpu_threads: None,
            };
            runtime.load(config).await?;
        }

        // Update current model
        {
            let mut current = self.current_model.write().await;
            *current = Some(model_name.to_string());
        }

        tracing::info!("Model {} loaded successfully", model_name);
        Ok(())
    }

    /// Unload the current model
    pub async fn unload_model(&self) -> Result<()> {
        let mut runtime = self.runtime.write().await;
        if runtime.status() == RuntimeStatus::Ready {
            runtime.unload().await?;
        }
        let mut current = self.current_model.write().await;
        *current = None;
        Ok(())
    }
}
