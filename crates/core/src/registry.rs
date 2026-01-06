use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use crate::models::ModelInfo;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    #[serde(skip)]
    registry_path: PathBuf,
}

impl ModelRegistry {
    pub fn load() -> Result<Self> {
        let base_dir = Config::base_dir()?;
        fs::create_dir_all(&base_dir)?;

        let registry_path = Config::registry_path()?;

        let mut registry = if registry_path.exists() {
            let content = fs::read_to_string(&registry_path)?;
            serde_json::from_str(&content)?
        } else {
            ModelRegistry::default()
        };

        registry.registry_path = registry_path;
        Ok(registry)
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self)?;
        fs::write(&self.registry_path, content)?;
        Ok(())
    }

    pub fn add(&mut self, model: ModelInfo) -> Result<()> {
        self.models.insert(model.name.clone(), model);
        self.save()?;
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<Option<ModelInfo>> {
        let removed = self.models.remove(name);
        self.save()?;
        Ok(removed)
    }

    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Get the models directory: ~/.config/ohmygpu/models/
    pub fn models_dir() -> Result<PathBuf> {
        let dir = Config::base_dir()?.join("models");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}
