use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use super::ModelInfo;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    #[serde(skip)]
    config_path: PathBuf,
}

impl ModelRegistry {
    pub fn load() -> Result<Self> {
        let config_dir = Self::config_dir()?;
        fs::create_dir_all(&config_dir)?;

        let config_path = config_dir.join("registry.json");

        let mut registry = if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            ModelRegistry::default()
        };

        registry.config_path = config_path;
        Ok(registry)
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self)?;
        fs::write(&self.config_path, content)?;
        Ok(())
    }

    pub fn add(&mut self, model: ModelInfo) -> Result<()> {
        self.models.insert(model.name.clone(), model);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<()> {
        self.models.remove(name);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    pub fn models_dir() -> Result<PathBuf> {
        let dir = Self::data_dir()?.join("models");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn config_dir() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "ohmygpu", "omg")
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        Ok(dirs.config_dir().to_path_buf())
    }

    fn data_dir() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "ohmygpu", "omg")
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;
        Ok(dirs.data_dir().to_path_buf())
    }
}
