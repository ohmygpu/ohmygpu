use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Daemon settings
    #[serde(default)]
    pub daemon: DaemonConfig,

    /// Model settings
    #[serde(default)]
    pub models: ModelsConfig,

    /// Inference settings
    #[serde(default)]
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Directory to store models
    #[serde(default)]
    pub directory: Option<PathBuf>,

    /// HuggingFace token for private models
    #[serde(default)]
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Default max tokens
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Default temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Default top-p
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Use GPU acceleration (Metal on macOS, CUDA on Linux)
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
}

fn default_port() -> u16 {
    11434
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_use_gpu() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            models: ModelsConfig::default(),
            inference: InferenceConfig::default(),
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            directory: None,
            hf_token: None,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            use_gpu: default_use_gpu(),
        }
    }
}

impl Config {
    /// Load config from default location
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Save config to default location
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        fs::write(&config_path, content)?;
        Ok(())
    }

    /// Get the config file path
    pub fn config_path() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "ohmygpu", "ohmygpu")
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    /// Get the models directory (from config or default)
    pub fn models_dir(&self) -> Result<PathBuf> {
        if let Some(dir) = &self.models.directory {
            Ok(dir.clone())
        } else {
            let dirs = directories::ProjectDirs::from("com", "ohmygpu", "ohmygpu")
                .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;
            Ok(dirs.data_dir().join("models"))
        }
    }
}
