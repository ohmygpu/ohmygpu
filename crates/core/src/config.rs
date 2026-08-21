//! Runtime configuration: `config.toml` under the base directory, with a few
//! environment overrides that matter for embedding.
//!
//! ```toml
//! [daemon]
//! host = "127.0.0.1"
//! port = 10692
//!
//! [models]
//! # storage_path = "/custom/models/dir"   # default: <base>/models
//! # hf_token = "hf_..."                   # or HF_TOKEN env
//!
//! [inference]
//! auto_start = false      # start an installed model on first inference request
//!
//! [backend.llamacpp]
//! # server_path = "/usr/local/bin/llama-server"   # or OHMYGPU_LLAMA_SERVER env
//! auto_install = true      # download the matching llama.cpp release if not found
//! release = "latest"       # or a tag like "b10437"
//! context_length = 8192
//! # gpu_layers = 999
//! # threads = 8
//! startup_timeout_secs = 600
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::Paths;

pub const DEFAULT_PORT: u16 = 10692;
pub const DEFAULT_HOST: &str = "127.0.0.1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Config {
    pub daemon: DaemonConfig,
    pub models: ModelsConfig,
    pub inference: InferenceConfig,
    pub backend: BackendConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ModelsConfig {
    /// Overrides `<base>/models` when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_path: Option<PathBuf>,
    /// Hugging Face token for gated repositories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct InferenceConfig {
    /// If true, an inference request for an installed-but-stopped model starts it
    /// (and waits) instead of failing with `model_not_running`.
    pub auto_start: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct BackendConfig {
    pub llamacpp: LlamaCppConfig,
    pub whisper: WhisperConfig,
}

/// whisper.cpp release the runtime installs by default (Linux/Windows: the
/// official assets of that tag; macOS: `whisper-server` built by our release
/// workflow for the same tag).
pub const WHISPER_DEFAULT_RELEASE: &str = "b4938";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WhisperConfig {
    /// Explicit `whisper-server` binary. When unset the daemon looks at
    /// `OHMYGPU_WHISPER_SERVER`, the managed install dir, then `PATH`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_path: Option<PathBuf>,
    /// Download a release into `<base>/runtimes/whisper/` if no binary is found.
    pub auto_install: bool,
    /// whisper.cpp release tag to install (pinned; see `WHISPER_DEFAULT_RELEASE`).
    pub release: String,
    /// CPU threads for transcription (`None` = whisper.cpp default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    /// How long to wait for the server to load the model before failing the start.
    pub startup_timeout_secs: u64,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            server_path: None,
            auto_install: true,
            release: WHISPER_DEFAULT_RELEASE.to_string(),
            threads: None,
            startup_timeout_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LlamaCppConfig {
    /// Explicit `llama-server` binary. When unset the daemon looks at
    /// `OHMYGPU_LLAMA_SERVER`, the managed install dir, then `PATH`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_path: Option<PathBuf>,
    /// Download the official release into `<base>/runtimes/llamacpp/` if no
    /// binary can be found.
    pub auto_install: bool,
    /// GitHub release tag to install, or `"latest"`.
    pub release: String,
    /// Default context window for started models.
    pub context_length: u32,
    /// Default GPU layer offload (`None` = let llama.cpp decide: all that fit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    /// Default CPU threads (`None` = llama.cpp default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    /// How long to wait for a model to become ready before failing the start.
    pub startup_timeout_secs: u64,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            server_path: None,
            auto_install: true,
            release: "latest".to_string(),
            context_length: 8192,
            gpu_layers: None,
            threads: None,
            startup_timeout_secs: 600,
        }
    }
}

impl Config {
    /// Load `<base>/config.toml` (defaults if missing) and apply env overrides.
    pub fn load(paths: &Paths) -> Result<Self> {
        let mut cfg = Self::load_file(&paths.config_path())?;
        cfg.apply_env();
        Ok(cfg)
    }

    /// Load from an explicit file without env overrides (used by tests).
    pub fn load_file(path: &Path) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let cfg: Config =
                toml::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
            Ok(cfg)
        } else {
            Ok(Config::default())
        }
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        let path = paths.config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Environment overrides: `OHMYGPU_HOST`, `OHMYGPU_PORT`,
    /// `OHMYGPU_LLAMA_SERVER`, `OHMYGPU_WHISPER_SERVER`, `HF_TOKEN`.
    pub fn apply_env(&mut self) {
        if let Some(h) = env_nonempty("OHMYGPU_HOST") {
            self.daemon.host = h;
        }
        if let Some(p) = env_nonempty("OHMYGPU_PORT").and_then(|p| p.parse().ok()) {
            self.daemon.port = p;
        }
        if let Some(p) = env_nonempty("OHMYGPU_LLAMA_SERVER") {
            self.backend.llamacpp.server_path = Some(PathBuf::from(p));
        }
        if let Some(p) = env_nonempty("OHMYGPU_WHISPER_SERVER") {
            self.backend.whisper.server_path = Some(PathBuf::from(p));
        }
        if let Some(t) = env_nonempty("HF_TOKEN") {
            self.models.hf_token = Some(t);
        }
    }

    /// Effective models directory.
    pub fn models_dir(&self, paths: &Paths) -> PathBuf {
        self.models
            .storage_path
            .clone()
            .unwrap_or_else(|| paths.models_dir())
    }

    /// Get a dotted key as a string (for `omg config <key>`).
    pub fn get(&self, key: &str) -> Option<String> {
        Some(match key {
            "daemon.host" => self.daemon.host.clone(),
            "daemon.port" => self.daemon.port.to_string(),
            "models.storage_path" => self
                .models
                .storage_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            "models.hf_token" => self
                .models
                .hf_token
                .as_ref()
                .map(|_| "***".to_string())
                .unwrap_or_default(),
            "inference.auto_start" => self.inference.auto_start.to_string(),
            "backend.llamacpp.server_path" => self
                .backend
                .llamacpp
                .server_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            "backend.llamacpp.auto_install" => self.backend.llamacpp.auto_install.to_string(),
            "backend.llamacpp.release" => self.backend.llamacpp.release.clone(),
            "backend.llamacpp.context_length" => self.backend.llamacpp.context_length.to_string(),
            "backend.llamacpp.gpu_layers" => self
                .backend
                .llamacpp
                .gpu_layers
                .map(|v| v.to_string())
                .unwrap_or_default(),
            "backend.llamacpp.threads" => self
                .backend
                .llamacpp
                .threads
                .map(|v| v.to_string())
                .unwrap_or_default(),
            "backend.llamacpp.startup_timeout_secs" => {
                self.backend.llamacpp.startup_timeout_secs.to_string()
            }
            _ => return None,
        })
    }

    /// Set a dotted key from a string. Empty value clears optional keys.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let opt = |v: &str| {
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        };
        match key {
            "daemon.host" => self.daemon.host = value.to_string(),
            "daemon.port" => self.daemon.port = value.parse().context("port must be a number")?,
            "models.storage_path" => self.models.storage_path = opt(value).map(PathBuf::from),
            "models.hf_token" => self.models.hf_token = opt(value),
            "inference.auto_start" => {
                self.inference.auto_start = value.parse().context("expected true/false")?
            }
            "backend.llamacpp.server_path" => {
                self.backend.llamacpp.server_path = opt(value).map(PathBuf::from)
            }
            "backend.llamacpp.auto_install" => {
                self.backend.llamacpp.auto_install = value.parse().context("expected true/false")?
            }
            "backend.llamacpp.release" => self.backend.llamacpp.release = value.to_string(),
            "backend.llamacpp.context_length" => {
                self.backend.llamacpp.context_length = value.parse().context("expected a number")?
            }
            "backend.llamacpp.gpu_layers" => {
                self.backend.llamacpp.gpu_layers = opt(value)
                    .map(|v| v.parse())
                    .transpose()
                    .context("expected a number")?
            }
            "backend.llamacpp.threads" => {
                self.backend.llamacpp.threads = opt(value)
                    .map(|v| v.parse())
                    .transpose()
                    .context("expected a number")?
            }
            "backend.llamacpp.startup_timeout_secs" => {
                self.backend.llamacpp.startup_timeout_secs =
                    value.parse().context("expected a number")?
            }
            _ => anyhow::bail!("unknown config key: {key}"),
        }
        Ok(())
    }

    /// All keys understood by [`Config::get`] / [`Config::set`].
    pub const KEYS: &'static [&'static str] = &[
        "daemon.host",
        "daemon.port",
        "models.storage_path",
        "models.hf_token",
        "inference.auto_start",
        "backend.llamacpp.server_path",
        "backend.llamacpp.auto_install",
        "backend.llamacpp.release",
        "backend.llamacpp.context_length",
        "backend.llamacpp.gpu_layers",
        "backend.llamacpp.threads",
        "backend.llamacpp.startup_timeout_secs",
    ];
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_local_and_sane() {
        let c = Config::default();
        assert_eq!(c.daemon.host, "127.0.0.1");
        assert_eq!(c.daemon.port, 10692);
        assert!(c.backend.llamacpp.auto_install);
        assert!(!c.inference.auto_start);
    }

    #[test]
    fn partial_toml_fills_defaults_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        std::fs::write(
            paths.config_path(),
            "[daemon]\nport = 1234\n[backend.llamacpp]\ncontext_length = 2048\n",
        )
        .unwrap();
        let c = Config::load_file(&paths.config_path()).unwrap();
        assert_eq!(c.daemon.port, 1234);
        assert_eq!(c.daemon.host, "127.0.0.1");
        assert_eq!(c.backend.llamacpp.context_length, 2048);
        assert_eq!(c.backend.llamacpp.release, "latest");
        c.save(&paths).unwrap();
        let again = Config::load_file(&paths.config_path()).unwrap();
        assert_eq!(again, c);
    }

    #[test]
    fn get_and_set_cover_all_keys() {
        let mut c = Config::default();
        for k in Config::KEYS {
            assert!(c.get(k).is_some(), "get {k}");
        }
        c.set("daemon.port", "9999").unwrap();
        assert_eq!(c.get("daemon.port").unwrap(), "9999");
        c.set("backend.llamacpp.gpu_layers", "12").unwrap();
        assert_eq!(c.backend.llamacpp.gpu_layers, Some(12));
        c.set("backend.llamacpp.gpu_layers", "").unwrap();
        assert_eq!(c.backend.llamacpp.gpu_layers, None);
        assert!(c.set("nope", "1").is_err());
        assert!(c.set("daemon.port", "abc").is_err());
    }
}
