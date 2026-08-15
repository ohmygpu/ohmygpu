//! Filesystem layout. Everything OhMyGPU writes lives under one base directory so
//! that an application bundling the runtime can give it a private data dir.
//!
//! ```text
//! <base>/
//! ├── config.toml
//! ├── registry.json
//! ├── models/<model-id>/<file>.gguf
//! ├── runtimes/llamacpp/<tag>/llama-server
//! └── daemon.json
//! ```

use std::path::{Path, PathBuf};

/// Environment variable that overrides the base directory.
pub const HOME_ENV: &str = "OHMYGPU_HOME";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    base: PathBuf,
}

impl Paths {
    /// Explicit base directory (used by `--data-dir` and by tests).
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// `$OHMYGPU_HOME`, else `~/.config/ohmygpu`.
    pub fn from_env() -> Self {
        if let Some(dir) = std::env::var_os(HOME_ENV).filter(|v| !v.is_empty()) {
            return Self::new(PathBuf::from(dir));
        }
        Self::new(Self::default_base_dir())
    }

    pub fn default_base_dir() -> PathBuf {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join(".config").join("ohmygpu")
    }

    pub fn base_dir(&self) -> &Path {
        &self.base
    }
    pub fn config_path(&self) -> PathBuf {
        self.base.join("config.toml")
    }
    pub fn registry_path(&self) -> PathBuf {
        self.base.join("registry.json")
    }
    pub fn models_dir(&self) -> PathBuf {
        self.base.join("models")
    }
    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.models_dir().join(model_id)
    }
    pub fn runtimes_dir(&self) -> PathBuf {
        self.base.join("runtimes")
    }
    /// Written by a running daemon: `{ "pid": .., "port": .., "host": .. }`.
    pub fn daemon_state_path(&self) -> PathBuf {
        self.base.join("daemon.json")
    }

    /// Create the directories the daemon needs.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.base)?;
        std::fs::create_dir_all(self.models_dir())?;
        std::fs::create_dir_all(self.runtimes_dir())?;
        Ok(())
    }
}
