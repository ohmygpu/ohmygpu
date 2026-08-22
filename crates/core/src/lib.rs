//! ohmygpu_core — GPU-agnostic foundations shared by the daemon and the CLI.
//!
//! * [`paths`]     — where everything lives (`$OHMYGPU_HOME`, default `~/.config/ohmygpu`)
//! * [`config`]    — `config.toml` + environment overrides
//! * [`hardware`]  — what machine are we on (platform, CPU, memory, GPU/backend)
//! * [`catalog`]   — the curated set of supported models + model reference parsing
//! * [`registry`]  — installed models (`registry.json`)
//! * [`download`]  — resumable HTTP downloads (Hugging Face)
//! * [`gguf`]      — GGUF header reader (architecture, native context length) for installed models
//! * [`recipe`]    — per-model recipes (schema v1): YAML/JSON loader, validation, JSON Schema
//! * [`lifecycle`] — the explicit model state machine shared with clients
//!
//! Nothing in here spawns inference or knows about GPUs at build time.

pub mod catalog;
pub mod config;
pub mod download;
pub mod gguf;
pub mod hardware;
pub mod lifecycle;
pub mod paths;
pub mod recipe;
pub mod registry;

pub use catalog::{CatalogEntry, ModelRef, CATALOG};
pub use config::Config;
pub use hardware::HardwareInfo;
pub use lifecycle::{DownloadProgress, ModelState};
pub use paths::Paths;
pub use recipe::{Recipe, RecipeError, ResolvedVariant};
pub use registry::{
    InstalledModel, Modalities, Modality, ModelCapabilities, ModelRegistry, ModelSource,
};

/// Crate/product version, reported by the daemon and CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
