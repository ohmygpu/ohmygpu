//! ohmygpu_core - Core library for model management
//!
//! This crate provides:
//! - HuggingFace API client and model downloads
//! - Local model repository and registry
//! - Model metadata and types

pub mod downloaders;
pub mod models;
pub mod registry;

pub use models::{ModelInfo, ModelSource, ModelType};
pub use registry::ModelRegistry;
