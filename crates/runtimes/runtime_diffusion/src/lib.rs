//! ohmygpu_runtime_diffusion - Diffusion model inference runtime
//!
//! This crate provides image generation using diffusion models.
//! Supports FLUX and Z-Image (S3-DiT) architectures.

mod zimage;

use anyhow::Result;
use candle_core::Device;
use std::path::Path;

pub use zimage::ZImagePipeline;

/// Image generation request
#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
}

impl Default for ImageGenRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            steps: 9,
            guidance_scale: 5.0,
            seed: None,
        }
    }
}

/// Image generation response
pub struct ImageGenResponse {
    /// Raw pixel data (RGB, u8)
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Trait for diffusion model backends
pub trait DiffusionModel: Send + Sync {
    /// Generate an image from a text prompt
    fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse>;

    /// Get the model name
    fn name(&self) -> &str;
}

/// Supported diffusion model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionModelType {
    /// FLUX.1-dev or FLUX.1-schnell
    Flux,
    /// Z-Image-Turbo, Z-Image-Base, or Z-Image-Edit
    ZImage,
}

/// Load a diffusion model from a path
pub fn load_model(
    model_path: &Path,
    model_type: DiffusionModelType,
    device: &Device,
) -> Result<Box<dyn DiffusionModel>> {
    match model_type {
        DiffusionModelType::Flux => {
            anyhow::bail!("FLUX model loading not yet implemented")
        }
        DiffusionModelType::ZImage => {
            let pipeline = ZImagePipeline::load(model_path, device)?;
            Ok(Box::new(pipeline))
        }
    }
}

/// Detect model type from config.json
pub fn detect_model_type(model_path: &Path) -> Result<DiffusionModelType> {
    // Check for Z-Image specific files
    let transformer_config = model_path.join("transformer").join("config.json");
    if transformer_config.exists() {
        let config_str = std::fs::read_to_string(&transformer_config)?;
        if config_str.contains("ZImage") || config_str.contains("z_image") {
            return Ok(DiffusionModelType::ZImage);
        }
    }

    // Check for FLUX specific files
    let flux_config = model_path.join("flux1-dev.safetensors");
    if flux_config.exists() {
        return Ok(DiffusionModelType::Flux);
    }

    // Try to infer from directory name
    let dir_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    if dir_name.to_lowercase().contains("z-image") || dir_name.to_lowercase().contains("zimage") {
        return Ok(DiffusionModelType::ZImage);
    }

    if dir_name.to_lowercase().contains("flux") {
        return Ok(DiffusionModelType::Flux);
    }

    anyhow::bail!("Could not detect diffusion model type from path: {:?}", model_path)
}
