//! Image generation command

use anyhow::Result;
use candle_core::Device;
use ohmygpu_core::Config;
use ohmygpu_runtime_diffusion::{detect_model_type, load_model, ImageGenRequest};
use std::path::PathBuf;
use chrono::Local;

pub async fn execute(
    model: &str,
    prompt: &str,
    output: &str,
    width: u32,
    height: u32,
    steps: u32,
    guidance_scale: f32,
    negative_prompt: Option<&str>,
    seed: Option<u64>,
    cpu: bool,
) -> Result<()> {
    println!("Image Generation");
    println!("================");
    println!("Model: {}", model);
    println!("Prompt: {}", prompt);
    println!("Size: {}x{}", width, height);
    println!("Steps: {}", steps);
    println!("Guidance scale: {}", guidance_scale);
    if let Some(seed) = seed {
        println!("Seed: {}", seed);
    }
    println!();

    // Resolve model path - try local first, then download from HuggingFace
    let model_path = resolve_model_path(model)?;

    println!("Loading model from: {}", model_path.display());

    // Detect model type
    let model_type = detect_model_type(&model_path)?;
    println!("Detected model type: {:?}", model_type);

    // Setup device
    let device = if cpu {
        println!("Using CPU (this will be slow)");
        Device::Cpu
    } else {
        #[cfg(feature = "metal")]
        {
            println!("Using Metal GPU");
            Device::new_metal(0)?
        }
        #[cfg(feature = "cuda")]
        {
            println!("Using CUDA GPU");
            Device::new_cuda(0)?
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            println!("No GPU acceleration available, using CPU");
            Device::Cpu
        }
    };

    // Load model
    println!("\nLoading model...");
    let pipeline = load_model(&model_path, model_type, &device)?;
    println!("Model loaded: {}", pipeline.name());

    // Create request
    let request = ImageGenRequest {
        prompt: prompt.to_string(),
        negative_prompt: negative_prompt.map(|s| s.to_string()),
        width,
        height,
        steps,
        guidance_scale,
        seed,
    };

    // Generate image
    println!("\nGenerating image...");
    let start = std::time::Instant::now();
    let response = pipeline.generate(&request)?;
    let elapsed = start.elapsed();
    println!("Generation completed in {:.2}s", elapsed.as_secs_f64());

    // Resolve output path
    let output_path = resolve_output_path(output)?;

    // Save image
    println!("\nSaving to: {}", output_path.display());
    save_image(&response.pixels, response.width, response.height, &output_path)?;

    println!("\nDone!");
    Ok(())
}

fn resolve_model_path(model: &str) -> Result<PathBuf> {

    // Check if it's an absolute path
    let path = PathBuf::from(model);
    if path.is_absolute() && path.exists() {
        return Ok(path);
    }

    // Check if it's a relative path that exists
    if path.exists() {
        return Ok(path.canonicalize()?);
    }

    // Try to load from config storage path
    if let Ok(config) = Config::load() {
        let storage_path = PathBuf::from(&config.models.storage_path);
        let model_path = storage_path.join(model);
        if model_path.exists() {
            return Ok(model_path);
        }

        // Try with repo name (e.g., "Tongyi-MAI--Z-Image-Turbo")
        let hf_name = model.replace('/', "--");
        let model_dir = storage_path.join(&hf_name);
        if model_dir.exists() {
            // Check if it has snapshots subdirectory (HuggingFace structure)
            let snapshots = model_dir.join("snapshots");
            if snapshots.exists() {
                if let Some(entry) = std::fs::read_dir(&snapshots)?.next() {
                    return Ok(entry?.path());
                }
            }
            return Ok(model_dir);
        }

        // Also check with "models--" prefix for backwards compatibility
        let hf_path = storage_path.join(format!("models--{}", hf_name));
        if hf_path.exists() {
            let snapshots = hf_path.join("snapshots");
            if snapshots.exists() {
                if let Some(entry) = std::fs::read_dir(&snapshots)?.next() {
                    return Ok(entry?.path());
                }
            }
            return Ok(hf_path);
        }
    }

    // Treat as HuggingFace repo ID - download all required files
    println!("Model not found locally, downloading from HuggingFace: {}", model);

    // Use configured storage path or default
    let cache_dir = Config::load()
        .map(|c| PathBuf::from(&c.models.storage_path))
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config/ohmygpu/models")
        });

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.clone())
        .build()?;
    let repo = api.model(model.to_string());

    // Check if this is a valid HuggingFace repo by trying to get a file
    // The repo path is derived from HuggingFace cache structure
    // Files are downloaded to ~/.cache/huggingface/hub/models--<repo>/snapshots/<hash>/

    // Force download of essential files for Z-Image
    println!("Downloading tokenizer...");
    let _ = repo.get("tokenizer/tokenizer.json")?;

    println!("Downloading text encoder...");
    let _ = repo.get("text_encoder/config.json");
    for i in 1..=3 {
        let file = format!("text_encoder/model-{:05}-of-00003.safetensors", i);
        if let Err(e) = repo.get(&file) {
            println!("Warning: could not download {}: {}", file, e);
        }
    }

    println!("Downloading transformer...");
    let _ = repo.get("transformer/config.json")?;
    for i in 1..=3 {
        let file = format!("transformer/diffusion_pytorch_model-{:05}-of-00003.safetensors", i);
        if let Err(e) = repo.get(&file) {
            println!("Warning: could not download {}: {}", file, e);
        }
    }

    println!("Downloading VAE...");
    let _ = repo.get("vae/config.json");
    let _ = repo.get("vae/diffusion_pytorch_model.safetensors")?;

    // Get the model path from our configured cache
    let hf_name = model.replace('/', "--");
    // hf-hub adds "models--" prefix
    let model_cache_with_prefix = cache_dir.join(format!("models--{}", hf_name));
    let model_cache_clean = cache_dir.join(&hf_name);

    // Auto-rename: remove "models--" prefix
    if model_cache_with_prefix.exists() && !model_cache_clean.exists() {
        std::fs::rename(&model_cache_with_prefix, &model_cache_clean)?;
        println!("Renamed to: {}", model_cache_clean.display());
    }

    let snapshots = model_cache_clean.join("snapshots");
    if snapshots.exists() {
        // Get the most recent snapshot
        let mut entries: Vec<_> = std::fs::read_dir(&snapshots)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));

        if let Some(entry) = entries.first() {
            return Ok(entry.path());
        }
    }

    anyhow::bail!(
        "Failed to download model: {}. Check your internet connection and try again.",
        model
    )
}

/// Resolve output path, defaulting to ~/Documents/ohmygpu/
fn resolve_output_path(output: &str) -> Result<PathBuf> {
    let path = PathBuf::from(output);

    // If it's an absolute path, use as-is
    if path.is_absolute() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        return Ok(path);
    }

    // Default output directory: ~/Documents/ohmygpu/
    let output_dir = dirs::document_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
        .join("ohmygpu");

    // Create directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;

    // If output is the default "output.png", generate a timestamped filename
    if output == "output.png" {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        Ok(output_dir.join(format!("image_{}.png", timestamp)))
    } else {
        Ok(output_dir.join(output))
    }
}

fn save_image(pixels: &[u8], width: u32, height: u32, path: &PathBuf) -> Result<()> {
    // pixels are in RGB format, convert to image
    let img = image::RgbImage::from_raw(width, height, pixels.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from pixels"))?;

    img.save(path)?;
    Ok(())
}
