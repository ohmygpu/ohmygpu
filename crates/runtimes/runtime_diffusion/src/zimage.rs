//! Z-Image pipeline implementation
//!
//! Wraps candle-transformers' z_image module for image generation.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, get_noise, postprocess_image, AutoEncoderKL, Config,
    FlowMatchEulerDiscreteScheduler, SchedulerConfig, TextEncoderConfig, VaeConfig,
    ZImageTextEncoder, ZImageTransformer2DModel,
};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use crate::{DiffusionModel, ImageGenRequest, ImageGenResponse};

/// Z-Image scheduler constants
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Z-Image generation pipeline
pub struct ZImagePipeline {
    tokenizer: Tokenizer,
    text_encoder: ZImageTextEncoder,
    transformer: ZImageTransformer2DModel,
    vae: AutoEncoderKL,
    device: Device,
    dtype: DType,
    scheduler: Mutex<FlowMatchEulerDiscreteScheduler>,
}

impl ZImagePipeline {
    /// Load Z-Image pipeline from a model directory
    pub fn load(model_path: &Path, device: &Device) -> Result<Self> {
        let dtype = device.bf16_default_to_f32();

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer").join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {:?}", tokenizer_path);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load text encoder config
        let text_encoder_config_path = model_path.join("text_encoder").join("config.json");
        let text_encoder_cfg: TextEncoderConfig = if text_encoder_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&text_encoder_config_path)?)?
        } else {
            TextEncoderConfig::z_image()
        };

        // Load text encoder weights
        let text_encoder_files: Vec<std::path::PathBuf> = (1..=3)
            .map(|i| {
                model_path
                    .join("text_encoder")
                    .join(format!("model-{:05}-of-00003.safetensors", i))
            })
            .filter(|p| p.exists())
            .collect();

        if text_encoder_files.is_empty() {
            anyhow::bail!("Text encoder weights not found in {:?}", model_path.join("text_encoder"));
        }

        let files: Vec<&str> = text_encoder_files.iter().map(|p| p.to_str().unwrap()).collect();
        let text_encoder_weights =
            unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        let text_encoder = ZImageTextEncoder::new(&text_encoder_cfg, text_encoder_weights)?;

        // Load transformer config
        let transformer_config_path = model_path.join("transformer").join("config.json");
        let transformer_cfg: Config = if transformer_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&transformer_config_path)?)?
        } else {
            Config::z_image_turbo()
        };

        // Load transformer weights
        let transformer_files: Vec<std::path::PathBuf> = (1..=3)
            .map(|i| {
                model_path
                    .join("transformer")
                    .join(format!("diffusion_pytorch_model-{:05}-of-00003.safetensors", i))
            })
            .filter(|p| p.exists())
            .collect();

        if transformer_files.is_empty() {
            anyhow::bail!(
                "Transformer weights not found in {:?}",
                model_path.join("transformer")
            );
        }

        let files: Vec<&str> = transformer_files.iter().map(|p| p.to_str().unwrap()).collect();
        let transformer_weights =
            unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        let transformer = ZImageTransformer2DModel::new(&transformer_cfg, transformer_weights)?;

        // Load VAE config
        let vae_config_path = model_path.join("vae").join("config.json");
        let vae_cfg: VaeConfig = if vae_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&vae_config_path)?)?
        } else {
            VaeConfig::z_image()
        };

        // Load VAE weights
        let vae_path = model_path.join("vae").join("diffusion_pytorch_model.safetensors");
        if !vae_path.exists() {
            anyhow::bail!("VAE weights not found at {:?}", vae_path);
        }

        let vae_weights = unsafe {
            VarBuilder::from_mmaped_safetensors(&[vae_path.to_str().unwrap()], dtype, device)?
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vae_weights)?;

        // Initialize scheduler
        let scheduler_cfg = SchedulerConfig::z_image_turbo();
        let scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);

        Ok(Self {
            tokenizer,
            text_encoder,
            transformer,
            vae,
            device: device.clone(),
            dtype,
            scheduler: Mutex::new(scheduler),
        })
    }

    /// Format prompt for Qwen3 chat template
    fn format_prompt(prompt: &str) -> String {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    }

    /// Generate image from request
    fn generate_internal(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        let num_steps = request.steps as usize;

        // Set seed if provided
        if let Some(seed) = request.seed {
            self.device.set_seed(seed)?;
        }

        // Tokenize prompt
        let formatted_prompt = Self::format_prompt(&request.prompt);
        let tokens = self
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();

        let input_ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &self.device)?;

        // Get text embeddings
        let cap_feats = self.text_encoder.forward(&input_ids)?;
        let cap_mask = Tensor::ones((1, tokens.len()), DType::U8, &self.device)?;

        // Process negative prompt for CFG
        let (neg_cap_feats, neg_cap_mask) = if let Some(ref neg_prompt) = request.negative_prompt {
            if !neg_prompt.is_empty() && request.guidance_scale > 1.0 {
                let formatted_neg = Self::format_prompt(neg_prompt);
                let neg_tokens = self
                    .tokenizer
                    .encode(formatted_neg.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("Negative prompt tokenization failed: {}", e))?
                    .get_ids()
                    .to_vec();
                let neg_input_ids =
                    Tensor::from_vec(neg_tokens.clone(), (1, neg_tokens.len()), &self.device)?;
                let neg_feats = self.text_encoder.forward(&neg_input_ids)?;
                let neg_mask = Tensor::ones((1, neg_tokens.len()), DType::U8, &self.device)?;
                (Some(neg_feats), Some(neg_mask))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Calculate latent dimensions
        let patch_size = self.transformer.config().all_patch_size[0];
        let vae_align = 16;

        let height = request.height as usize;
        let width = request.width as usize;

        if height % vae_align != 0 || width % vae_align != 0 {
            anyhow::bail!(
                "Image dimensions must be divisible by {}. Got {}x{}",
                vae_align,
                width,
                height
            );
        }

        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        // Calculate shift
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );

        // Reset scheduler and set timesteps
        let mut scheduler = self.scheduler.lock().unwrap();
        *scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        scheduler.set_timesteps(num_steps, Some(mu));

        // Generate initial noise
        let mut latents = get_noise(1, 16, latent_h, latent_w, &self.device)?.to_dtype(self.dtype)?;
        latents = latents.unsqueeze(2)?; // Add frame dimension

        // Denoising loop
        for _step in 0..num_steps {
            let t = scheduler.current_timestep_normalized();
            let t_tensor =
                Tensor::from_vec(vec![t as f32], (1,), &self.device)?.to_dtype(self.dtype)?;

            // Model prediction
            let noise_pred = self
                .transformer
                .forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

            // Apply CFG
            let noise_pred = if request.guidance_scale > 1.0 {
                if let (Some(ref neg_feats), Some(ref neg_mask)) = (&neg_cap_feats, &neg_cap_mask) {
                    let neg_pred = self
                        .transformer
                        .forward(&latents, &t_tensor, neg_feats, neg_mask)?;
                    let diff = (&noise_pred - &neg_pred)?;
                    (&neg_pred + (diff * request.guidance_scale as f64)?)?
                } else {
                    noise_pred
                }
            } else {
                noise_pred
            };

            // Negate prediction (Z-Image specific)
            let noise_pred = noise_pred.neg()?;

            // Scheduler step
            let noise_pred_4d = noise_pred.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;
            let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;
            latents = prev_latents.unsqueeze(2)?;
        }

        drop(scheduler); // Release lock

        // VAE decode
        let latents = latents.squeeze(2)?;
        let image = self.vae.decode(&latents)?;

        // Post-process
        let image = postprocess_image(&image)?;
        let image = image.i(0)?; // Remove batch dimension

        // Convert to RGB u8 pixels
        let (c, h, w) = image.dims3()?;
        assert_eq!(c, 3, "Expected 3 channels");

        let image_data: Vec<u8> = image.flatten_all()?.to_vec1()?;

        Ok(ImageGenResponse {
            pixels: image_data,
            width: w as u32,
            height: h as u32,
        })
    }
}

impl DiffusionModel for ZImagePipeline {
    fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        self.generate_internal(request)
    }

    fn name(&self) -> &str {
        "Z-Image-Turbo"
    }
}

// ZImagePipeline is Send + Sync because:
// - tokenizer is thread-safe
// - candle models don't have interior mutability
// - scheduler is wrapped in Mutex
unsafe impl Send for ZImagePipeline {}
unsafe impl Sync for ZImagePipeline {}
