//! Model loading and inference for various architectures

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as llama_model;
use candle_transformers::models::phi as phi_model;
use ohmygpu_runtime_api::ChatToken;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use crate::sampling::Sampler;

pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: String,
}

pub struct LoadedModel {
    model: ModelType,
    tokenizer: Tokenizer,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    eos_token_id: Option<u32>,
}

enum ModelType {
    Llama {
        model: llama_model::Llama,
        cache: Mutex<llama_model::Cache>,
    },
    Phi(Mutex<phi_model::Model>),
}

impl LoadedModel {
    pub fn load(model_path: &Path, device: &Device) -> Result<Self> {
        tracing::info!("Loading model from {:?}", model_path);

        // Determine dtype based on device
        let dtype = match device {
            Device::Metal(_) => DType::F32, // Metal works best with F32
            Device::Cuda(_) => DType::BF16, // CUDA can use BF16
            Device::Cpu => DType::F32,
        };

        // Find model files
        let config_path = find_file(model_path, "config.json")?;
        let tokenizer_path = find_file(model_path, "tokenizer.json")?;
        let weights_path = find_weights(model_path)?;

        tracing::info!("Config: {:?}", config_path);
        tracing::info!("Tokenizer: {:?}", tokenizer_path);
        tracing::info!("Weights: {:?}", weights_path);

        // Load config to determine model type
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

        let model_type_str = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        tracing::info!("Model type: {}", model_type_str);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Get EOS token ID
        let eos_token_id = get_eos_token_id(&tokenizer);
        tracing::info!("EOS token ID: {:?}", eos_token_id);

        // Load model weights
        let vb = if weights_path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], dtype, device)?
            }
        } else {
            anyhow::bail!("Only safetensors format is supported currently");
        };

        // Load model based on type
        let model = match model_type_str {
            "phi" | "phi-msft" | "phi2" => {
                tracing::info!("Loading Phi model");
                let config: phi_model::Config = serde_json::from_str(&config_str)?;
                let model = phi_model::Model::new(&config, vb)?;
                ModelType::Phi(Mutex::new(model))
            }
            _ => {
                // Default to Llama for llama, mistral, etc.
                tracing::info!("Loading Llama-style model");
                let config: llama_model::LlamaConfig = serde_json::from_str(&config_str)?;
                let config = config.into_config(false); // use_flash_attn = false
                let model = llama_model::Llama::load(vb, &config)?;
                let cache = llama_model::Cache::new(true, dtype, &config, device)?;
                ModelType::Llama {
                    model,
                    cache: Mutex::new(cache),
                }
            }
        };

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            eos_token_id,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResult> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let input_ids = tokens.get_ids();
        let mut all_tokens = input_ids.to_vec();

        let mut sampler = Sampler::new(temperature, 0.9, 42);

        let mut generated = 0;
        let mut finish_reason = "length".to_string();

        for _ in 0..max_tokens {
            let input = Tensor::new(&all_tokens[..], &self.device)?
                .unsqueeze(0)?;

            let logits = self.forward(&input, all_tokens.len())?;
            let logits = logits.squeeze(0)?;
            let last_logits = logits.get(logits.dim(0)? - 1)?;

            let next_token = sampler.sample(&last_logits)?;

            if Some(next_token) == self.eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            all_tokens.push(next_token);
            generated += 1;
        }

        // Decode only the generated tokens
        let generated_tokens = &all_tokens[input_ids.len()..];
        let text = self
            .tokenizer
            .decode(generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        Ok(GenerationResult {
            text,
            tokens_generated: generated,
            finish_reason,
        })
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<ChatToken>,
    ) -> Result<()> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let input_ids = tokens.get_ids();
        let mut all_tokens = input_ids.to_vec();

        let mut sampler = Sampler::new(temperature, 0.9, 42);

        let mut prev_text_len = 0;

        for i in 0..max_tokens {
            let input = Tensor::new(&all_tokens[..], &self.device)?
                .unsqueeze(0)?;

            let logits = self.forward(&input, all_tokens.len())?;
            let logits = logits.squeeze(0)?;
            let last_logits = logits.get(logits.dim(0)? - 1)?;

            let next_token = sampler.sample(&last_logits)?;

            if Some(next_token) == self.eos_token_id {
                let _ = tx
                    .send(ChatToken {
                        content: String::new(),
                        finish_reason: Some("stop".to_string()),
                    })
                    .await;
                return Ok(());
            }

            all_tokens.push(next_token);

            // Decode current text and send delta
            let generated_tokens = &all_tokens[input_ids.len()..];
            let current_text = self
                .tokenizer
                .decode(generated_tokens, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

            if current_text.len() > prev_text_len {
                let delta = current_text[prev_text_len..].to_string();
                prev_text_len = current_text.len();

                if tx
                    .send(ChatToken {
                        content: delta,
                        finish_reason: None,
                    })
                    .await
                    .is_err()
                {
                    // Receiver dropped
                    return Ok(());
                }
            }

            // Small yield to allow other tasks
            if i % 10 == 0 {
                tokio::task::yield_now().await;
            }
        }

        // Send final token indicating we hit length limit
        let _ = tx
            .send(ChatToken {
                content: String::new(),
                finish_reason: Some("length".to_string()),
            })
            .await;

        Ok(())
    }

    fn forward(&self, input: &Tensor, seq_len: usize) -> Result<Tensor> {
        match &self.model {
            ModelType::Llama { model, cache } => {
                let mut cache_guard = cache.lock().unwrap();
                Ok(model.forward(input, seq_len - 1, &mut cache_guard)?)
            }
            ModelType::Phi(m) => {
                let mut model_guard = m.lock().unwrap();
                Ok(model_guard.forward(input)?)
            }
        }
    }

}

fn get_eos_token_id(tokenizer: &Tokenizer) -> Option<u32> {
    let vocab = tokenizer.get_vocab(true);
    vocab
        .get("</s>")
        .or_else(|| vocab.get("<|endoftext|>"))
        .or_else(|| vocab.get("<eos>"))
        .or_else(|| vocab.get("<|end|>"))
        .copied()
}

fn find_file(model_path: &Path, filename: &str) -> Result<std::path::PathBuf> {
    let direct = model_path.join(filename);
    if direct.exists() {
        return Ok(direct);
    }

    // Search in subdirectories
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.file_name().map(|n| n == filename).unwrap_or(false) {
            return Ok(path);
        }
    }

    anyhow::bail!("Could not find {} in {:?}", filename, model_path)
}

fn find_weights(model_path: &Path) -> Result<std::path::PathBuf> {
    // Look for safetensors files
    let patterns = [
        "model.safetensors",
        "pytorch_model.safetensors",
    ];

    for pattern in &patterns {
        let path = model_path.join(pattern);
        if path.exists() {
            return Ok(path);
        }
    }

    // Search for any .safetensors file
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "Could not find model weights (safetensors) in {:?}",
        model_path
    )
}
