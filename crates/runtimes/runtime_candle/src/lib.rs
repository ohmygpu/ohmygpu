//! ohmygpu_runtime_candle - Candle-based inference runtime
//!
//! This crate provides LLM inference using HuggingFace's candle library.
//! Supports Metal (macOS) and CUDA (Linux/Windows) acceleration.

mod model;
mod sampling;

use anyhow::Result;
use async_trait::async_trait;
use ohmygpu_runtime_api::{
    ChatRequest, ChatResponse, ChatToken, Runtime, RuntimeCaps, RuntimeConfig, RuntimeStatus,
};
use std::sync::Arc;
use tokio::sync::RwLock;

use model::LoadedModel;

pub struct CandleRuntime {
    status: RuntimeStatus,
    config: Option<RuntimeConfig>,
    model: Arc<RwLock<Option<LoadedModel>>>,
}

impl CandleRuntime {
    pub fn new() -> Self {
        Self {
            status: RuntimeStatus::Unloaded,
            config: None,
            model: Arc::new(RwLock::new(None)),
        }
    }

    fn get_device() -> Result<candle_core::Device> {
        #[cfg(feature = "metal")]
        {
            tracing::info!("Using Metal device");
            Ok(candle_core::Device::new_metal(0)?)
        }
        #[cfg(feature = "cuda")]
        {
            tracing::info!("Using CUDA device");
            Ok(candle_core::Device::new_cuda(0)?)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            tracing::info!("Using CPU device (no GPU features enabled)");
            Ok(candle_core::Device::Cpu)
        }
    }
}

impl Default for CandleRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runtime for CandleRuntime {
    fn caps(&self) -> RuntimeCaps {
        RuntimeCaps {
            chat: true,
            completions: true,
            embeddings: false,
            images: false,
            audio: false,
            streaming: true,
        }
    }

    fn status(&self) -> RuntimeStatus {
        self.status
    }

    async fn load(&mut self, config: RuntimeConfig) -> Result<()> {
        self.status = RuntimeStatus::Loading;
        tracing::info!("Loading model from {:?}", config.model_path);

        let device = Self::get_device()?;
        tracing::info!("Device: {:?}", device);

        // Load the model
        let model_path = config.model_path.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            LoadedModel::load(&model_path, &device)
        })
        .await??;

        *self.model.write().await = Some(loaded);
        self.config = Some(config);
        self.status = RuntimeStatus::Ready;

        tracing::info!("Model loaded successfully");
        Ok(())
    }

    async fn unload(&mut self) -> Result<()> {
        tracing::info!("Unloading model");
        *self.model.write().await = None;
        self.config = None;
        self.status = RuntimeStatus::Unloaded;
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if self.status != RuntimeStatus::Ready {
            anyhow::bail!("Model not loaded");
        }

        let model_guard = self.model.read().await;
        let model = model_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;

        // Build prompt from messages
        let prompt = build_chat_prompt(&request.messages);
        tracing::debug!("Prompt: {}", prompt);

        // Generate response
        let response = model.generate(
            &prompt,
            request.max_tokens as usize,
            request.temperature,
        )?;

        Ok(ChatResponse {
            content: response.text,
            tokens_used: response.tokens_generated as u32,
            finish_reason: response.finish_reason,
        })
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<ChatToken>> {
        if self.status != RuntimeStatus::Ready {
            anyhow::bail!("Model not loaded");
        }

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let model = self.model.clone();
        let prompt = build_chat_prompt(&request.messages);
        let max_tokens = request.max_tokens as usize;
        let temperature = request.temperature;

        tokio::spawn(async move {
            let model_guard = model.read().await;
            if let Some(loaded_model) = model_guard.as_ref() {
                if let Err(e) = loaded_model.generate_stream(&prompt, max_tokens, temperature, tx).await {
                    tracing::error!("Generation error: {}", e);
                }
            }
        });

        Ok(rx)
    }
}

fn build_chat_prompt(messages: &[ohmygpu_runtime_api::ChatMessage]) -> String {
    // Simple chat template (Llama-style)
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!(" {} ", msg.content));
            }
            _ => {
                prompt.push_str(&msg.content);
            }
        }
    }

    prompt
}
