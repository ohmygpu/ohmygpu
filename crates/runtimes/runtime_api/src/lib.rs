//! ohmygpu_runtime_api - Runtime API traits and types
//!
//! This crate defines the contract for pluggable inference runtimes.
//! Each runtime (candle, llamacpp, etc.) implements these traits.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Capabilities that a runtime can provide
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeCaps {
    pub chat: bool,
    pub completions: bool,
    pub embeddings: bool,
    pub images: bool,
    pub audio: bool,
    pub streaming: bool,
}

/// Runtime status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeStatus {
    Unloaded,
    Loading,
    Ready,
    Error,
}

/// Configuration for a runtime instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model_path: PathBuf,
    pub gpu_id: Option<u32>,
    pub vram_budget_mb: Option<u64>,
    pub cpu_threads: Option<u32>,
}

/// Chat message for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request for chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

/// Response from chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    pub tokens_used: u32,
    pub finish_reason: String,
}

/// A single token from streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToken {
    pub content: String,
    pub finish_reason: Option<String>,
}

/// The main Runtime trait that all backends must implement
#[async_trait]
pub trait Runtime: Send + Sync {
    /// Get the capabilities of this runtime
    fn caps(&self) -> RuntimeCaps;

    /// Get current status
    fn status(&self) -> RuntimeStatus;

    /// Load a model
    async fn load(&mut self, config: RuntimeConfig) -> Result<()>;

    /// Unload the current model
    async fn unload(&mut self) -> Result<()>;

    /// Run chat completion (non-streaming)
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Run chat completion with streaming
    /// Returns a channel receiver for tokens
    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<ChatToken>>;
}
