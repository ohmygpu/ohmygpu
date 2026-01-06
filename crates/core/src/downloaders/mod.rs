pub mod huggingface;

use anyhow::Result;
use async_trait::async_trait;

use crate::models::ModelInfo;

#[async_trait]
pub trait Downloader: Send + Sync {
    async fn download(&self, model_id: &str, file: Option<&str>) -> Result<ModelInfo>;
}

pub use huggingface::HuggingFaceDownloader;
