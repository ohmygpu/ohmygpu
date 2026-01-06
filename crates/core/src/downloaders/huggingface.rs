use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use super::Downloader;
use crate::models::{ModelInfo, ModelSource, ModelType};
use crate::registry::ModelRegistry;

const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_CDN_BASE: &str = "https://huggingface.co";

pub struct HuggingFaceDownloader {
    client: Client,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HfModelInfo {
    pub id: String,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: i64,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HfSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct HfSearchResult {
    #[serde(rename = "modelId")]
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: i64,
}

impl HuggingFaceDownloader {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("ohmygpu/0.1.0")
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<HfSearchResult>> {
        let url = format!(
            "{}/models?search={}&sort=downloads&direction=-1&limit=20",
            HF_API_BASE, query
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to search HuggingFace")?;

        let results: Vec<HfSearchResult> = response.json().await?;
        Ok(results)
    }

    pub async fn get_model_info(&self, repo_id: &str) -> Result<HfModelInfo> {
        let url = format!("{}/models/{}", HF_API_BASE, repo_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch model info")?;

        if !response.status().is_success() {
            anyhow::bail!("Model '{}' not found on HuggingFace", repo_id);
        }

        let info: HfModelInfo = response.json().await?;
        Ok(info)
    }

    fn select_files(&self, model_info: &HfModelInfo, requested_file: Option<&str>) -> Vec<String> {
        if let Some(file) = requested_file {
            return vec![file.to_string()];
        }

        // Prioritize GGUF files for LLMs (quantized, smaller)
        let gguf_files: Vec<_> = model_info
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".gguf"))
            .map(|s| s.rfilename.clone())
            .collect();

        if !gguf_files.is_empty() {
            // Prefer Q4_K_M quantization as a good balance
            if let Some(q4) = gguf_files.iter().find(|f| f.contains("Q4_K_M")) {
                return vec![q4.clone()];
            }
            // Otherwise take the first GGUF
            return vec![gguf_files[0].clone()];
        }

        // For other model types, get safetensors or pytorch files
        let model_files: Vec<_> = model_info
            .siblings
            .iter()
            .filter(|s| {
                s.rfilename.ends_with(".safetensors")
                    || s.rfilename.ends_with(".bin")
                    || s.rfilename.ends_with(".pt")
                    || s.rfilename == "config.json"
                    || s.rfilename == "tokenizer.json"
                    || s.rfilename == "tokenizer_config.json"
            })
            .map(|s| s.rfilename.clone())
            .collect();

        model_files
    }

    async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        dest_dir: &PathBuf,
    ) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{}", HF_CDN_BASE, repo_id, filename);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to start download")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download {}: {}", filename, response.status());
        }

        let total_size = response.content_length().unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        pb.set_message(filename.to_string());

        // Create subdirectories if needed
        let dest_path = dest_dir.join(filename);
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(&dest_path)?;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error downloading chunk")?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message(format!("Downloaded {}", filename));
        Ok(downloaded)
    }
}

impl Default for HuggingFaceDownloader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Downloader for HuggingFaceDownloader {
    async fn download(&self, model_id: &str, file: Option<&str>) -> Result<ModelInfo> {
        println!("Fetching model info from HuggingFace...");
        let hf_info = self.get_model_info(model_id).await?;

        let model_type = hf_info
            .pipeline_tag
            .as_ref()
            .map(|t| ModelType::from_pipeline_tag(t))
            .unwrap_or(ModelType::Unknown);

        // Create model directory
        let models_dir = ModelRegistry::models_dir()?;
        let model_name = model_id.replace('/', "--");
        let model_dir = models_dir.join(&model_name);
        fs::create_dir_all(&model_dir)?;

        // Select and download files
        let files_to_download = self.select_files(&hf_info, file);

        if files_to_download.is_empty() {
            anyhow::bail!("No suitable files found for model '{}'", model_id);
        }

        println!(
            "Downloading {} file(s) to {:?}",
            files_to_download.len(),
            model_dir
        );

        let mut total_size = 0u64;
        for filename in &files_to_download {
            total_size += self.download_file(model_id, filename, &model_dir).await?;
        }

        Ok(ModelInfo {
            name: model_name,
            source: ModelSource::HuggingFace {
                repo_id: model_id.to_string(),
            },
            model_type,
            path: model_dir,
            size_bytes: total_size,
            files: files_to_download,
            downloaded_at: chrono::Utc::now(),
        })
    }
}
