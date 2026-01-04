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
use crate::models::{registry::ModelRegistry, ModelInfo, ModelSource, ModelType};

const GITHUB_API_BASE: &str = "https://api.github.com";

pub struct GitHubDownloader {
    client: Client,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    assets: Vec<GhAsset>,
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    size: u64,
    browser_download_url: String,
}

impl GitHubDownloader {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("ohmygpu/0.1.0")
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    fn parse_repo(&self, model_id: &str) -> Result<String> {
        // Remove "github:" prefix if present
        let repo = model_id.strip_prefix("github:").unwrap_or(model_id);
        Ok(repo.to_string())
    }

    async fn get_latest_release(&self, repo: &str) -> Result<GhRelease> {
        let url = format!("{}/repos/{}/releases/latest", GITHUB_API_BASE, repo);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch GitHub release")?;

        if !response.status().is_success() {
            anyhow::bail!("No releases found for '{}'", repo);
        }

        let release: GhRelease = response.json().await?;
        Ok(release)
    }

    fn select_assets<'a>(&self, release: &'a GhRelease, requested_file: Option<&str>) -> Vec<&'a GhAsset> {
        if let Some(file) = requested_file {
            return release
                .assets
                .iter()
                .filter(|a| a.name.contains(file))
                .collect();
        }

        // Prioritize GGUF files, then other model files
        let model_extensions = [".gguf", ".safetensors", ".bin", ".pt", ".onnx"];

        release
            .assets
            .iter()
            .filter(|a| model_extensions.iter().any(|ext| a.name.ends_with(ext)))
            .collect()
    }

    async fn download_asset(&self, asset: &GhAsset, dest_dir: &PathBuf) -> Result<u64> {
        let response = self
            .client
            .get(&asset.browser_download_url)
            .send()
            .await
            .context("Failed to start download")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download {}: {}", asset.name, response.status());
        }

        let pb = ProgressBar::new(asset.size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        pb.set_message(asset.name.clone());

        let dest_path = dest_dir.join(&asset.name);
        let mut file = File::create(&dest_path)?;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error downloading chunk")?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message(format!("Downloaded {}", asset.name));
        Ok(downloaded)
    }
}

#[async_trait]
impl Downloader for GitHubDownloader {
    async fn download(&self, model_id: &str, file: Option<&str>) -> Result<ModelInfo> {
        let repo = self.parse_repo(model_id)?;

        println!("Fetching latest release from GitHub...");
        let release = self.get_latest_release(&repo).await?;

        let assets = self.select_assets(&release, file);
        if assets.is_empty() {
            anyhow::bail!("No model files found in release '{}'", release.tag_name);
        }

        // Create model directory
        let models_dir = ModelRegistry::models_dir()?;
        let model_name = format!("github--{}", repo.replace('/', "--"));
        let model_dir = models_dir.join(&model_name);
        fs::create_dir_all(&model_dir)?;

        println!(
            "Downloading {} file(s) from release {}",
            assets.len(),
            release.tag_name
        );

        let mut total_size = 0u64;
        let mut files = Vec::new();

        for asset in assets {
            total_size += self.download_asset(asset, &model_dir).await?;
            files.push(asset.name.clone());
        }

        Ok(ModelInfo {
            name: model_name,
            source: ModelSource::GitHub {
                repo,
                release: Some(release.tag_name),
            },
            model_type: ModelType::Unknown, // GitHub doesn't provide this info
            path: model_dir,
            size_bytes: total_size,
            files,
            downloaded_at: chrono::Utc::now(),
        })
    }
}
