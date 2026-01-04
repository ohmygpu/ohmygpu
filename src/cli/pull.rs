use anyhow::Result;

use crate::downloaders::{github::GitHubDownloader, huggingface::HuggingFaceDownloader, Downloader};
use crate::models::registry::ModelRegistry;

pub async fn execute(model: &str, file: Option<&str>) -> Result<()> {
    let registry = ModelRegistry::load()?;

    // Determine the source based on the model identifier
    let downloader: Box<dyn Downloader> = if model.starts_with("github:") {
        Box::new(GitHubDownloader::new())
    } else {
        Box::new(HuggingFaceDownloader::new())
    };

    println!("Pulling model: {}", model);

    let model_info = downloader.download(model, file).await?;

    // Register the model
    let mut registry = registry;
    registry.add(model_info)?;
    registry.save()?;

    println!("Successfully pulled: {}", model);
    Ok(())
}
