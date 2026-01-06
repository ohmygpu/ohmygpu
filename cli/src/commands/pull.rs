use anyhow::Result;
use ohmygpu_core::downloaders::{Downloader, HuggingFaceDownloader};
use ohmygpu_core::ModelRegistry;

pub async fn execute(model: &str, file: Option<&str>) -> Result<()> {
    println!("Pulling model: {}", model);

    let downloader = HuggingFaceDownloader::new();
    let model_info = downloader.download(model, file).await?;

    // Register the model
    let mut registry = ModelRegistry::load()?;
    registry.add(model_info.clone())?;

    println!("\nModel downloaded successfully!");
    println!("  Name: {}", model_info.name);
    println!("  Type: {}", model_info.model_type.as_str());
    println!("  Size: {:.2} GB", model_info.size_bytes as f64 / 1_073_741_824.0);
    println!("  Path: {:?}", model_info.path);

    Ok(())
}
