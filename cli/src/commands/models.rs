use anyhow::Result;
use ohmygpu_core::ModelRegistry;

pub async fn execute() -> Result<()> {
    let registry = ModelRegistry::load()?;
    let models = registry.list();

    if models.is_empty() {
        println!("No models installed.");
        println!("\nRun `omg pull <model>` to download a model.");
        return Ok(());
    }

    println!("{:<40} {:<12} {:<10} {}", "NAME", "TYPE", "SIZE", "DOWNLOADED");
    println!("{}", "-".repeat(80));

    for model in models {
        let size = format!("{:.2} GB", model.size_bytes as f64 / 1_073_741_824.0);
        let date = model.downloaded_at.format("%Y-%m-%d").to_string();
        println!(
            "{:<40} {:<12} {:<10} {}",
            model.name,
            model.model_type.as_str(),
            size,
            date
        );
    }

    Ok(())
}
