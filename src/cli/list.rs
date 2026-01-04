use anyhow::Result;

use crate::models::registry::ModelRegistry;

pub async fn execute() -> Result<()> {
    let registry = ModelRegistry::load()?;
    let models = registry.list();

    if models.is_empty() {
        println!("No models downloaded yet.");
        println!("Run 'omg pull <model>' to download a model.");
        return Ok(());
    }

    println!("{:<40} {:<15} {:<10}", "MODEL", "TYPE", "SIZE");
    println!("{}", "-".repeat(65));

    for model in models {
        println!(
            "{:<40} {:<15} {:<10}",
            model.name,
            model.model_type.as_str(),
            format_size(model.size_bytes)
        );
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
