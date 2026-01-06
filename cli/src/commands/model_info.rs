//! Show model information

use anyhow::Result;
use ohmygpu_core::ModelRegistry;

pub async fn execute(model: &str) -> Result<()> {
    let registry = ModelRegistry::load()?;

    match registry.get(model) {
        Some(info) => {
            println!("Model: {}", info.name);
            println!("Source: {:?}", info.source);
            println!("Type: {:?}", info.model_type);
            println!("Path: {}", info.path.display());

            // Show file sizes if available
            if info.path.exists() {
                let size = dir_size(&info.path)?;
                println!("Size: {}", format_size(size));
            }

            // TODO: Show more details like architecture, quantization, etc.
        }
        None => {
            eprintln!("Model '{}' not found", model);
            eprintln!();
            eprintln!("Use `omg model list` to see installed models");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn dir_size(path: &std::path::Path) -> Result<u64> {
    let mut size = 0;
    if path.is_file() {
        size = path.metadata()?.len();
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                size += path.metadata()?.len();
            } else if path.is_dir() {
                size += dir_size(&path)?;
            }
        }
    }
    Ok(size)
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
