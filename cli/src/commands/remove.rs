use anyhow::Result;
use ohmygpu_core::ModelRegistry;
use std::fs;

pub async fn execute(model: &str) -> Result<()> {
    let mut registry = ModelRegistry::load()?;

    // Check if model exists
    let model_info = registry.get(model).cloned();

    match model_info {
        Some(info) => {
            // Remove files
            if info.path.exists() {
                println!("Removing model files from {:?}...", info.path);
                fs::remove_dir_all(&info.path)?;
            }

            // Remove from registry
            registry.remove(model)?;
            println!("Model '{}' removed.", model);
        }
        None => {
            // Try fuzzy match
            let models = registry.list();
            let matches: Vec<_> = models
                .iter()
                .filter(|m| m.name.contains(model))
                .collect();

            if matches.is_empty() {
                println!("Model '{}' not found.", model);
                println!("\nRun `omg models` to see installed models.");
            } else if matches.len() == 1 {
                println!("Did you mean '{}'?", matches[0].name);
            } else {
                println!("Model '{}' not found. Similar models:", model);
                for m in matches {
                    println!("  - {}", m.name);
                }
            }
        }
    }

    Ok(())
}
