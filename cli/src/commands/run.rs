use anyhow::Result;
use ohmygpu_core::ModelRegistry;

pub async fn execute(model: &str) -> Result<()> {
    let registry = ModelRegistry::load()?;

    // Check if model exists
    let model_info = registry.get(model);

    match model_info {
        Some(info) => {
            println!("Loading model: {}", info.name);
            println!("Path: {:?}", info.path);

            // TODO: Connect to daemon and request model load
            // For now, print instructions
            println!("\nModel loading not yet implemented.");
            println!("Start the daemon with `omg serve` first, then the model will be loaded on first request.");
        }
        None => {
            // Try to find similar models
            let models = registry.list();
            let matches: Vec<_> = models
                .iter()
                .filter(|m| m.name.contains(model))
                .collect();

            if matches.is_empty() {
                println!("Model '{}' not found.", model);
                println!("\nRun `omg pull {}` to download it first.", model);
            } else {
                println!("Model '{}' not found. Did you mean:", model);
                for m in matches {
                    println!("  - {}", m.name);
                }
            }
        }
    }

    Ok(())
}
