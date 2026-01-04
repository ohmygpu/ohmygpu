use anyhow::Result;
use std::fs;

use crate::models::registry::ModelRegistry;

pub async fn execute(model: &str) -> Result<()> {
    let mut registry = ModelRegistry::load()?;

    let model_info = registry.get(model).ok_or_else(|| {
        anyhow::anyhow!("Model '{}' not found.", model)
    })?;

    // Remove the model files
    if model_info.path.exists() {
        fs::remove_dir_all(&model_info.path)?;
    }

    // Remove from registry
    registry.remove(model)?;
    registry.save()?;

    println!("Removed model: {}", model);
    Ok(())
}
