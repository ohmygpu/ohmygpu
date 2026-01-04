use anyhow::Result;

use crate::models::registry::ModelRegistry;
use crate::runners::Runner;

pub async fn execute(model: &str, prompt: Option<&str>) -> Result<()> {
    let registry = ModelRegistry::load()?;

    let model_info = registry.get(model).ok_or_else(|| {
        anyhow::anyhow!("Model '{}' not found. Run 'omg pull {}' first.", model, model)
    })?;

    let runner = Runner::for_model(model_info)?;

    match prompt {
        Some(p) => {
            // Non-interactive mode
            let response = runner.generate(p).await?;
            println!("{}", response);
        }
        None => {
            // Interactive mode
            runner.interactive().await?;
        }
    }

    Ok(())
}
