use anyhow::Result;

pub async fn execute(model: Option<&str>) -> Result<()> {
    match model {
        Some(name) => {
            println!("Stopping model: {}", name);
            // TODO: Connect to daemon and request model unload
            println!("Not yet implemented - connect to daemon to stop model.");
        }
        None => {
            println!("Stopping all models...");
            // TODO: Connect to daemon and request all models unload
            println!("Not yet implemented - connect to daemon to stop all models.");
        }
    }

    Ok(())
}
