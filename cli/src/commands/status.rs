use anyhow::Result;
use ohmygpu_core::ModelRegistry;

pub async fn execute() -> Result<()> {
    println!("ohmygpu status\n");

    // Check daemon status
    // TODO: Actually ping the daemon
    let daemon_running = false;
    println!("Daemon: {}", if daemon_running { "running" } else { "not running" });

    // List installed models
    let registry = ModelRegistry::load()?;
    let models = registry.list();
    println!("Installed models: {}", models.len());

    // Show model storage location
    let models_dir = ModelRegistry::models_dir()?;
    println!("Models directory: {:?}", models_dir);

    // TODO: Show GPU info
    println!("\nGPU info: not yet implemented");

    if !daemon_running {
        println!("\nRun `omg serve` to start the daemon.");
    }

    Ok(())
}
