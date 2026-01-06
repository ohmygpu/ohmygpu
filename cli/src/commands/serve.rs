//! Daemon server management

use anyhow::Result;
use std::net::SocketAddr;

const DAEMON_URL: &str = "http://localhost:11434";

/// Start the daemon in foreground
pub async fn execute(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;

    println!("Starting ohmygpu daemon...");
    println!("Listening on http://{}", addr);
    println!();
    println!("API endpoints:");
    println!("  OpenAI:  POST /v1/chat/completions");
    println!("  Ollama:  POST /api/chat");
    println!("  Models:  GET  /v1/models");
    println!("  Health:  GET  /health");
    println!();
    println!("Press Ctrl+C to stop.");
    println!();

    ohmygpu_daemon::run_server(addr).await?;

    Ok(())
}

/// Start the daemon in background
pub async fn execute_background(port: u16) -> Result<()> {
    // Check if already running
    let client = reqwest::Client::new();
    if let Ok(response) = client.get(format!("{}/health", DAEMON_URL)).send().await {
        if response.status().is_success() {
            println!("Daemon is already running on port {}", port);
            return Ok(());
        }
    }

    // For now, just print instructions
    // TODO: Actually spawn a background process
    println!("To run in background, use:");
    println!("  nohup omg serve > /tmp/ohmygpu.log 2>&1 &");
    println!();
    println!("Or use a process manager like systemd/launchd.");

    Ok(())
}

/// Check daemon status
pub async fn status() -> Result<()> {
    let client = reqwest::Client::new();

    match client.get(format!("{}/health", DAEMON_URL)).send().await {
        Ok(response) if response.status().is_success() => {
            println!("Daemon: running");
            println!("URL: {}", DAEMON_URL);

            // Get loaded models
            if let Ok(models_response) = client.get(format!("{}/v1/models", DAEMON_URL)).send().await
            {
                if let Ok(models) = models_response.json::<serde_json::Value>().await {
                    if let Some(data) = models["data"].as_array() {
                        if data.is_empty() {
                            println!("Models loaded: none");
                        } else {
                            println!("Models loaded:");
                            for model in data {
                                if let Some(id) = model["id"].as_str() {
                                    println!("  - {}", id);
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            println!("Daemon: not running");
            println!();
            println!("Start with: omg serve");
        }
    }

    Ok(())
}

/// Stop the daemon
pub async fn stop() -> Result<()> {
    let client = reqwest::Client::new();

    // Check if running
    match client.get(format!("{}/health", DAEMON_URL)).send().await {
        Ok(response) if response.status().is_success() => {
            // TODO: Send shutdown signal to daemon
            // For now, just print instructions
            println!("To stop the daemon, press Ctrl+C in the terminal where it's running,");
            println!("or kill the process:");
            println!("  pkill -f 'omg serve'");
        }
        _ => {
            println!("Daemon is not running.");
        }
    }

    Ok(())
}
