//! Daemon server management

use crate::daemon;
use anyhow::Result;
use std::net::SocketAddr;

/// Start the daemon in foreground
pub async fn execute(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;

    // Write PID file for this process
    let pid = std::process::id();
    daemon::write_pid(pid)?;

    // Set up cleanup on exit
    let cleanup = || {
        let _ = daemon::remove_pid_file();
    };

    println!("Starting ohmygpu daemon (PID: {})...", pid);
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

    // Run server - cleanup on exit
    let result = ohmygpu_daemon::run_server(addr).await;
    cleanup();
    result?;

    Ok(())
}

/// Start the daemon in background
pub async fn execute_background(port: u16) -> Result<()> {
    // Check if already running
    let status = daemon::get_status().await;
    if status.running {
        if let Some(pid) = status.pid {
            println!("Daemon is already running (PID: {})", pid);
        } else {
            println!("Daemon is already running on port {}", port);
        }
        return Ok(());
    }

    // For now, provide instructions for background mode
    // Full background spawn would require fork() or similar
    println!("To run in background, use:");
    println!("  nohup omg serve > /tmp/ohmygpu.log 2>&1 &");
    println!();
    println!("Or use a process manager like systemd/launchd.");
    println!();
    println!("The daemon will write its PID to ~/.config/ohmygpu/daemon.pid");

    Ok(())
}

/// Check daemon status
pub async fn status() -> Result<()> {
    let status = daemon::get_status().await;

    if status.running {
        print!("Daemon: running");
        if let Some(pid) = status.pid {
            println!(" (PID: {})", pid);
        } else {
            println!();
        }
        println!("URL: {}", status.url);

        // Get loaded models
        let client = reqwest::Client::new();
        if let Ok(response) = client
            .get(format!("{}/v1/models", status.url))
            .send()
            .await
        {
            if let Ok(models) = response.json::<serde_json::Value>().await {
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
    } else {
        println!("Daemon: not running");
        if let Some(pid) = status.pid {
            println!("  (stale PID file found: {})", pid);
        }
        println!();
        println!("Start with: omg serve");
    }

    Ok(())
}

/// Stop the daemon
pub async fn stop() -> Result<()> {
    let status = daemon::get_status().await;

    if !status.running {
        println!("Daemon is not running.");
        return Ok(());
    }

    if daemon::stop_daemon().await? {
        // Successfully stopped
    } else {
        println!("Failed to stop daemon automatically.");
        println!();
        println!("Try manually:");
        if let Some(pid) = status.pid {
            println!("  kill {}", pid);
        } else {
            println!("  pkill -f 'ohmygpu.*serve'");
        }
    }

    Ok(())
}
