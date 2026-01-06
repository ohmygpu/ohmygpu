//! Self-update command with daemon handling

use crate::daemon;
use anyhow::Result;

pub async fn execute() -> Result<()> {
    println!("Checking for updates...");

    // Check if daemon is running
    let daemon_was_running = daemon::check_health().await;
    let daemon_pid = daemon::read_pid().ok().flatten();

    if daemon_was_running {
        println!();
        if let Some(pid) = daemon_pid {
            println!("Daemon is running (PID: {})", pid);
        } else {
            println!("Daemon is running");
        }
        println!("It will be stopped before update and restarted after.");
        println!();

        // Confirm with user
        if !confirm_update()? {
            println!("Update cancelled.");
            return Ok(());
        }

        // Stop the daemon
        println!("Stopping daemon...");
        if !daemon::stop_daemon().await? {
            eprintln!("Warning: Could not stop daemon automatically.");
            eprintln!("Please stop it manually and run update again.");
            return Ok(());
        }
        println!("Daemon stopped.");
        println!();
    }

    // Perform the update
    println!("Updating ohmygpu...");

    match do_self_update().await {
        Ok(updated) => {
            if updated {
                println!("Update complete!");
            } else {
                println!("Already running the latest version.");
            }
        }
        Err(e) => {
            eprintln!("Update failed: {}", e);

            // Try to restart daemon if it was running
            if daemon_was_running {
                println!();
                println!("Attempting to restart daemon...");
                restart_daemon_hint();
            }
            return Err(e);
        }
    }

    // Restart daemon if it was running
    if daemon_was_running {
        println!();
        println!("To restart the daemon:");
        restart_daemon_hint();
    }

    Ok(())
}

fn confirm_update() -> Result<bool> {
    use dialoguer::Confirm;

    let confirmed = Confirm::new()
        .with_prompt("Proceed with update?")
        .default(true)
        .interact()?;

    Ok(confirmed)
}

async fn do_self_update() -> Result<bool> {
    use self_update::backends::github::Update;
    use self_update::cargo_crate_version;

    let current_version = cargo_crate_version!();
    println!("Current version: {}", current_version);

    // Check for updates from GitHub releases
    let status = Update::configure()
        .repo_owner("anthropics") // TODO: Update to actual repo owner
        .repo_name("ohmygpu") // TODO: Update to actual repo name
        .bin_name("ohmygpu")
        .current_version(current_version)
        .show_download_progress(true)
        .no_confirm(true)
        .build()?
        .update()?;

    Ok(status.updated())
}

fn restart_daemon_hint() {
    println!("  omg serve");
    println!();
    println!("Or in background:");
    println!("  nohup omg serve > /tmp/ohmygpu.log 2>&1 &");
}
