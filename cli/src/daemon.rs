//! Daemon process management with PID file tracking

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const DAEMON_URL: &str = "http://localhost:10692";

/// Get the PID file path
pub fn pid_file_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .context("Failed to get config directory")?
        .join("ohmygpu");
    fs::create_dir_all(&config_dir)?;
    Ok(config_dir.join("daemon.pid"))
}

/// Write PID to file
pub fn write_pid(pid: u32) -> Result<()> {
    let path = pid_file_path()?;
    fs::write(&path, pid.to_string())?;
    Ok(())
}

/// Read PID from file
pub fn read_pid() -> Result<Option<u32>> {
    let path = pid_file_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)?;
    let pid: u32 = content.trim().parse().context("Invalid PID in file")?;
    Ok(Some(pid))
}

/// Remove PID file
pub fn remove_pid_file() -> Result<()> {
    let path = pid_file_path()?;
    if path.exists() {
        fs::remove_file(&path)?;
    }
    Ok(())
}

/// Check if a process with given PID is running
#[cfg(unix)]
pub fn is_process_running(pid: u32) -> bool {
    // Use kill -0 to check if process exists (signal 0 = just check)
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
pub fn is_process_running(_pid: u32) -> bool {
    // On non-Unix, fall back to HTTP health check
    false
}

/// Get daemon status
pub struct DaemonStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub url: String,
}

/// Check daemon status using PID file and health check
pub async fn get_status() -> DaemonStatus {
    let mut status = DaemonStatus {
        running: false,
        pid: None,
        url: DAEMON_URL.to_string(),
    };

    // Check PID file first
    if let Ok(Some(pid)) = read_pid() {
        status.pid = Some(pid);
        if is_process_running(pid) {
            // Process exists, verify it's actually the daemon via health check
            if check_health().await {
                status.running = true;
                return status;
            }
        }
        // PID file exists but process is dead - clean up
        let _ = remove_pid_file();
    }

    // No PID file or stale PID - check health directly
    // (daemon might be running in foreground without PID file)
    if check_health().await {
        status.running = true;
    }

    status
}

/// Check daemon health via HTTP
pub async fn check_health() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    match client.get(format!("{}/health", DAEMON_URL)).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Stop daemon by PID
#[cfg(unix)]
pub fn stop_by_pid(pid: u32) -> Result<bool> {
    use std::thread;
    use std::time::Duration;

    // Send SIGTERM first
    unsafe {
        if libc::kill(pid as i32, libc::SIGTERM) != 0 {
            return Ok(false);
        }
    }

    // Wait up to 5 seconds for graceful shutdown
    for _ in 0..50 {
        thread::sleep(Duration::from_millis(100));
        if !is_process_running(pid) {
            remove_pid_file()?;
            return Ok(true);
        }
    }

    // Force kill if still running
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
    thread::sleep(Duration::from_millis(100));
    remove_pid_file()?;

    Ok(!is_process_running(pid))
}

#[cfg(not(unix))]
pub fn stop_by_pid(_pid: u32) -> Result<bool> {
    anyhow::bail!("PID-based stop not supported on this platform")
}

/// Stop daemon - tries PID first, then pkill as fallback
pub async fn stop_daemon() -> Result<bool> {
    // Try PID file first
    if let Ok(Some(pid)) = read_pid() {
        if is_process_running(pid) {
            println!("Stopping daemon (PID: {})...", pid);
            if stop_by_pid(pid)? {
                println!("Daemon stopped.");
                return Ok(true);
            }
        }
    }

    // Fallback: check if running via health and use pkill
    if check_health().await {
        println!("Stopping daemon via pkill...");
        let status = Command::new("pkill")
            .args(["-f", "ohmygpu.*serve"])
            .status();

        match status {
            Ok(s) if s.success() => {
                // Wait a bit and verify
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if !check_health().await {
                    remove_pid_file()?;
                    println!("Daemon stopped.");
                    return Ok(true);
                }
            }
            _ => {}
        }
    }

    Ok(false)
}
