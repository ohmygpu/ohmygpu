use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

pub async fn generate(model_path: &Path, prompt: &str) -> Result<String> {
    // Check if llama-cli or llama.cpp is available
    let llama_cmd = find_llama_command()?;

    let output = Command::new(&llama_cmd)
        .args([
            "-m",
            model_path.to_str().unwrap(),
            "-p",
            prompt,
            "-n",
            "512", // max tokens
            "--temp",
            "0.7",
            "-ngl",
            "99", // offload layers to GPU
        ])
        .output()
        .context("Failed to run llama.cpp")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("llama.cpp failed: {}", stderr);
    }

    let response = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(response)
}

fn find_llama_command() -> Result<String> {
    // Try common llama.cpp binary names
    let candidates = [
        "llama-cli",
        "llama",
        "llama-cpp",
        "main", // older llama.cpp builds
    ];

    for cmd in candidates {
        if Command::new(cmd).arg("--version").output().is_ok() {
            return Ok(cmd.to_string());
        }
    }

    anyhow::bail!(
        "llama.cpp not found. Please install it:\n\
         - macOS: brew install llama.cpp\n\
         - Linux: Build from https://github.com/ggerganov/llama.cpp\n\
         - Or download pre-built binaries from the releases page"
    )
}

#[allow(dead_code)]
pub fn is_available() -> bool {
    find_llama_command().is_ok()
}
