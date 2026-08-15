//! Hardware detection: what machine is the runtime on, and which acceleration
//! backend can llama.cpp use here. Detection only — no benchmarking.
//!
//! Exposed as `GET /ohmygpu/v1/hardware`:
//!
//! ```json
//! {
//!   "platform": "macos",
//!   "architecture": "arm64",
//!   "cpu": { "name": "Apple M4 Max", "cores": 16 },
//!   "system_memory_bytes": 137438953472,
//!   "gpu": { "vendor": "apple", "name": "Apple M4 Max", "memory_bytes": 137438953472, "backend": "metal" },
//!   "backend": "metal"
//! }
//! ```

use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareInfo {
    /// `macos` | `linux` | `windows` | other `std::env::consts::OS` value
    pub platform: String,
    /// `arm64` | `x86_64` | …
    pub architecture: String,
    pub cpu: CpuInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuInfo>,
    /// The acceleration backend llama.cpp is expected to use here:
    /// `metal` | `cuda` | `vulkan` | `cpu`.
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuInfo {
    /// `apple` | `nvidia` | `amd` | `unknown`
    pub vendor: String,
    pub name: String,
    /// Dedicated VRAM, or unified memory on Apple Silicon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    /// `metal` | `cuda` | `vulkan`
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
}

impl HardwareInfo {
    /// Detect the local machine. Never fails; unknown parts are `None`/"unknown".
    pub fn detect() -> Self {
        let platform = std::env::consts::OS.to_string();
        let architecture = match std::env::consts::ARCH {
            "aarch64" => "arm64".to_string(),
            other => other.to_string(),
        };

        let (cpu_name, cores, system_memory_bytes) = system_basics();

        let gpu = detect_nvidia()
            .or_else(|| detect_apple(&platform, &architecture, &cpu_name, system_memory_bytes));

        let backend = gpu
            .as_ref()
            .map(|g| g.backend.clone())
            .unwrap_or_else(|| "cpu".to_string());

        HardwareInfo {
            platform,
            architecture,
            cpu: CpuInfo {
                name: cpu_name,
                cores,
            },
            system_memory_bytes,
            gpu,
            backend,
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

fn system_basics() -> (String, usize, Option<u64>) {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    let mut name = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_default();
    if name.is_empty() {
        name = macos_sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".to_string());
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mem = sys.total_memory();
    let mem = if mem > 0 { Some(mem) } else { None };
    (name, cores, mem)
}

fn macos_sysctl(key: &str) -> Option<String> {
    if std::env::consts::OS != "macos" {
        return None;
    }
    let out = Command::new("sysctl").args(["-n", key]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Apple Silicon: unified memory, Metal backend.
fn detect_apple(platform: &str, arch: &str, cpu_name: &str, mem: Option<u64>) -> Option<GpuInfo> {
    if platform != "macos" || arch != "arm64" {
        return None;
    }
    let name = if cpu_name.starts_with("Apple") {
        cpu_name.to_string()
    } else {
        "Apple Silicon".to_string()
    };
    Some(GpuInfo {
        vendor: "apple".into(),
        name,
        memory_bytes: mem,
        backend: "metal".into(),
        driver_version: None,
    })
}

/// NVIDIA via `nvidia-smi` (present wherever the driver is installed).
fn detect_nvidia() -> Option<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| !l.trim().is_empty())?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 2 {
        return None;
    }
    let memory_bytes = parts[1].parse::<u64>().ok().map(|mib| mib * 1024 * 1024);
    Some(GpuInfo {
        vendor: "nvidia".into(),
        name: parts[0].to_string(),
        memory_bytes,
        backend: "cuda".into(),
        driver_version: parts.get(2).map(|s| s.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics_and_reports_platform() {
        let hw = HardwareInfo::detect();
        assert!(!hw.platform.is_empty());
        assert!(!hw.architecture.is_empty());
        assert!(hw.cpu.cores >= 1);
        assert!(["metal", "cuda", "vulkan", "cpu"].contains(&hw.backend.as_str()));
        if let Some(gpu) = &hw.gpu {
            assert_eq!(gpu.backend, hw.backend);
        }
    }
}
