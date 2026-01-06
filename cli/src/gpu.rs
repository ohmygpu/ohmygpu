use anyhow::Result;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub backend: GpuBackend,
    pub vram_mb: u64,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum GpuBackend {
    Metal,
    Cuda,
    None,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::None => write!(f, "None"),
        }
    }
}

const MIN_VRAM_MB: u64 = 8 * 1024; // 8GB

pub fn detect_gpu() -> GpuInfo {
    // Check compile-time feature flags first
    #[cfg(feature = "metal")]
    {
        if let Some(info) = detect_metal() {
            return info;
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(info) = detect_cuda() {
            return info;
        }
    }

    // No GPU detected
    GpuInfo {
        backend: GpuBackend::None,
        vram_mb: 0,
        name: "No GPU".to_string(),
    }
}

#[cfg(feature = "metal")]
fn detect_metal() -> Option<GpuInfo> {
    // On Apple Silicon, GPU memory is unified with system RAM
    // Use sysctl to get total memory
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let mem_bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    let vram_mb = mem_bytes / (1024 * 1024);

    // Get chip name from system_profiler
    let name = get_apple_chip_name().unwrap_or_else(|| "Apple Silicon".to_string());

    Some(GpuInfo {
        backend: GpuBackend::Metal,
        vram_mb,
        name,
    })
}

#[cfg(feature = "metal")]
fn get_apple_chip_name() -> Option<String> {
    let output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda() -> Option<GpuInfo> {
    // Use nvidia-smi to get GPU info
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let output_str = String::from_utf8_lossy(&output.stdout);
    let line = output_str.lines().next()?;
    let parts: Vec<&str> = line.split(", ").collect();

    if parts.len() < 2 {
        return None;
    }

    let name = parts[0].trim().to_string();
    let vram_mb: u64 = parts[1].trim().parse().ok()?;

    Some(GpuInfo {
        backend: GpuBackend::Cuda,
        vram_mb,
        name,
    })
}

#[allow(dead_code)]
pub enum GpuCheckResult {
    /// GPU meets all requirements
    Ok(GpuInfo),
    /// GPU detected but low VRAM - needs user confirmation
    LowVram(GpuInfo),
    /// No GPU detected - cannot proceed
    NoGpu,
}

pub fn check_gpu_requirements() -> GpuCheckResult {
    let info = detect_gpu();

    match info.backend {
        GpuBackend::None => GpuCheckResult::NoGpu,
        _ if info.vram_mb < MIN_VRAM_MB => GpuCheckResult::LowVram(info),
        _ => GpuCheckResult::Ok(info),
    }
}

pub fn prompt_low_vram_confirmation(info: &GpuInfo) -> Result<bool> {
    use dialoguer::Confirm;

    println!();
    println!("  GPU: {} ({})", info.name, info.backend);
    println!(
        "  VRAM: {:.1} GB (minimum recommended: 8 GB)",
        info.vram_mb as f64 / 1024.0
    );
    println!();

    let confirmed = Confirm::new()
        .with_prompt("I understand, but I want to use omg anyway")
        .default(false)
        .interact()?;

    Ok(confirmed)
}

pub fn print_no_gpu_error() {
    eprintln!();
    eprintln!("  Error: This does not meet the minimum requirements of omg.");
    eprintln!();
    eprintln!("  omg requires one of the following:");
    eprintln!("    - Apple Silicon Mac with Metal support");
    eprintln!("    - NVIDIA GPU with CUDA support");
    eprintln!();
    eprintln!("  With at least 8 GB of GPU memory (recommended).");
    eprintln!();
}
