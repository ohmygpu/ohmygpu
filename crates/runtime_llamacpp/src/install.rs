//! Locating and (managed) installing the `llama-server` binary.
//!
//! Resolution order:
//! 1. explicit path (`backend.llamacpp.server_path` / `OHMYGPU_LLAMA_SERVER`)
//! 2. managed install dir `<runtimes>/llamacpp/<tag>/`
//! 3. `llama-server` on `PATH`
//!
//! Managed install downloads the official prebuilt release asset for this
//! platform from GitHub (`ggml-org/llama.cpp`) and extracts it. The application
//! developer never sees any of this.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use ohmygpu_core::download::Downloader;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_runtime_api::{ProgressFn, ProgressUpdate, RuntimeError};
use serde::{Deserialize, Serialize};

pub const GITHUB_REPO: &str = "ggml-org/llama.cpp";
pub const BINARY_NAME: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};

/// Where a binary came from (reported in `/ohmygpu/v1/status`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinarySource {
    Config,
    Managed,
    Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocatedBinary {
    pub path: PathBuf,
    pub source: BinarySource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Marker file written into a managed install directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstallRecord {
    tag: String,
    asset: String,
    binary: PathBuf,
}

pub struct Locator {
    pub explicit: Option<PathBuf>,
    pub managed_root: PathBuf,
}

impl Locator {
    pub fn managed_root(runtimes_dir: &Path) -> PathBuf {
        runtimes_dir.join("llamacpp")
    }

    /// Find a usable binary without installing anything.
    pub async fn locate(&self) -> Option<LocatedBinary> {
        if let Some(p) = &self.explicit {
            if p.is_file() {
                let version = binary_version(p).await;
                return Some(LocatedBinary {
                    path: p.clone(),
                    source: BinarySource::Config,
                    version,
                });
            }
            tracing::warn!(
                "configured llama-server path {} does not exist",
                p.display()
            );
        }
        if let Some(p) = self.managed_binary() {
            let version = binary_version(&p).await;
            return Some(LocatedBinary {
                path: p,
                source: BinarySource::Managed,
                version,
            });
        }
        if let Some(p) = which(BINARY_NAME) {
            let version = binary_version(&p).await;
            return Some(LocatedBinary {
                path: p,
                source: BinarySource::Path,
                version,
            });
        }
        None
    }

    /// Newest managed install (highest tag) that still has its binary.
    fn managed_binary(&self) -> Option<PathBuf> {
        let mut records: Vec<(u64, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(&self.managed_root).ok()?.flatten() {
            let rec_path = entry.path().join("install.json");
            let Ok(text) = std::fs::read_to_string(&rec_path) else {
                continue;
            };
            let Ok(rec) = serde_json::from_str::<InstallRecord>(&text) else {
                continue;
            };
            if rec.binary.is_file() {
                records.push((tag_number(&rec.tag), rec.binary));
            }
        }
        records.sort_by(|a, b| b.0.cmp(&a.0));
        records.into_iter().next().map(|(_, p)| p)
    }
}

/// `b10437` → 10437 (unknown formats sort last).
fn tag_number(tag: &str) -> u64 {
    tag.trim_start_matches('b').parse().unwrap_or(0)
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// `llama-server --version` → e.g. `b10437` (parsed from "build 10437").
pub async fn binary_version(path: &Path) -> Option<String> {
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(path)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&text)
}

pub fn parse_version(text: &str) -> Option<String> {
    let idx = text.find("build ")?;
    let rest = &text[idx + 6..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        Some(format!("b{digits}"))
    }
}

// ---------------------------------------------------------------------------
// Managed install
// ---------------------------------------------------------------------------

/// The release asset to download for this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetChoice {
    /// e.g. `macos-arm64`
    pub target: &'static str,
    /// `tar.gz` or `zip`
    pub ext: &'static str,
}

impl AssetChoice {
    pub fn for_hardware(hw: &HardwareInfo) -> Result<AssetChoice, RuntimeError> {
        let has_gpu = hw.gpu.is_some();
        let choice = match (hw.platform.as_str(), hw.architecture.as_str()) {
            ("macos", "arm64") => AssetChoice { target: "macos-arm64", ext: "tar.gz" },
            ("macos", "x86_64") => AssetChoice { target: "macos-x64", ext: "tar.gz" },
            // No prebuilt CUDA for Linux; Vulkan works on NVIDIA/AMD/Intel and the
            // CPU backend is bundled, so it degrades gracefully without a driver.
            ("linux", "x86_64") if has_gpu => AssetChoice { target: "ubuntu-vulkan-x64", ext: "tar.gz" },
            ("linux", "x86_64") => AssetChoice { target: "ubuntu-x64", ext: "tar.gz" },
            ("linux", "arm64") if has_gpu => AssetChoice { target: "ubuntu-vulkan-arm64", ext: "tar.gz" },
            ("linux", "arm64") => AssetChoice { target: "ubuntu-arm64", ext: "tar.gz" },
            ("windows", "x86_64") if has_gpu => AssetChoice { target: "win-vulkan-x64", ext: "zip" },
            ("windows", "x86_64") => AssetChoice { target: "win-cpu-x64", ext: "zip" },
            ("windows", "arm64") => AssetChoice { target: "win-cpu-arm64", ext: "zip" },
            (p, a) => {
                return Err(RuntimeError::NotAvailable(format!(
                    "no prebuilt llama.cpp release for {p}/{a}; set backend.llamacpp.server_path to your own llama-server build"
                )))
            }
        };
        Ok(choice)
    }

    pub fn asset_name(&self, tag: &str) -> String {
        format!("llama-{tag}-bin-{}.{}", self.target, self.ext)
    }

    pub fn download_url(&self, tag: &str) -> String {
        format!(
            "https://github.com/{GITHUB_REPO}/releases/download/{tag}/{}",
            self.asset_name(tag)
        )
    }
}

pub struct Installer {
    pub managed_root: PathBuf,
    pub hardware: HardwareInfo,
    pub release: String,
}

impl Installer {
    /// Resolve `"latest"` to a concrete tag via GitHub's redirect (no API quota).
    pub async fn resolve_tag(&self) -> Result<String, RuntimeError> {
        if self.release != "latest" {
            return Ok(self.release.clone());
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(format!("ohmygpu/{}", ohmygpu_core::VERSION))
            .build()
            .map_err(|e| RuntimeError::Install(e.to_string()))?;
        let url = format!("https://github.com/{GITHUB_REPO}/releases/latest");
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Install(format!("{url}: {e}")))?;
        let location = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                RuntimeError::Install(format!(
                    "{url} did not redirect to a release tag (status {})",
                    resp.status()
                ))
            })?;
        let tag = location.rsplit('/').next().unwrap_or_default().to_string();
        if tag.is_empty() {
            return Err(RuntimeError::Install(format!(
                "could not parse release tag from {location}"
            )));
        }
        Ok(tag)
    }

    /// Download + extract the release; returns the binary path.
    pub async fn install(
        &self,
        progress: Option<ProgressFn>,
    ) -> Result<LocatedBinary, RuntimeError> {
        let choice = AssetChoice::for_hardware(&self.hardware)?;
        let tag = self.resolve_tag().await?;
        let asset = choice.asset_name(&tag);
        let url = choice.download_url(&tag);
        let dest_dir = self.managed_root.join(&tag);
        std::fs::create_dir_all(&dest_dir).map_err(|e| RuntimeError::Install(e.to_string()))?;

        // Already installed?
        let record_path = dest_dir.join("install.json");
        if let Ok(text) = std::fs::read_to_string(&record_path) {
            if let Ok(rec) = serde_json::from_str::<InstallRecord>(&text) {
                if rec.binary.is_file() {
                    let version = binary_version(&rec.binary).await;
                    return Ok(LocatedBinary {
                        path: rec.binary,
                        source: BinarySource::Managed,
                        version,
                    });
                }
            }
        }

        tracing::info!(
            "installing llama.cpp {tag} ({asset}) into {}",
            dest_dir.display()
        );
        let archive_path = dest_dir.join(&asset);
        let msg = format!("downloading llama.cpp {tag}");
        let progress_cb: Option<ohmygpu_core::download::ProgressCallback> =
            progress.clone().map(|p| {
                let msg = msg.clone();
                Arc::new(move |dp: ohmygpu_core::lifecycle::DownloadProgress| {
                    p(ProgressUpdate {
                        message: msg.clone(),
                        done_bytes: Some(dp.downloaded_bytes),
                        total_bytes: dp.total_bytes,
                    })
                }) as ohmygpu_core::download::ProgressCallback
            });
        Downloader::new(None)
            .download(&url, &archive_path, progress_cb, None)
            .await
            .map_err(|e| RuntimeError::Install(format!("downloading {url}: {e}")))?;

        if let Some(p) = &progress {
            p(ProgressUpdate {
                message: format!("extracting llama.cpp {tag}"),
                done_bytes: None,
                total_bytes: None,
            });
        }
        let extract_dir = dest_dir.clone();
        let archive_for_extract = archive_path.clone();
        let ext = choice.ext;
        tokio::task::spawn_blocking(move || extract(&archive_for_extract, &extract_dir, ext))
            .await
            .map_err(|e| RuntimeError::Install(e.to_string()))??;
        std::fs::remove_file(&archive_path).ok();

        let binary = find_binary(&dest_dir).ok_or_else(|| {
            RuntimeError::Install(format!("{asset} did not contain {BINARY_NAME}"))
        })?;
        make_executable(&binary)?;
        let rec = InstallRecord {
            tag: tag.clone(),
            asset,
            binary: binary.clone(),
        };
        std::fs::write(&record_path, serde_json::to_string_pretty(&rec).unwrap())
            .map_err(|e| RuntimeError::Install(e.to_string()))?;

        let version = binary_version(&binary).await;
        tracing::info!("llama.cpp {tag} installed at {}", binary.display());
        Ok(LocatedBinary {
            path: binary,
            source: BinarySource::Managed,
            version,
        })
    }
}

fn extract(archive: &Path, into: &Path, ext: &str) -> Result<(), RuntimeError> {
    let file = std::fs::File::open(archive).map_err(|e| RuntimeError::Install(e.to_string()))?;
    match ext {
        "tar.gz" => {
            let gz = flate2::read::GzDecoder::new(file);
            let mut ar = tar::Archive::new(gz);
            ar.unpack(into)
                .map_err(|e| RuntimeError::Install(format!("extracting tar.gz: {e}")))?;
        }
        "zip" => {
            let mut z = zip::ZipArchive::new(file)
                .map_err(|e| RuntimeError::Install(format!("opening zip: {e}")))?;
            z.extract(into)
                .map_err(|e| RuntimeError::Install(format!("extracting zip: {e}")))?;
        }
        other => {
            return Err(RuntimeError::Install(format!(
                "unsupported archive type {other}"
            )))
        }
    }
    Ok(())
}

/// Recursively find the server binary (release layouts differ per platform).
fn find_binary(dir: &Path) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).ok()?.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().map(|n| n == BINARY_NAME).unwrap_or(false) {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), RuntimeError> {
    use std::os::unix::fs::PermissionsExt;
    // The bundled dylibs/.so files and other tools should be executable too.
    if let Some(dir) = path.parent() {
        for entry in std::fs::read_dir(dir)
            .map_err(|e| RuntimeError::Install(e.to_string()))?
            .flatten()
        {
            let p = entry.path();
            if p.is_file() {
                if let Ok(meta) = std::fs::metadata(&p) {
                    let mut perms = meta.permissions();
                    perms.set_mode(perms.mode() | 0o755);
                    std::fs::set_permissions(&p, perms).ok();
                }
            }
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), RuntimeError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ohmygpu_core::hardware::{CpuInfo, GpuInfo};

    fn hw(platform: &str, arch: &str, gpu: bool) -> HardwareInfo {
        HardwareInfo {
            platform: platform.into(),
            architecture: arch.into(),
            cpu: CpuInfo {
                name: "cpu".into(),
                cores: 4,
            },
            system_memory_bytes: None,
            gpu: gpu.then(|| GpuInfo {
                vendor: "nvidia".into(),
                name: "x".into(),
                memory_bytes: None,
                backend: "cuda".into(),
                driver_version: None,
            }),
            backend: if gpu { "cuda".into() } else { "cpu".into() },
        }
    }

    #[test]
    fn asset_names_match_release_naming() {
        let c = AssetChoice::for_hardware(&hw("macos", "arm64", true)).unwrap();
        assert_eq!(
            c.asset_name("b10437"),
            "llama-b10437-bin-macos-arm64.tar.gz"
        );
        assert_eq!(
            c.download_url("b10437"),
            "https://github.com/ggml-org/llama.cpp/releases/download/b10437/llama-b10437-bin-macos-arm64.tar.gz"
        );
        assert_eq!(
            AssetChoice::for_hardware(&hw("linux", "x86_64", false))
                .unwrap()
                .target,
            "ubuntu-x64"
        );
        assert_eq!(
            AssetChoice::for_hardware(&hw("linux", "x86_64", true))
                .unwrap()
                .target,
            "ubuntu-vulkan-x64"
        );
        assert_eq!(
            AssetChoice::for_hardware(&hw("windows", "x86_64", true))
                .unwrap()
                .asset_name("b1"),
            "llama-b1-bin-win-vulkan-x64.zip"
        );
        assert!(AssetChoice::for_hardware(&hw("freebsd", "x86_64", false)).is_err());
    }

    #[test]
    fn parses_version_output() {
        assert_eq!(
            parse_version("version: 0.1.0-dev (build 10437, commit 16d222fc5)\n"),
            Some("b10437".into())
        );
        assert_eq!(parse_version("garbage"), None);
        assert_eq!(tag_number("b10437"), 10437);
    }

    #[test]
    fn finds_binary_recursively() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("llama-b1").join("bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join(BINARY_NAME), b"x").unwrap();
        assert_eq!(find_binary(dir.path()).unwrap(), nested.join(BINARY_NAME));
    }
}
