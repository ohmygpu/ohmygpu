//! Finding or installing `llama-server`.
//!
//! Lookup order: explicit path from config / `OHMYGPU_LLAMA_SERVER` → the
//! newest managed install under `<data>/runtimes/llamacpp/<tag>/` → `PATH`.
//! Managed install downloads the official prebuilt release asset for this
//! machine from GitHub and extracts it next to an `install.json` record.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_runtime_api::{ProgressFn, ProgressUpdate, RuntimeError};
use ohmygpu_runtime_common::install::{
    download_archive, extract_archive, find_binary, make_executable, newest_managed,
    resolve_latest_tag, tag_number, which, InstallRecord,
};
pub use ohmygpu_runtime_common::install::{BinarySource, LocatedBinary};

pub const GITHUB_REPO: &str = "ggml-org/llama.cpp";
pub const BINARY_NAME: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};

pub struct Locator {
    pub explicit: Option<PathBuf>,
    pub managed_root: PathBuf,
}

impl Locator {
    pub fn managed_root(runtimes_dir: &Path) -> PathBuf {
        runtimes_dir.join("llamacpp")
    }

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
        if let Some(rec) = newest_managed(&self.managed_root, tag_number) {
            let version = binary_version(&rec.binary).await;
            return Some(LocatedBinary {
                path: rec.binary,
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
}

/// `llama-server --version` → `b1234`.
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

/// The release asset to download for this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetChoice {
    /// e.g. `macos-arm64`
    pub target: &'static str,
    pub ext: &'static str,
}

impl AssetChoice {
    pub fn for_hardware(hw: &HardwareInfo) -> Result<AssetChoice, RuntimeError> {
        let has_gpu = hw.gpu.is_some();
        let choice = match (hw.platform.as_str(), hw.architecture.as_str()) {
            ("macos", "arm64") => AssetChoice { target: "macos-arm64", ext: "tar.gz" },
            ("macos", "x86_64") => AssetChoice { target: "macos-x64", ext: "tar.gz" },
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
    /// A tag like `b10437`, or `latest`.
    pub release: String,
}

impl Installer {
    pub async fn resolve_tag(&self) -> Result<String, RuntimeError> {
        if self.release != "latest" {
            return Ok(self.release.clone());
        }
        resolve_latest_tag(GITHUB_REPO).await
    }

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
        if let Some(rec) = InstallRecord::read(&dest_dir) {
            let version = binary_version(&rec.binary).await;
            return Ok(LocatedBinary {
                path: rec.binary,
                source: BinarySource::Managed,
                version,
            });
        }

        tracing::info!(
            "installing llama.cpp {tag} ({asset}) into {}",
            dest_dir.display()
        );
        let archive_path = dest_dir.join(&asset);
        download_archive(
            &url,
            &archive_path,
            progress.clone(),
            &format!("downloading llama.cpp {tag}"),
        )
        .await?;
        if let Some(p) = &progress {
            p(ProgressUpdate {
                message: format!("extracting llama.cpp {tag}"),
                done_bytes: None,
                total_bytes: None,
            });
        }
        extract_archive(&archive_path, &dest_dir, choice.ext).await?;
        std::fs::remove_file(&archive_path).ok();
        let binary = find_binary(&dest_dir, BINARY_NAME).ok_or_else(|| {
            RuntimeError::Install(format!("{asset} did not contain {BINARY_NAME}"))
        })?;
        make_executable(&binary)?;
        InstallRecord {
            tag: tag.clone(),
            asset,
            binary: binary.clone(),
        }
        .write(&dest_dir)?;
        let version = binary_version(&binary).await;
        tracing::info!("llama.cpp {tag} installed at {}", binary.display());
        Ok(LocatedBinary {
            path: binary,
            source: BinarySource::Managed,
            version,
        })
    }
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
            parse_version("version: 4567 (abc)\nbuild 4567 (abc) with clang"),
            Some("b4567".into())
        );
        assert_eq!(parse_version("nothing"), None);
    }
}
