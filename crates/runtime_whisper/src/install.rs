//! Finding or installing `whisper-server`.
//!
//! Lookup order: explicit path from config / `OHMYGPU_WHISPER_SERVER` → the
//! newest managed install under `<data>/runtimes/whisper/<tag>/` → `PATH`.
//!
//! Managed install: Linux and Windows use the official whisper.cpp release
//! assets; whisper.cpp publishes no macOS binaries, so macOS downloads the
//! `whisper-server` our own release workflow builds (static, Metal) and attaches
//! to the OhMyGPU release of the running version.

use std::path::{Path, PathBuf};

use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_runtime_api::{ProgressFn, ProgressUpdate, RuntimeError};
use ohmygpu_runtime_common::install::{
    download_archive, extract_archive, find_binary, make_executable, newest_managed, tag_number,
    which, InstallRecord,
};
pub use ohmygpu_runtime_common::install::{BinarySource, LocatedBinary};

pub const WHISPER_REPO: &str = "ggml-org/whisper.cpp";
pub const OHMYGPU_REPO: &str = "ohmygpu/ohmygpu";
pub const BINARY_NAME: &str = if cfg!(windows) {
    "whisper-server.exe"
} else {
    "whisper-server"
};

pub struct Locator {
    pub explicit: Option<PathBuf>,
    pub managed_root: PathBuf,
}

impl Locator {
    pub fn managed_root(runtimes_dir: &Path) -> PathBuf {
        runtimes_dir.join("whisper")
    }

    pub async fn locate(&self) -> Option<LocatedBinary> {
        if let Some(p) = &self.explicit {
            if p.is_file() {
                return Some(LocatedBinary {
                    path: p.clone(),
                    source: BinarySource::Config,
                    version: None,
                });
            }
            tracing::warn!(
                "configured whisper-server path {} does not exist",
                p.display()
            );
        }
        if let Some(rec) = newest_managed(&self.managed_root, tag_number) {
            return Some(LocatedBinary {
                path: rec.binary,
                source: BinarySource::Managed,
                version: Some(rec.tag),
            });
        }
        if let Some(p) = which(BINARY_NAME) {
            return Some(LocatedBinary {
                path: p,
                source: BinarySource::Path,
                version: None,
            });
        }
        None
    }
}

/// The release asset to download for this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetChoice {
    pub asset: String,
    pub url: String,
    pub ext: &'static str,
}

impl AssetChoice {
    /// `tag` is the whisper.cpp release (`b4938`); `ohmygpu_version` (e.g.
    /// `0.5.0`) selects the OhMyGPU release that carries the macOS builds.
    pub fn for_hardware(
        hw: &HardwareInfo,
        tag: &str,
        ohmygpu_version: &str,
    ) -> Result<AssetChoice, RuntimeError> {
        let official = |asset: &str, ext: &'static str| AssetChoice {
            asset: asset.to_string(),
            url: format!("https://github.com/{WHISPER_REPO}/releases/download/{tag}/{asset}"),
            ext,
        };
        let ours = |arch: &str| {
            let asset = format!("whisper-{tag}-bin-macos-{arch}.tar.gz");
            AssetChoice {
                url: format!(
                    "https://github.com/{OHMYGPU_REPO}/releases/download/v{ohmygpu_version}/{asset}"
                ),
                asset,
                ext: "tar.gz",
            }
        };
        let choice = match (hw.platform.as_str(), hw.architecture.as_str()) {
            ("macos", "arm64") => ours("arm64"),
            ("macos", "x86_64") => ours("x64"),
            ("linux", "x86_64") => official("whisper-bin-ubuntu-x64.tar.gz", "tar.gz"),
            ("linux", "arm64") => official("whisper-bin-ubuntu-arm64.tar.gz", "tar.gz"),
            ("windows", "x86_64") => official("whisper-bin-x64.zip", "zip"),
            (p, a) => {
                return Err(RuntimeError::NotAvailable(format!(
                    "no prebuilt whisper-server for {p}/{a}; set backend.whisper.server_path to your own whisper.cpp build"
                )))
            }
        };
        Ok(choice)
    }
}

pub struct Installer {
    pub managed_root: PathBuf,
    pub hardware: HardwareInfo,
    /// whisper.cpp tag, e.g. `b4938` (pinned; `latest` is not supported because
    /// the macOS build must match).
    pub release: String,
}

impl Installer {
    pub async fn install(
        &self,
        progress: Option<ProgressFn>,
    ) -> Result<LocatedBinary, RuntimeError> {
        let tag = self.release.trim().to_string();
        if tag.is_empty() || tag == "latest" {
            return Err(RuntimeError::Install(
                "backend.whisper.release must be a whisper.cpp tag like b4938".into(),
            ));
        }
        let choice = AssetChoice::for_hardware(&self.hardware, &tag, ohmygpu_core::VERSION)?;
        let dest_dir = self.managed_root.join(&tag);
        std::fs::create_dir_all(&dest_dir).map_err(|e| RuntimeError::Install(e.to_string()))?;
        if let Some(rec) = InstallRecord::read(&dest_dir) {
            return Ok(LocatedBinary {
                path: rec.binary,
                source: BinarySource::Managed,
                version: Some(rec.tag),
            });
        }

        tracing::info!(
            "installing whisper.cpp {tag} ({}) into {}",
            choice.asset,
            dest_dir.display()
        );
        let archive_path = dest_dir.join(&choice.asset);
        download_archive(
            &choice.url,
            &archive_path,
            progress.clone(),
            &format!("downloading whisper.cpp {tag}"),
        )
        .await
        .map_err(|e| {
            if hw_is_macos(&self.hardware) {
                RuntimeError::Install(format!(
                    "{e}\nmacOS builds of whisper-server ship with OhMyGPU releases; this build \
                     (v{}) may not have one. Build whisper.cpp yourself (cmake, target \
                     whisper-server) and set OHMYGPU_WHISPER_SERVER, or `brew install whisper-cpp`.",
                    ohmygpu_core::VERSION
                ))
            } else {
                e
            }
        })?;
        if let Some(p) = &progress {
            p(ProgressUpdate {
                message: format!("extracting whisper.cpp {tag}"),
                done_bytes: None,
                total_bytes: None,
            });
        }
        extract_archive(&archive_path, &dest_dir, choice.ext).await?;
        std::fs::remove_file(&archive_path).ok();
        let binary = find_binary(&dest_dir, BINARY_NAME).ok_or_else(|| {
            RuntimeError::Install(format!("{} did not contain {BINARY_NAME}", choice.asset))
        })?;
        make_executable(&binary)?;
        InstallRecord {
            tag: tag.clone(),
            asset: choice.asset,
            binary: binary.clone(),
        }
        .write(&dest_dir)?;
        tracing::info!("whisper.cpp {tag} installed at {}", binary.display());
        Ok(LocatedBinary {
            path: binary,
            source: BinarySource::Managed,
            version: Some(tag),
        })
    }
}

fn hw_is_macos(hw: &HardwareInfo) -> bool {
    hw.platform == "macos"
}

#[cfg(test)]
mod tests {
    use super::*;
    use ohmygpu_core::hardware::CpuInfo;

    fn hw(platform: &str, arch: &str) -> HardwareInfo {
        HardwareInfo {
            platform: platform.into(),
            architecture: arch.into(),
            cpu: CpuInfo {
                name: "cpu".into(),
                cores: 4,
            },
            system_memory_bytes: None,
            gpu: None,
            backend: "cpu".into(),
        }
    }

    #[test]
    fn asset_choice_per_platform() {
        let c = AssetChoice::for_hardware(&hw("linux", "x86_64"), "b4938", "0.5.0").unwrap();
        assert_eq!(c.asset, "whisper-bin-ubuntu-x64.tar.gz");
        assert_eq!(
            c.url,
            "https://github.com/ggml-org/whisper.cpp/releases/download/b4938/whisper-bin-ubuntu-x64.tar.gz"
        );
        let c = AssetChoice::for_hardware(&hw("macos", "arm64"), "b4938", "0.5.0").unwrap();
        assert_eq!(c.asset, "whisper-b4938-bin-macos-arm64.tar.gz");
        assert_eq!(
            c.url,
            "https://github.com/ohmygpu/ohmygpu/releases/download/v0.5.0/whisper-b4938-bin-macos-arm64.tar.gz"
        );
        let c = AssetChoice::for_hardware(&hw("windows", "x86_64"), "b4938", "0.5.0").unwrap();
        assert_eq!(c.ext, "zip");
        assert!(AssetChoice::for_hardware(&hw("freebsd", "x86_64"), "b4938", "0.5.0").is_err());
    }
}
