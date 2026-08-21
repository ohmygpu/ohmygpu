//! Managed binary installs, the parts that are the same for every backend:
//! where a binary was found, install records, `PATH` lookup, downloading and
//! extracting a release archive, and resolving GitHub's `latest` tag.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ohmygpu_core::download::Downloader;
use ohmygpu_runtime_api::{ProgressFn, ProgressUpdate, RuntimeError};
use serde::{Deserialize, Serialize};

/// Where a backend binary came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinarySource {
    /// Explicit path from config / environment.
    Config,
    /// Installed by us under `<data>/runtimes/<backend>/<tag>/`.
    Managed,
    /// Found on `PATH`.
    Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocatedBinary {
    pub path: PathBuf,
    pub source: BinarySource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// `install.json` next to a managed install.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRecord {
    pub tag: String,
    pub asset: String,
    pub binary: PathBuf,
}

impl InstallRecord {
    pub fn path_in(dir: &Path) -> PathBuf {
        dir.join("install.json")
    }

    /// The record in `dir`, if present and its binary still exists.
    pub fn read(dir: &Path) -> Option<InstallRecord> {
        let text = std::fs::read_to_string(Self::path_in(dir)).ok()?;
        let rec: InstallRecord = serde_json::from_str(&text).ok()?;
        rec.binary.is_file().then_some(rec)
    }

    pub fn write(&self, dir: &Path) -> Result<(), RuntimeError> {
        std::fs::write(
            Self::path_in(dir),
            serde_json::to_string_pretty(self).expect("record serializes"),
        )
        .map_err(|e| RuntimeError::Install(e.to_string()))
    }
}

/// The newest managed install under `managed_root` (one subdirectory per tag),
/// ranked by `rank(tag)`.
pub fn newest_managed(managed_root: &Path, rank: fn(&str) -> u64) -> Option<InstallRecord> {
    let mut records: Vec<(u64, InstallRecord)> = Vec::new();
    for entry in std::fs::read_dir(managed_root).ok()?.flatten() {
        if let Some(rec) = InstallRecord::read(&entry.path()) {
            records.push((rank(&rec.tag), rec));
        }
    }
    records.sort_by(|a, b| b.0.cmp(&a.0));
    records.into_iter().next().map(|(_, r)| r)
}

/// `b1234` → 1234 (ggml-style build tags); anything else → 0.
pub fn tag_number(tag: &str) -> u64 {
    tag.trim_start_matches('b').parse().unwrap_or(0)
}

/// First `name` on `PATH`.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// The tag GitHub's `releases/latest` redirects to for `owner/repo`.
pub async fn resolve_latest_tag(repo: &str) -> Result<String, RuntimeError> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(format!("ohmygpu/{}", ohmygpu_core::VERSION))
        .build()
        .map_err(|e| RuntimeError::Install(e.to_string()))?;
    let url = format!("https://github.com/{repo}/releases/latest");
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

/// Download `url` to `dest`, reporting `message` with byte progress.
pub async fn download_archive(
    url: &str,
    dest: &Path,
    progress: Option<ProgressFn>,
    message: &str,
) -> Result<(), RuntimeError> {
    let msg = message.to_string();
    let progress_cb: Option<ohmygpu_core::download::ProgressCallback> = progress.map(|p| {
        Arc::new(move |dp: ohmygpu_core::lifecycle::DownloadProgress| {
            p(ProgressUpdate {
                message: msg.clone(),
                done_bytes: Some(dp.downloaded_bytes),
                total_bytes: dp.total_bytes,
            })
        }) as ohmygpu_core::download::ProgressCallback
    });
    Downloader::new(None)
        .download(url, dest, progress_cb, None)
        .await
        .map_err(|e| RuntimeError::Install(format!("downloading {url}: {e}")))?;
    Ok(())
}

/// Extract a `tar.gz` or `zip` archive into `into` (on a blocking thread).
pub async fn extract_archive(archive: &Path, into: &Path, ext: &str) -> Result<(), RuntimeError> {
    let archive = archive.to_path_buf();
    let into = into.to_path_buf();
    let ext = ext.to_string();
    tokio::task::spawn_blocking(move || extract_sync(&archive, &into, &ext))
        .await
        .map_err(|e| RuntimeError::Install(e.to_string()))?
}

fn extract_sync(archive: &Path, into: &Path, ext: &str) -> Result<(), RuntimeError> {
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

/// Find a file called `name` anywhere under `dir`.
pub fn find_binary(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).ok()?.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().map(|n| n == name).unwrap_or(false) {
                return Some(p);
            }
        }
    }
    None
}

/// Mark every file in the binary's directory executable (archives from CI
/// sometimes lose the bit; shared libraries live next to the binary).
#[cfg(unix)]
pub fn make_executable(path: &Path) -> Result<(), RuntimeError> {
    use std::os::unix::fs::PermissionsExt;
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
pub fn make_executable(_path: &Path) -> Result<(), RuntimeError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_numbers_and_records() {
        assert_eq!(tag_number("b6234"), 6234);
        assert_eq!(tag_number("latest"), 0);
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("x");
        std::fs::write(&bin, b"x").unwrap();
        let sub = dir.path().join("b10");
        std::fs::create_dir_all(&sub).unwrap();
        InstallRecord {
            tag: "b10".into(),
            asset: "a".into(),
            binary: bin.clone(),
        }
        .write(&sub)
        .unwrap();
        let sub2 = dir.path().join("b7");
        std::fs::create_dir_all(&sub2).unwrap();
        InstallRecord {
            tag: "b7".into(),
            asset: "a".into(),
            binary: bin,
        }
        .write(&sub2)
        .unwrap();
        assert_eq!(newest_managed(dir.path(), tag_number).unwrap().tag, "b10");
        assert!(find_binary(dir.path(), "x").is_some());
        assert!(find_binary(dir.path(), "nope").is_none());
    }
}
