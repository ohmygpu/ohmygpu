//! `omg upgrade` — replace the installed `ohmygpu` and `ohmygpu-runtime` with a
//! GitHub release (the latest by default), verifying the release's
//! `SHA256SUMS.txt`. Client-side tooling only: no runtime state is touched, and
//! a runtime that is already running keeps serving the old version until it is
//! restarted.
//!
//! The binaries are replaced where they currently live (the CLI's own path and
//! the `ohmygpu-runtime` next to it, or on `PATH`), by writing a temporary file
//! in the same directory and renaming it over the old one — the same thing
//! `install.sh` does, without a shell.

use std::cmp::Ordering;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use ohmygpu_runtime_api::{ProgressFn, ProgressUpdate, RuntimeError};
use ohmygpu_runtime_common::install as managed;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::client::Client;

/// GitHub repository the releases come from.
pub const REPO: &str = "ohmygpu/ohmygpu";
const CLI_BIN: &str = "ohmygpu";
const RUNTIME_BIN: &str = "ohmygpu-runtime";

pub struct UpgradeOptions<'a> {
    /// A specific release (`v0.4.0` / `0.4.0`); `None` = latest.
    pub version: Option<&'a str>,
    /// Only report whether a newer release exists.
    pub check: bool,
    /// Install even when that version is already installed (or newer).
    pub force: bool,
    pub json: bool,
}

// ---------------------------------------------------------------------------
// pure helpers (unit-tested)
// ---------------------------------------------------------------------------

/// The release asset target of this build. Must match the matrix in
/// `.github/workflows/release.yml`; `None` means no prebuilt binary exists.
pub fn release_target() -> Option<&'static str> {
    release_target_for(
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg!(target_env = "musl"),
    )
}

fn release_target_for(os: &str, arch: &str, musl: bool) -> Option<&'static str> {
    match (os, arch) {
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("macos", "x86_64") => Some("x86_64-apple-darwin"),
        ("linux", "x86_64") if !musl => Some("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") if !musl => Some("aarch64-unknown-linux-gnu"),
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        _ => None,
    }
}

/// `ohmygpu-<target>.tar.gz` (`.zip` on Windows), as published by the release workflow.
pub fn asset_name(target: &str) -> String {
    format!("ohmygpu-{target}.{}", archive_ext(target))
}

fn archive_ext(target: &str) -> &'static str {
    if target.contains("windows") {
        "zip"
    } else {
        "tar.gz"
    }
}

/// `0.4.0` / `v0.4.0` → `v0.4.0`.
pub fn normalize_tag(version: &str) -> String {
    let v = version.trim();
    if v.starts_with('v') {
        v.to_string()
    } else {
        format!("v{v}")
    }
}

/// How `tag` compares to the running `current` version (`Greater` = the tag is
/// newer). `None` when either side is not a semantic version.
pub fn compare(current: &str, tag: &str) -> Option<Ordering> {
    let cur = semver::Version::parse(current.trim_start_matches('v')).ok()?;
    let new = semver::Version::parse(tag.trim_start_matches('v')).ok()?;
    Some(new.cmp(&cur))
}

/// The hex digest listed for `asset` in a `SHA256SUMS.txt` (`<hex>  <name>` lines).
pub fn checksum_for(sums: &str, asset: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        let hex = parts.next()?;
        let name = parts.next()?.trim_start_matches('*');
        (name == asset && hex.len() == 64).then(|| hex.to_ascii_lowercase())
    })
}

fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// True for a binary living in a Homebrew cellar (`/opt/homebrew/Cellar/ohmygpu/…`,
/// `/home/linuxbrew/.linuxbrew/Cellar/…`): those installs belong to `brew upgrade`.
pub fn is_homebrew_install(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str().to_str() == Some("Cellar"))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

// ---------------------------------------------------------------------------
// where the binaries live
// ---------------------------------------------------------------------------

/// `(binary name in the archive, path to replace)` for this installation: the
/// running CLI itself (symlinks such as `omg` resolved), an `ohmygpu` sibling if
/// the CLI runs under another name, and `ohmygpu-runtime` next to it — or, if
/// there is none, the one on `PATH`; a missing runtime is installed next to the CLI.
fn destinations() -> Result<Vec<(&'static str, PathBuf)>> {
    let me = std::env::current_exe().context("locating the running executable")?;
    let me = me.canonicalize().unwrap_or(me);
    let dir = me
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", me.display()))?
        .to_path_buf();

    let mut dests = vec![(CLI_BIN, me.clone())];
    let cli_sibling = dir.join(exe(CLI_BIN));
    if me.file_name().and_then(|n| n.to_str()) != Some(&exe(CLI_BIN)) && cli_sibling.is_file() {
        dests.push((CLI_BIN, cli_sibling));
    }

    let sibling_runtime = dir.join(exe(RUNTIME_BIN));
    let runtime = if sibling_runtime.is_file() {
        sibling_runtime
    } else {
        managed::which(&exe(RUNTIME_BIN))
            .map(|p| p.canonicalize().unwrap_or(p))
            .unwrap_or(sibling_runtime)
    };
    dests.push((RUNTIME_BIN, runtime));
    Ok(dests)
}

/// Fail early — before downloading anything — if we cannot write next to the binaries.
fn check_writable(dir: &Path, me: &Path) -> Result<()> {
    tempfile::Builder::new()
        .prefix(".ohmygpu-upgrade-")
        .tempfile_in(dir)
        .map(|_| ())
        .map_err(|e| {
            let hint = if cfg!(windows) {
                "re-run from an elevated (administrator) prompt".to_string()
            } else {
                format!("re-run as: sudo {} upgrade", me.display())
            };
            let why = match e.kind() {
                std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
                _ => e.to_string(),
            };
            anyhow!("cannot write to {} ({why}) — {hint}", dir.display())
        })
}

/// Put `src` at `dest` atomically: copy into a temporary file in the same
/// directory, then rename over the old binary. A running process keeps its old
/// inode, so a running runtime or the CLI itself is not disturbed. On Windows a
/// running executable cannot be replaced, but it can be renamed away first.
fn replace_binary(src: &Path, dest: &Path) -> Result<()> {
    let dir = dest
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", dest.display()))?;
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("{} has no file name", dest.display()))?;
    let tmp = dir.join(format!(".{name}.new"));
    std::fs::copy(src, &tmp).with_context(|| format!("writing {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    #[cfg(windows)]
    if dest.exists() {
        let old = dir.join(format!("{name}.old"));
        let _ = std::fs::remove_file(&old);
        std::fs::rename(dest, &old)
            .with_context(|| format!("moving the old {} aside", dest.display()))?;
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("installing {}", dest.display()))?;
    Ok(())
}

/// Remove `<name>.old` leftovers from a previous Windows upgrade (best effort).
fn sweep_old(dests: &[(&str, PathBuf)]) {
    for (_, dest) in dests {
        if let Some(name) = dest.file_name().and_then(|n| n.to_str()) {
            if let Some(dir) = dest.parent() {
                let _ = std::fs::remove_file(dir.join(format!("{name}.old")));
            }
        }
    }
}

/// The plain message of a managed-install error (its Display is phrased for backends).
fn install_msg(e: RuntimeError) -> String {
    match e {
        RuntimeError::Install(m) => m,
        other => other.to_string(),
    }
}

/// `<binary> --version` → "0.5.0".
fn binary_version(path: &Path) -> Option<String> {
    let out = Command::new(path).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().nth(1).map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// the command
// ---------------------------------------------------------------------------

pub async fn upgrade(client: &Client, opts: UpgradeOptions<'_>) -> Result<()> {
    let current = ohmygpu_core::VERSION;
    let target = release_target().ok_or_else(|| {
        anyhow!(
            "no prebuilt release for {}/{}{} — build from source (see README) or download an archive from https://github.com/{REPO}/releases",
            std::env::consts::OS,
            std::env::consts::ARCH,
            if cfg!(target_env = "musl") { " (musl)" } else { "" }
        )
    })?;

    // Which release?
    let tag = match opts.version {
        Some(v) => normalize_tag(v),
        None => managed::resolve_latest_tag(REPO)
            .await
            .map_err(|e| anyhow!("could not determine the latest release: {}", install_msg(e)))?,
    };
    let order = compare(current, &tag);
    let pinned = opts.version.is_some();

    if opts.check {
        let newer = order == Some(Ordering::Greater);
        if opts.json {
            println!(
                "{}",
                json!({ "current": current, "available": tag.trim_start_matches('v'),
                        "target": target, "update_available": newer })
            );
        } else if newer {
            println!("ohmygpu v{current} — {tag} is available: run `omg upgrade`");
        } else if pinned {
            println!("ohmygpu v{current} — {tag} {}", describe(order));
        } else {
            println!("ohmygpu v{current} is up to date ({tag} is the latest release)");
        }
        return Ok(());
    }

    if !opts.force {
        match order {
            Some(Ordering::Equal) => {
                if !opts.json {
                    println!("ohmygpu v{current} is already installed — nothing to do (use --force to reinstall)");
                }
                return Ok(());
            }
            Some(Ordering::Less) if !pinned => {
                if !opts.json {
                    println!(
                        "ohmygpu v{current} is newer than the latest release ({tag}) — nothing to do (use `omg upgrade {tag} --force` to install it anyway)"
                    );
                }
                return Ok(());
            }
            _ => {}
        }
    }

    // Where do the binaries go? Fail before downloading if we cannot write there.
    let dests = destinations()?;
    let me = dests[0].1.clone();
    if is_homebrew_install(&me) {
        bail!(
            "ohmygpu is installed with Homebrew ({}) — upgrade it with: brew upgrade ohmygpu",
            me.display()
        );
    }
    let mut dirs: Vec<&Path> = dests.iter().filter_map(|(_, p)| p.parent()).collect();
    dirs.dedup();
    for dir in &dirs {
        check_writable(dir, &me)?;
    }
    sweep_old(&dests);

    let verb = match order {
        Some(Ordering::Less) => "Downgrading",
        Some(Ordering::Equal) => "Reinstalling",
        _ => "Upgrading",
    };
    if !opts.json {
        println!("{verb} ohmygpu v{current} → {tag} ({target})");
        for (_, p) in &dests {
            println!("  {}", p.display());
        }
    }

    // Download the archive and its checksum.
    let asset = asset_name(target);
    let base = format!("https://github.com/{REPO}/releases/download/{tag}");
    let work = tempfile::Builder::new()
        .prefix("ohmygpu-upgrade-")
        .tempdir()
        .context("creating a temporary directory")?;
    let archive = work.path().join(&asset);
    let tty = !opts.json && std::io::stdout().is_terminal();
    if !opts.json {
        println!("Downloading {base}/{asset} …");
    }
    // Live progress only on a terminal, at most ~10 updates/s (the downloader reports every chunk).
    let progress: Option<ProgressFn> = tty.then(|| {
        let last = Mutex::new(Instant::now() - Duration::from_secs(1));
        Arc::new(move |u: ProgressUpdate| {
            let finished = u.done_bytes.is_some() && u.done_bytes == u.total_bytes;
            let mut last = last.lock().unwrap_or_else(|p| p.into_inner());
            if !finished && last.elapsed() < Duration::from_millis(100) {
                return;
            }
            *last = Instant::now();
            let line = match (u.done_bytes, u.total_bytes) {
                (Some(d), Some(t)) if t > 0 => format!(
                    "  {:>5.1}%  {:.1} / {:.1} MB",
                    d as f64 / t as f64 * 100.0,
                    d as f64 / 1_048_576.0,
                    t as f64 / 1_048_576.0
                ),
                (Some(d), _) => format!("  {:.1} MB", d as f64 / 1_048_576.0),
                _ => String::new(),
            };
            print!("\r{line:<40}");
            std::io::stdout().flush().ok();
        }) as ProgressFn
    });
    let url = format!("{base}/{asset}");
    managed::download_archive(&url, &archive, progress, "")
        .await
        .map_err(|e| {
            let m = install_msg(e);
            let m = m.strip_prefix(&format!("downloading {url}: ")).unwrap_or(&m);
            anyhow!(
                "could not download {asset} for {tag}: {m} — is there such a release, with a build for {target}? https://github.com/{REPO}/releases"
            )
        })?;
    if tty {
        println!();
    }
    let sums_path = work.path().join("SHA256SUMS.txt");
    managed::download_archive(&format!("{base}/SHA256SUMS.txt"), &sums_path, None, "")
        .await
        .map_err(|e| {
            anyhow!(
                "could not download SHA256SUMS.txt for {tag}: {} — refusing to install an unverified archive",
                install_msg(e)
            )
        })?;
    let sums = std::fs::read_to_string(&sums_path)?;
    let expected = checksum_for(&sums, &asset)
        .ok_or_else(|| anyhow!("SHA256SUMS.txt of {tag} has no entry for {asset}"))?;
    let actual = sha256_file(&archive)?;
    if actual != expected {
        bail!("checksum mismatch for {asset}: expected {expected}, got {actual} — refusing to install");
    }

    // Extract and install.
    let extracted = work.path().join("extracted");
    managed::extract_archive(&archive, &extracted, archive_ext(target))
        .await
        .map_err(install_msg)
        .map_err(anyhow::Error::msg)?;
    let mut installed = Vec::new();
    for (name, dest) in &dests {
        let src = managed::find_binary(&extracted, &exe(name))
            .ok_or_else(|| anyhow!("{asset} does not contain {}", exe(name)))?;
        replace_binary(&src, dest)?;
        installed.push(dest.clone());
    }
    sweep_old(&dests);

    // Verify what is on disk now.
    let new_version = binary_version(&me).ok_or_else(|| {
        anyhow!(
            "{} was installed but does not run — try downloading the archive by hand from https://github.com/{REPO}/releases",
            me.display()
        )
    })?;

    // A runtime that is already up keeps the old version until restarted.
    let running = if client.is_up().await {
        client
            .get("/ohmygpu/v1/health")
            .await
            .ok()
            .and_then(|v| v["version"].as_str().map(|s| s.to_string()))
            .or_else(|| Some("?".into()))
    } else {
        None
    };

    if opts.json {
        println!(
            "{}",
            json!({ "from": current, "to": new_version, "tag": tag, "target": target,
                    "installed": installed, "runtime_running_version": running })
        );
        return Ok(());
    }
    println!(
        "Installed ohmygpu v{new_version} ({})",
        fmt_paths(&installed)
    );
    if let Some(v) = running {
        println!(
            "A runtime (v{v}) is still running at {} — restart it to use v{new_version}:  omg shutdown && omg serve",
            client.base_url()
        );
    }
    Ok(())
}

fn describe(order: Option<Ordering>) -> &'static str {
    match order {
        Some(Ordering::Greater) => "is newer",
        Some(Ordering::Equal) => "is what you have",
        Some(Ordering::Less) => "is older",
        None => "is a different (non-semver) version",
    }
}

fn fmt_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_match_the_release_matrix() {
        assert_eq!(
            release_target_for("macos", "aarch64", false),
            Some("aarch64-apple-darwin")
        );
        assert_eq!(
            release_target_for("linux", "x86_64", false),
            Some("x86_64-unknown-linux-gnu")
        );
        assert_eq!(release_target_for("linux", "x86_64", true), None);
        assert_eq!(
            release_target_for("windows", "x86_64", false),
            Some("x86_64-pc-windows-msvc")
        );
        assert_eq!(release_target_for("freebsd", "x86_64", false), None);
        assert_eq!(
            asset_name("aarch64-apple-darwin"),
            "ohmygpu-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            asset_name("x86_64-pc-windows-msvc"),
            "ohmygpu-x86_64-pc-windows-msvc.zip"
        );
        assert!(
            release_target().is_some(),
            "this CI platform has a release build"
        );
    }

    #[test]
    fn homebrew_installs_are_recognised() {
        assert!(is_homebrew_install(Path::new(
            "/opt/homebrew/Cellar/ohmygpu/0.5.0/bin/ohmygpu"
        )));
        assert!(is_homebrew_install(Path::new(
            "/home/linuxbrew/.linuxbrew/Cellar/ohmygpu/0.5.0/bin/ohmygpu"
        )));
        assert!(!is_homebrew_install(Path::new("/usr/local/bin/ohmygpu")));
        assert!(!is_homebrew_install(Path::new(
            "/Users/x/.local/bin/ohmygpu"
        )));
    }

    #[test]
    fn tags_and_versions() {
        assert_eq!(normalize_tag("0.4.0"), "v0.4.0");
        assert_eq!(normalize_tag(" v0.4.0 "), "v0.4.0");
        assert_eq!(compare("0.3.2", "v0.5.0"), Some(Ordering::Greater));
        assert_eq!(compare("0.5.0", "v0.5.0"), Some(Ordering::Equal));
        assert_eq!(compare("0.5.0", "v0.4.0"), Some(Ordering::Less));
        assert_eq!(compare("0.5.0", "v0.6.0-beta.1"), Some(Ordering::Greater));
        assert_eq!(compare("0.5.0", "nightly"), None);
    }

    #[test]
    fn checksums_are_looked_up_by_asset_name() {
        let sums = "dac6c416ca663113cafae2549fa0012a4719c064e0ada47b12afa4810cc3940b  ohmygpu-aarch64-apple-darwin.tar.gz\n\
                    70ACC69C1FC56E8E2C7DB2EC51A9B7A36994F5E2E6D651C0849949AA1BE60E8E *ohmygpu-x86_64-pc-windows-msvc.zip\n";
        assert_eq!(
            checksum_for(sums, "ohmygpu-aarch64-apple-darwin.tar.gz").as_deref(),
            Some("dac6c416ca663113cafae2549fa0012a4719c064e0ada47b12afa4810cc3940b")
        );
        assert_eq!(
            checksum_for(sums, "ohmygpu-x86_64-pc-windows-msvc.zip").as_deref(),
            Some("70acc69c1fc56e8e2c7db2ec51a9b7a36994f5e2e6d651c0849949aa1be60e8e")
        );
        assert_eq!(
            checksum_for(sums, "ohmygpu-x86_64-apple-darwin.tar.gz"),
            None
        );
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x");
        std::fs::write(&f, b"abc").unwrap();
        assert_eq!(
            sha256_file(&f).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn replace_binary_is_a_rename_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let dest = dir.path().join("bin").join("ohmygpu");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&src, b"new").unwrap();
        std::fs::write(&dest, b"old").unwrap();
        replace_binary(&src, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"new");
        assert!(!dest.parent().unwrap().join(".ohmygpu.new").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&dest).unwrap().permissions().mode() & 0o777,
                0o755
            );
        }
    }
}
