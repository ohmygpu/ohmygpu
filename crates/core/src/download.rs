//! Resumable HTTP downloads (Hugging Face `resolve` URLs).
//!
//! * streams to `<dest>.part` and renames on completion (never leaves a partial
//!   file at the final path)
//! * resumes with `Range` when a `.part` file exists
//! * reports progress through a callback (no terminal output in the library)
//! * cooperative cancellation via an `AtomicBool`

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use reqwest::StatusCode;
use tokio::io::AsyncWriteExt;

use crate::lifecycle::DownloadProgress;

pub const HF_BASE_URL: &str = "https://huggingface.co";

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("access denied ({status}) for {url}; the repository may be gated — set an HF token (config models.hf_token or HF_TOKEN)")]
    AccessDenied { url: String, status: u16 },
    #[error("unexpected HTTP status {status} for {url}")]
    Http { url: String, status: u16 },
    #[error("network error: {0}")]
    Network(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("download cancelled")]
    Cancelled,
}

impl From<reqwest::Error> for DownloadError {
    fn from(e: reqwest::Error) -> Self {
        DownloadError::Network(e.to_string())
    }
}

pub type ProgressCallback = Arc<dyn Fn(DownloadProgress) + Send + Sync>;

#[derive(Clone)]
pub struct Downloader {
    client: reqwest::Client,
    hf_token: Option<String>,
}

impl Downloader {
    pub fn new(hf_token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(format!("ohmygpu/{}", crate::VERSION))
            .connect_timeout(Duration::from_secs(30))
            // no overall timeout: multi-GB files legitimately take a long time
            .build()
            .expect("reqwest client");
        Self {
            client,
            hf_token: hf_token.filter(|t| !t.trim().is_empty()),
        }
    }

    /// `https://huggingface.co/{repo}/resolve/main/{file}`
    pub fn hf_url(repo: &str, file: &str) -> String {
        format!("{HF_BASE_URL}/{repo}/resolve/main/{file}")
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.hf_token {
            Some(t) => req.header(AUTHORIZATION, format!("Bearer {t}")),
            None => req,
        }
    }

    /// Size of the remote file (via a redirected `HEAD`), if the server says.
    pub async fn remote_size(&self, url: &str) -> Result<Option<u64>, DownloadError> {
        let resp = self.auth(self.client.head(url)).send().await?;
        check_status(url, resp.status())?;
        Ok(resp
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok()))
    }

    /// Download `url` to `dest`, resuming a previous `.part` if present.
    /// Returns the final file size.
    pub async fn download(
        &self,
        url: &str,
        dest: &Path,
        progress: Option<ProgressCallback>,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<u64, DownloadError> {
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let part: PathBuf = part_path(dest);
        let mut have: u64 = match tokio::fs::metadata(&part).await {
            Ok(m) => m.len(),
            Err(_) => 0,
        };

        let mut req = self.auth(self.client.get(url));
        if have > 0 {
            req = req.header(RANGE, format!("bytes={have}-"));
        }
        let resp = req.send().await?;
        let status = resp.status();

        let (append, total) = match status {
            StatusCode::PARTIAL_CONTENT => {
                let total = resp
                    .headers()
                    .get(CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.rsplit('/').next().and_then(|t| t.parse::<u64>().ok()));
                (true, total)
            }
            StatusCode::RANGE_NOT_SATISFIABLE => {
                // Our .part may already be complete: confirm with the remote size.
                let remote = self.remote_size(url).await?;
                if remote == Some(have) {
                    tokio::fs::rename(&part, dest).await?;
                    report(&progress, have, remote);
                    return Ok(have);
                }
                // Otherwise start over.
                tokio::fs::remove_file(&part).await.ok();
                return Box::pin(self.download(url, dest, progress, cancel)).await;
            }
            s if s.is_success() => {
                // Server ignored the range (or nothing to resume): start from zero.
                have = 0;
                (false, resp.content_length())
            }
            s => return Err(status_error(url, s)),
        };

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(&part)
            .await?;

        let total_bytes = total;
        report(&progress, have, total_bytes);

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if cancel
                .as_ref()
                .map(|c| c.load(Ordering::Relaxed))
                .unwrap_or(false)
            {
                file.flush().await.ok();
                return Err(DownloadError::Cancelled);
            }
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            have += chunk.len() as u64;
            report(&progress, have, total_bytes);
        }
        file.flush().await?;
        drop(file);

        if let Some(t) = total_bytes {
            if have != t {
                return Err(DownloadError::Network(format!(
                    "incomplete download: got {have} of {t} bytes (will resume on retry)"
                )));
            }
        }
        tokio::fs::rename(&part, dest).await?;
        Ok(have)
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

fn report(progress: &Option<ProgressCallback>, downloaded: u64, total: Option<u64>) {
    if let Some(cb) = progress {
        cb(DownloadProgress {
            downloaded_bytes: downloaded,
            total_bytes: total,
        });
    }
}

fn check_status(url: &str, status: StatusCode) -> Result<(), DownloadError> {
    if status.is_success() || status == StatusCode::PARTIAL_CONTENT {
        Ok(())
    } else {
        Err(status_error(url, status))
    }
}

fn status_error(url: &str, status: StatusCode) -> DownloadError {
    match status {
        StatusCode::NOT_FOUND => DownloadError::NotFound(url.to_string()),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => DownloadError::AccessDenied {
            url: url.to_string(),
            status: status.as_u16(),
        },
        s => DownloadError::Http {
            url: url.to_string(),
            status: s.as_u16(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, Response, StatusCode},
        routing::get,
        Router,
    };
    use std::sync::Mutex;

    /// Tiny file server with Range support and an optional "drop after N bytes" fault.
    async fn serve(data: Vec<u8>, cut_first_at: Option<usize>) -> String {
        #[derive(Clone)]
        struct S {
            data: Arc<Vec<u8>>,
            cut: Arc<Mutex<Option<usize>>>,
        }
        async fn handler(State(s): State<S>, headers: HeaderMap) -> Response<Body> {
            let range = headers
                .get("range")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.to_string());
            let start = range
                .as_deref()
                .and_then(|r| r.strip_prefix("bytes="))
                .and_then(|r| r.split('-').next())
                .and_then(|s| s.parse::<usize>().ok());
            let total = s.data.len();
            let cut = s.cut.lock().unwrap().take();
            match start {
                Some(st) if st >= total => {
                    Response::builder().status(416).body(Body::empty()).unwrap()
                }
                Some(st) => {
                    let body = s.data[st..].to_vec();
                    Response::builder()
                        .status(206)
                        .header("content-range", format!("bytes {st}-{}/{total}", total - 1))
                        .header("content-length", body.len())
                        .body(Body::from(body))
                        .unwrap()
                }
                None => {
                    let body = match cut {
                        // Send the first `n` bytes, let them flush, then fail the transfer
                        // (content-length promises more) — looks like a dropped connection.
                        Some(n) => {
                            let head = s.data[..n].to_vec();
                            let stream = futures_util::stream::iter(vec![Ok::<_, std::io::Error>(
                                bytes::Bytes::from(head),
                            )])
                            .chain(futures_util::stream::once(async {
                                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                                Err(std::io::Error::other("simulated network drop"))
                            }));
                            Body::from_stream(stream)
                        }
                        None => Body::from(s.data.to_vec()),
                    };
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-length", total)
                        .body(body)
                        .unwrap()
                }
            }
        }
        let state = S {
            data: Arc::new(data),
            cut: Arc::new(Mutex::new(cut_first_at)),
        };
        let app = Router::new()
            .route("/f.gguf", get(handler))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}/f.gguf")
    }

    #[tokio::test]
    async fn downloads_and_reports_progress() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let url = serve(data.clone(), None).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.gguf");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let n = Downloader::new(None)
            .download(
                &url,
                &dest,
                Some(Arc::new(move |p| seen2.lock().unwrap().push(p))),
                None,
            )
            .await
            .unwrap();
        assert_eq!(n, data.len() as u64);
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        assert!(!part_path(&dest).exists());
        let last = *seen.lock().unwrap().last().unwrap();
        assert_eq!(last.downloaded_bytes, data.len() as u64);
        assert_eq!(last.total_bytes, Some(data.len() as u64));
    }

    #[tokio::test]
    async fn resumes_from_existing_part_file() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 13) as u8).collect();
        let url = serve(data.clone(), None).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.gguf");
        // A previous run left the first 100k bytes behind.
        std::fs::write(part_path(&dest), &data[..100_000]).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let n = Downloader::new(None)
            .download(
                &url,
                &dest,
                Some(Arc::new(move |p| seen2.lock().unwrap().push(p))),
                None,
            )
            .await
            .unwrap();
        assert_eq!(n, data.len() as u64);
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        // Progress starts from the resumed offset, not zero.
        assert_eq!(
            seen.lock().unwrap().first().unwrap().downloaded_bytes,
            100_000
        );
    }

    #[tokio::test]
    async fn interrupted_download_keeps_part_and_resumes() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 13) as u8).collect();
        let url = serve(data.clone(), Some(100_000)).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.gguf");
        let d = Downloader::new(None);
        // First attempt is cut short by the server → error, nothing at `dest`, .part kept.
        let err = d.download(&url, &dest, None, None).await.unwrap_err();
        assert!(matches!(err, DownloadError::Network(_)), "{err}");
        assert!(!dest.exists());
        assert!(part_path(&dest).exists());
        // Second attempt (server now healthy) resumes/restarts and completes.
        let n = d.download(&url, &dest, None, None).await.unwrap();
        assert_eq!(n, data.len() as u64);
        assert_eq!(std::fs::read(&dest).unwrap(), data);
    }

    #[tokio::test]
    async fn cancellation_keeps_part_file() {
        let data = vec![7u8; 500_000];
        let url = serve(data, None).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.gguf");
        let cancel = Arc::new(AtomicBool::new(true));
        let err = Downloader::new(None)
            .download(&url, &dest, None, Some(cancel))
            .await
            .unwrap_err();
        assert!(matches!(err, DownloadError::Cancelled));
        assert!(!dest.exists());
    }

    #[tokio::test]
    async fn not_found_is_reported() {
        let url = serve(vec![1, 2, 3], None)
            .await
            .replace("f.gguf", "missing.gguf");
        let dir = tempfile::tempdir().unwrap();
        let err = Downloader::new(None)
            .download(&url, &dir.path().join("x"), None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, DownloadError::NotFound(_)));
    }

    #[test]
    fn hf_url_shape() {
        assert_eq!(
            Downloader::hf_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        );
    }
}
