//! Tiny HTTP client for the Management API. The CLI never touches models,
//! registry files or the backend directly — it only talks to the daemon.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: &str) -> Self {
        Self {
            base: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(3))
                .build()
                .expect("reqwest client"),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// True if a daemon answers at the base URL.
    pub async fn is_up(&self) -> bool {
        self.http
            .get(format!("{}/ohmygpu/v1/health", self.base))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub async fn get(&self, path: &str) -> Result<Value> {
        let resp = self
            .http
            .get(format!("{}{}", self.base, path))
            .send()
            .await
            .map_err(|e| self.conn_err(e))?;
        Self::body(resp).await
    }

    pub async fn post(&self, path: &str, body: Option<Value>) -> Result<Value> {
        let mut req = self
            .http
            .post(format!("{}{}", self.base, path))
            .timeout(Duration::from_secs(60 * 30));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.map_err(|e| self.conn_err(e))?;
        Self::body(resp).await
    }

    pub async fn delete(&self, path: &str) -> Result<Value> {
        let resp = self
            .http
            .delete(format!("{}{}", self.base, path))
            .send()
            .await
            .map_err(|e| self.conn_err(e))?;
        Self::body(resp).await
    }

    fn conn_err(&self, e: reqwest::Error) -> anyhow::Error {
        if e.is_connect() || e.is_timeout() {
            anyhow!(
                "cannot reach the OhMyGPU runtime at {} — start it with `omg serve`",
                self.base
            )
        } else {
            anyhow!("request failed: {e}")
        }
    }

    async fn body(resp: reqwest::Response) -> Result<Value> {
        let status = resp.status();
        let text = resp.text().await.context("reading response")?;
        let value: Value = serde_json::from_str(&text).unwrap_or(Value::String(text.clone()));
        if !status.is_success() {
            let msg = value
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .unwrap_or(text);
            bail!("{msg} (HTTP {status})");
        }
        Ok(value)
    }
}
