//! The explicit model lifecycle shared between the daemon and its clients.
//!
//! ```text
//! not_installed ─pull─▶ downloading ─▶ installed ─start─▶ starting ─▶ running ─stop─▶ stopping ─▶ stopped
//!                           │                                 │           │
//!                           └──▶ error ◀──────────────────────┴───────────┘   (failed start / crash)
//! ```
//!
//! `installed` and `stopped` are both startable; they are kept distinct so a
//! client can tell "never started" from "was running, then stopped".

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
}

impl DownloadProgress {
    pub fn percent(&self) -> Option<f64> {
        self.total_bytes
            .filter(|t| *t > 0)
            .map(|t| (self.downloaded_bytes as f64 / t as f64 * 100.0).min(100.0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ModelState {
    NotInstalled,
    Downloading {
        #[serde(flatten)]
        progress: DownloadProgress,
    },
    Installed,
    Starting {
        /// What the runtime is doing right now ("installing backend", "loading model").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    Running,
    Stopping,
    Stopped,
    Error {
        message: String,
    },
}

impl ModelState {
    pub fn name(&self) -> &'static str {
        match self {
            ModelState::NotInstalled => "not_installed",
            ModelState::Downloading { .. } => "downloading",
            ModelState::Installed => "installed",
            ModelState::Starting { .. } => "starting",
            ModelState::Running => "running",
            ModelState::Stopping => "stopping",
            ModelState::Stopped => "stopped",
            ModelState::Error { .. } => "error",
        }
    }

    /// The model file is present locally (regardless of run state).
    pub fn is_installed(&self) -> bool {
        !matches!(
            self,
            ModelState::NotInstalled | ModelState::Downloading { .. }
        )
    }

    /// A start request is legal from this state.
    pub fn can_start(&self) -> bool {
        matches!(
            self,
            ModelState::Installed | ModelState::Stopped | ModelState::Error { .. }
        )
    }

    pub fn is_running(&self) -> bool {
        matches!(self, ModelState::Running)
    }

    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            ModelState::Downloading { .. } | ModelState::Starting { .. } | ModelState::Stopping
        )
    }

    pub fn error_message(&self) -> Option<&str> {
        match self {
            ModelState::Error { message } => Some(message),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_predicates() {
        assert!(!ModelState::NotInstalled.is_installed());
        assert!(!ModelState::Downloading {
            progress: DownloadProgress::default()
        }
        .is_installed());
        assert!(ModelState::Installed.can_start());
        assert!(ModelState::Stopped.can_start());
        assert!(ModelState::Error {
            message: "x".into()
        }
        .can_start());
        assert!(!ModelState::Running.can_start());
        assert!(ModelState::Starting { message: None }.is_busy());
    }

    #[test]
    fn serializes_with_state_tag() {
        let s = serde_json::to_value(ModelState::Downloading {
            progress: DownloadProgress {
                downloaded_bytes: 5,
                total_bytes: Some(10),
            },
        })
        .unwrap();
        assert_eq!(s["state"], "downloading");
        assert_eq!(s["downloaded_bytes"], 5);
        assert_eq!(
            serde_json::to_value(ModelState::Running).unwrap()["state"],
            "running"
        );
        assert_eq!(
            DownloadProgress {
                downloaded_bytes: 5,
                total_bytes: Some(10)
            }
            .percent(),
            Some(50.0)
        );
    }
}
