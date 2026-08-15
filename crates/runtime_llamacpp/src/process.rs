//! Supervised `llama-server` child process: spawn, forward logs, detect exit,
//! stop gracefully (SIGTERM → wait → SIGKILL).

use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ohmygpu_runtime_api::{InstanceStatus, RuntimeError, StartSpec};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, watch};

/// How many recent log lines to keep for error reporting.
const LOG_TAIL: usize = 60;

#[derive(Debug, Clone, Copy)]
enum Signal {
    Terminate,
    Kill,
}

/// Handle to a running (or exited) llama-server process.
pub struct ServerProcess {
    pub pid: Option<u32>,
    pub port: u16,
    exit_rx: watch::Receiver<Option<InstanceStatus>>,
    signal_tx: mpsc::UnboundedSender<Signal>,
    log_tail: Arc<Mutex<VecDeque<String>>>,
}

impl ServerProcess {
    /// Spawn `llama-server` for `spec` listening on `127.0.0.1:port`.
    pub fn spawn(
        binary: &Path,
        spec: &StartSpec,
        port: u16,
    ) -> Result<ServerProcess, RuntimeError> {
        let mut cmd = Command::new(binary);
        cmd.arg("--model")
            .arg(&spec.model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--alias")
            .arg(&spec.model_id)
            .arg("--jinja")
            .arg("--no-webui");
        if let Some(ctx) = spec.context_length {
            cmd.arg("--ctx-size").arg(ctx.to_string());
        }
        if let Some(ngl) = spec.gpu_layers {
            cmd.arg("--n-gpu-layers").arg(ngl.to_string());
        }
        if let Some(t) = spec.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Do not let the child inherit an unrelated CWD-dependent behaviour.
        if let Some(dir) = binary.parent() {
            cmd.current_dir(dir);
        }

        tracing::info!(model = %spec.model_id, port, "spawning llama-server: {}", binary.display());
        let mut child = cmd.spawn().map_err(|e| {
            RuntimeError::Start(format!("failed to spawn {}: {e}", binary.display()))
        })?;
        let pid = child.id();

        let log_tail = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_TAIL)));
        if let Some(out) = child.stdout.take() {
            forward_logs(out, spec.model_id.clone(), log_tail.clone());
        }
        if let Some(err) = child.stderr.take() {
            forward_logs(err, spec.model_id.clone(), log_tail.clone());
        }

        let (exit_tx, exit_rx) = watch::channel(None);
        let (signal_tx, mut signal_rx) = mpsc::unbounded_channel::<Signal>();
        let model_id = spec.model_id.clone();
        let tail_for_exit = log_tail.clone();
        tokio::spawn(async move {
            // Once every handle to this process is gone nobody can stop it, so
            // treat a dropped handle as "kill" (covers aborted start-ups).
            let mut handle_gone = false;
            loop {
                tokio::select! {
                    status = child.wait() => {
                        let status = match status {
                            Ok(s) => s,
                            Err(e) => {
                                let _ = exit_tx.send(Some(InstanceStatus::Exited { code: None, message: format!("wait failed: {e}") }));
                                return;
                            }
                        };
                        let code = status.code();
                        let tail = tail_string(&tail_for_exit);
                        let message = match code {
                            Some(0) => "llama-server exited".to_string(),
                            Some(c) => format!("llama-server exited with code {c}{tail}"),
                            None => format!("llama-server terminated by signal{tail}"),
                        };
                        tracing::info!(model = %model_id, "{message}");
                        let _ = exit_tx.send(Some(InstanceStatus::Exited { code, message }));
                        return;
                    }
                    sig = signal_rx.recv(), if !handle_gone => {
                        match sig {
                            Some(Signal::Terminate) => {
                                #[cfg(unix)]
                                {
                                    if let Some(pid) = child.id() {
                                        unsafe { libc::kill(pid as i32, libc::SIGTERM); }
                                    }
                                }
                                #[cfg(not(unix))]
                                {
                                    let _ = child.start_kill();
                                }
                            }
                            Some(Signal::Kill) => { let _ = child.start_kill(); }
                            None => { handle_gone = true; let _ = child.start_kill(); }
                        }
                    }
                }
            }
        });

        Ok(ServerProcess {
            pid,
            port,
            exit_rx,
            signal_tx,
            log_tail,
        })
    }

    pub fn has_exited(&self) -> Option<InstanceStatus> {
        self.exit_rx.borrow().clone()
    }

    /// Resolves with the exit status (immediately if already exited).
    pub async fn wait(&self) -> InstanceStatus {
        let mut rx = self.exit_rx.clone();
        loop {
            if let Some(st) = rx.borrow().clone() {
                return st;
            }
            if rx.changed().await.is_err() {
                return InstanceStatus::Exited {
                    code: None,
                    message: "supervisor gone".into(),
                };
            }
        }
    }

    /// SIGTERM, wait up to `grace`, then SIGKILL.
    pub async fn stop(&self, grace: Duration) {
        if self.has_exited().is_some() {
            return;
        }
        let _ = self.signal_tx.send(Signal::Terminate);
        if tokio::time::timeout(grace, self.wait()).await.is_ok() {
            return;
        }
        tracing::warn!(
            "llama-server (pid {:?}) did not exit after SIGTERM; killing",
            self.pid
        );
        let _ = self.signal_tx.send(Signal::Kill);
        let _ = tokio::time::timeout(Duration::from_secs(5), self.wait()).await;
    }

    /// Recent stdout/stderr lines (for error messages).
    pub fn log_tail(&self) -> String {
        tail_string(&self.log_tail)
    }
}

fn tail_string(tail: &Arc<Mutex<VecDeque<String>>>) -> String {
    let lines = tail.lock().unwrap();
    // Keep only the most informative recent lines.
    let picked: Vec<&String> = lines.iter().rev().take(8).collect();
    if picked.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n--- llama-server log tail ---\n");
    for l in picked.into_iter().rev() {
        out.push_str(l);
        out.push('\n');
    }
    out
}

fn forward_logs<R>(reader: R, model_id: String, tail: Arc<Mutex<VecDeque<String>>>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim_end().to_string();
            if line.is_empty() {
                continue;
            }
            tracing::debug!(target: "llamacpp", model = %model_id, "{line}");
            let mut t = tail.lock().unwrap();
            if t.len() >= LOG_TAIL {
                t.pop_front();
            }
            t.push_back(line);
        }
    });
}

/// Pick a free localhost port (bind to :0 and release it).
pub async fn free_port() -> Result<u16, RuntimeError> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| RuntimeError::Start(format!("no free port: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| RuntimeError::Start(e.to_string()))?
        .port();
    drop(listener);
    Ok(port)
}
