//! Test doubles: an in-process `RuntimeBackend` that needs no GPU, no model
//! file contents and no network. Enabled with the `testing` feature.
//!
//! Behaviour of `MockInstance::infer_stream` is driven by the request so tests
//! can exercise every path of the protocol adapters:
//!
//! * last user text `"fail"`        → one text delta, then a backend error
//! * tools present and last user text starts with `"call "` → a tool call to the
//!   first tool with arguments `{"city":"Paris"}`
//! * `max_output_tokens == Some(1)` → finish reason `length`
//! * otherwise                       → `"echo: <last user text>"` in two deltas
//!
//! Usage counts: `input_tokens = number of input items`, `output_tokens = 2`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use ohmygpu_inference::{
    FinishReason, InferenceError, InferenceRequest, InferenceStream, InputItem, ModelKind, Role,
    StreamEvent, TranscriptionRequest, TranscriptionResponse, TranscriptionSegment, Usage,
};
use ohmygpu_runtime_api::{
    BackendAvailability, InstanceInfo, InstanceStatus, ModelInstance, ProgressFn, RuntimeBackend,
    RuntimeError, StartSpec,
};
use tokio::sync::watch;

#[derive(Default)]
pub struct MockBackend {
    /// When set, `start` fails with this message.
    pub fail_start: Mutex<Option<String>>,
    /// Artificial start latency (lets tests observe `starting`).
    pub start_delay: Mutex<Duration>,
    pub instances: Mutex<Vec<Arc<MockInstance>>>,
    pub prepare_calls: AtomicUsize,
}

impl MockBackend {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn last_instance(&self) -> Option<Arc<MockInstance>> {
        self.instances.lock().unwrap().last().cloned()
    }

    pub fn set_fail_start(&self, msg: Option<&str>) {
        *self.fail_start.lock().unwrap() = msg.map(|s| s.to_string());
    }

    pub fn set_start_delay(&self, d: Duration) {
        *self.start_delay.lock().unwrap() = d;
    }
}

#[async_trait]
impl RuntimeBackend for MockBackend {
    fn id(&self) -> &'static str {
        "mock"
    }

    async fn available(&self) -> BackendAvailability {
        BackendAvailability {
            available: true,
            version: Some("mock".into()),
            path: None,
            message: None,
        }
    }

    async fn prepare(
        &self,
        _progress: Option<ProgressFn>,
    ) -> Result<BackendAvailability, RuntimeError> {
        self.prepare_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.available().await)
    }

    async fn start(&self, spec: StartSpec) -> Result<Arc<dyn ModelInstance>, RuntimeError> {
        let delay = *self.start_delay.lock().unwrap();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        if let Some(msg) = self.fail_start.lock().unwrap().clone() {
            return Err(RuntimeError::Start(msg));
        }
        let (exit_tx, exit_rx) = watch::channel(None);
        let inst = Arc::new(MockInstance {
            model_id: spec.model_id,
            kind: spec.kind,
            spec_context: spec.context_length,
            spec_mmproj: spec.mmproj_path,
            exit_tx,
            exit_rx,
        });
        self.instances.lock().unwrap().push(inst.clone());
        Ok(inst)
    }
}

pub struct MockInstance {
    pub model_id: String,
    pub kind: ModelKind,
    pub spec_context: Option<u32>,
    /// Projector path the manager handed us (vision models).
    pub spec_mmproj: Option<std::path::PathBuf>,
    exit_tx: watch::Sender<Option<InstanceStatus>>,
    exit_rx: watch::Receiver<Option<InstanceStatus>>,
}

impl MockInstance {
    /// Simulate the backend process dying.
    pub fn crash(&self) {
        let _ = self.exit_tx.send(Some(InstanceStatus::Exited {
            code: Some(1),
            message: "simulated crash".into(),
        }));
    }
}

/// Text and image count of the last user message.
fn last_user_message(req: &InferenceRequest) -> (String, usize) {
    req.input
        .iter()
        .rev()
        .find_map(|i| match i {
            InputItem::Message {
                role: Role::User, ..
            } => Some((i.text().unwrap_or_default(), i.image_count())),
            _ => None,
        })
        .unwrap_or_default()
}

#[async_trait]
impl ModelInstance for MockInstance {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn info(&self) -> InstanceInfo {
        InstanceInfo {
            backend: "mock".into(),
            pid: Some(4242),
            port: Some(1),
            backend_version: Some("mock".into()),
        }
    }

    async fn status(&self) -> InstanceStatus {
        self.exit_rx
            .borrow()
            .clone()
            .unwrap_or(InstanceStatus::Running)
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, InferenceError> {
        if let Some(InstanceStatus::Exited { message, .. }) = self.exit_rx.borrow().clone() {
            return Err(InferenceError::Unavailable(message));
        }
        if self.kind == ModelKind::Whisper {
            return Err(InferenceError::Unsupported(format!(
                "model '{}' is a speech-to-text model; use POST /v1/audio/transcriptions",
                request.model
            )));
        }
        let (text, images) = last_user_message(&request);
        let usage = Usage {
            input_tokens: request.input.len() as u32,
            output_tokens: 2,
        };
        let mut events: Vec<Result<StreamEvent, InferenceError>> = Vec::new();
        if text == "fail" {
            events.push(Ok(StreamEvent::TextDelta {
                text: "partial".into(),
            }));
            events.push(Err(InferenceError::Backend("simulated failure".into())));
        } else if !request.tools.is_empty() && text.starts_with("call ") {
            let name = request.tools[0].name.clone();
            events.push(Ok(StreamEvent::ToolCallStart {
                index: 0,
                id: "call_mock1".into(),
                name,
            }));
            events.push(Ok(StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: "{\"city\":".into(),
            }));
            events.push(Ok(StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: "\"Paris\"}".into(),
            }));
            events.push(Ok(StreamEvent::Completed {
                finish_reason: FinishReason::ToolCalls,
                usage,
            }));
        } else if request.options.max_output_tokens == Some(1) {
            events.push(Ok(StreamEvent::TextDelta {
                text: "echo".into(),
            }));
            events.push(Ok(StreamEvent::Completed {
                finish_reason: FinishReason::Length,
                usage,
            }));
        } else {
            events.push(Ok(StreamEvent::TextDelta {
                text: if images > 0 {
                    format!("saw {images} image(s); echo: ")
                } else {
                    "echo: ".into()
                },
            }));
            events.push(Ok(StreamEvent::TextDelta { text }));
            events.push(Ok(StreamEvent::Completed {
                finish_reason: FinishReason::Stop,
                usage,
            }));
        }
        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, InferenceError> {
        if self.kind != ModelKind::Whisper {
            return Err(InferenceError::Unsupported(format!(
                "model '{}' does not transcribe audio (not a speech-to-text model)",
                request.model
            )));
        }
        request.validate()?;
        let secs = request.audio.duration_secs();
        let lang = request.language.clone().unwrap_or_else(|| "auto".into());
        let text = format!(
            "transcribed {secs:.1}s at {} Hz ({lang})",
            request.audio.sample_rate
        );
        Ok(TranscriptionResponse {
            model: request.model,
            text: text.clone(),
            language: Some(lang),
            duration_secs: secs,
            segments: vec![TranscriptionSegment {
                id: 0,
                start_secs: 0.0,
                end_secs: secs,
                text,
            }],
        })
    }

    async fn wait(&self) -> InstanceStatus {
        let mut rx = self.exit_rx.clone();
        loop {
            if let Some(s) = rx.borrow().clone() {
                return s;
            }
            if rx.changed().await.is_err() {
                return InstanceStatus::Exited {
                    code: None,
                    message: "gone".into(),
                };
            }
        }
    }

    async fn stop(&self) -> Result<(), RuntimeError> {
        let _ = self.exit_tx.send(Some(InstanceStatus::Exited {
            code: Some(0),
            message: "stopped".into(),
        }));
        Ok(())
    }
}
