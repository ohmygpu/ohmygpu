//! HTTP-level tests of the daemon against the mock backend: health, hardware,
//! management API, lifecycle transitions, error format, and both inference
//! protocols mapping onto the same internal pipeline.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use ohmygpu_core::config::Config;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_core::lifecycle::ModelState;
use ohmygpu_core::paths::Paths;
use ohmygpu_core::registry::{InstalledModel, ModelCapabilities, ModelRegistry, ModelSource};
use ohmygpu_daemon::api::router;
use ohmygpu_daemon::manager::Backends;
use ohmygpu_daemon::server::build_state;
use ohmygpu_daemon::state::SharedState;
use ohmygpu_daemon::testing::MockBackend;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "mock-model";

struct TestApp {
    app: Router,
    state: SharedState,
    backend: Arc<MockBackend>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    _dir: tempfile::TempDir,
}

fn install_fake_model(paths: &Paths, id: &str) {
    let dir = paths.model_dir(id);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{id}.gguf"));
    std::fs::write(&file, b"GGUF-not-really").unwrap();
    let mut reg = ModelRegistry::load(paths.registry_path()).unwrap();
    reg.add(InstalledModel {
        id: id.into(),
        display_name: "Mock Model".into(),
        source: ModelSource::HuggingFace {
            repo: "mock/repo".into(),
            file: format!("{id}.gguf"),
        },
        format: "gguf".into(),
        path: file,
        mmproj_path: None,
        kind: Default::default(),
        size_bytes: 15,
        installed_at: chrono::Utc::now(),
        capabilities: ModelCapabilities {
            tools: true,
            vision: false,
        },
        context_length: None,
        curated: false,
    })
    .unwrap();
}

/// A minimal GGUF v3 header (no tensors) declaring `general.architecture =
/// llama` and `llama.context_length = ctx` — enough for the runtime to read the
/// native context window.
fn tiny_gguf(ctx: u32) -> Vec<u8> {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&0u64.to_le_bytes()); // tensors
    out.extend_from_slice(&2u64.to_le_bytes()); // kv pairs
    put_str(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes()); // string
    put_str(&mut out, "llama");
    put_str(&mut out, "llama.context_length");
    out.extend_from_slice(&4u32.to_le_bytes()); // u32
    out.extend_from_slice(&ctx.to_le_bytes());
    out
}

/// A fake model whose file is a real GGUF header with `context_length = ctx`,
/// registered *without* `context_length` (as older installs are) — the
/// manager backfills it from the file on load.
fn install_fake_model_with_gguf_header(paths: &Paths, id: &str, ctx: u32) {
    let dir = paths.model_dir(id);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{id}.gguf"));
    std::fs::write(&file, tiny_gguf(ctx)).unwrap();
    let mut reg = ModelRegistry::load(paths.registry_path()).unwrap();
    reg.add(InstalledModel {
        id: id.into(),
        display_name: "Mock Model With Header".into(),
        source: ModelSource::Url {
            url: format!("https://example.invalid/{id}.gguf"),
        },
        format: "gguf".into(),
        path: file,
        mmproj_path: None,
        kind: Default::default(),
        size_bytes: 100,
        installed_at: chrono::Utc::now(),
        capabilities: ModelCapabilities::default(),
        context_length: None,
        curated: false,
    })
    .unwrap();
}

/// A fake *vision* model: weights + projector, `capabilities.vision = true`.
fn install_fake_vision_model(paths: &Paths, id: &str) {
    let dir = paths.model_dir(id);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{id}.gguf"));
    let mmproj = dir.join("mmproj.gguf");
    std::fs::write(&file, b"GGUF-not-really").unwrap();
    std::fs::write(&mmproj, b"GGUF-proj").unwrap();
    let mut reg = ModelRegistry::load(paths.registry_path()).unwrap();
    reg.add(InstalledModel {
        id: id.into(),
        display_name: "Mock Vision Model".into(),
        source: ModelSource::HuggingFace {
            repo: "mock/vision".into(),
            file: format!("{id}.gguf"),
        },
        format: "gguf".into(),
        path: file,
        mmproj_path: Some(mmproj),
        kind: Default::default(),
        size_bytes: 24,
        installed_at: chrono::Utc::now(),
        capabilities: ModelCapabilities {
            tools: false,
            vision: true,
        },
        context_length: None,
        curated: false,
    })
    .unwrap();
}

async fn setup_with(config: Config) -> TestApp {
    setup_full(config, |_| {}).await
}

/// Like `setup_with`, with a hook to install more fake models before the
/// manager loads the registry.
async fn setup_full(config: Config, extra: impl FnOnce(&Paths)) -> TestApp {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::new(dir.path());
    paths.ensure_dirs().unwrap();
    install_fake_model(&paths, MODEL);
    extra(&paths);
    let backend = MockBackend::new();
    let (state, shutdown_rx) = build_state(
        paths,
        config,
        HardwareInfo::detect(),
        Backends::single(backend.clone()),
        "127.0.0.1".into(),
        10692,
    )
    .unwrap();
    TestApp {
        app: router(state.clone()),
        state,
        backend,
        shutdown_rx,
        _dir: dir,
    }
}

async fn setup() -> TestApp {
    setup_with(Config::default()).await
}

impl TestApp {
    async fn req(&self, method: &str, path: &str, body: Option<Value>) -> (StatusCode, String) {
        let mut b = Request::builder().method(method).uri(path);
        let body = match body {
            Some(v) => {
                b = b.header("content-type", "application/json");
                Body::from(v.to_string())
            }
            None => Body::empty(),
        };
        let resp = self
            .app
            .clone()
            .oneshot(b.body(body).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    async fn json(&self, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let (status, text) = self.req(method, path, body).await;
        let v =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("non-JSON body ({e}): {text}"));
        (status, v)
    }

    async fn raw(
        &self,
        method: &str,
        path: &str,
        body: &str,
        content_type: &str,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", content_type)
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn start_and_wait(&self) {
        let (status, body) = self
            .json(
                "POST",
                &format!("/ohmygpu/v1/models/{MODEL}/start?wait=true"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"]["state"], "running");
    }
}

/// Parse an SSE body into (event, data) pairs.
fn sse_events(body: &str) -> Vec<(Option<String>, String)> {
    let mut out = Vec::new();
    for block in body.split("\n\n") {
        let mut event = None;
        let mut data = Vec::new();
        for line in block.lines() {
            if let Some(e) = line.strip_prefix("event:") {
                event = Some(e.trim().to_string());
            } else if let Some(d) = line.strip_prefix("data:") {
                data.push(d.trim().to_string());
            }
        }
        if !data.is_empty() {
            out.push((event, data.join("\n")));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Health / hardware / status / catalog
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_endpoints() {
    let t = setup().await;
    let (s, v) = t.json("GET", "/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["status"], "ok");
    let (s, v) = t.json("GET", "/ohmygpu/v1/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["version"], ohmygpu_core::VERSION);
}

#[tokio::test]
async fn hardware_endpoint_reports_machine() {
    let t = setup().await;
    let (s, v) = t.json("GET", "/ohmygpu/v1/hardware", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["platform"].is_string());
    assert!(v["architecture"].is_string());
    assert!(v["cpu"]["cores"].as_u64().unwrap() >= 1);
    assert!(["metal", "cuda", "vulkan", "cpu"].contains(&v["backend"].as_str().unwrap()));
}

#[tokio::test]
async fn status_and_catalog() {
    let t = setup().await;
    let (s, v) = t.json("GET", "/ohmygpu/v1/status", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["backend"]["id"], "mock");
    assert_eq!(v["backend"]["available"], true);
    assert_eq!(v["models"]["installed"], 1);
    assert_eq!(v["port"], 10692);

    let (s, v) = t.json("GET", "/ohmygpu/v1/catalog", None).await;
    assert_eq!(s, StatusCode::OK);
    let models = v["models"].as_array().unwrap();
    let qwen = models
        .iter()
        .find(|m| m["id"] == "qwen2.5-0.5b-instruct")
        .expect("catalog entry");
    assert_eq!(qwen["installed"], false);
    assert_eq!(qwen["state"], "not_installed");
    assert_eq!(qwen["tools"], true);
    assert!(qwen["repo"].as_str().unwrap().contains('/'));
}

// ---------------------------------------------------------------------------
// Models listing
// ---------------------------------------------------------------------------

/// `/v1/models` tells an application what it needs to *choose* a model: kind,
/// capabilities, modalities and the native context window — for every kind.
#[tokio::test]
async fn v1_models_describe_capabilities_modalities_and_context_length() {
    let t = setup_full(Config::default(), |p| {
        install_fake_vision_model(p, "eyes");
        install_fake_whisper_model(p, "ears");
        install_fake_model_with_gguf_header(p, "ctx", 8192);
    })
    .await;
    let (_, v) = t.json("GET", "/v1/models", None).await;
    let by_id = |id: &str| {
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .cloned()
            .unwrap_or_else(|| panic!("{id} listed in {v}"))
    };

    let eyes = by_id("eyes");
    assert_eq!(eyes["kind"], "llm");
    assert_eq!(eyes["capabilities"]["vision"], true);
    assert_eq!(eyes["modalities"]["input"], json!(["text", "image"]));
    assert_eq!(eyes["modalities"]["output"], json!(["text"]));

    let ears = by_id("ears");
    assert_eq!(ears["kind"], "whisper");
    assert_eq!(ears["modalities"]["input"], json!(["audio"]));
    assert_eq!(ears["modalities"]["output"], json!(["text"]));
    assert!(ears.get("context_length").is_none());

    // Installed before context_length existed → backfilled from the GGUF on load …
    let ctx = by_id("ctx");
    assert_eq!(ctx["context_length"], 8192, "{ctx}");
    // … and persisted, so the next daemon start does not re-read the file.
    let reg = ModelRegistry::load(t.state.paths.registry_path()).unwrap();
    assert_eq!(reg.get("ctx").unwrap().context_length, Some(8192));
    // The Management API shows the same model facts.
    let (_, m) = t.json("GET", "/ohmygpu/v1/models/ctx", None).await;
    assert_eq!(m["context_length"], 8192);
    assert_eq!(m["modalities"]["input"], json!(["text"]));
    // Not running → no served window yet.
    assert!(m.get("runtime").is_none());
}

#[tokio::test]
async fn v1_models_lists_only_installed_models() {
    let t = setup().await;
    let (s, v) = t.json("GET", "/v1/models", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], MODEL);
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["state"], "installed");
    assert_eq!(data[0]["kind"], "llm");
    assert_eq!(data[0]["capabilities"]["tools"], true);
    assert_eq!(data[0]["capabilities"]["vision"], false);
    assert_eq!(data[0]["modalities"]["input"], json!(["text"]));
    assert_eq!(data[0]["modalities"]["output"], json!(["text"]));
    assert!(
        data[0].get("context_length").is_none(),
        "unknown context length is absent, not null/0: {}",
        data[0]
    );

    let (s, v) = t.json("GET", &format!("/v1/models/{MODEL}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["id"], MODEL);
    assert_eq!(v["capabilities"]["tools"], true);

    let (s, v) = t.json("GET", "/v1/models/nope", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(v["error"]["code"], "model_not_found");

    let (_, v) = t
        .json("GET", "/ohmygpu/v1/models?installed=true", None)
        .await;
    assert_eq!(v["models"].as_array().unwrap().len(), 1);
    let (_, v) = t.json("GET", "/ohmygpu/v1/models", None).await;
    assert!(
        v["models"].as_array().unwrap().len() > 1,
        "catalog entries appear as not_installed"
    );

    let (s, v) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["state"], "installed");
    assert_eq!(v["installed"], true);
    assert_eq!(v["capabilities"]["tools"], true);
    assert_eq!(v["source"]["type"], "hugging_face");
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_not_found_and_not_running_errors_on_both_apis() {
    let t = setup().await;
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let body = if path.contains("chat") {
            json!({"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
        } else {
            json!({"model": "nope", "input": "hi"})
        };
        let (s, v) = t.json("POST", path, Some(body)).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{path}: {v}");
        assert_eq!(v["error"]["code"], "model_not_found");
        assert_eq!(v["error"]["type"], "not_found_error");
        assert_eq!(v["error"]["param"], "model");

        let body = if path.contains("chat") {
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
        } else {
            json!({"model": MODEL, "input": "hi"})
        };
        let (s, v) = t.json("POST", path, Some(body)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{path}: {v}");
        assert_eq!(v["error"]["code"], "model_not_running");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("installed"));
    }
}

#[tokio::test]
async fn invalid_json_and_missing_fields_are_400_in_openai_error_shape() {
    let t = setup().await;
    let (s, v) = t
        .raw(
            "POST",
            "/v1/chat/completions",
            "{not json",
            "application/json",
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "invalid_json");

    let (s, v) = t
        .json("POST", "/v1/responses", Some(json!({"model": MODEL})))
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "input");

    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "n": 3})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["code"], "unsupported");
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_start_stop_crash_restart_delete() {
    let t = setup().await;

    // start (wait) → running with runtime info
    t.start_and_wait().await;
    let (_, v) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(v["state"], "running");
    assert_eq!(v["runtime"]["backend"], "mock");
    assert!(v["runtime"]["started_at"].is_string());
    let (_, v) = t.json("GET", "/ohmygpu/v1/status", None).await;
    assert_eq!(v["models"]["running"], json!([MODEL]));

    // start again is idempotent (200 running)
    let (s, v) = t
        .json("POST", &format!("/ohmygpu/v1/models/{MODEL}/start"), None)
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["model"]["state"], "running");

    // stop → stopped
    let (s, v) = t
        .json("POST", &format!("/ohmygpu/v1/models/{MODEL}/stop"), None)
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["model"]["state"], "stopped");
    // stopping again is fine
    let (s, v) = t
        .json("POST", &format!("/ohmygpu/v1/models/{MODEL}/stop"), None)
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["model"]["state"], "stopped");

    // async start with a slow backend → 202 starting, then running
    t.backend.set_start_delay(Duration::from_millis(200));
    let (s, v) = t
        .json(
            "POST",
            &format!("/ohmygpu/v1/models/{MODEL}/start"),
            Some(json!({"context_length": 1234})),
        )
        .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    assert_eq!(v["model"]["state"], "starting");
    let st = t
        .state
        .manager
        .wait_started(MODEL, Duration::from_secs(5))
        .await;
    assert_eq!(st, Some(ModelState::Running));
    assert_eq!(
        t.backend.last_instance().unwrap().spec_context,
        Some(1234),
        "start options reach the backend"
    );
    let (_, v) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(
        v["runtime"]["context_length"], 1234,
        "the window a running model serves is reported under runtime: {v}"
    );
    t.backend.set_start_delay(Duration::ZERO);

    // crash → error, with message; error state is startable
    t.backend.last_instance().unwrap().crash();
    let st = t
        .state
        .manager
        .wait_for(MODEL, Duration::from_secs(5), |s| {
            matches!(s, ModelState::Error { .. })
        })
        .await
        .unwrap();
    assert!(matches!(st, Some(ModelState::Error { .. })));
    let (_, v) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(v["state"], "error");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("simulated crash"));
    assert!(v.get("runtime").is_none());
    t.start_and_wait().await;

    // delete while running → stopped + removed
    let (s, v) = t
        .json("DELETE", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["deleted"], true);
    assert_eq!(v["model"]["state"], "not_installed");
    let (s, _) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (_, v) = t.json("GET", "/v1/models", None).await;
    assert!(v["data"].as_array().unwrap().is_empty());
    assert!(!t.state.paths.model_dir(MODEL).exists());
}

#[tokio::test]
async fn start_failure_is_reported_and_recoverable() {
    let t = setup().await;
    t.backend.set_fail_start(Some("no gpu memory"));
    let (s, v) = t
        .json(
            "POST",
            &format!("/ohmygpu/v1/models/{MODEL}/start?wait=true"),
            None,
        )
        .await;
    assert_eq!(s, StatusCode::BAD_GATEWAY);
    assert_eq!(v["error"]["code"], "model_start_failed");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("no gpu memory"));
    let (_, v) = t
        .json("GET", &format!("/ohmygpu/v1/models/{MODEL}"), None)
        .await;
    assert_eq!(v["state"], "error");
    t.backend.set_fail_start(None);
    t.start_and_wait().await;
}

#[tokio::test]
async fn invalid_lifecycle_requests() {
    let t = setup().await;
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/qwen2.5-0.5b-instruct/start",
            None,
        )
        .await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "model_not_installed");
    let (s, v) = t
        .json("POST", "/ohmygpu/v1/models/does-not-exist/start", None)
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": "not-a-thing"})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "invalid_model_reference");
    let (s, _) = t
        .json("DELETE", "/ohmygpu/v1/models/does-not-exist", None)
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

/// Pull from a local HTTP server: downloading → installed, then usable.
#[tokio::test]
async fn pull_downloads_registers_and_can_run() {
    let t = setup().await;
    // Serve a fake gguf: a real header (context_length 4096) padded to 300 kB.
    let mut payload = tiny_gguf(4096);
    payload.resize(300_000, 7u8);
    let file_app = Router::new().route(
        "/models/tiny.gguf",
        axum::routing::get({
            let payload = payload.clone();
            move || async move { payload }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, file_app).await.unwrap() });
    let url = format!("http://{addr}/models/tiny.gguf");

    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url, "id": "tiny"})),
        )
        .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["model"]["state"], "downloading");
    assert_eq!(v["model"]["id"], "tiny");
    let st = t
        .state
        .manager
        .wait_for("tiny", Duration::from_secs(10), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .unwrap();
    assert_eq!(st, Some(ModelState::Installed));

    let (_, v) = t.json("GET", "/ohmygpu/v1/models/tiny", None).await;
    assert_eq!(v["state"], "installed");
    assert_eq!(v["size_bytes"], 300_000);
    assert_eq!(v["source"]["type"], "url");
    assert_eq!(
        v["context_length"], 4096,
        "native context length is read from the GGUF at install: {v}"
    );
    assert!(t.state.paths.model_dir("tiny").join("tiny.gguf").exists());
    // pulling again is a no-op 200
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url, "id": "tiny"})),
        )
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["model"]["state"], "installed");
    // and it shows up for OpenAI clients, context length included
    let (_, v) = t.json("GET", "/v1/models", None).await;
    let tiny = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "tiny")
        .expect("tiny listed");
    assert_eq!(tiny["context_length"], 4096);
    // survives a registry reload
    let reg = ModelRegistry::load(t.state.paths.registry_path()).unwrap();
    assert!(reg.contains("tiny"));

    let (s, v) = t
        .json("POST", "/ohmygpu/v1/models/tiny/start?wait=true", None)
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": "tiny", "input": "hello"})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["output"][0]["content"][0]["text"], "echo: hello");
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_non_streaming() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, v) = t
        .json("POST", "/v1/chat/completions", Some(json!({"model": MODEL, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]})))
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], MODEL);
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "echo: hi");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 2);
    assert_eq!(v["usage"]["completion_tokens"], 2);
    assert_eq!(v["usage"]["total_tokens"], 4);
}

#[tokio::test]
async fn chat_completions_tool_call() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "call weather"}],
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
            })),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let msg = &v["choices"][0]["message"];
    assert!(msg["content"].is_null());
    assert_eq!(msg["tool_calls"][0]["type"], "function");
    assert_eq!(msg["tool_calls"][0]["id"], "call_mock1");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(
        msg["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn chat_completions_streaming() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, body) = t
        .req(
            "POST",
            "/v1/chat/completions",
            Some(json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": true, "stream_options": {"include_usage": true}})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let events = sse_events(&body);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let chunks: Vec<Value> = events
        .iter()
        .filter(|(_, d)| d != "[DONE]")
        .map(|(_, d)| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks
        .iter()
        .all(|c| c["object"] == "chat.completion.chunk"));
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "echo: hi");
    assert!(chunks
        .iter()
        .any(|c| c["choices"][0]["finish_reason"] == "stop"));
    let usage_chunk = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk");
    assert_eq!(usage_chunk["usage"]["total_tokens"], 3);
    assert!(usage_chunk["choices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn chat_completions_streaming_tool_call_and_midstream_error() {
    let t = setup().await;
    t.start_and_wait().await;
    let (_, body) = t
        .req(
            "POST",
            "/v1/chat/completions",
            Some(json!({"model": MODEL, "messages": [{"role": "user", "content": "call it"}], "stream": true,
                        "tools": [{"type": "function", "function": {"name": "f"}}]})),
        )
        .await;
    let chunks: Vec<Value> = sse_events(&body)
        .iter()
        .filter(|(_, d)| d != "[DONE]")
        .map(|(_, d)| serde_json::from_str(d).unwrap())
        .collect();
    let tc: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["delta"]["tool_calls"].is_array())
        .collect();
    assert_eq!(
        tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "f"
    );
    assert_eq!(
        tc[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_mock1"
    );
    let args: String = tc
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
        .collect();
    assert_eq!(args, "{\"city\":\"Paris\"}");
    assert!(chunks
        .iter()
        .any(|c| c["choices"][0]["finish_reason"] == "tool_calls"));

    let (_, body) = t
        .req("POST", "/v1/chat/completions", Some(json!({"model": MODEL, "messages": [{"role": "user", "content": "fail"}], "stream": true})))
        .await;
    let events = sse_events(&body);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let err = events
        .iter()
        .find_map(|(_, d)| {
            serde_json::from_str::<Value>(d)
                .ok()
                .filter(|v| v["error"].is_object())
        })
        .expect("error chunk");
    assert_eq!(err["error"]["code"], "backend_error");
}

#[tokio::test]
async fn chat_completions_length_finish_reason() {
    let t = setup().await;
    t.start_and_wait().await;
    let (_, v) = t
        .json("POST", "/v1/chat/completions", Some(json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})))
        .await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_non_streaming() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "hi", "metadata": {"a": "b"}})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["model"], MODEL);
    assert_eq!(v["output"][0]["type"], "message");
    assert_eq!(v["output"][0]["role"], "assistant");
    assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(v["output"][0]["content"][0]["text"], "echo: hi");
    assert_eq!(v["usage"]["input_tokens"], 1);
    assert_eq!(v["usage"]["output_tokens"], 2);
    assert_eq!(v["usage"]["total_tokens"], 3);
    assert_eq!(v["metadata"]["a"], "b");
    assert!(v["error"].is_null());
    assert!(v["incomplete_details"].is_null());
    assert_eq!(v["store"], false);
}

#[tokio::test]
async fn responses_tool_call_and_incomplete() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "call weather", "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["output"][0]["type"], "function_call");
    assert_eq!(v["output"][0]["name"], "get_weather");
    assert_eq!(v["output"][0]["call_id"], "call_mock1");
    assert_eq!(v["output"][0]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(v["output"][0]["status"], "completed");
    assert_eq!(v["tools"][0]["name"], "get_weather");

    let (_, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "hi", "max_output_tokens": 1})),
        )
        .await;
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
}

#[tokio::test]
async fn responses_streaming_events() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, body) = t
        .req(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "hi", "stream": true})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let events = sse_events(&body);
    let kinds: Vec<&str> = events.iter().map(|(e, _)| e.as_deref().unwrap()).collect();
    assert_eq!(
        kinds,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (i, (_, d)) in events.iter().enumerate() {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_eq!(v["sequence_number"], i as u64);
        assert_eq!(v["type"], kinds[i]);
    }
    let completed: Value = serde_json::from_str(&events.last().unwrap().1).unwrap();
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "echo: hi"
    );

    // mid-stream failure → response.failed + error
    let (_, body) = t
        .req(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "fail", "stream": true})),
        )
        .await;
    let kinds: Vec<String> = sse_events(&body)
        .iter()
        .map(|(e, _)| e.clone().unwrap())
        .collect();
    assert!(kinds.contains(&"response.failed".to_string()), "{kinds:?}");
    assert_eq!(kinds.last().unwrap(), "error");
}

// ---------------------------------------------------------------------------
// One pipeline
// ---------------------------------------------------------------------------

#[test]
fn both_protocols_produce_the_same_internal_request() {
    use ohmygpu_daemon::api::{chat_completions, responses};
    let chat: chat_completions::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "thanks"}
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
        "temperature": 0.4, "top_p": 0.8, "max_completion_tokens": 33
    }))
    .unwrap();
    let resp: responses::ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "instructions": "sys",
        "input": [
            {"role": "user", "content": "weather?"},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "thanks"}]}
        ],
        "tools": [{"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}],
        "tool_choice": "required",
        "temperature": 0.4, "top_p": 0.8, "max_output_tokens": 33
    }))
    .unwrap();
    let a = chat_completions::to_inference_request(chat).unwrap();
    let (b, _) = responses::to_inference_request(resp).unwrap();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// auto_start + shutdown
// ---------------------------------------------------------------------------

#[tokio::test]
async fn auto_start_starts_installed_model_on_inference() {
    let mut cfg = Config::default();
    cfg.inference.auto_start = true;
    let t = setup_with(cfg).await;
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": MODEL, "input": "hi"})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["output"][0]["content"][0]["text"], "echo: hi");
    assert_eq!(t.state.manager.state_of(MODEL), Some(ModelState::Running));
}

#[tokio::test]
async fn shutdown_endpoint_signals_the_server() {
    let t = setup().await;
    assert!(!*t.shutdown_rx.borrow());
    let (s, v) = t.json("POST", "/ohmygpu/v1/shutdown", None).await;
    assert_eq!(s, StatusCode::ACCEPTED);
    assert_eq!(v["status"], "shutting_down");
    assert!(*t.shutdown_rx.borrow());
}

// ---------------------------------------------------------------------------
// Vision (image input)
// ---------------------------------------------------------------------------

const VISION_MODEL: &str = "mock-vision";
/// 64×64 solid red PNG (132 bytes).
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC";

fn red_png_data_url() -> String {
    format!("data:image/png;base64,{RED_PNG_B64}")
}

#[tokio::test]
async fn images_need_a_vision_model() {
    let t = setup().await;
    t.start_and_wait().await;
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(
                json!({"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": red_png_data_url()}}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "unsupported");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("vision"),
        "{v}"
    );
    assert_eq!(v["error"]["param"], "messages");

    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(
                json!({"model": MODEL, "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": red_png_data_url()}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "unsupported");
    assert_eq!(v["error"]["param"], "input");
}

#[tokio::test]
async fn vision_model_accepts_inline_and_remote_images() {
    let t = setup_full(Config::default(), |p| {
        install_fake_vision_model(p, VISION_MODEL)
    })
    .await;
    let (s, v) = t
        .json(
            "POST",
            &format!("/ohmygpu/v1/models/{VISION_MODEL}/start?wait=true"),
            None,
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["model"]["capabilities"]["vision"], true);
    assert_eq!(v["model"]["modalities"]["input"], json!(["text", "image"]));
    assert!(t.backend.last_instance().unwrap().spec_mmproj.is_some());

    // Inline data: URL, chat completions shape.
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(
                json!({"model": VISION_MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": red_png_data_url()}}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "saw 1 image(s); echo: what is this?"
    );

    // Remote image: served locally, fetched and inlined by the daemon.
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .decode(RED_PNG_B64)
        .unwrap();
    let file_app = Router::new()
        .route(
            "/red.png",
            axum::routing::get({
                let png = png.clone();
                move || async move { ([("content-type", "image/png")], png) }
            }),
        )
        .route("/note.txt", axum::routing::get(|| async { "hello" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, file_app).await.unwrap() });

    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(json!({"model": VISION_MODEL, "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "image_url": format!("http://{addr}/red.png"), "detail": "auto"}
            ]}]})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["output"][0]["content"][0]["text"],
        "saw 1 image(s); echo: describe"
    );

    // A remote URL that is not an image.
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(
                json!({"model": VISION_MODEL, "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": format!("http://{addr}/note.txt")}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "invalid_request");

    // Wrong media type inline.
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(
                json!({"model": VISION_MODEL, "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:text/plain;base64,aGk="}}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "unsupported");

    // Images only in user messages.
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(json!({"model": VISION_MODEL, "messages": [
                {"role": "system", "content": [{"type": "image_url", "image_url": {"url": red_png_data_url()}}]},
                {"role": "user", "content": "hi"}
            ]})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn pull_with_mmproj_downloads_both_files_and_adds_vision() {
    let t = setup().await;
    let weights = vec![7u8; 300_000];
    let proj = vec![9u8; 50_000];
    let file_app = Router::new()
        .route(
            "/m.gguf",
            axum::routing::get({
                let w = weights.clone();
                move || async move { w }
            }),
        )
        .route(
            "/mmproj.gguf",
            axum::routing::get({
                let p = proj.clone();
                move || async move { p }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, file_app).await.unwrap() });
    let url_m = format!("http://{addr}/m.gguf");
    let url_p = format!("http://{addr}/mmproj.gguf");

    // 1. Plain pull: text-only model.
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url_m, "id": "eyes"})),
        )
        .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    t.state
        .manager
        .wait_for("eyes", Duration::from_secs(10), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .unwrap();
    let (_, v) = t.json("GET", "/ohmygpu/v1/models/eyes", None).await;
    assert_eq!(v["state"], "installed");
    assert_eq!(v["capabilities"]["vision"], false);
    assert_eq!(v["size_bytes"], 300_000);

    // 2. Pull again with a projector: only the projector is downloaded,
    //    the model becomes a vision model.
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url_m, "id": "eyes", "mmproj": url_p})),
        )
        .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["model"]["state"], "downloading");
    assert_eq!(v["model"]["capabilities"]["vision"], true);
    let st = t
        .state
        .manager
        .wait_for("eyes", Duration::from_secs(10), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .unwrap();
    assert_eq!(st, Some(ModelState::Installed));
    let (_, v) = t.json("GET", "/ohmygpu/v1/models/eyes", None).await;
    assert_eq!(v["capabilities"]["vision"], true);
    assert_eq!(v["size_bytes"], 350_000);
    assert!(t.state.paths.model_dir("eyes").join("m.gguf").exists());
    assert!(t.state.paths.model_dir("eyes").join("mmproj.gguf").exists());
    let reg = ModelRegistry::load(t.state.paths.registry_path()).unwrap();
    assert!(reg.get("eyes").unwrap().mmproj_path.is_some());

    // 3. Pulling once more is a no-op.
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url_m, "id": "eyes", "mmproj": url_p})),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["model"]["state"], "installed");

    // 4. Start hands the projector to the backend, and images now work.
    let (s, v) = t
        .json("POST", "/ohmygpu/v1/models/eyes/start?wait=true", None)
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(t.backend.last_instance().unwrap().spec_mmproj.is_some());
    let (s, v) = t
        .json(
            "POST",
            "/v1/responses",
            Some(
                json!({"model": "eyes", "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": "colour?"},
                    {"type": "input_image", "image_url": red_png_data_url()}
                ]}]}),
            ),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["output"][0]["content"][0]["text"],
        "saw 1 image(s); echo: colour?"
    );

    // A bad projector reference is rejected up front.
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": url_m, "id": "eyes2", "mmproj": "not-a-gguf.bin"})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
}

// ---------------------------------------------------------------------------
// Speech to text (whisper)
// ---------------------------------------------------------------------------

const WHISPER_MODEL: &str = "mock-whisper";

/// A fake whisper model (ggml .bin, kind = whisper).
fn install_fake_whisper_model(paths: &Paths, id: &str) {
    let dir = paths.model_dir(id);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("ggml-tiny.bin");
    std::fs::write(&file, b"ggml-not-really").unwrap();
    let mut reg = ModelRegistry::load(paths.registry_path()).unwrap();
    reg.add(InstalledModel {
        id: id.into(),
        display_name: "Mock Whisper".into(),
        source: ModelSource::HuggingFace {
            repo: "ggerganov/whisper.cpp".into(),
            file: "ggml-tiny.bin".into(),
        },
        kind: ohmygpu_inference::ModelKind::Whisper,
        format: "ggml".into(),
        path: file,
        mmproj_path: None,
        size_bytes: 15,
        installed_at: chrono::Utc::now(),
        capabilities: ModelCapabilities::default(),
        context_length: None,
        curated: false,
    })
    .unwrap();
}

/// 16-bit PCM mono WAV: `secs` of a 440 Hz tone at `rate`.
fn tone_wav(rate: u32, secs: f32) -> Vec<u8> {
    let frames = (rate as f32 * secs) as usize;
    let data_len = (frames * 2) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..frames {
        let v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin();
        out.extend_from_slice(&((v * 20000.0) as i16).to_le_bytes());
    }
    out
}

/// Build a multipart/form-data body: `(content_type, body)`.
fn multipart(fields: &[(&str, &str)], file: Option<(&str, &[u8], &str)>) -> (String, Vec<u8>) {
    let b = "----ohmygpu-test-boundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{b}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    if let Some((file_name, bytes, ct)) = file {
        body.extend_from_slice(
            format!("--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: {ct}\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{b}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={b}"), body)
}

impl TestApp {
    async fn post_bytes(
        &self,
        path: &str,
        content_type: &str,
        body: Vec<u8>,
    ) -> (StatusCode, String) {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    async fn transcribe(
        &self,
        fields: &[(&str, &str)],
        file: Option<(&str, &[u8], &str)>,
    ) -> (StatusCode, String) {
        let (ct, body) = multipart(fields, file);
        self.post_bytes("/v1/audio/transcriptions", &ct, body).await
    }
}

fn json_of(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("non-JSON body ({e}): {text}"))
}

#[tokio::test]
async fn transcriptions_with_a_mock_whisper_model() {
    let t = setup_full(Config::default(), |p| {
        install_fake_whisper_model(p, WHISPER_MODEL)
    })
    .await;
    let wav = tone_wav(16_000, 0.5);

    // Not running yet → 409, like the text APIs.
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL)],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::CONFLICT, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "model_not_running");

    // Start it: the view says what kind it is.
    let (s, v) = t
        .json(
            "POST",
            &format!("/ohmygpu/v1/models/{WHISPER_MODEL}/start?wait=true"),
            None,
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["model"]["kind"], "whisper");
    let (_, v) = t.json("GET", "/v1/models", None).await;
    let me = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == WHISPER_MODEL)
        .cloned()
        .unwrap();
    assert_eq!(me["kind"], "whisper");

    // Chat on a speech model is refused.
    let (s, v) = t
        .json(
            "POST",
            "/v1/chat/completions",
            Some(json!({"model": WHISPER_MODEL, "messages": [{"role": "user", "content": "hi"}]})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["code"], "unsupported");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("transcriptions"),
        "{v}"
    );

    // json (default)
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL)],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(
        json_of(&body)["text"],
        "transcribed 0.5s at 16000 Hz (auto)"
    );

    // language + verbose_json (+ a 44.1 kHz file that gets resampled)
    let wav44 = tone_wav(44_100, 1.0);
    let (s, body) = t
        .transcribe(
            &[
                ("model", WHISPER_MODEL),
                ("language", "zh"),
                ("response_format", "verbose_json"),
                ("temperature", "0.2"),
                ("timestamp_granularities[]", "segment"),
            ],
            Some(("a.wav", &wav44, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v = json_of(&body);
    assert_eq!(v["task"], "transcribe");
    assert_eq!(v["language"], "zh");
    assert_eq!(v["text"], "transcribed 1.0s at 16000 Hz (zh)");
    assert_eq!(v["segments"].as_array().unwrap().len(), 1);
    assert_eq!(v["segments"][0]["end"], 1.0);

    // text / srt / vtt
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL), ("response_format", "text")],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body, "transcribed 0.5s at 16000 Hz (auto)\n");
    let (_, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL), ("response_format", "srt")],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert!(
        body.starts_with("1\n00:00:00,000 --> 00:00:00,500\n"),
        "{body}"
    );
    let (_, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL), ("response_format", "vtt")],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert!(
        body.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:00.500\n"),
        "{body}"
    );

    // Compressed input is decoded too.
    let mp3 = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/audio/tone.mp3"
    ))
    .unwrap();
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL)],
            Some(("tone.mp3", &mp3, "audio/mpeg")),
        )
        .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert!(
        json_of(&body)["text"]
            .as_str()
            .unwrap()
            .starts_with("transcribed 1."),
        "{body}"
    );

    // Errors: no file, no model, garbage audio, opus, word timestamps, streaming.
    let (s, body) = t.transcribe(&[("model", WHISPER_MODEL)], None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_of(&body)["error"]["param"], "file");
    let (s, body) = t.transcribe(&[], Some(("a.wav", &wav, "audio/wav"))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_of(&body)["error"]["param"], "model");
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL)],
            Some(("x.txt", b"not audio at all", "text/plain")),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        json_of(&body)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unrecognised audio format"),
        "{body}"
    );
    let webm = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/audio/tone.webm"
    ))
    .unwrap();
    let (s, body) = t
        .transcribe(
            &[("model", WHISPER_MODEL)],
            Some(("tone.webm", &webm, "audio/webm")),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        json_of(&body)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("opus"),
        "{body}"
    );
    let (s, body) = t
        .transcribe(
            &[
                ("model", WHISPER_MODEL),
                ("timestamp_granularities[]", "word"),
            ],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "unsupported");
    let (s, _) = t
        .transcribe(
            &[("model", WHISPER_MODEL), ("stream", "true")],
            Some(("a.wav", &wav, "audio/wav")),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);

    // Unknown model → 404; an LLM → 400 unsupported.
    let (s, body) = t
        .transcribe(&[("model", "nope")], Some(("a.wav", &wav, "audio/wav")))
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{body}");
    t.start_and_wait().await;
    let (s, body) = t
        .transcribe(&[("model", MODEL)], Some(("a.wav", &wav, "audio/wav")))
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "unsupported");

    // Not multipart at all.
    let (s, body) = t
        .post_bytes(
            "/v1/audio/transcriptions",
            "application/json",
            br#"{"model":"x"}"#.to_vec(),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn pull_whisper_reference_by_file_name_and_status_lists_backends() {
    let t = setup().await;
    // A `ggml-*.bin` file is recognised as a whisper model from its name alone.
    let payload = vec![3u8; 20_000];
    let file_app = Router::new().route(
        "/ggml-tiny.bin",
        axum::routing::get({
            let payload = payload.clone();
            move || async move { payload }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, file_app).await.unwrap() });
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": format!("http://{addr}/ggml-tiny.bin"), "id": "w"})),
        )
        .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["model"]["kind"], "whisper");
    t.state
        .manager
        .wait_for("w", Duration::from_secs(10), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .unwrap();
    let (_, v) = t.json("GET", "/ohmygpu/v1/models/w", None).await;
    assert_eq!(v["state"], "installed", "{v}");
    assert_eq!(v["kind"], "whisper");
    assert_eq!(v["format"], "ggml");
    let reg = ModelRegistry::load(t.state.paths.registry_path()).unwrap();
    assert_eq!(
        reg.get("w").unwrap().kind,
        ohmygpu_inference::ModelKind::Whisper
    );

    // An explicit kind wins, and mmproj is refused on whisper models.
    let (s, v) = t
        .json(
            "POST",
            "/ohmygpu/v1/models/pull",
            Some(json!({"model": "https://example.com/whisper-custom.bin", "id": "w2", "kind": "whisper", "mmproj": "x.gguf"})),
        )
        .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    // Catalog exposes the whisper entries with their kind.
    let (_, v) = t.json("GET", "/ohmygpu/v1/catalog", None).await;
    let whisper: Vec<&Value> = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["kind"] == "whisper")
        .collect();
    assert!(whisper.iter().any(|m| m["id"] == "whisper-base"), "{v}");
    // Status lists every backend.
    let (_, v) = t.json("GET", "/ohmygpu/v1/status", None).await;
    assert!(!v["backends"].as_array().unwrap().is_empty(), "{v}");
    let (_, v) = t.json("GET", "/ohmygpu/v1/backend", None).await;
    assert!(v["backends"].is_array(), "{v}");
}
