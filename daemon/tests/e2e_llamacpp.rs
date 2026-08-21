//! End-to-end test with the real llama.cpp backend and a real (small) model.
//!
//! Ignored by default because it needs network (~11 MB llama.cpp release +
//! ~470 MB model) and a machine that can run inference. Run with:
//!
//! ```bash
//! OHMYGPU_E2E=1 cargo test -p ohmygpu_daemon --test e2e_llamacpp -- --ignored --nocapture
//! ```
//!
//! Set `OHMYGPU_E2E_HOME=/some/dir` to reuse downloads across runs.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use ohmygpu_core::config::Config;
use ohmygpu_core::hardware::HardwareInfo;
use ohmygpu_core::lifecycle::ModelState;
use ohmygpu_core::paths::Paths;
use ohmygpu_daemon::api::router;
use ohmygpu_daemon::manager::Backends;
use ohmygpu_daemon::server::build_state;
use ohmygpu_runtime_llamacpp::LlamaCppBackend;
use ohmygpu_runtime_whisper::WhisperBackend;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "qwen2.5-0.5b-instruct";

async fn call(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, String) {
    let mut b = Request::builder().method(method).uri(path);
    let body = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let resp = app.clone().oneshot(b.body(body).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// Real backend + real data dir (`OHMYGPU_E2E_HOME`, or a temp dir kept alive
/// by the returned guard).
fn real_app() -> (
    ohmygpu_daemon::state::SharedState,
    axum::Router,
    Option<tempfile::TempDir>,
) {
    let (paths, tmp) = match std::env::var("OHMYGPU_E2E_HOME") {
        Ok(dir) => (Paths::new(dir), None),
        Err(_) => {
            let t = tempfile::tempdir().unwrap();
            (Paths::new(t.path()), Some(t))
        }
    };
    paths.ensure_dirs().unwrap();
    let mut config = Config::default();
    config.apply_env();
    let hardware = HardwareInfo::detect();
    let llm = Arc::new(LlamaCppBackend::new(
        config.backend.llamacpp.clone(),
        &paths.runtimes_dir(),
        hardware.clone(),
    ));
    let whisper = Arc::new(WhisperBackend::new(
        config.backend.whisper.clone(),
        &paths.runtimes_dir(),
        hardware.clone(),
    ));
    let (state, _rx) = build_state(
        paths.clone(),
        config,
        hardware,
        Backends::new(llm, Some(whisper)),
        "127.0.0.1".into(),
        0,
    )
    .unwrap();
    let app = router(state.clone());
    (state, app, tmp)
}

#[tokio::test]
#[ignore = "needs network + real inference; run with OHMYGPU_E2E=1 -- --ignored"]
async fn pull_start_infer_stop_with_real_llamacpp() {
    if std::env::var("OHMYGPU_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping: set OHMYGPU_E2E=1");
        return;
    }
    let (state, app, _tmp) = real_app();

    // pull
    let (s, body) = call(
        &app,
        "POST",
        "/ohmygpu/v1/models/pull",
        Some(json!({"model": MODEL})),
    )
    .await;
    assert!(
        s == StatusCode::ACCEPTED || s == StatusCode::OK,
        "{s} {body}"
    );
    let st = state
        .manager
        .wait_for(MODEL, Duration::from_secs(1800), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .expect("download did not finish in time");
    assert_eq!(
        st,
        Some(ModelState::Installed),
        "{:?}",
        state.manager.get(MODEL)
    );

    // start (installs llama.cpp on first use)
    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{MODEL}/start?wait=true&timeout=900"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"]["state"], "running");
    assert_eq!(v["model"]["runtime"]["backend"], "llamacpp");

    // responses
    let (s, body) = call(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": MODEL, "input": "Reply with exactly: pong", "max_output_tokens": 8})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "response");
    let text = v["output"][0]["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    eprintln!("responses → {text:?}");

    // chat completions, streaming
    let (s, body) = call(&app, "POST", "/v1/chat/completions", Some(json!({"model": MODEL, "messages": [{"role": "user", "content": "Reply with exactly: pong"}], "max_tokens": 8, "stream": true}))).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");

    // tool call
    let (s, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
                       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "max_tokens": 80
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let tc = &v["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tc["function"]["name"], "get_weather", "{v}");
    assert!(
        tc["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("Paris"),
        "{v}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    // stop
    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{MODEL}/stop"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"]["state"], "stopped");
    state.manager.stop_all().await;
}

const VISION_MODEL: &str = "smolvlm-256m-instruct";
/// 64×64 solid red PNG.
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC";

/// Vision: pull a tiny VLM (weights + projector), start it with `--mmproj`, and
/// ask about a solid red image through both APIs.
#[tokio::test]
#[ignore = "needs network + real inference; run with OHMYGPU_E2E=1 -- --ignored"]
async fn vision_model_sees_an_image_with_real_llamacpp() {
    if std::env::var("OHMYGPU_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping: set OHMYGPU_E2E=1");
        return;
    }
    let (state, app, _tmp) = real_app();
    let data_url = format!("data:image/png;base64,{RED_PNG_B64}");

    let (s, body) = call(
        &app,
        "POST",
        "/ohmygpu/v1/models/pull",
        Some(json!({"model": VISION_MODEL})),
    )
    .await;
    assert!(
        s == StatusCode::ACCEPTED || s == StatusCode::OK,
        "{s} {body}"
    );
    let st = state
        .manager
        .wait_for(VISION_MODEL, Duration::from_secs(1800), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .expect("download did not finish in time");
    assert_eq!(
        st,
        Some(ModelState::Installed),
        "{:?}",
        state.manager.get(VISION_MODEL)
    );
    let (_, body) = call(
        &app,
        "GET",
        &format!("/ohmygpu/v1/models/{VISION_MODEL}"),
        None,
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["capabilities"]["vision"], true, "{v}");

    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{VISION_MODEL}/start?wait=true&timeout=900"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");

    // chat completions with an inline image
    let (s, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the dominant color of this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}],
            "max_tokens": 12
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let answer = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    eprintln!("chat/completions (image) → {answer:?}");
    assert!(answer.to_ascii_lowercase().contains("red"), "{v}");

    // responses with input_image
    let (s, body) = call(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({
            "model": VISION_MODEL,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "What color is this image? One word."},
                {"type": "input_image", "image_url": data_url}
            ]}],
            "max_output_tokens": 12
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let answer = v["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    eprintln!("responses (image) → {answer:?}");
    assert!(answer.to_ascii_lowercase().contains("red"), "{v}");

    // text-only still works on the same model
    let (s, body) = call(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": VISION_MODEL, "input": "Reply with exactly: pong", "max_output_tokens": 8})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");

    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{VISION_MODEL}/stop"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    state.manager.stop_all().await;
}

const WHISPER_MODEL: &str = "whisper-tiny";
const JFK_WAV_URL: &str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav";

/// Speech to text: pull a tiny whisper model, start `whisper-server` (found via
/// `OHMYGPU_WHISPER_SERVER` / managed install), transcribe whisper.cpp's JFK sample.
#[tokio::test]
#[ignore = "needs network + real inference; run with OHMYGPU_E2E=1 -- --ignored"]
async fn whisper_transcribes_jfk_with_real_whisper_cpp() {
    if std::env::var("OHMYGPU_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping: set OHMYGPU_E2E=1");
        return;
    }
    let (state, app, _tmp) = real_app();

    let (s, body) = call(
        &app,
        "POST",
        "/ohmygpu/v1/models/pull",
        Some(json!({"model": WHISPER_MODEL})),
    )
    .await;
    assert!(
        s == StatusCode::ACCEPTED || s == StatusCode::OK,
        "{s} {body}"
    );
    let st = state
        .manager
        .wait_for(WHISPER_MODEL, Duration::from_secs(1800), |s| {
            !matches!(s, ModelState::Downloading { .. })
        })
        .await
        .expect("download did not finish in time");
    assert_eq!(
        st,
        Some(ModelState::Installed),
        "{:?}",
        state.manager.get(WHISPER_MODEL)
    );

    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{WHISPER_MODEL}/start?wait=true&timeout=900"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"]["runtime"]["backend"], "whisper", "{v}");

    // The sample clip (11 s of JFK).
    let wav = reqwest::get(JFK_WAV_URL)
        .await
        .expect("download jfk.wav")
        .bytes()
        .await
        .expect("read jfk.wav")
        .to_vec();
    assert!(
        wav.len() > 100_000,
        "jfk.wav looks truncated: {} bytes",
        wav.len()
    );

    let boundary = "----ohmygpu-e2e";
    let mut body_bytes = Vec::new();
    for (name, value) in [
        ("model", WHISPER_MODEL),
        ("response_format", "verbose_json"),
        ("language", "en"),
    ] {
        body_bytes.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
        );
    }
    body_bytes.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"jfk.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes(),
    );
    body_bytes.extend_from_slice(&wav);
    body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let s = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).to_string();
    assert_eq!(s, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let text = v["text"].as_str().unwrap().to_string();
    eprintln!(
        "transcription → {text:?} ({}s, {} segments)",
        v["duration"],
        v["segments"].as_array().map(|a| a.len()).unwrap_or(0)
    );
    assert!(
        text.to_ascii_lowercase().contains("fellow americans"),
        "{v}"
    );
    assert!(v["duration"].as_f64().unwrap() > 10.0, "{v}");
    assert!(!v["segments"].as_array().unwrap().is_empty(), "{v}");

    // Chat on the speech model is refused with a pointer to the right endpoint.
    let (s, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({"model": WHISPER_MODEL, "messages": [{"role": "user", "content": "hi"}]})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");

    let (s, body) = call(
        &app,
        "POST",
        &format!("/ohmygpu/v1/models/{WHISPER_MODEL}/stop"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    state.manager.stop_all().await;
}
