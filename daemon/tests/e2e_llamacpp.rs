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
use ohmygpu_daemon::server::build_state;
use ohmygpu_runtime_llamacpp::LlamaCppBackend;
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

#[tokio::test]
#[ignore = "needs network + real inference; run with OHMYGPU_E2E=1 -- --ignored"]
async fn pull_start_infer_stop_with_real_llamacpp() {
    if std::env::var("OHMYGPU_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping: set OHMYGPU_E2E=1");
        return;
    }
    let tmp;
    let paths = match std::env::var("OHMYGPU_E2E_HOME") {
        Ok(dir) => Paths::new(dir),
        Err(_) => {
            tmp = tempfile::tempdir().unwrap();
            Paths::new(tmp.path())
        }
    };
    paths.ensure_dirs().unwrap();
    let mut config = Config::default();
    config.apply_env();
    let hardware = HardwareInfo::detect();
    let backend = Arc::new(LlamaCppBackend::new(
        config.backend.llamacpp.clone(),
        &paths.runtimes_dir(),
        hardware.clone(),
    ));
    let (state, _rx) = build_state(
        paths.clone(),
        config,
        hardware,
        backend,
        "127.0.0.1".into(),
        0,
    )
    .unwrap();
    let app = router(state.clone());

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
