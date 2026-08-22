//! CLI command implementations — thin wrappers over the Management API.

use std::io::Write;
use std::time::Duration;

use anyhow::{bail, Result};
use ohmygpu_core::config::Config;
use ohmygpu_core::paths::Paths;
use serde_json::{json, Value};

use crate::client::Client;

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

pub async fn serve(
    paths: Paths,
    mut config: Config,
    host: Option<String>,
    port: Option<u16>,
    log: String,
) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&log).unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();
    if let Some(h) = host {
        config.daemon.host = h;
    }
    if let Some(p) = port {
        config.daemon.port = p;
    }
    ohmygpu_daemon::serve(ohmygpu_daemon::ServeOptions { paths, config }).await
}

// ---------------------------------------------------------------------------
// status / hardware
// ---------------------------------------------------------------------------

pub async fn status(client: &Client, json: bool) -> Result<()> {
    if !client.is_up().await {
        if json {
            println!("{}", json!({ "running": false, "url": client.base_url() }));
        } else {
            println!("Runtime: not running ({})", client.base_url());
            println!("Start it with: omg serve");
        }
        return Ok(());
    }
    let v = client.get("/ohmygpu/v1/status").await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!(
        "Runtime:   running (v{}, pid {}, up {}s)",
        v["version"].as_str().unwrap_or("?"),
        v["pid"],
        v["uptime_seconds"]
    );
    println!(
        "URL:       http://{}:{}",
        v["host"].as_str().unwrap_or("?"),
        v["port"]
    );
    println!("Data dir:  {}", v["data_dir"].as_str().unwrap_or("?"));
    let backends: Vec<Value> = match v["backends"].as_array() {
        Some(list) if !list.is_empty() => list.clone(),
        _ => vec![v["backend"].clone()],
    };
    for (i, b) in backends.iter().enumerate() {
        let avail = if b["available"].as_bool().unwrap_or(false) {
            format!("available ({})", b["version"].as_str().unwrap_or("?"))
        } else {
            format!("not available — {}", b["message"].as_str().unwrap_or(""))
        };
        println!(
            "{:<11}{} — {}",
            if i == 0 { "Backends:" } else { "" },
            b["id"].as_str().unwrap_or("?"),
            avail
        );
    }
    println!(
        "Hardware:  {}",
        v["hardware_backend"].as_str().unwrap_or("?")
    );
    let m = &v["models"];
    println!("Models:    {} installed", m["installed"]);
    let running: Vec<&str> = m["running"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    println!(
        "Running:   {}",
        if running.is_empty() {
            "none".to_string()
        } else {
            running.join(", ")
        }
    );
    let dl: Vec<&str> = m["downloading"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    if !dl.is_empty() {
        println!("Downloading: {}", dl.join(", "));
    }
    Ok(())
}

pub async fn hardware(client: &Client, json: bool) -> Result<()> {
    // Hardware detection has no daemon-side state, so fall back to local detection.
    let v = if client.is_up().await {
        client.get("/ohmygpu/v1/hardware").await?
    } else {
        serde_json::to_value(ohmygpu_core::hardware::HardwareInfo::detect())?
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!(
        "Platform:      {} / {}",
        v["platform"].as_str().unwrap_or("?"),
        v["architecture"].as_str().unwrap_or("?")
    );
    println!(
        "CPU:           {} ({} cores)",
        v["cpu"]["name"].as_str().unwrap_or("?"),
        v["cpu"]["cores"]
    );
    println!(
        "System memory: {}",
        v["system_memory_bytes"]
            .as_u64()
            .map(human_bytes)
            .unwrap_or_else(|| "?".into())
    );
    match v.get("gpu").filter(|g| !g.is_null()) {
        Some(g) => println!(
            "GPU:           {} ({}){}",
            g["name"].as_str().unwrap_or("?"),
            g["vendor"].as_str().unwrap_or("?"),
            g["memory_bytes"]
                .as_u64()
                .map(|b| format!(", {}", human_bytes(b)))
                .unwrap_or_default()
        ),
        None => println!("GPU:           none detected"),
    }
    println!("Backend:       {}", v["backend"].as_str().unwrap_or("?"));
    Ok(())
}

// ---------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------

pub async fn model_list(client: &Client, paths: &Paths, json: bool) -> Result<()> {
    let models: Vec<Value> = if client.is_up().await {
        client.get("/ohmygpu/v1/models?installed=true").await?["models"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        // Read-only fallback when the runtime is down: what is on disk.
        let reg = ohmygpu_core::registry::ModelRegistry::load(paths.registry_path())?;
        reg.list()
            .into_iter()
            .map(|m| {
                json!({ "id": m.id, "display_name": m.display_name, "kind": m.kind, "state": "(offline)",
                        "size_bytes": m.size_bytes, "capabilities": m.capabilities,
                        "context_length": m.context_length, "installed": true })
            })
            .collect()
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }
    if models.is_empty() {
        println!("No models installed. Try: omg model pull qwen2.5-0.5b-instruct");
        println!("See supported models with: omg model catalog");
        return Ok(());
    }
    println!(
        "{:<28} {:<8} {:<12} {:>9}  {:<6} {:<7} {:>7}  NAME",
        "ID", "KIND", "STATE", "SIZE", "TOOLS", "VISION", "CONTEXT"
    );
    for m in &models {
        println!(
            "{:<28} {:<8} {:<12} {:>9}  {:<6} {:<7} {:>7}  {}",
            m["id"].as_str().unwrap_or("?"),
            m["kind"].as_str().unwrap_or("llm"),
            m["state"].as_str().unwrap_or("?"),
            m["size_bytes"]
                .as_u64()
                .map(human_bytes)
                .unwrap_or_else(|| "?".into()),
            yes_no(m["capabilities"]["tools"].as_bool().unwrap_or(false)),
            yes_no(m["capabilities"]["vision"].as_bool().unwrap_or(false)),
            m["context_length"]
                .as_u64()
                .map(human_context)
                .unwrap_or_else(|| "-".into()),
            m["display_name"].as_str().unwrap_or(""),
        );
    }
    Ok(())
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// Native context window for a table cell: `32768` → `32k`, `131072` → `128k`;
/// odd sizes stay exact.
fn human_context(tokens: u64) -> String {
    if tokens > 0 && tokens.is_multiple_of(1024) {
        format!("{}k", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

pub async fn model_catalog(client: &Client, json: bool) -> Result<()> {
    let models: Vec<Value> = if client.is_up().await {
        client.get("/ohmygpu/v1/catalog").await?["models"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        ohmygpu_core::catalog::CATALOG
            .iter()
            .map(|e| {
                let mut v = serde_json::to_value(e).unwrap();
                v["installed"] = json!(false);
                v["state"] = json!("(offline)");
                v
            })
            .collect()
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }
    println!(
        "{:<28} {:<8} {:>9}  {:<6} {:<7} {:<14} NAME",
        "ID", "KIND", "SIZE", "TOOLS", "VISION", "STATE"
    );
    for m in &models {
        let size = m["size_bytes_approx"]
            .as_u64()
            .map(|s| s + m["mmproj_size_bytes_approx"].as_u64().unwrap_or(0));
        println!(
            "{:<28} {:<8} {:>9}  {:<6} {:<7} {:<14} {}",
            m["id"].as_str().unwrap_or("?"),
            m["kind"].as_str().unwrap_or("llm"),
            size.map(human_bytes).unwrap_or_else(|| "?".into()),
            yes_no(m["tools"].as_bool().unwrap_or(false)),
            yes_no(m["mmproj_file"].is_string()),
            m["state"].as_str().unwrap_or("?"),
            m["display_name"].as_str().unwrap_or(""),
        );
    }
    println!();
    println!("Pull with: omg model pull <ID>   (advanced: hf:owner/repo/file.gguf, hf:owner/repo/ggml-x.bin, or a direct URL)");
    Ok(())
}

pub async fn model_pull(
    client: &Client,
    model: &str,
    id: Option<&str>,
    mmproj: Option<&str>,
    kind: Option<&str>,
    json: bool,
) -> Result<()> {
    let mut body = json!({ "model": model });
    if let Some(id) = id {
        body["id"] = json!(id);
    }
    if let Some(mmproj) = mmproj {
        body["mmproj"] = json!(mmproj);
    }
    if let Some(kind) = kind {
        body["kind"] = json!(kind);
    }
    let v = client.post("/ohmygpu/v1/models/pull", Some(body)).await?;
    let m = &v["model"];
    let id = m["id"].as_str().unwrap_or(model).to_string();
    if m["state"] != "downloading" {
        if json {
            println!("{}", serde_json::to_string_pretty(m)?);
        } else {
            println!("{id}: already {}", m["state"].as_str().unwrap_or("?"));
        }
        return Ok(());
    }
    if !json {
        println!("Pulling {id} …");
    }
    let mut stdout = std::io::stdout();
    loop {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let m = client.get(&format!("/ohmygpu/v1/models/{id}")).await?;
        match m["state"].as_str().unwrap_or("") {
            "downloading" => {
                if !json {
                    let d = &m["download"];
                    let done = d["downloaded_bytes"].as_u64().unwrap_or(0);
                    let line = match d["total_bytes"].as_u64() {
                        Some(t) if t > 0 => format!(
                            "  {:>5.1}%  {} / {}",
                            done as f64 / t as f64 * 100.0,
                            human_bytes(done),
                            human_bytes(t)
                        ),
                        _ => format!("  {}", human_bytes(done)),
                    };
                    print!("\r{line:<60}");
                    stdout.flush().ok();
                }
            }
            "installed" => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&m)?);
                } else {
                    println!("\r{:<60}", "");
                    println!(
                        "Installed {id} ({})",
                        m["size_bytes"]
                            .as_u64()
                            .map(human_bytes)
                            .unwrap_or_default()
                    );
                    println!("Start it with: omg run {id}");
                }
                return Ok(());
            }
            "error" => {
                println!();
                bail!(
                    "pull failed: {}",
                    m["error"]["message"].as_str().unwrap_or("unknown error")
                );
            }
            other => {
                println!();
                bail!("unexpected state '{other}' while pulling {id}");
            }
        }
    }
}

pub async fn model_remove(client: &Client, model: &str) -> Result<()> {
    let v = client
        .delete(&format!("/ohmygpu/v1/models/{model}"))
        .await?;
    println!("Removed {}", v["id"].as_str().unwrap_or(model));
    Ok(())
}

pub async fn model_info(client: &Client, model: &str) -> Result<()> {
    let v = client.get(&format!("/ohmygpu/v1/models/{model}")).await?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// run / stop / shutdown
// ---------------------------------------------------------------------------

pub async fn run(
    client: &Client,
    model: &str,
    context_length: Option<u32>,
    gpu_layers: Option<i32>,
    threads: Option<u32>,
    json: bool,
) -> Result<()> {
    let mut opts = serde_json::Map::new();
    if let Some(c) = context_length {
        opts.insert("context_length".into(), json!(c));
    }
    if let Some(g) = gpu_layers {
        opts.insert("gpu_layers".into(), json!(g));
    }
    if let Some(t) = threads {
        opts.insert("threads".into(), json!(t));
    }
    let v = client
        .post(
            &format!("/ohmygpu/v1/models/{model}/start"),
            Some(Value::Object(opts)),
        )
        .await?;
    if v["model"]["state"] == "running" {
        if json {
            println!("{}", serde_json::to_string_pretty(&v["model"])?);
        } else {
            println!("{model} is running");
        }
        return Ok(());
    }
    if !json {
        println!("Starting {model} …");
    }
    let mut last_msg = String::new();
    loop {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let m = client.get(&format!("/ohmygpu/v1/models/{model}")).await?;
        match m["state"].as_str().unwrap_or("") {
            "starting" => {
                if !json {
                    let msg = m["message"].as_str().unwrap_or("").to_string();
                    if msg != last_msg && !msg.is_empty() {
                        println!("  {msg}");
                        last_msg = msg;
                    }
                }
            }
            "running" => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&m)?);
                } else {
                    println!(
                        "{model} is running (backend {} pid {})",
                        m["runtime"]["backend"].as_str().unwrap_or("?"),
                        m["runtime"]["pid"]
                    );
                    println!();
                    println!("Try it:");
                    println!(
                        "  curl {}/v1/responses -H 'Content-Type: application/json' \\",
                        client.base_url()
                    );
                    println!("    -d '{{\"model\": \"{model}\", \"input\": \"Explain why the sky is blue.\"}}'");
                }
                return Ok(());
            }
            "error" => bail!(
                "start failed: {}",
                m["error"]["message"].as_str().unwrap_or("unknown error")
            ),
            other => bail!("model ended up in state '{other}'"),
        }
    }
}

pub async fn stop(client: &Client, model: &str) -> Result<()> {
    let v = client
        .post(&format!("/ohmygpu/v1/models/{model}/stop"), None)
        .await?;
    println!("{model}: {}", v["model"]["state"].as_str().unwrap_or("?"));
    Ok(())
}

pub async fn shutdown(client: &Client) -> Result<()> {
    if !client.is_up().await {
        println!("Runtime is not running.");
        return Ok(());
    }
    client.post("/ohmygpu/v1/shutdown", None).await?;
    println!("Shutdown requested.");
    Ok(())
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

pub fn config(paths: &Paths, key: Option<&str>, value: Option<&str>) -> Result<()> {
    let mut cfg = Config::load_file(&paths.config_path())?;
    match (key, value) {
        (None, _) => {
            println!("# {}", paths.config_path().display());
            for k in Config::KEYS {
                println!("{k} = {}", cfg.get(k).unwrap_or_default());
            }
        }
        (Some(k), None) => match cfg.get(k) {
            Some(v) => println!("{v}"),
            None => bail!(
                "unknown config key '{k}'. Known keys: {}",
                Config::KEYS.join(", ")
            ),
        },
        (Some(k), Some(v)) => {
            cfg.set(k, v)?;
            cfg.save(paths)?;
            println!("{k} = {v}  (restart the runtime to apply)");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn human_bytes(b: u64) -> String {
    const KB: f64 = 1024.0;
    let b = b as f64;
    if b >= KB * KB * KB {
        format!("{:.2} GB", b / (KB * KB * KB))
    } else if b >= KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{b} B")
    }
}
