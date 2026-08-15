//! `ohmygpu-runtime` — the standalone daemon binary.
//!
//! Applications that bundle OhMyGPU launch this executable (optionally with
//! `--data-dir` and `--port`) and talk to it over HTTP. No CLI or GUI needed.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use ohmygpu_core::config::Config;
use ohmygpu_core::paths::Paths;
use ohmygpu_daemon::{serve, ServeOptions};

#[derive(Parser, Debug)]
#[command(
    name = "ohmygpu-runtime",
    version,
    about = "OhMyGPU Runtime — headless local AI runtime"
)]
struct Args {
    /// Address to bind (default 127.0.0.1; overrides config/OHMYGPU_HOST)
    #[arg(long, env = "OHMYGPU_HOST")]
    host: Option<String>,
    /// Port to listen on (default 10692; overrides config/OHMYGPU_PORT)
    #[arg(long, short, env = "OHMYGPU_PORT")]
    port: Option<u16>,
    /// Data directory for config, models and runtimes (default ~/.config/ohmygpu)
    #[arg(long, env = "OHMYGPU_HOME")]
    data_dir: Option<PathBuf>,
    /// Log filter (same syntax as RUST_LOG), e.g. `debug` or `info,llamacpp=debug`
    #[arg(long, env = "OHMYGPU_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&args.log).unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    let paths = args
        .data_dir
        .map(Paths::new)
        .unwrap_or_else(Paths::from_env);
    let mut config = Config::load(&paths)?;
    if let Some(h) = args.host {
        config.daemon.host = h;
    }
    if let Some(p) = args.port {
        config.daemon.port = p;
    }
    serve(ServeOptions { paths, config }).await
}
