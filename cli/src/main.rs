//! `ohmygpu` / `omg` — the official *administrative client* for the OhMyGPU
//! Runtime. It is deliberately thin: every model/runtime operation goes through
//! the same Management API third-party applications use. `omg serve` runs the
//! runtime in-process (same code as the `ohmygpu-runtime` binary).

mod client;
mod commands;
mod upgrade;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ohmygpu_core::config::Config;
use ohmygpu_core::paths::Paths;

use client::Client;

#[derive(Parser, Debug)]
#[command(name = "omg", bin_name = "omg", version, about = "OhMyGPU Runtime — administrative CLI", long_about = None)]
struct Cli {
    /// Runtime base URL (default: http://<config host>:<config port>)
    #[arg(long, global = true, env = "OHMYGPU_URL")]
    url: Option<String>,
    /// Data directory (default ~/.config/ohmygpu)
    #[arg(long, global = true, env = "OHMYGPU_HOME")]
    data_dir: Option<PathBuf>,
    /// Machine-readable JSON output where applicable
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the runtime in the foreground (Ctrl-C to stop)
    Serve {
        /// Address to bind (default 127.0.0.1)
        #[arg(long)]
        host: Option<String>,
        /// Port to listen on (default 10692)
        #[arg(long, short)]
        port: Option<u16>,
        /// Log filter, e.g. `debug` or `info,llamacpp=debug`
        #[arg(long, env = "OHMYGPU_LOG", default_value = "info")]
        log: String,
    },
    /// Show runtime status (backend, models)
    Status,
    /// Show detected hardware
    Hardware,
    /// Manage model files (static assets)
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },
    /// Start a model (blocks until it is running)
    Run {
        model: String,
        /// Context window (tokens)
        #[arg(long)]
        context_length: Option<u32>,
        /// GPU layers to offload (default: all that fit)
        #[arg(long)]
        gpu_layers: Option<i32>,
        /// CPU threads
        #[arg(long)]
        threads: Option<u32>,
    },
    /// Stop a running model
    Stop { model: String },
    /// Ask the runtime to shut down cleanly
    Shutdown,
    /// View or set configuration (`omg config`, `omg config daemon.port`, `omg config daemon.port 1234`)
    Config {
        key: Option<String>,
        value: Option<String>,
    },
    /// Upgrade ohmygpu and ohmygpu-runtime in place from the latest GitHub release
    #[command(alias = "self-update")]
    Upgrade {
        /// Install this release instead of the latest (e.g. v0.4.0)
        version: Option<String>,
        /// Only report whether a newer release exists
        #[arg(long)]
        check: bool,
        /// Install even if that version is already installed
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    /// List installed models
    #[command(alias = "ls")]
    List,
    /// Show the curated catalog of supported models
    Catalog,
    /// Download a model (catalog id, hf:owner/repo/file.gguf, or a direct URL)
    Pull {
        model: String,
        /// Install a non-catalog model under this id
        #[arg(long)]
        id: Option<String>,
        /// Vision projector for a non-catalog model: a .gguf file name in the same
        /// repo (mmproj-….gguf) or a full URL; lets the model accept images
        #[arg(long)]
        mmproj: Option<String>,
        /// Model kind for non-catalog references: llm or whisper (inferred from
        /// the file name when omitted: *.gguf → llm, ggml-*.bin → whisper)
        #[arg(long, value_parser = ["llm", "whisper"])]
        kind: Option<String>,
    },
    /// Remove an installed model (stops it first)
    #[command(alias = "rm")]
    Remove { model: String },
    /// Show details for a model
    Info { model: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Behave like a normal Unix tool when piped into `head` etc.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    let paths = cli
        .data_dir
        .clone()
        .map(Paths::new)
        .unwrap_or_else(Paths::from_env);
    let config = Config::load(&paths)?;
    let base_url = cli
        .url
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}", config.daemon.host, config.daemon.port));
    let client = Client::new(&base_url);
    let json = cli.json;

    match cli.command {
        Commands::Serve { host, port, log } => {
            commands::serve(paths, config, host, port, log).await
        }
        Commands::Status => commands::status(&client, json).await,
        Commands::Hardware => commands::hardware(&client, json).await,
        Commands::Model { action } => match action {
            ModelCommands::List => commands::model_list(&client, &paths, json).await,
            ModelCommands::Catalog => commands::model_catalog(&client, json).await,
            ModelCommands::Pull {
                model,
                id,
                mmproj,
                kind,
            } => {
                commands::model_pull(
                    &client,
                    &model,
                    id.as_deref(),
                    mmproj.as_deref(),
                    kind.as_deref(),
                    json,
                )
                .await
            }
            ModelCommands::Remove { model } => commands::model_remove(&client, &model).await,
            ModelCommands::Info { model } => commands::model_info(&client, &model).await,
        },
        Commands::Run {
            model,
            context_length,
            gpu_layers,
            threads,
        } => commands::run(&client, &model, context_length, gpu_layers, threads, json).await,
        Commands::Stop { model } => commands::stop(&client, &model).await,
        Commands::Shutdown => commands::shutdown(&client).await,
        Commands::Config { key, value } => {
            commands::config(&paths, key.as_deref(), value.as_deref())
        }
        Commands::Upgrade {
            version,
            check,
            force,
        } => {
            upgrade::upgrade(
                &client,
                upgrade::UpgradeOptions {
                    version: version.as_deref(),
                    check,
                    force,
                    json,
                },
            )
            .await
        }
    }
}
