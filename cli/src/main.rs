mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ohmygpu")]
#[command(author, version, about = "Download and run AI models locally", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from HuggingFace
    Pull {
        /// Model identifier (e.g., "microsoft/phi-2")
        model: String,

        /// Specific file to download
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Run/load a model for inference
    Run {
        /// Model name to run
        model: String,
    },

    /// Stop a running model
    Stop {
        /// Model name to stop (optional, stops all if not specified)
        model: Option<String>,
    },

    /// List downloaded models
    #[command(alias = "ls")]
    Models,

    /// Remove a downloaded model
    #[command(alias = "rm")]
    Remove {
        /// Model name to remove
        model: String,
    },

    /// Show daemon and GPU status
    Status,

    /// Start the daemon server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "11434")]
        port: u16,
    },

    /// Self-update to the latest version
    Update,

    /// Search for models on HuggingFace
    Search {
        /// Search query
        query: String,
    },

    /// View or set configuration
    Config {
        /// Config key (e.g., "daemon.port", "inference.temperature")
        key: Option<String>,

        /// Value to set (if omitted, shows current value)
        value: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model, file } => {
            commands::pull::execute(&model, file.as_deref()).await?;
        }
        Commands::Run { model } => {
            commands::run::execute(&model).await?;
        }
        Commands::Stop { model } => {
            commands::stop::execute(model.as_deref()).await?;
        }
        Commands::Models => {
            commands::models::execute().await?;
        }
        Commands::Remove { model } => {
            commands::remove::execute(&model).await?;
        }
        Commands::Status => {
            commands::status::execute().await?;
        }
        Commands::Serve { port } => {
            commands::serve::execute(port).await?;
        }
        Commands::Update => {
            commands::update::execute().await?;
        }
        Commands::Search { query } => {
            commands::search::execute(&query).await?;
        }
        Commands::Config { key, value } => {
            commands::config::execute(key.as_deref(), value.as_deref()).await?;
        }
    }

    Ok(())
}
