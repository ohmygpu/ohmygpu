mod cli;
mod downloaders;
mod models;
mod runners;
mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "omg")]
#[command(author, version, about = "Download and run AI models locally", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from HuggingFace or GitHub
    Pull {
        /// Model identifier (e.g., "meta-llama/Llama-3.2-3B" or "github:user/repo")
        model: String,

        /// Specific file or variant to download
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Run a downloaded model
    Run {
        /// Model name to run
        model: String,

        /// Optional prompt to run (non-interactive mode)
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// List downloaded models
    List,

    /// Remove a downloaded model
    #[command(alias = "rm")]
    Remove {
        /// Model name to remove
        model: String,
    },

    /// Search for models on HuggingFace
    Search {
        /// Search query
        query: String,
    },

    /// Open interactive TUI
    Ui,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Pull { model, file }) => {
            cli::pull::execute(&model, file.as_deref()).await?;
        }
        Some(Commands::Run { model, prompt }) => {
            cli::run::execute(&model, prompt.as_deref()).await?;
        }
        Some(Commands::List) => {
            cli::list::execute().await?;
        }
        Some(Commands::Remove { model }) => {
            cli::remove::execute(&model).await?;
        }
        Some(Commands::Search { query }) => {
            cli::search::execute(&query).await?;
        }
        Some(Commands::Ui) | None => {
            tui::run().await?;
        }
    }

    Ok(())
}
