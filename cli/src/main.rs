mod commands;
mod gpu;

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

    /// Generate an image from a text prompt
    Generate {
        /// Model to use (e.g., "Tongyi-MAI/Z-Image-Turbo" or local path)
        #[arg(short, long, default_value = "Tongyi-MAI/Z-Image-Turbo")]
        model: String,

        /// Text prompt for image generation
        prompt: String,

        /// Output file path
        #[arg(short, long, default_value = "output.png")]
        output: String,

        /// Image width in pixels
        #[arg(long, default_value_t = 1024)]
        width: u32,

        /// Image height in pixels
        #[arg(long, default_value_t = 1024)]
        height: u32,

        /// Number of inference steps
        #[arg(short, long, default_value_t = 9)]
        steps: u32,

        /// Guidance scale for CFG
        #[arg(short, long, default_value_t = 5.0)]
        guidance_scale: f32,

        /// Negative prompt (for CFG)
        #[arg(long)]
        negative_prompt: Option<String>,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Run on CPU instead of GPU
        #[arg(long)]
        cpu: bool,
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

    // Check GPU requirements at startup
    match gpu::check_gpu_requirements() {
        gpu::GpuCheckResult::NoGpu => {
            gpu::print_no_gpu_error();
            std::process::exit(1);
        }
        gpu::GpuCheckResult::LowVram(info) => {
            if !gpu::prompt_low_vram_confirmation(&info)? {
                std::process::exit(0);
            }
        }
        gpu::GpuCheckResult::Ok(_) => {
            // GPU meets requirements, continue
        }
    }

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
        Commands::Generate {
            model,
            prompt,
            output,
            width,
            height,
            steps,
            guidance_scale,
            negative_prompt,
            seed,
            cpu,
        } => {
            commands::generate::execute(
                &model,
                &prompt,
                &output,
                width,
                height,
                steps,
                guidance_scale,
                negative_prompt.as_deref(),
                seed,
                cpu,
            )
            .await?;
        }
    }

    Ok(())
}
