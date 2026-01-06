mod commands;
mod daemon;
mod gpu;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ohmygpu")]
#[command(author, version, about = "Unified local AI infrastructure", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Model management (list, pull, remove, info, gc)
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },

    /// Daemon server management
    Serve {
        #[command(subcommand)]
        action: Option<ServeCommands>,

        /// Run in background (daemon mode)
        #[arg(short, long)]
        daemon: bool,

        /// Port to listen on
        #[arg(short, long, default_value = "10692")]
        port: u16,
    },

    /// Generate content (image, video, audio)
    Gen {
        #[command(subcommand)]
        action: GenCommands,
    },

    /// Interactive chat with a model
    Chat {
        /// Model to chat with
        model: String,
    },

    /// Start MCP server for Claude Desktop integration
    Mcp,

    /// View or set configuration
    Config {
        /// Config key (e.g., "daemon.port", "inference.temperature")
        key: Option<String>,

        /// Value to set (if omitted, shows current value)
        value: Option<String>,
    },

    /// Search for models on HuggingFace
    Search {
        /// Search query
        query: String,
    },

    /// Self-update to the latest version
    Update,
}

#[derive(Subcommand)]
enum ModelCommands {
    /// List installed models
    #[command(alias = "ls")]
    List,

    /// Pull/download a model from HuggingFace
    Pull {
        /// Model identifier (e.g., "microsoft/phi-2")
        model: String,

        /// Specific file to download
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Remove an installed model
    #[command(alias = "rm")]
    Remove {
        /// Model name to remove
        model: String,
    },

    /// Show model information
    Info {
        /// Model name
        model: String,
    },

    /// Garbage collect unused cache files
    Gc,
}

#[derive(Subcommand)]
enum ServeCommands {
    /// Check daemon status
    Status,

    /// Stop the daemon
    Stop,
}

#[derive(Subcommand)]
enum GenCommands {
    /// Generate an image from a text prompt
    Image {
        /// Text prompt for image generation
        prompt: String,

        /// Model to use
        #[arg(short, long, default_value = "Tongyi-MAI/Z-Image-Turbo")]
        model: String,

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

    /// Generate a video (coming soon)
    Video {
        /// Text prompt
        prompt: String,
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

    // MCP command skips GPU check (it just connects to daemon via HTTP)
    if matches!(cli.command, Commands::Mcp) {
        return commands::mcp::execute().await;
    }

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
        // Model management
        Commands::Model { action } => match action {
            ModelCommands::List => {
                commands::models::execute().await?;
            }
            ModelCommands::Pull { model, file } => {
                commands::pull::execute(&model, file.as_deref()).await?;
            }
            ModelCommands::Remove { model } => {
                commands::remove::execute(&model).await?;
            }
            ModelCommands::Info { model } => {
                commands::model_info::execute(&model).await?;
            }
            ModelCommands::Gc => {
                commands::model_gc::execute().await?;
            }
        },

        // Serve daemon
        Commands::Serve { action, daemon, port } => match action {
            None => {
                // Start server
                if daemon {
                    commands::serve::execute_background(port).await?;
                } else {
                    commands::serve::execute(port).await?;
                }
            }
            Some(ServeCommands::Status) => {
                commands::serve::status().await?;
            }
            Some(ServeCommands::Stop) => {
                commands::serve::stop().await?;
            }
        },

        // Generate content
        Commands::Gen { action } => match action {
            GenCommands::Image {
                prompt,
                model,
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
            GenCommands::Video { prompt: _ } => {
                println!("Video generation coming soon!");
            }
        },

        // Interactive chat
        Commands::Chat { model } => {
            commands::chat::execute(&model).await?;
        }

        // MCP (handled above with early return)
        Commands::Mcp => unreachable!(),

        // Config
        Commands::Config { key, value } => {
            commands::config::execute(key.as_deref(), value.as_deref()).await?;
        }

        // Search
        Commands::Search { query } => {
            commands::search::execute(&query).await?;
        }

        // Update
        Commands::Update => {
            commands::update::execute().await?;
        }
    }

    Ok(())
}
