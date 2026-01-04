pub mod llama;

use anyhow::Result;
use std::io::{self, BufRead, Write};

use crate::models::ModelInfo;

pub struct Runner {
    model_info: ModelInfo,
}

impl Runner {
    pub fn for_model(model_info: &ModelInfo) -> Result<Self> {
        Ok(Self {
            model_info: model_info.clone(),
        })
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        // Find GGUF file in model directory
        let gguf_file = self
            .model_info
            .files
            .iter()
            .find(|f| f.ends_with(".gguf"))
            .ok_or_else(|| anyhow::anyhow!("No GGUF file found. Model may not be supported yet."))?;

        let model_path = self.model_info.path.join(gguf_file);

        // For now, we'll use llama.cpp via command line if available
        // In the future, we can add native bindings
        llama::generate(&model_path, prompt).await
    }

    pub async fn interactive(&self) -> Result<()> {
        println!("Starting interactive mode with {}...", self.model_info.name);
        println!("Type 'exit' or 'quit' to end the session.\n");

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("> ");
            stdout.flush()?;

            let mut input = String::new();
            stdin.lock().read_line(&mut input)?;

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            if input == "exit" || input == "quit" {
                println!("Goodbye!");
                break;
            }

            match self.generate(input).await {
                Ok(response) => println!("\n{}\n", response),
                Err(e) => eprintln!("Error: {}\n", e),
            }
        }

        Ok(())
    }
}
