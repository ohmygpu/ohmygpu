//! Interactive chat command

use anyhow::Result;
use std::io::{self, BufRead, Write};

const DAEMON_URL: &str = "http://localhost:10692";

pub async fn execute(model: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Check if daemon is running
    match client.get(format!("{}/health", DAEMON_URL)).send().await {
        Ok(r) if r.status().is_success() => {}
        _ => {
            eprintln!("Error: Daemon is not running. Start it with `omg serve`");
            std::process::exit(1);
        }
    }

    println!("Chatting with {} (Ctrl+C to exit)", model);
    println!("---");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        print!("> ");
        stdout.flush()?;

        // Read user input
        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Send to daemon
        let request = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": input}],
            "stream": false
        });

        match client
            .post(format!("{}/v1/chat/completions", DAEMON_URL))
            .json(&request)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let result: serde_json::Value = response.json().await?;
                if let Some(content) = result["choices"][0]["message"]["content"].as_str() {
                    println!("{}", content);
                    println!();
                }
            }
            Ok(response) => {
                let error = response.text().await?;
                eprintln!("Error: {}", error);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    Ok(())
}
