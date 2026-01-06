//! MCP Server command - exposes ohmygpu to Claude Desktop and other MCP clients.
//!
//! Usage in Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "ohmygpu": {
//!       "command": "ohmygpu",
//!       "args": ["mcp"]
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use serde::Deserialize;

const DAEMON_URL: &str = "http://localhost:11434";

// ============================================================================
// Request types for tools
// ============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ChatRequest {
    /// The model to use (e.g., "microsoft/phi-2", "meta-llama/Llama-2-7b")
    pub model: String,
    /// The message to send to the model
    pub message: String,
}

// ============================================================================
// MCP Server implementation
// ============================================================================

#[derive(Debug, Clone)]
pub struct OhmyGpuMcp {
    client: reqwest::Client,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl OhmyGpuMcp {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            tool_router: Self::tool_router(),
        }
    }

    /// Chat with a local AI model
    #[tool(description = "Chat with a local AI model running on ohmygpu. Returns the model's response.")]
    async fn chat(&self, Parameters(req): Parameters<ChatRequest>) -> Result<String, String> {
        let request = serde_json::json!({
            "model": req.model,
            "messages": [{"role": "user", "content": req.message}],
            "stream": false
        });

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", DAEMON_URL))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to connect to ohmygpu daemon: {}. Is it running? Start with `omg serve`", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Chat request failed: {}", error_text));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response")
            .to_string();

        Ok(content)
    }

    /// List available models
    #[tool(description = "List all AI models installed on ohmygpu")]
    async fn list_models(&self) -> Result<String, String> {
        let response = self
            .client
            .get(format!("{}/v1/models", DAEMON_URL))
            .send()
            .await
            .map_err(|e| format!("Failed to connect to ohmygpu daemon: {}. Is it running? Start with `omg serve`", e))?;

        if !response.status().is_success() {
            return Err("Failed to list models".to_string());
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let models: Vec<String> = result["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if models.is_empty() {
            Ok("No models installed. Use `omg pull <model>` to download a model.".to_string())
        } else {
            Ok(format!(
                "Installed models:\n{}",
                models
                    .iter()
                    .map(|m| format!("- {}", m))
                    .collect::<Vec<_>>()
                    .join("\n")
            ))
        }
    }

    /// Check daemon status
    #[tool(description = "Check if the ohmygpu daemon is running and healthy")]
    async fn status(&self) -> Result<String, String> {
        match self
            .client
            .get(format!("{}/health", DAEMON_URL))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                Ok("ohmygpu daemon is running and healthy".to_string())
            }
            Ok(response) => Err(format!(
                "ohmygpu daemon returned status: {}",
                response.status()
            )),
            Err(_) => Err("ohmygpu daemon is not running. Start it with `omg serve`".to_string()),
        }
    }
}

#[tool_handler]
impl ServerHandler for OhmyGpuMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "ohmygpu MCP server - chat with local AI models. \
                 Make sure the ohmygpu daemon is running (`omg serve`)."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

/// Run the MCP server (stdio mode for Claude Desktop)
pub async fn execute() -> Result<()> {
    // Note: No logging to stdout - MCP uses stdout for protocol
    // Logging goes to stderr
    let server = OhmyGpuMcp::new();
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
