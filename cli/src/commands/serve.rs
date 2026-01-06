use anyhow::Result;
use std::net::SocketAddr;

pub async fn execute(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;

    println!("Starting ohmygpu daemon...");
    println!("Listening on http://{}", addr);
    println!("\nAPI endpoints:");
    println!("  GET  /health              - Health check");
    println!("  GET  /v1/models           - List models");
    println!("  POST /v1/chat/completions - Chat completion (OpenAI-compatible)");
    println!("\nPress Ctrl+C to stop.\n");

    ohmygpu_daemon::run_server(addr).await?;

    Ok(())
}
