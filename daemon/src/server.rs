use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::api;
use crate::state::AppState;

pub async fn run_server(addr: SocketAddr) -> Result<()> {
    let state = Arc::new(AppState::new()?);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = api::routes(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    tracing::info!("Starting daemon on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
