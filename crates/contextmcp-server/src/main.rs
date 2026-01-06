//! ContextMCP - Local-first context MCP server
//!
//! A high-performance, privacy-preserving context management MCP server
//! that runs entirely on your local machine.

use contextmcp_server::ContextMcpServer;
use rmcp::ServiceExt;
use std::env;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging (to stderr so it doesn't interfere with MCP stdio)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("Starting ContextMCP server v{}", env!("CARGO_PKG_VERSION"));

    // Determine root directory
    let root = env::var("CONTEXTMCP_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    tracing::info!("Root directory: {}", root.display());

    // Create server
    let server = ContextMcpServer::new(root.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create server: {}", e))?;

    // Initial indexing
    tracing::info!("Starting initial index...");
    if let Err(e) = server.index() {
        tracing::warn!("Initial indexing failed: {}. Continuing without index.", e);
    } else {
        let stats = server.db.get_stats().unwrap_or_default();
        tracing::info!(
            "Indexed {} files, {} symbols",
            stats.file_count,
            stats.symbol_count
        );
    }

    // List available tools
    tracing::info!("Available tools:");
    for tool in server.list_tools() {
        tracing::info!("  - {}: {}", tool.name, tool.description.unwrap_or_default());
    }

    tracing::info!("ContextMCP server ready, starting MCP protocol...");

    // Start MCP server with stdio transport
    let transport = rmcp::transport::stdio();
    let service = server.serve(transport).await?;

    // Wait for the service to complete
    service.waiting().await?;

    tracing::info!("ContextMCP server shutdown complete.");
    Ok(())
}
