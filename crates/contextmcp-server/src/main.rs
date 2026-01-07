//! CogMCP - Local-first context MCP server
//!
//! A high-performance, privacy-preserving context management MCP server
//! that runs entirely on your local machine.

use cogmcp_server::runner::{RunnerConfig, ServerRunner};
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

    tracing::info!("Starting CogMCP server v{}", env!("CARGO_PKG_VERSION"));

    // Determine root directory from environment or current directory
    let root = env::var("COGMCP_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    tracing::info!("Root directory: {}", root.display());

    // Create and run the server
    let config = RunnerConfig::new(root).index_on_startup(true);

    let runner = ServerRunner::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create server: {}", e))?;

    // Log available tools before starting
    tracing::info!("Available tools:");
    for tool in runner.server().list_tools() {
        tracing::info!("  - {}: {}", tool.name, tool.description.unwrap_or_default());
    }

    // Run the server (blocks until shutdown)
    runner.run().await
}
