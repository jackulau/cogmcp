//! Server runner for managing the MCP server lifecycle
//!
//! This module provides a `ServerRunner` that combines all components
//! (server, transport, protocol) into a unified abstraction for running
//! the CogMCP server.

use crate::CogMcpServer;
use cogmcp_core::Result;
use rmcp::ServiceExt;
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

/// Configuration for the server runner
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Root directory for the server to index
    pub root: PathBuf,
    /// Whether to perform initial indexing on startup
    pub index_on_startup: bool,
    /// Use in-memory storage (for testing)
    pub in_memory: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            index_on_startup: true,
            in_memory: false,
        }
    }
}

impl RunnerConfig {
    /// Create a new config with the specified root directory
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            ..Default::default()
        }
    }

    /// Set whether to index on startup
    pub fn index_on_startup(mut self, enabled: bool) -> Self {
        self.index_on_startup = enabled;
        self
    }

    /// Use in-memory storage
    pub fn in_memory(mut self, enabled: bool) -> Self {
        self.in_memory = enabled;
        self
    }
}

/// Server runner combining all MCP components
pub struct ServerRunner {
    config: RunnerConfig,
    server: CogMcpServer,
}

impl ServerRunner {
    /// Create a new server runner with the given configuration
    pub fn new(config: RunnerConfig) -> Result<Self> {
        let server = if config.in_memory {
            CogMcpServer::in_memory(config.root.clone())?
        } else {
            CogMcpServer::new(config.root.clone())?
        };

        Ok(Self { config, server })
    }

    /// Create a runner with default configuration
    pub fn with_root(root: PathBuf) -> Result<Self> {
        Self::new(RunnerConfig::new(root))
    }

    /// Get a reference to the underlying server
    pub fn server(&self) -> &CogMcpServer {
        &self.server
    }

    /// Perform initial indexing if configured
    pub fn index(&self) -> Result<()> {
        info!("Starting initial index of {}", self.config.root.display());
        self.server.index()?;

        let stats = self.server.db.get_stats().unwrap_or_default();
        info!(
            "Indexed {} files, {} symbols",
            stats.file_count, stats.symbol_count
        );

        Ok(())
    }

    /// Run the server with stdio transport
    ///
    /// This method:
    /// 1. Optionally performs initial indexing
    /// 2. Starts the MCP protocol over stdio
    /// 3. Processes requests until EOF or shutdown
    /// 4. Returns when the service completes
    pub async fn run(self) -> std::result::Result<(), anyhow::Error> {
        info!(
            "CogMCP server starting (root: {})",
            self.config.root.display()
        );

        // Perform initial indexing if configured
        if self.config.index_on_startup {
            if let Err(e) = self.index() {
                warn!("Initial indexing failed: {}. Continuing without index.", e);
            }
        }

        // Log available tools
        debug!("Available tools:");
        for tool in self.server.list_tools() {
            debug!("  - {}: {}", tool.name, tool.description.unwrap_or_default());
        }

        info!("Starting MCP protocol over stdio...");

        // Create stdio transport and start the service
        let transport = rmcp::transport::stdio();
        let service = self.server.serve(transport).await?;

        info!("MCP server ready, waiting for requests...");

        // Wait for the service to complete (EOF or shutdown)
        if let Err(e) = service.waiting().await {
            error!("Service error: {}", e);
            return Err(e.into());
        }

        info!("CogMCP server shutdown complete.");
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert!(config.index_on_startup);
        assert!(!config.in_memory);
    }

    #[test]
    fn test_runner_config_builder() {
        let config = RunnerConfig::new(PathBuf::from("/tmp/test"))
            .index_on_startup(false)
            .in_memory(true);

        assert_eq!(config.root, PathBuf::from("/tmp/test"));
        assert!(!config.index_on_startup);
        assert!(config.in_memory);
    }

    #[test]
    fn test_runner_creation_in_memory() {
        let config = RunnerConfig::new(PathBuf::from("/tmp/test")).in_memory(true);
        let runner = ServerRunner::new(config);
        assert!(runner.is_ok());
    }
}
