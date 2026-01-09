//! Server runner for managing the MCP server lifecycle
//!
//! This module provides a `ServerRunner` that combines all components
//! (server, transport, protocol) into a unified abstraction for running
//! the CogMCP server.

use crate::CogMcpServer;
use cogmcp_core::Result;
use cogmcp_watcher::{spawn_debounce_checker, IndexCallback, WatcherEventHandler};
use rmcp::ServiceExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
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

/// Callback implementation for the watcher event handler
///
/// This bridges the watcher events to the server's incremental indexing methods.
struct ServerIndexCallback {
    server: CogMcpServer,
}

impl IndexCallback for ServerIndexCallback {
    fn index_file(&self, path: &Path) {
        if let Err(e) = self.server.index_file_incremental(path) {
            warn!(path = %path.display(), error = %e, "Failed to index file incrementally");
        }
    }

    fn queue_file(&self, path: &Path) {
        // Queue the file for debounced indexing
        if let Some(debouncer) = self.server.debouncer() {
            debouncer.mark_pending(path);
            debug!(path = %path.display(), "File queued for debounced indexing");
        }
    }

    fn remove_file(&self, path: &Path) {
        if let Err(e) = self.server.remove_file_from_index(path) {
            warn!(path = %path.display(), error = %e, "Failed to remove file from index");
        }
    }
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
    /// 2. Starts the file watcher if enabled
    /// 3. Starts the MCP protocol over stdio
    /// 4. Processes requests until EOF or shutdown
    /// 5. Gracefully shuts down the watcher
    /// 6. Returns when the service completes
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

        // Start watcher tasks if file watching is enabled
        let watcher_tasks = self.start_watcher_tasks();

        info!("Starting MCP protocol over stdio...");

        // Create stdio transport and start the service
        let transport = rmcp::transport::stdio();
        let service = self.server.clone().serve(transport).await?;

        info!("MCP server ready, waiting for requests...");

        // Wait for the service to complete (EOF or shutdown)
        if let Err(e) = service.waiting().await {
            error!("Service error: {}", e);
            // Abort watcher tasks on error
            if let Some((event_handle, debounce_handle, _)) = watcher_tasks {
                event_handle.abort();
                debounce_handle.abort();
            }
            return Err(e.into());
        }

        // Gracefully shutdown watcher tasks
        if let Some((event_handle, debounce_handle, _)) = watcher_tasks {
            info!("Shutting down file watcher...");
            event_handle.abort();
            debounce_handle.abort();
        }

        info!("CogMCP server shutdown complete.");
        Ok(())
    }

    /// Start the file watcher event handler and debounce checker tasks
    ///
    /// Returns the task handles if watching is enabled, None otherwise.
    fn start_watcher_tasks(
        &self,
    ) -> Option<(
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
    )> {
        // Check if watcher components are available
        let watcher = self.server.watcher()?;
        let prioritizer = self.server.prioritizer()?;
        let debouncer = self.server.debouncer()?;

        info!("Starting file watcher for {}", self.config.root.display());

        // Create the callback that bridges watcher events to server indexing
        let callback = Arc::new(ServerIndexCallback {
            server: self.server.clone(),
        });

        // Create and spawn the event handler
        let event_handler = Arc::new(WatcherEventHandler::new(
            Arc::clone(prioritizer),
            callback,
        ));
        let event_rx = watcher.subscribe();
        let event_handle = event_handler.spawn(event_rx);

        // Create channel for debounced files
        let (debounce_tx, debounce_rx) = mpsc::channel(100);

        // Spawn the debounce checker (polls every 100ms)
        let debounce_checker_handle =
            spawn_debounce_checker(Arc::clone(debouncer), debounce_tx, 100);

        // Spawn task to process debounced files
        let server = self.server.clone();
        let debounce_processor_handle = tokio::spawn(async move {
            Self::process_debounced_files(server, debounce_rx).await;
        });

        info!("File watcher started successfully");

        Some((event_handle, debounce_checker_handle, debounce_processor_handle))
    }

    /// Process files that have been debounced and are ready for indexing
    async fn process_debounced_files(server: CogMcpServer, mut rx: mpsc::Receiver<PathBuf>) {
        while let Some(path) = rx.recv().await {
            debug!(path = %path.display(), "Processing debounced file");
            if let Err(e) = server.index_file_incremental(&path) {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to index debounced file"
                );
            }
        }
        debug!("Debounce processor shutting down");
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
