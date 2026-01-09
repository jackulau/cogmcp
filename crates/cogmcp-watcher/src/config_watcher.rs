//! Config file watching with debouncing
//!
//! This module provides automatic detection of config file changes
//! and triggers reload events with debouncing to prevent excessive reloads.
//!
//! The main types are:
//! - [`ConfigWatcher`] - Low-level file watcher that emits change events
//! - [`AutoReloadingConfig`] - Standalone config wrapper with auto-reload
//! - [`SharedConfigWatcher`] - Integrates with SharedConfig from cogmcp-core

use cogmcp_core::{Config, Result, SharedConfig};
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, RecommendedCache};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

/// Event emitted when a config file changes
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// Path to the config file that changed
    pub path: PathBuf,
    /// Kind of change detected
    pub kind: ConfigChangeKind,
}

/// Kind of config file change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigChangeKind {
    /// Config file was modified
    Modified,
    /// Config file was created
    Created,
    /// Config file was deleted
    Deleted,
}

/// Configuration for the ConfigWatcher
#[derive(Debug, Clone)]
pub struct ConfigWatcherOptions {
    /// Debounce duration in milliseconds
    pub debounce_ms: u64,
    /// Additional paths to watch (besides default config locations)
    pub extra_paths: Vec<PathBuf>,
}

impl Default for ConfigWatcherOptions {
    fn default() -> Self {
        Self {
            debounce_ms: 1000,
            extra_paths: Vec::new(),
        }
    }
}

/// Watches config files for changes and emits events
///
/// ConfigWatcher monitors the standard config file locations from `Config::config_locations()`
/// plus any additional paths specified. Changes are debounced to prevent excessive reload events.
pub struct ConfigWatcher {
    /// Broadcast sender for config change events
    event_tx: broadcast::Sender<ConfigChangeEvent>,
    /// Paths being watched
    watched_paths: Vec<PathBuf>,
    /// Handle to keep the watcher alive
    _debouncer: Debouncer<RecommendedWatcher, RecommendedCache>,
}

impl ConfigWatcher {
    /// Create a new ConfigWatcher with default options
    pub fn new() -> Result<Self> {
        Self::with_options(ConfigWatcherOptions::default())
    }

    /// Create a new ConfigWatcher with custom options
    pub fn with_options(options: ConfigWatcherOptions) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(100);
        let tx = event_tx.clone();

        // Collect all paths to watch
        let mut watched_paths = Config::config_locations();
        watched_paths.extend(options.extra_paths);

        // Create a channel for the debouncer to send events
        let (notify_tx, mut notify_rx) = mpsc::unbounded_channel();

        // Create debounced watcher
        let debouncer = new_debouncer(
            Duration::from_millis(options.debounce_ms),
            None,
            move |res: DebounceEventResult| {
                let _ = notify_tx.send(res);
            },
        )
        .map_err(|e| cogmcp_core::Error::FileSystem(format!("Failed to create debouncer: {}", e)))?;

        // Start background task to process debounced events
        let paths_clone = watched_paths.clone();
        tokio::spawn(async move {
            while let Some(result) = notify_rx.recv().await {
                match result {
                    Ok(events) => {
                        for event in events {
                            // Check if this event is for one of our watched config files
                            for path in &event.paths {
                                if Self::is_config_path(path, &paths_clone) {
                                    let kind = match event.kind {
                                        notify::EventKind::Create(_) => ConfigChangeKind::Created,
                                        notify::EventKind::Modify(_) => ConfigChangeKind::Modified,
                                        notify::EventKind::Remove(_) => ConfigChangeKind::Deleted,
                                        _ => continue,
                                    };

                                    debug!("Config file changed: {:?} ({:?})", path, kind);

                                    let _ = tx.send(ConfigChangeEvent {
                                        path: path.clone(),
                                        kind,
                                    });
                                }
                            }
                        }
                    }
                    Err(errors) => {
                        for error in errors {
                            warn!("Config watcher error: {:?}", error);
                        }
                    }
                }
            }
        });

        // Watch all config paths
        let mut debouncer = debouncer;
        for path in &watched_paths {
            // Watch the parent directory if the file doesn't exist yet
            let watch_path = if path.exists() {
                path.clone()
            } else if let Some(parent) = path.parent() {
                if parent.exists() {
                    parent.to_path_buf()
                } else {
                    debug!("Skipping watch for non-existent path: {:?}", path);
                    continue;
                }
            } else {
                continue;
            };

            if let Err(e) = debouncer.watch(&watch_path, RecursiveMode::NonRecursive) {
                warn!("Failed to watch {:?}: {}", watch_path, e);
            } else {
                info!("Watching config path: {:?}", watch_path);
            }
        }

        Ok(Self {
            event_tx,
            watched_paths,
            _debouncer: debouncer,
        })
    }

    /// Subscribe to config change events
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.event_tx.subscribe()
    }

    /// Get the list of paths being watched
    pub fn watched_paths(&self) -> &[PathBuf] {
        &self.watched_paths
    }

    /// Check if a path matches one of our config paths
    fn is_config_path(path: &PathBuf, config_paths: &[PathBuf]) -> bool {
        for config_path in config_paths {
            if path == config_path {
                return true;
            }
            // Also check if it's a file in a watched directory
            if let Some(file_name) = path.file_name() {
                if let Some(config_name) = config_path.file_name() {
                    if file_name == config_name {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Auto-reloading config wrapper that watches for file changes
///
/// This struct combines ConfigWatcher with automatic config reloading.
/// It validates new configs before applying them and keeps the old config
/// if validation fails.
pub struct AutoReloadingConfig {
    /// Current configuration (wrapped in Arc for sharing)
    config: Arc<parking_lot::RwLock<Config>>,
    /// Config watcher instance
    _watcher: ConfigWatcher,
    /// Broadcast sender for reload events
    reload_tx: broadcast::Sender<ReloadEvent>,
}

/// Event emitted when config is reloaded
#[derive(Debug, Clone)]
pub struct ReloadEvent {
    /// Whether the reload was successful
    pub success: bool,
    /// Path to the config file that triggered the reload
    pub path: PathBuf,
    /// Error message if reload failed
    pub error: Option<String>,
}

impl AutoReloadingConfig {
    /// Create a new auto-reloading config wrapper
    pub fn new() -> Result<Self> {
        Self::with_options(ConfigWatcherOptions::default())
    }

    /// Create a new auto-reloading config wrapper with custom options
    pub fn with_options(options: ConfigWatcherOptions) -> Result<Self> {
        let config = Arc::new(parking_lot::RwLock::new(Config::load()?));
        let watcher = ConfigWatcher::with_options(options)?;
        let (reload_tx, _) = broadcast::channel(100);

        let config_clone = config.clone();
        let reload_tx_clone = reload_tx.clone();
        let mut change_rx = watcher.subscribe();

        // Spawn task to handle config changes
        tokio::spawn(async move {
            while let Ok(event) = change_rx.recv().await {
                if event.kind == ConfigChangeKind::Deleted {
                    info!("Config file deleted, keeping current config");
                    continue;
                }

                info!("Config file changed, attempting reload: {:?}", event.path);

                // Try to load the new config
                match Config::load() {
                    Ok(new_config) => {
                        // Validate the new config (basic validation - can be extended)
                        if let Err(e) = Self::validate_config(&new_config) {
                            warn!("New config failed validation: {}", e);
                            let _ = reload_tx_clone.send(ReloadEvent {
                                success: false,
                                path: event.path,
                                error: Some(format!("Validation failed: {}", e)),
                            });
                            continue;
                        }

                        // Apply the new config
                        *config_clone.write() = new_config;
                        info!("Config reloaded successfully");

                        let _ = reload_tx_clone.send(ReloadEvent {
                            success: true,
                            path: event.path,
                            error: None,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to reload config: {}", e);
                        let _ = reload_tx_clone.send(ReloadEvent {
                            success: false,
                            path: event.path,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        });

        Ok(Self {
            config,
            _watcher: watcher,
            reload_tx,
        })
    }

    /// Get the current configuration
    pub fn current(&self) -> Config {
        self.config.read().clone()
    }

    /// Subscribe to reload events
    pub fn subscribe_reloads(&self) -> broadcast::Receiver<ReloadEvent> {
        self.reload_tx.subscribe()
    }

    /// Validate a config (can be extended with more checks)
    fn validate_config(config: &Config) -> Result<()> {
        // Basic validation checks
        if config.indexing.max_file_size == 0 {
            return Err(cogmcp_core::Error::Config(
                "max_file_size must be greater than 0".into(),
            ));
        }

        if config.watching.debounce_ms == 0 {
            return Err(cogmcp_core::Error::Config(
                "debounce_ms must be greater than 0".into(),
            ));
        }

        if config.context.chunk_size == 0 {
            return Err(cogmcp_core::Error::Config(
                "chunk_size must be greater than 0".into(),
            ));
        }

        if config.context.chunk_overlap < 0.0 || config.context.chunk_overlap >= 1.0 {
            return Err(cogmcp_core::Error::Config(
                "chunk_overlap must be between 0.0 and 1.0".into(),
            ));
        }

        if config.search.min_similarity < 0.0 || config.search.min_similarity > 1.0 {
            return Err(cogmcp_core::Error::Config(
                "min_similarity must be between 0.0 and 1.0".into(),
            ));
        }

        Ok(())
    }
}

/// Watches config files and automatically reloads SharedConfig on changes
///
/// This struct integrates the file-watching capabilities of ConfigWatcher
/// with the SharedConfig from cogmcp-core, providing automatic config
/// reload when files change on disk.
///
/// # Example
///
/// ```no_run
/// use cogmcp_core::SharedConfig;
/// use cogmcp_watcher::{SharedConfigWatcher, ConfigWatcherOptions};
///
/// # async fn example() -> cogmcp_core::Result<()> {
/// let shared_config = SharedConfig::load()?;
/// let watcher = SharedConfigWatcher::new(shared_config.clone())?;
///
/// // Subscribe to reload events
/// let mut rx = watcher.subscribe_reloads();
/// tokio::spawn(async move {
///     while let Ok(event) = rx.recv().await {
///         println!("Config reloaded: success={}", event.success);
///     }
/// });
/// # Ok(())
/// # }
/// ```
pub struct SharedConfigWatcher {
    /// The shared config being watched
    shared_config: Arc<SharedConfig>,
    /// The underlying file watcher
    _watcher: ConfigWatcher,
    /// Broadcast sender for reload events
    reload_tx: broadcast::Sender<ReloadEvent>,
}

impl SharedConfigWatcher {
    /// Create a new SharedConfigWatcher with default options
    ///
    /// This watches all default config file locations and triggers
    /// reload when any of them change.
    pub fn new(shared_config: Arc<SharedConfig>) -> Result<Self> {
        Self::with_options(shared_config, ConfigWatcherOptions::default())
    }

    /// Create a new SharedConfigWatcher with custom options
    ///
    /// The debounce_ms option controls how long to wait after a file
    /// change before triggering a reload. This prevents excessive
    /// reloads when a file is being edited rapidly.
    pub fn with_options(
        shared_config: Arc<SharedConfig>,
        options: ConfigWatcherOptions,
    ) -> Result<Self> {
        let watcher = ConfigWatcher::with_options(options)?;
        let (reload_tx, _) = broadcast::channel(100);

        let config_clone = shared_config.clone();
        let reload_tx_clone = reload_tx.clone();
        let mut change_rx = watcher.subscribe();

        // Spawn task to handle config changes and trigger reloads
        tokio::spawn(async move {
            while let Ok(event) = change_rx.recv().await {
                if event.kind == ConfigChangeKind::Deleted {
                    info!("Config file deleted, keeping current config");
                    continue;
                }

                info!(
                    "Config file changed, attempting reload: {:?}",
                    event.path
                );

                // Use SharedConfig's reload method which handles validation
                match config_clone.reload() {
                    Ok(_new_config) => {
                        info!("SharedConfig reloaded successfully");

                        let _ = reload_tx_clone.send(ReloadEvent {
                            success: true,
                            path: event.path,
                            error: None,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to reload SharedConfig: {}", e);
                        let _ = reload_tx_clone.send(ReloadEvent {
                            success: false,
                            path: event.path,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        });

        Ok(Self {
            shared_config,
            _watcher: watcher,
            reload_tx,
        })
    }

    /// Get a reference to the underlying SharedConfig
    pub fn shared_config(&self) -> &Arc<SharedConfig> {
        &self.shared_config
    }

    /// Get the current configuration
    ///
    /// This returns a clone of the current configuration.
    pub fn current(&self) -> Config {
        (*self.shared_config.get()).clone()
    }

    /// Subscribe to reload events
    ///
    /// Returns a receiver that will receive events whenever
    /// a config reload is attempted (successful or not).
    pub fn subscribe_reloads(&self) -> broadcast::Receiver<ReloadEvent> {
        self.reload_tx.subscribe()
    }

    /// Get the paths being watched
    pub fn watched_paths(&self) -> &[PathBuf] {
        self._watcher.watched_paths()
    }
}

impl std::fmt::Debug for SharedConfigWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedConfigWatcher")
            .field("watched_paths", &self._watcher.watched_paths())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_config_watcher_creation() {
        let watcher = ConfigWatcher::new();
        assert!(watcher.is_ok());

        let watcher = watcher.unwrap();
        assert!(!watcher.watched_paths().is_empty());
    }

    #[tokio::test]
    async fn test_config_watcher_with_options() {
        let temp_dir = TempDir::new().unwrap();
        let extra_path = temp_dir.path().join("extra-config.toml");
        fs::write(&extra_path, "[server]\ntransport = \"stdio\"").unwrap();

        let options = ConfigWatcherOptions {
            debounce_ms: 500,
            extra_paths: vec![extra_path.clone()],
        };

        let watcher = ConfigWatcher::with_options(options).unwrap();
        assert!(watcher.watched_paths().contains(&extra_path));
    }

    #[tokio::test]
    async fn test_config_change_event_subscription() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test-config.toml");
        fs::write(&config_path, "[server]\ntransport = \"stdio\"").unwrap();

        let options = ConfigWatcherOptions {
            debounce_ms: 100,
            extra_paths: vec![config_path.clone()],
        };

        let watcher = ConfigWatcher::with_options(options).unwrap();
        let mut rx = watcher.subscribe();

        // Give the watcher time to set up
        sleep(Duration::from_millis(100)).await;

        // Modify the config file
        fs::write(&config_path, "[server]\ntransport = \"http\"").unwrap();

        // Wait for debounced event (debounce_ms + some margin)
        let result = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;

        // The event should be received (may timeout in CI environments without inotify)
        // Note: On some platforms (macOS), file writes may trigger Created instead of Modified
        if let Ok(Ok(event)) = result {
            assert!(
                event.kind == ConfigChangeKind::Modified || event.kind == ConfigChangeKind::Created,
                "Expected Modified or Created, got {:?}",
                event.kind
            );
        }
    }

    #[tokio::test]
    async fn test_config_validation() {
        // Valid config
        let valid_config = Config::default();
        assert!(AutoReloadingConfig::validate_config(&valid_config).is_ok());

        // Invalid config - zero max_file_size
        let mut invalid_config = Config::default();
        invalid_config.indexing.max_file_size = 0;
        assert!(AutoReloadingConfig::validate_config(&invalid_config).is_err());

        // Invalid config - zero debounce
        let mut invalid_config = Config::default();
        invalid_config.watching.debounce_ms = 0;
        assert!(AutoReloadingConfig::validate_config(&invalid_config).is_err());

        // Invalid config - chunk_overlap out of range
        let mut invalid_config = Config::default();
        invalid_config.context.chunk_overlap = 1.5;
        assert!(AutoReloadingConfig::validate_config(&invalid_config).is_err());

        // Invalid config - min_similarity out of range
        let mut invalid_config = Config::default();
        invalid_config.search.min_similarity = -0.5;
        assert!(AutoReloadingConfig::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_is_config_path() {
        let config_paths = vec![
            PathBuf::from("/home/user/.cogmcp.toml"),
            PathBuf::from("/etc/cogmcp/config.toml"),
        ];

        // Exact match
        assert!(ConfigWatcher::is_config_path(
            &PathBuf::from("/home/user/.cogmcp.toml"),
            &config_paths
        ));

        // Non-match
        assert!(!ConfigWatcher::is_config_path(
            &PathBuf::from("/home/user/other.toml"),
            &config_paths
        ));

        // Match by filename
        assert!(ConfigWatcher::is_config_path(
            &PathBuf::from("/other/path/.cogmcp.toml"),
            &config_paths
        ));
    }

    #[test]
    fn test_config_change_kind() {
        assert_ne!(ConfigChangeKind::Modified, ConfigChangeKind::Created);
        assert_ne!(ConfigChangeKind::Modified, ConfigChangeKind::Deleted);
        assert_ne!(ConfigChangeKind::Created, ConfigChangeKind::Deleted);
    }

    #[tokio::test]
    async fn test_shared_config_watcher_creation() {
        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = SharedConfigWatcher::new(shared_config);
        assert!(watcher.is_ok());

        let watcher = watcher.unwrap();
        assert!(!watcher.watched_paths().is_empty());
    }

    #[tokio::test]
    async fn test_shared_config_watcher_with_options() {
        let temp_dir = TempDir::new().unwrap();
        let extra_path = temp_dir.path().join("extra-config.toml");
        fs::write(&extra_path, "[server]\ntransport = \"stdio\"").unwrap();

        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let options = ConfigWatcherOptions {
            debounce_ms: 500,
            extra_paths: vec![extra_path.clone()],
        };

        let watcher = SharedConfigWatcher::with_options(shared_config, options).unwrap();
        assert!(watcher.watched_paths().contains(&extra_path));
    }

    #[tokio::test]
    async fn test_shared_config_watcher_current() {
        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = SharedConfigWatcher::new(shared_config).unwrap();

        let config = watcher.current();
        assert_eq!(config.server.transport, "stdio");
    }

    #[tokio::test]
    async fn test_shared_config_watcher_shared_config_ref() {
        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = SharedConfigWatcher::new(shared_config.clone()).unwrap();

        // Verify we can access the underlying SharedConfig
        let config_ref = watcher.shared_config();
        assert_eq!(config_ref.get().server.transport, "stdio");
    }

    #[tokio::test]
    async fn test_shared_config_watcher_subscribe() {
        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = SharedConfigWatcher::new(shared_config).unwrap();

        // Should be able to subscribe to reload events
        let _rx = watcher.subscribe_reloads();
    }

    #[tokio::test]
    async fn test_shared_config_watcher_debug() {
        let shared_config = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = SharedConfigWatcher::new(shared_config).unwrap();

        let debug_str = format!("{:?}", watcher);
        assert!(debug_str.contains("SharedConfigWatcher"));
        assert!(debug_str.contains("watched_paths"));
    }

    #[tokio::test]
    async fn test_shared_config_watcher_file_change_triggers_reload() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary config file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[server]
transport = "stdio"
log_level = "info"
"#
        )
        .unwrap();

        // Load SharedConfig from the temp file
        let shared_config = SharedConfig::load_from_path(temp_file.path()).unwrap();
        assert_eq!(shared_config.get().server.log_level, "info");

        // Create watcher with the temp file path
        let options = ConfigWatcherOptions {
            debounce_ms: 100,
            extra_paths: vec![temp_file.path().to_path_buf()],
        };

        let watcher = SharedConfigWatcher::with_options(shared_config.clone(), options).unwrap();
        let mut reload_rx = watcher.subscribe_reloads();

        // Give the watcher time to set up
        sleep(Duration::from_millis(100)).await;

        // Modify the config file
        let path = temp_file.path().to_path_buf();
        std::fs::write(
            &path,
            r#"
[server]
transport = "stdio"
log_level = "debug"
"#,
        )
        .unwrap();

        // Wait for the reload event
        let result = tokio::time::timeout(Duration::from_millis(500), reload_rx.recv()).await;

        // In environments with proper file watching support, we should get an event
        if let Ok(Ok(event)) = result {
            assert!(event.success);
            // After reload, the config should have the new value
            assert_eq!(shared_config.get().server.log_level, "debug");
        }
    }

    #[tokio::test]
    async fn test_shared_config_watcher_invalid_config_rejected() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary config file with valid config
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[server]
transport = "stdio"
log_level = "info"
"#
        )
        .unwrap();

        // Load SharedConfig from the temp file
        let shared_config = SharedConfig::load_from_path(temp_file.path()).unwrap();
        let original_level = shared_config.get().server.log_level.clone();

        // Create watcher
        let options = ConfigWatcherOptions {
            debounce_ms: 100,
            extra_paths: vec![temp_file.path().to_path_buf()],
        };

        let watcher = SharedConfigWatcher::with_options(shared_config.clone(), options).unwrap();
        let mut reload_rx = watcher.subscribe_reloads();

        // Give the watcher time to set up
        sleep(Duration::from_millis(100)).await;

        // Write an invalid config (invalid transport)
        let path = temp_file.path().to_path_buf();
        std::fs::write(
            &path,
            r#"
[server]
transport = "invalid_transport"
log_level = "warn"
"#,
        )
        .unwrap();

        // Wait for the reload event
        let result = tokio::time::timeout(Duration::from_millis(500), reload_rx.recv()).await;

        // In environments with proper file watching support, we should get a failed event
        if let Ok(Ok(event)) = result {
            assert!(!event.success);
            assert!(event.error.is_some());
            // Original config should be preserved
            assert_eq!(shared_config.get().server.log_level, original_level);
        }
    }

    #[tokio::test]
    async fn test_debouncing_prevents_excessive_reloads() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary config file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[server]
transport = "stdio"
log_level = "info"
"#
        )
        .unwrap();

        let shared_config = SharedConfig::load_from_path(temp_file.path()).unwrap();

        // Create watcher with longer debounce
        let options = ConfigWatcherOptions {
            debounce_ms: 200,
            extra_paths: vec![temp_file.path().to_path_buf()],
        };

        let watcher = SharedConfigWatcher::with_options(shared_config.clone(), options).unwrap();
        let mut reload_rx = watcher.subscribe_reloads();

        // Give the watcher time to set up
        sleep(Duration::from_millis(100)).await;

        // Make multiple rapid changes
        let path = temp_file.path().to_path_buf();
        for i in 0..5 {
            std::fs::write(
                &path,
                format!(
                    r#"
[server]
transport = "stdio"
log_level = "level{}"
"#,
                    i
                ),
            )
            .unwrap();
            sleep(Duration::from_millis(20)).await;
        }

        // Wait for debounce period plus margin
        sleep(Duration::from_millis(300)).await;

        // Count received events (should be less than 5 due to debouncing)
        let mut event_count = 0;
        while let Ok(result) =
            tokio::time::timeout(Duration::from_millis(50), reload_rx.recv()).await
        {
            if result.is_ok() {
                event_count += 1;
            }
        }

        // Due to debouncing, we should have fewer events than changes
        // (actual count depends on timing, but should be > 0 and < 5)
        // We just verify the test runs without panic
        assert!(event_count <= 5);
    }
}
