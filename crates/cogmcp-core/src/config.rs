//! Configuration management for CogMCP

use arc_swap::{ArcSwap, Guard};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};

/// Main configuration for CogMCP
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub indexing: IndexingConfig,
    pub watching: WatchingConfig,
    pub context: ContextConfig,
    pub git: GitConfig,
    pub search: SearchConfig,
    pub cache: CacheConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            indexing: IndexingConfig::default(),
            watching: WatchingConfig::default(),
            context: ContextConfig::default(),
            git: GitConfig::default(),
            search: SearchConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from default locations
    pub fn load() -> Result<Self> {
        // Try loading from multiple locations in order of priority
        let locations = Self::config_locations();

        for path in locations {
            if path.exists() {
                return Self::load_from_path(&path);
            }
        }

        // Return default config if no file found
        Ok(Self::default())
    }

    /// Load configuration from a specific path
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Get default configuration file locations
    pub fn config_locations() -> Vec<PathBuf> {
        let mut locations = Vec::new();

        // 1. Current directory
        locations.push(PathBuf::from(".cogmcp.toml"));

        // 2. User config directory
        if let Some(config_dir) = dirs::config_dir() {
            locations.push(config_dir.join("cogmcp").join("config.toml"));
        }

        // 3. Home directory
        if let Some(home) = dirs::home_dir() {
            locations.push(home.join(".cogmcp.toml"));
        }

        locations
    }

    /// Get the data directory for storing index and cache
    pub fn data_dir() -> Result<PathBuf> {
        if let Some(data_dir) = dirs::data_local_dir() {
            let path = data_dir.join("cogmcp");
            std::fs::create_dir_all(&path)?;
            Ok(path)
        } else {
            Err(Error::Config("Could not determine data directory".into()))
        }
    }

    /// Get the path to the SQLite database
    pub fn database_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("index.db"))
    }

    /// Get the path to the Tantivy index
    pub fn tantivy_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("tantivy"))
    }

    /// Get the path to the embeddings model
    pub fn model_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("models").join("all-MiniLM-L6-v2.onnx"))
    }

    /// Validate the configuration
    ///
    /// Returns Ok(()) if the configuration is valid, or an error describing
    /// what's invalid.
    pub fn validate(&self) -> Result<()> {
        // Validate server config
        if self.server.transport != "stdio" && self.server.transport != "http" {
            return Err(Error::Config(format!(
                "Invalid transport '{}': must be 'stdio' or 'http'",
                self.server.transport
            )));
        }

        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.server.log_level.to_lowercase().as_str()) {
            return Err(Error::Config(format!(
                "Invalid log_level '{}': must be one of {:?}",
                self.server.log_level, valid_log_levels
            )));
        }

        // Validate context config
        if self.context.chunk_overlap < 0.0 || self.context.chunk_overlap > 1.0 {
            return Err(Error::Config(format!(
                "Invalid chunk_overlap {}: must be between 0.0 and 1.0",
                self.context.chunk_overlap
            )));
        }

        let valid_strategies = ["semantic", "recursive", "fixed"];
        if !valid_strategies.contains(&self.context.chunking_strategy.as_str()) {
            return Err(Error::Config(format!(
                "Invalid chunking_strategy '{}': must be one of {:?}",
                self.context.chunking_strategy, valid_strategies
            )));
        }

        // Validate search config
        if self.search.min_similarity < 0.0 || self.search.min_similarity > 1.0 {
            return Err(Error::Config(format!(
                "Invalid min_similarity {}: must be between 0.0 and 1.0",
                self.search.min_similarity
            )));
        }

        if self.search.keyword_weight < 0.0 || self.search.keyword_weight > 1.0 {
            return Err(Error::Config(format!(
                "Invalid keyword_weight {}: must be between 0.0 and 1.0",
                self.search.keyword_weight
            )));
        }

        if self.search.semantic_weight < 0.0 || self.search.semantic_weight > 1.0 {
            return Err(Error::Config(format!(
                "Invalid semantic_weight {}: must be between 0.0 and 1.0",
                self.search.semantic_weight
            )));
        }

        let valid_modes = ["keyword", "semantic", "hybrid"];
        if !valid_modes.contains(&self.search.default_mode.as_str()) {
            return Err(Error::Config(format!(
                "Invalid default_mode '{}': must be one of {:?}",
                self.search.default_mode, valid_modes
            )));
        }

        Ok(())
    }
}

/// Event emitted when configuration changes
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// The new configuration after the change
    pub new_config: Arc<Config>,
    /// The previous configuration before the change
    pub old_config: Arc<Config>,
    /// The path from which the config was reloaded (if any)
    pub source_path: Option<PathBuf>,
}

/// Thread-safe shared configuration wrapper
///
/// This type provides atomic access to configuration and supports
/// hot-reloading from disk. Use `current()` to get a read guard
/// to the current configuration, and `reload()` to refresh from disk.
pub struct SharedConfig {
    /// The current configuration, wrapped in ArcSwap for atomic updates
    config: ArcSwap<Config>,
    /// The path to the config file (if loaded from a file)
    config_path: parking_lot::RwLock<Option<PathBuf>>,
    /// Broadcast channel for config change notifications
    change_sender: broadcast::Sender<ConfigChangeEvent>,
}

impl SharedConfig {
    /// Load configuration from default locations
    ///
    /// This searches for configuration files in the standard locations
    /// and loads the first one found. If no config file exists, uses defaults.
    pub fn load() -> Result<Arc<Self>> {
        let locations = Config::config_locations();

        for path in &locations {
            if path.exists() {
                return Self::load_from_path(path);
            }
        }

        // No config file found, use defaults
        info!("No config file found, using defaults");
        let config = Config::default();
        config.validate()?;

        let (change_sender, _) = broadcast::channel(16);

        Ok(Arc::new(Self {
            config: ArcSwap::from_pointee(config),
            config_path: parking_lot::RwLock::new(None),
            change_sender,
        }))
    }

    /// Load configuration from a specific path
    pub fn load_from_path(path: &Path) -> Result<Arc<Self>> {
        info!("Loading config from: {}", path.display());
        let config = Config::load_from_path(path)?;
        config.validate()?;

        let (change_sender, _) = broadcast::channel(16);

        Ok(Arc::new(Self {
            config: ArcSwap::from_pointee(config),
            config_path: parking_lot::RwLock::new(Some(path.to_path_buf())),
            change_sender,
        }))
    }

    /// Create a SharedConfig from an existing Config
    pub fn from_config(config: Config) -> Result<Arc<Self>> {
        config.validate()?;

        let (change_sender, _) = broadcast::channel(16);

        Ok(Arc::new(Self {
            config: ArcSwap::from_pointee(config),
            config_path: parking_lot::RwLock::new(None),
            change_sender,
        }))
    }

    /// Get a read guard to the current configuration
    ///
    /// This is very cheap (just an atomic load) and the returned guard
    /// can be held for as long as needed. The configuration it points
    /// to will remain valid even if a reload happens.
    pub fn current(&self) -> Guard<Arc<Config>> {
        self.config.load()
    }

    /// Get a clone of the current configuration
    ///
    /// This is slightly more expensive than `current()` but returns
    /// an owned Arc that can be stored or passed around freely.
    pub fn get(&self) -> Arc<Config> {
        self.config.load_full()
    }

    /// Reload configuration from disk
    ///
    /// If the SharedConfig was created without a file path, this will
    /// search the default config locations. The new configuration is
    /// validated before being applied.
    ///
    /// Returns the new configuration on success, or an error if the
    /// file cannot be read or validation fails.
    pub fn reload(&self) -> Result<Arc<Config>> {
        let path = self.config_path.read().clone();

        let (new_config, source_path) = if let Some(ref path) = path {
            debug!("Reloading config from: {}", path.display());
            (Config::load_from_path(path)?, Some(path.clone()))
        } else {
            // Try to find a config file
            let locations = Config::config_locations();
            let mut found_config = None;
            let mut found_path = None;

            for location in &locations {
                if location.exists() {
                    debug!("Found config at: {}", location.display());
                    found_config = Some(Config::load_from_path(location)?);
                    found_path = Some(location.clone());
                    break;
                }
            }

            match found_config {
                Some(config) => (config, found_path),
                None => {
                    debug!("No config file found, using defaults");
                    (Config::default(), None)
                }
            }
        };

        // Validate before applying
        new_config.validate()?;

        let old_config = self.config.load_full();
        let new_config = Arc::new(new_config);

        // Atomically swap the configuration
        self.config.store(Arc::clone(&new_config));

        // Update the path if we found a new one
        if source_path.is_some() && path.is_none() {
            *self.config_path.write() = source_path.clone();
        }

        info!("Configuration reloaded successfully");

        // Broadcast the change event
        let event = ConfigChangeEvent {
            new_config: Arc::clone(&new_config),
            old_config,
            source_path,
        };

        // Ignore send errors (no receivers is fine)
        let _ = self.change_sender.send(event);

        Ok(new_config)
    }

    /// Subscribe to configuration change events
    ///
    /// Returns a receiver that will receive events whenever the
    /// configuration is reloaded.
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.change_sender.subscribe()
    }

    /// Get the path to the config file (if any)
    pub fn config_path(&self) -> Option<PathBuf> {
        self.config_path.read().clone()
    }

    /// Set the config file path
    ///
    /// This updates the path that will be used for future reloads.
    pub fn set_config_path(&self, path: Option<PathBuf>) {
        *self.config_path.write() = path;
    }
}

impl std::fmt::Debug for SharedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedConfig")
            .field("config", &*self.config.load())
            .field("config_path", &*self.config_path.read())
            .finish()
    }
}

/// Watches a configuration file for changes and triggers reloads
///
/// This struct monitors the config file path and can be used to
/// trigger reloads when the file changes.
pub struct ConfigWatcher {
    /// The shared config to reload
    shared_config: Arc<SharedConfig>,
    /// Path to watch
    watch_path: PathBuf,
}

impl ConfigWatcher {
    /// Create a new ConfigWatcher
    ///
    /// The watcher will monitor the specified path and can reload
    /// the shared config when triggered.
    pub fn new(shared_config: Arc<SharedConfig>, watch_path: PathBuf) -> Self {
        Self {
            shared_config,
            watch_path,
        }
    }

    /// Get the path being watched
    pub fn watch_path(&self) -> &Path {
        &self.watch_path
    }

    /// Trigger a reload of the configuration
    ///
    /// This should be called when the config file changes.
    /// Returns the new configuration on success.
    pub fn trigger_reload(&self) -> Result<Arc<Config>> {
        info!("Config file change detected, reloading...");

        // Update the path in shared config if different
        let current_path = self.shared_config.config_path();
        if current_path.as_ref() != Some(&self.watch_path) {
            self.shared_config.set_config_path(Some(self.watch_path.clone()));
        }

        match self.shared_config.reload() {
            Ok(config) => {
                info!("Config reloaded successfully");
                Ok(config)
            }
            Err(e) => {
                warn!("Failed to reload config: {}", e);
                Err(e)
            }
        }
    }

    /// Get a reference to the shared config
    pub fn shared_config(&self) -> &Arc<SharedConfig> {
        &self.shared_config
    }
}

impl std::fmt::Debug for ConfigWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigWatcher")
            .field("watch_path", &self.watch_path)
            .finish()
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Transport mode: stdio or http (future)
    pub transport: String,
    /// Log level
    pub log_level: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            transport: "stdio".to_string(),
            log_level: "info".to_string(),
        }
    }
}

/// Indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexingConfig {
    /// Maximum file size to index (in KB)
    pub max_file_size: u64,
    /// Patterns to ignore (in addition to .gitignore)
    pub ignore_patterns: Vec<String>,
    /// File types to include
    pub include_types: Vec<String>,
    /// Enable embedding generation
    pub enable_embeddings: bool,
    /// Path to embedding model
    pub embedding_model: Option<String>,
    /// Enable parallel indexing for improved performance
    pub enable_parallel: bool,
    /// Number of threads for parallel operations (0 = auto-detect based on CPU count)
    pub parallel_threads: usize,
    /// Batch size for database operations during parallel indexing
    pub batch_size: usize,
    /// Batch size for embedding generation during parallel indexing
    pub embedding_batch_size: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            max_file_size: 500,
            ignore_patterns: vec![
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "dist/**".to_string(),
                ".git/**".to_string(),
                "*.lock".to_string(),
                "__pycache__/**".to_string(),
                ".venv/**".to_string(),
                "venv/**".to_string(),
            ],
            include_types: vec![
                "rs".to_string(),
                "ts".to_string(),
                "tsx".to_string(),
                "js".to_string(),
                "jsx".to_string(),
                "py".to_string(),
                "go".to_string(),
                "java".to_string(),
                "md".to_string(),
                "json".to_string(),
                "toml".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
            ],
            enable_embeddings: true,
            embedding_model: None,
            enable_parallel: true,
            parallel_threads: 0, // 0 means auto-detect based on CPU count
            batch_size: 100,
            embedding_batch_size: 32,
        }
    }
}

/// File watching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WatchingConfig {
    /// Enable real-time file watching
    pub enabled: bool,
    /// Debounce interval for warm files (milliseconds)
    pub debounce_ms: u64,
    /// Hot file threshold (files accessed within N seconds are hot)
    pub hot_threshold_seconds: u64,
}

impl Default for WatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debounce_ms: 1000,
            hot_threshold_seconds: 300,
        }
    }
}

/// Context management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Default maximum tokens for context
    pub default_max_tokens: u32,
    /// Chunking strategy: semantic, recursive, fixed
    pub chunking_strategy: String,
    /// Chunk size in tokens
    pub chunk_size: u32,
    /// Chunk overlap percentage (0.0 - 1.0)
    pub chunk_overlap: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            default_max_tokens: 8000,
            chunking_strategy: "semantic".to_string(),
            chunk_size: 512,
            chunk_overlap: 0.1,
        }
    }
}

/// Git integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitConfig {
    /// Enable git history indexing
    pub enabled: bool,
    /// Maximum commits to index
    pub max_commits: u32,
    /// Include blame information
    pub include_blame: bool,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_commits: 1000,
            include_blame: true,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Maximum number of entries in the query embedding cache
    pub query_cache_capacity: usize,
    /// TTL for query cache in seconds
    pub query_cache_ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            query_cache_capacity: 1000,
            query_cache_ttl_seconds: 300,
        }
    }
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Default search mode: keyword, semantic, or hybrid
    pub default_mode: String,
    /// Minimum similarity threshold for semantic search (0.0 - 1.0)
    pub min_similarity: f32,
    /// Weight for keyword results in hybrid search (0.0 - 1.0)
    pub keyword_weight: f32,
    /// Weight for semantic results in hybrid search (0.0 - 1.0)
    pub semantic_weight: f32,
    /// RRF constant k (higher = more weight to lower-ranked results)
    pub rrf_k: f32,
    /// Default result limit
    pub default_limit: usize,
    /// Enable HNSW approximate nearest neighbor search
    pub use_hnsw: bool,
    /// HNSW ef_construction parameter (higher = better recall, slower build)
    pub hnsw_ef_construction: u32,
    /// HNSW ef_search parameter (higher = better recall, slower search)
    pub hnsw_ef_search: u32,
    /// HNSW m parameter (connections per layer)
    pub hnsw_m: u32,
    /// Minimum embeddings count to trigger HNSW (below this, brute-force is used)
    pub hnsw_min_embeddings: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_mode: "hybrid".to_string(),
            min_similarity: 0.3,
            keyword_weight: 0.5,
            semantic_weight: 0.5,
            rrf_k: 60.0,
            default_limit: 20,
            use_hnsw: true,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            hnsw_m: 16,
            hnsw_min_embeddings: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.transport, "stdio");
        assert_eq!(config.indexing.max_file_size, 500);
        assert!(config.watching.enabled);
        assert_eq!(config.context.default_max_tokens, 8000);
        assert!(config.git.enabled);
        assert!(config.cache.enabled);
    }

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig::default();
        assert_eq!(config.default_mode, "hybrid");
        assert_eq!(config.min_similarity, 0.3);
        assert_eq!(config.keyword_weight, 0.5);
        assert_eq!(config.semantic_weight, 0.5);
        assert_eq!(config.rrf_k, 60.0);
        assert_eq!(config.default_limit, 20);
        // HNSW config
        assert!(config.use_hnsw);
        assert_eq!(config.hnsw_ef_construction, 200);
        assert_eq!(config.hnsw_ef_search, 100);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_min_embeddings, 1000);
    }

    #[test]
    fn test_config_validation_default() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_transport() {
        let mut config = Config::default();
        config.server.transport = "invalid".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid transport"));
    }

    #[test]
    fn test_config_validation_invalid_log_level() {
        let mut config = Config::default();
        config.server.log_level = "invalid".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid log_level"));
    }

    #[test]
    fn test_config_validation_invalid_chunk_overlap() {
        let mut config = Config::default();
        config.context.chunk_overlap = 1.5;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid chunk_overlap"));
    }

    #[test]
    fn test_config_validation_invalid_chunking_strategy() {
        let mut config = Config::default();
        config.context.chunking_strategy = "invalid".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid chunking_strategy"));
    }

    #[test]
    fn test_config_validation_invalid_search_mode() {
        let mut config = Config::default();
        config.search.default_mode = "invalid".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid default_mode"));
    }

    #[test]
    fn test_config_validation_invalid_min_similarity() {
        let mut config = Config::default();
        config.search.min_similarity = -0.5;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid min_similarity"));
    }

    #[test]
    fn test_shared_config_from_default() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let config = shared.current();
        assert_eq!(config.server.transport, "stdio");
    }

    #[test]
    fn test_shared_config_current() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let config1 = shared.current();
        let config2 = shared.current();
        // Both should see the same config
        assert_eq!(config1.server.transport, config2.server.transport);
    }

    #[test]
    fn test_shared_config_get() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let config = shared.get();
        assert_eq!(config.server.transport, "stdio");
    }

    #[test]
    fn test_shared_config_load_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[server]
transport = "http"
log_level = "debug"

[indexing]
max_file_size = 1000
"#
        )
        .unwrap();

        let shared = SharedConfig::load_from_path(temp_file.path()).unwrap();
        let config = shared.current();
        assert_eq!(config.server.transport, "http");
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.indexing.max_file_size, 1000);
    }

    #[test]
    fn test_shared_config_reload() {
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

        let shared = SharedConfig::load_from_path(temp_file.path()).unwrap();
        assert_eq!(shared.current().server.log_level, "info");

        // Modify the file
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

        // Reload and verify
        shared.reload().unwrap();
        assert_eq!(shared.current().server.log_level, "debug");
    }

    #[test]
    fn test_shared_config_reload_invalid_rejected() {
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

        let shared = SharedConfig::load_from_path(temp_file.path()).unwrap();
        let original_level = shared.current().server.log_level.clone();

        // Write invalid config
        let path = temp_file.path().to_path_buf();
        std::fs::write(
            &path,
            r#"
[server]
transport = "invalid_transport"
log_level = "info"
"#,
        )
        .unwrap();

        // Reload should fail
        let result = shared.reload();
        assert!(result.is_err());

        // Original config should be preserved
        assert_eq!(shared.current().server.log_level, original_level);
    }

    #[tokio::test]
    async fn test_shared_config_subscribe() {
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

        let shared = SharedConfig::load_from_path(temp_file.path()).unwrap();
        let mut receiver = shared.subscribe();

        // Modify the file
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

        // Reload
        shared.reload().unwrap();

        // Should receive the event
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.old_config.server.log_level, "info");
        assert_eq!(event.new_config.server.log_level, "debug");
    }

    #[test]
    fn test_config_watcher_creation() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = ConfigWatcher::new(shared.clone(), PathBuf::from("/tmp/test.toml"));
        assert_eq!(watcher.watch_path(), Path::new("/tmp/test.toml"));
    }

    #[test]
    fn test_config_watcher_trigger_reload() {
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

        let shared = SharedConfig::load_from_path(temp_file.path()).unwrap();
        let watcher = ConfigWatcher::new(shared.clone(), temp_file.path().to_path_buf());

        // Modify the file
        let path = temp_file.path().to_path_buf();
        std::fs::write(
            &path,
            r#"
[server]
transport = "stdio"
log_level = "warn"
"#,
        )
        .unwrap();

        // Trigger reload via watcher
        watcher.trigger_reload().unwrap();
        assert_eq!(shared.current().server.log_level, "warn");
    }

    #[test]
    fn test_shared_config_debug() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let debug_str = format!("{:?}", shared);
        assert!(debug_str.contains("SharedConfig"));
    }

    #[test]
    fn test_config_watcher_debug() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        let watcher = ConfigWatcher::new(shared, PathBuf::from("/tmp/test.toml"));
        let debug_str = format!("{:?}", watcher);
        assert!(debug_str.contains("ConfigWatcher"));
        assert!(debug_str.contains("/tmp/test.toml"));
    }

    #[test]
    fn test_shared_config_set_path() {
        let shared = SharedConfig::from_config(Config::default()).unwrap();
        assert!(shared.config_path().is_none());

        shared.set_config_path(Some(PathBuf::from("/tmp/new_config.toml")));
        assert_eq!(
            shared.config_path(),
            Some(PathBuf::from("/tmp/new_config.toml"))
        );

        shared.set_config_path(None);
        assert!(shared.config_path().is_none());
    }

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert!(config.enabled);
        assert_eq!(config.embedding_cache_capacity, 10000);
        assert!(config.embedding_cache_ttl_seconds.is_none());
        assert_eq!(config.query_cache_ttl_seconds, 300);
    }

    #[test]
    fn test_config_includes_cache() {
        let config = Config::default();
        assert!(config.cache.enabled);
        assert_eq!(config.cache.embedding_cache_capacity, 10000);
    }

    #[test]
    fn test_cache_config_toml_parsing() {
        let toml_str = r#"
            [cache]
            enabled = false
            embedding_cache_capacity = 5000
            embedding_cache_ttl_seconds = 600
            query_cache_ttl_seconds = 120
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(!config.cache.enabled);
        assert_eq!(config.cache.embedding_cache_capacity, 5000);
        assert_eq!(config.cache.embedding_cache_ttl_seconds, Some(600));
        assert_eq!(config.cache.query_cache_ttl_seconds, 120);
    }

    #[test]
    fn test_cache_config_partial_toml_parsing() {
        let toml_str = r#"
            [cache]
            embedding_cache_capacity = 20000
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        // Specified value
        assert_eq!(config.cache.embedding_cache_capacity, 20000);
        // Defaults for unspecified values
        assert!(config.cache.enabled);
        assert!(config.cache.embedding_cache_ttl_seconds.is_none());
        assert_eq!(config.cache.query_cache_ttl_seconds, 300);
    }

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert!(config.enabled);
        assert_eq!(config.query_cache_capacity, 1000);
        assert_eq!(config.query_cache_ttl_seconds, 300);
    }
}
