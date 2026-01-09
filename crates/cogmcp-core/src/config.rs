//! Configuration management for CogMCP

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    pub pool: PoolConfig,
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
            pool: PoolConfig::default(),
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

    /// Find which config file would be loaded
    pub fn find_config_path() -> Option<PathBuf> {
        Self::config_locations().into_iter().find(|p| p.exists())
    }

    /// Generate a summary of key config settings
    pub fn summary(&self) -> String {
        format!(
            "Server: transport={}, log_level={}\n\
             Indexing: max_file_size={}KB, embeddings={}\n\
             Search: mode={}, min_similarity={:.2}\n\
             Watching: enabled={}, debounce={}ms",
            self.server.transport,
            self.server.log_level,
            self.indexing.max_file_size,
            self.indexing.enable_embeddings,
            self.search.default_mode,
            self.search.min_similarity,
            self.watching.enabled,
            self.watching.debounce_ms,
        )
    }
}

/// Thread-safe shared configuration that can be reloaded at runtime
#[derive(Clone)]
pub struct SharedConfig {
    inner: Arc<RwLock<Config>>,
    source_path: Arc<RwLock<Option<PathBuf>>>,
}

impl SharedConfig {
    /// Create a new SharedConfig from an existing Config
    pub fn new(config: Config) -> Self {
        Self {
            inner: Arc::new(RwLock::new(config)),
            source_path: Arc::new(RwLock::new(Config::find_config_path())),
        }
    }

    /// Load SharedConfig from default locations
    pub fn load() -> Result<Self> {
        let source_path = Config::find_config_path();
        let config = Config::load()?;
        Ok(Self {
            inner: Arc::new(RwLock::new(config)),
            source_path: Arc::new(RwLock::new(source_path)),
        })
    }

    /// Get a read-only clone of the current config
    pub fn get(&self) -> Config {
        self.inner.read().clone()
    }

    /// Get the path of the config file being used (if any)
    pub fn source_path(&self) -> Option<PathBuf> {
        self.source_path.read().clone()
    }

    /// Reload configuration from disk
    ///
    /// Returns a summary of what changed or an error
    pub fn reload(&self) -> Result<ReloadResult> {
        let source_path = Config::find_config_path();
        let new_config = Config::load()?;

        // Update source path
        *self.source_path.write() = source_path.clone();

        // Update config
        *self.inner.write() = new_config.clone();

        Ok(ReloadResult {
            source_path,
            config_summary: new_config.summary(),
        })
    }

    /// Validate configuration from disk without applying
    ///
    /// Returns the config summary if valid, or an error
    pub fn validate(&self) -> Result<ReloadResult> {
        let source_path = Config::find_config_path();
        let new_config = Config::load()?;

        Ok(ReloadResult {
            source_path,
            config_summary: new_config.summary(),
        })
    }
}

impl Default for SharedConfig {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl std::fmt::Debug for SharedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedConfig")
            .field("config", &*self.inner.read())
            .field("source_path", &*self.source_path.read())
            .finish()
    }
}

/// Result of a config reload operation
#[derive(Debug, Clone)]
pub struct ReloadResult {
    /// Path to the config file that was loaded (None if using defaults)
    pub source_path: Option<PathBuf>,
    /// Summary of the loaded configuration
    pub config_summary: String,
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
    /// Enable parallel file processing during indexing
    pub enable_parallel_indexing: bool,
    /// Number of parallel workers for file processing (0 = auto-detect based on CPU cores)
    pub parallel_workers: usize,
    /// Number of chunks to accumulate before generating embeddings in batch
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
            enable_parallel_indexing: true,
            parallel_workers: 0, // 0 = auto-detect based on CPU cores
            embedding_batch_size: 64,
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

/// Connection pool configuration
///
/// Controls the behavior of database connection pooling for SQLite.
/// These settings affect performance and resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    ///
    /// Higher values allow more concurrent database access but use more resources.
    /// Default: 10
    pub max_connections: u32,

    /// Minimum number of idle connections to maintain
    ///
    /// Keeping idle connections reduces latency for new requests.
    /// Set to None to disable idle connection maintenance.
    /// Default: Some(2)
    pub min_idle: Option<u32>,

    /// Connection timeout in seconds
    ///
    /// Maximum time to wait when acquiring a connection from the pool.
    /// Default: 30
    pub connection_timeout_secs: u64,

    /// Idle timeout in seconds
    ///
    /// Connections idle longer than this will be closed.
    /// Set to None to disable idle timeout (connections never expire).
    /// Default: Some(600) (10 minutes)
    pub idle_timeout_secs: Option<u64>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_idle: Some(2),
            connection_timeout_secs: 30,
            idle_timeout_secs: Some(600),
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
    /// Cache configuration
    pub cache: CacheConfig,
}

/// Cache configuration for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Maximum number of cached search results
    pub max_entries: usize,
    /// TTL for cached results in seconds
    pub result_ttl_secs: u64,
    /// TTL for cached embeddings in seconds
    pub embedding_ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 1000,
            result_ttl_secs: 300,      // 5 minutes
            embedding_ttl_secs: 3600,  // 1 hour
        }
    }
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
            cache: CacheConfig::default(),
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
        assert!(config.cache.enabled);
        assert_eq!(config.cache.max_entries, 1000);
        assert_eq!(config.cache.result_ttl_secs, 300);
        assert_eq!(config.cache.embedding_ttl_secs, 3600);
    }
}
