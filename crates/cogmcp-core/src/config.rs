//! Configuration management for CogMCP

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Main configuration for CogMCP
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub indexing: IndexingConfig,
    pub watching: WatchingConfig,
    pub context: ContextConfig,
    pub git: GitConfig,
    pub search: SearchConfig,
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

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.transport, "stdio");
        assert_eq!(config.indexing.max_file_size, 500);
        assert!(config.watching.enabled);
        assert_eq!(config.context.default_max_tokens, 8000);
        assert!(config.git.enabled);
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
}
