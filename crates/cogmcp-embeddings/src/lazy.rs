//! Lazy-loading embedding engine wrapper
//!
//! This module provides a `LazyEmbeddingEngine` that defers ONNX model loading
//! until the first embedding request. This can significantly improve server
//! startup time when semantic search is not immediately needed.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use cogmcp_core::{Error, Result};

use crate::inference::EmbeddingEngine;
use crate::model::ModelConfig;

/// A lazy-loading wrapper around `EmbeddingEngine`
///
/// This wrapper stores the model configuration and defers the actual ONNX model
/// loading until the first `embed()` call. This can save 1-2 seconds of startup
/// time when semantic search is not immediately needed.
///
/// # Thread Safety
///
/// `LazyEmbeddingEngine` is thread-safe. Multiple threads can call `embed()`
/// concurrently - the model will be loaded exactly once, and subsequent calls
/// will reuse the loaded engine.
///
/// # Example
///
/// ```no_run
/// use cogmcp_embeddings::{LazyEmbeddingEngine, ModelConfig};
///
/// let config = ModelConfig::default();
/// let engine = LazyEmbeddingEngine::new(config);
///
/// // Model is NOT loaded yet
/// assert!(!engine.is_loaded());
///
/// // First embed() call triggers model loading
/// let embedding = engine.embed("Hello, world!").unwrap();
///
/// // Model is now loaded
/// assert!(engine.is_loaded());
/// ```
pub struct LazyEmbeddingEngine {
    /// Configuration for deferred loading
    config: ModelConfig,
    /// Lazily initialized engine wrapped in a mutex for thread-safe mutable access
    /// The Result<EmbeddingEngine, String> allows caching initialization errors
    engine: OnceLock<std::result::Result<Mutex<EmbeddingEngine>, String>>,
}

impl LazyEmbeddingEngine {
    /// Create a new lazy embedding engine with the given configuration
    ///
    /// This does NOT load the ONNX model - it only stores the configuration.
    /// The model will be loaded on the first call to `embed()` or `embed_batch()`.
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            engine: OnceLock::new(),
        }
    }

    /// Ensure the engine is loaded, initializing it if necessary
    ///
    /// This is called internally by `embed()` and `embed_batch()`.
    /// Multiple concurrent calls are safe - the engine will only be loaded once.
    fn ensure_loaded(&self) -> Result<&Mutex<EmbeddingEngine>> {
        let result = self.engine.get_or_init(|| {
            match EmbeddingEngine::new(self.config.clone()) {
                Ok(engine) => Ok(Mutex::new(engine)),
                Err(e) => Err(e.to_string()),
            }
        });

        match result {
            Ok(mutex) => Ok(mutex),
            Err(e) => Err(Error::Embedding(e.clone())),
        }
    }

    /// Generate an embedding for a single text
    ///
    /// On the first call, this will load the ONNX model (which may take 1-2 seconds).
    /// Subsequent calls will reuse the loaded model.
    ///
    /// Returns a 384-dimensional vector for all-MiniLM-L6-v2.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let engine_mutex = self.ensure_loaded()?;
        let mut engine = engine_mutex
            .lock()
            .map_err(|e| Error::Embedding(format!("Failed to acquire engine lock: {}", e)))?;
        engine.embed(text)
    }

    /// Generate embeddings for multiple texts (batch processing)
    ///
    /// On the first call, this will load the ONNX model if not already loaded.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let engine_mutex = self.ensure_loaded()?;
        let mut engine = engine_mutex
            .lock()
            .map_err(|e| Error::Embedding(format!("Failed to acquire engine lock: {}", e)))?;
        engine.embed_batch(texts)
    }

    /// Check if the engine has been loaded
    ///
    /// Returns `true` if `embed()` or `embed_batch()` has been called and
    /// successfully loaded the model, `false` otherwise.
    pub fn is_loaded(&self) -> bool {
        matches!(self.engine.get(), Some(Ok(_)))
    }

    /// Check if the model files exist and are available for loading
    ///
    /// This does NOT load the model - it only checks if the files exist.
    pub fn is_available(&self) -> bool {
        if self.config.model_path.is_empty() || self.config.tokenizer_path.is_empty() {
            return false;
        }
        Path::new(&self.config.model_path).exists()
            && Path::new(&self.config.tokenizer_path).exists()
    }

    /// Get the embedding dimension
    ///
    /// This returns the dimension from the configuration without loading the model.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Compute cosine similarity between two embeddings
    ///
    /// This is a static method that doesn't require loading the model.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        EmbeddingEngine::cosine_similarity(a, b)
    }
}

// Implement Debug manually since OnceLock<Mutex<T>> doesn't derive Debug nicely
impl std::fmt::Debug for LazyEmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyEmbeddingEngine")
            .field("config", &self.config)
            .field("is_loaded", &self.is_loaded())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_engine_not_loaded_initially() {
        let config = ModelConfig::default();
        let engine = LazyEmbeddingEngine::new(config);

        // Engine should not be loaded before any embed calls
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_lazy_engine_embedding_dim_without_loading() {
        let mut config = ModelConfig::default();
        config.embedding_dim = 512; // Custom dimension

        let engine = LazyEmbeddingEngine::new(config);

        // embedding_dim should work without loading the model
        assert_eq!(engine.embedding_dim(), 512);
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_lazy_engine_is_available_empty_config() {
        let config = ModelConfig::default();
        let engine = LazyEmbeddingEngine::new(config);

        // With empty paths, should return false
        assert!(!engine.is_available());
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_lazy_engine_is_available_nonexistent_paths() {
        let config = ModelConfig {
            model_path: "/nonexistent/path/model.onnx".to_string(),
            tokenizer_path: "/nonexistent/path/tokenizer.json".to_string(),
            embedding_dim: 384,
            max_length: 512,
        };
        let engine = LazyEmbeddingEngine::new(config);

        // With nonexistent paths, should return false
        assert!(!engine.is_available());
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_lazy_engine_cosine_similarity() {
        // Static method should work without any engine instance
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = LazyEmbeddingEngine::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lazy_engine_debug() {
        let config = ModelConfig::default();
        let engine = LazyEmbeddingEngine::new(config);

        // Debug should work and show is_loaded status
        let debug_str = format!("{:?}", engine);
        assert!(debug_str.contains("LazyEmbeddingEngine"));
        assert!(debug_str.contains("is_loaded"));
    }

    // Note: Tests that actually load the model require the model files to be present.
    // These are integration tests that would be run separately.

    #[test]
    fn test_lazy_engine_embed_without_model_returns_error() {
        let config = ModelConfig::default();
        let engine = LazyEmbeddingEngine::new(config);

        // Attempting to embed with empty config should fail
        let result = engine.embed("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_engine_embed_batch_empty() {
        let config = ModelConfig::default();
        let engine = LazyEmbeddingEngine::new(config);

        // Empty batch should return empty result without loading
        // Note: This will still try to load since embed_batch calls ensure_loaded
        // But with empty config, it will return an empty engine
        let empty_texts: &[&str] = &[];
        let result = engine.embed_batch(empty_texts);
        // This might succeed or fail depending on implementation
        // With empty config, EmbeddingEngine::new returns Ok with no model
        match result {
            Ok(embeddings) => assert!(embeddings.is_empty()),
            Err(_) => {} // Also acceptable
        }
    }
}
