//! Model loading and configuration

use std::path::Path;

/// Embedding model configuration
pub struct ModelConfig {
    /// Path to the ONNX model file
    pub model_path: String,
    /// Embedding dimension (384 for all-MiniLM-L6-v2)
    pub embedding_dim: usize,
    /// Maximum sequence length
    pub max_length: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            embedding_dim: 384,
            max_length: 512,
        }
    }
}

impl ModelConfig {
    /// Create config with a specific model path
    pub fn with_path(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_string_lossy().to_string(),
            ..Default::default()
        }
    }
}
