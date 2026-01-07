//! Embedding inference using ONNX Runtime

use cogmcp_core::{Error, Result};

use crate::model::ModelConfig;

/// Embedding engine for generating text embeddings
pub struct EmbeddingEngine {
    config: ModelConfig,
    // session: Option<ort::Session>, // TODO: Initialize when model is available
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        // TODO: Load the ONNX model when available
        Ok(Self {
            config,
            // session: None,
        })
    }

    /// Create an embedding engine without a model (for testing)
    pub fn without_model() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }

    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        !self.config.model_path.is_empty()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Generate an embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if !self.is_loaded() {
            return Err(Error::Embedding("Model not loaded".into()));
        }

        // TODO: Implement actual inference
        // For now, return a placeholder
        let _ = text;
        Ok(vec![0.0; self.config.embedding_dim])
    }

    /// Generate embeddings for multiple texts (batch)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Compute cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((EmbeddingEngine::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((EmbeddingEngine::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
}
