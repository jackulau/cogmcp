//! Embedding inference using ONNX Runtime

use std::fs;
use std::path::Path;

use ndarray_017::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tracing::{debug, info};

use cogmcp_core::{Error, Result};

use crate::model::ModelConfig;
use crate::tokenizer::Tokenizer;

/// Embedding engine for generating text embeddings using ONNX Runtime
#[derive(Debug)]
pub struct EmbeddingEngine {
    /// Model configuration
    config: ModelConfig,
    /// ONNX Runtime session
    session: Option<Session>,
    /// Tokenizer for text preprocessing
    tokenizer: Option<Tokenizer>,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the given configuration
    ///
    /// This will load the ONNX model and tokenizer from the paths specified in the config.
    pub fn new(config: ModelConfig) -> Result<Self> {
        if config.model_path.is_empty() {
            return Ok(Self {
                config,
                session: None,
                tokenizer: None,
            });
        }

        let model_path = Path::new(&config.model_path);
        if !model_path.exists() {
            return Err(Error::Embedding(format!(
                "Model file not found: {}",
                config.model_path
            )));
        }

        let tokenizer_path = Path::new(&config.tokenizer_path);
        if !tokenizer_path.exists() {
            return Err(Error::Embedding(format!(
                "Tokenizer file not found: {}",
                config.tokenizer_path
            )));
        }

        info!("Loading ONNX model from {}", config.model_path);

        // Read model file into memory
        let model_bytes = fs::read(&config.model_path).map_err(|e| {
            Error::Embedding(format!("Failed to read model file: {}", e))
        })?;

        // Initialize ONNX Runtime session
        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Embedding(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| Error::Embedding(format!("Failed to set thread count: {}", e)))?
            .commit_from_memory(&model_bytes)
            .map_err(|e| Error::Embedding(format!("Failed to load ONNX model: {}", e)))?;

        debug!("ONNX model loaded successfully");

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)?
            .with_max_length(config.max_length);

        debug!("Tokenizer loaded successfully");

        Ok(Self {
            config,
            session: Some(session),
            tokenizer: Some(tokenizer),
        })
    }

    /// Create an embedding engine without a model (for testing)
    pub fn without_model() -> Self {
        Self {
            config: ModelConfig::default(),
            session: None,
            tokenizer: None,
        }
    }

    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        self.session.is_some() && self.tokenizer.is_some()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Generate an embedding for a single text
    ///
    /// Returns a 384-dimensional vector for all-MiniLM-L6-v2
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // First, tokenize the text using the tokenizer
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Embedding("Tokenizer not loaded".into())
        })?;

        let encoded = tokenizer.encode(text)?;
        let seq_len = encoded.input_ids.len();
        let attention_mask_clone = encoded.attention_mask.clone();

        // Create input tensors with shape [1, seq_len]
        let input_ids: Array2<i64> =
            Array2::from_shape_vec((1, seq_len), encoded.input_ids).map_err(|e| {
                Error::Embedding(format!("Failed to create input_ids tensor: {}", e))
            })?;

        let attention_mask: Array2<i64> =
            Array2::from_shape_vec((1, seq_len), encoded.attention_mask).map_err(|e| {
                Error::Embedding(format!("Failed to create attention_mask tensor: {}", e))
            })?;

        let token_type_ids: Array2<i64> =
            Array2::from_shape_vec((1, seq_len), encoded.token_type_ids).map_err(|e| {
                Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e))
            })?;

        // Create Tensor values for ort using ndarray
        let input_ids_tensor = Tensor::from_array(input_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask)
            .map_err(|e| Error::Embedding(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Run inference and extract data in a block to limit borrows
        let (hidden_dim, raw_data) = {
            let session = self.session.as_mut().ok_or_else(|| {
                Error::Embedding("Model not loaded. Call ensure_model_available() first.".into())
            })?;

            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor, token_type_ids_tensor])
                .map_err(|e| Error::Embedding(format!("ONNX inference failed: {}", e)))?;

            // Extract the sentence embedding from the output
            let output_value = outputs.iter().next()
                .ok_or_else(|| Error::Embedding("No output tensor found".into()))?;

            let (shape, data) = output_value.1
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Embedding(format!("Failed to extract output tensor: {}", e)))?;

            // Get dimensions: should be [1, seq_len, hidden_dim]
            if shape.len() != 3 {
                return Err(Error::Embedding(format!(
                    "Expected 3D output tensor, got {}D with shape {:?}",
                    shape.len(),
                    &**shape
                )));
            }

            let hidden_dim = shape[2] as usize;
            // Copy the data to owned Vec before dropping outputs
            (hidden_dim, data.to_vec())
        };

        // Apply mean pooling over the sequence dimension
        let embedding = Self::mean_pooling_from_flat_static(
            &raw_data,
            &attention_mask_clone,
            seq_len,
            hidden_dim,
        )?;

        // Normalize the embedding (L2 normalization)
        let normalized = Self::l2_normalize_static(&embedding);

        Ok(normalized)
    }

    /// Generate embeddings for multiple texts (batch processing)
    ///
    /// More efficient than calling embed() multiple times for large batches
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Process sequentially for simplicity and to avoid complex batch handling
        // True batch processing could be added for larger batches
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Apply mean pooling to get sentence embeddings from flattened output
    fn mean_pooling_from_flat_static(
        embeddings: &[f32],
        attention_mask: &[i64],
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        // Sum embeddings weighted by attention mask
        let mut sum = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;

        for (i, mask_val) in attention_mask.iter().enumerate().take(seq_len) {
            if *mask_val == 1 {
                let start = i * hidden_dim;
                let end = start + hidden_dim;
                if end <= embeddings.len() {
                    for (j, val) in sum.iter_mut().enumerate() {
                        *val += embeddings[start + j];
                    }
                    count += 1.0;
                }
            }
        }

        // Compute mean
        if count > 0.0 {
            for val in &mut sum {
                *val /= count;
            }
        }

        Ok(sum)
    }

    /// L2 normalize a vector
    fn l2_normalize_static(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
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
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = EmbeddingEngine::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = EmbeddingEngine::cosine_similarity(&a, &b);
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let similarity = EmbeddingEngine::cosine_similarity(&a, &b);
        assert!((similarity - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = EmbeddingEngine::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = EmbeddingEngine::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_without_model() {
        let engine = EmbeddingEngine::without_model();
        assert!(!engine.is_loaded());
        assert_eq!(engine.embedding_dim(), 384);
    }

    #[test]
    fn test_embed_without_model_returns_error() {
        let mut engine = EmbeddingEngine::without_model();
        let result = engine.embed("test text");
        assert!(result.is_err());
    }

    #[test]
    fn test_l2_normalize() {
        let vec = vec![3.0, 4.0]; // 3-4-5 triangle
        let normalized = EmbeddingEngine::l2_normalize_static(&vec);

        // Should have unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Check values
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let vec = vec![0.0, 0.0, 0.0];
        let normalized = EmbeddingEngine::l2_normalize_static(&vec);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mean_pooling() {
        // 2 tokens, 3 hidden dims
        // Token 0: [1.0, 2.0, 3.0]
        // Token 1: [4.0, 5.0, 6.0]
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attention_mask = vec![1, 1];

        let result = EmbeddingEngine::mean_pooling_from_flat_static(&embeddings, &attention_mask, 2, 3).unwrap();

        // Mean should be [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pooling_with_mask() {
        // 2 tokens, 3 hidden dims, but only first token is active
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attention_mask = vec![1, 0]; // Only first token

        let result = EmbeddingEngine::mean_pooling_from_flat_static(&embeddings, &attention_mask, 2, 3).unwrap();

        // Mean should be [1.0, 2.0, 3.0] (only first token)
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }
}
