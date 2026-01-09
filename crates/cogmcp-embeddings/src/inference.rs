//! Embedding inference using ONNX Runtime

use std::fs;
use std::path::Path;

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tracing::{debug, info};

use cogmcp_core::{Error, Result};

use crate::model::ModelConfig;
use crate::tokenizer::Tokenizer;

/// Default batch size for batch inference operations
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Embedding engine for generating text embeddings using ONNX Runtime
#[derive(Debug)]
pub struct EmbeddingEngine {
    /// Model configuration
    config: ModelConfig,
    /// ONNX Runtime session
    session: Option<Session>,
    /// Tokenizer for text preprocessing
    tokenizer: Option<Tokenizer>,
    /// Batch size for batch inference operations
    batch_size: usize,
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
                batch_size: DEFAULT_BATCH_SIZE,
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
            batch_size: DEFAULT_BATCH_SIZE,
        })
    }

    /// Create a new embedding engine with a custom batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Get the current batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set the batch size for batch inference operations
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size.max(1);
    }

    /// Create an embedding engine without a model (for testing)
    pub fn without_model() -> Self {
        Self {
            config: ModelConfig::default(),
            session: None,
            tokenizer: None,
            batch_size: DEFAULT_BATCH_SIZE,
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

    /// Generate embeddings for multiple texts using true batch inference
    ///
    /// This processes all texts in a single ONNX forward pass, which is
    /// significantly more efficient than calling embed() multiple times.
    /// The embeddings are identical to sequential processing within floating
    /// point tolerance.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // For single text, use the simpler single-text path
        if texts.len() == 1 {
            return Ok(vec![self.embed(texts[0])?]);
        }

        // Tokenize all texts at once using batch encoding
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Embedding("Tokenizer not loaded".into())
        })?;

        let batch = tokenizer.encode_batch(texts)?;
        let batch_size = batch.batch_size;
        let seq_len = batch.seq_length;

        if batch_size == 0 || seq_len == 0 {
            return Ok(vec![]);
        }

        // Create input tensors with shape [batch_size, seq_len]
        let input_ids: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch.input_ids).map_err(|e| {
                Error::Embedding(format!("Failed to create input_ids tensor: {}", e))
            })?;

        let attention_mask: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch.attention_mask.clone())
                .map_err(|e| {
                    Error::Embedding(format!("Failed to create attention_mask tensor: {}", e))
                })?;

        let token_type_ids: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch.token_type_ids).map_err(|e| {
                Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e))
            })?;

        // Create Tensor values for ort
        let input_ids_tensor = Tensor::from_array(input_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask)
            .map_err(|e| Error::Embedding(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Run inference and extract data
        let (hidden_dim, raw_data) = {
            let session = self.session.as_mut().ok_or_else(|| {
                Error::Embedding("Model not loaded. Call ensure_model_available() first.".into())
            })?;

            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor, token_type_ids_tensor])
                .map_err(|e| Error::Embedding(format!("ONNX batch inference failed: {}", e)))?;

            // Extract the embeddings from the output
            let output_value = outputs.iter().next()
                .ok_or_else(|| Error::Embedding("No output tensor found".into()))?;

            let (shape, data) = output_value.1
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Embedding(format!("Failed to extract output tensor: {}", e)))?;

            // Get dimensions: should be [batch_size, seq_len, hidden_dim]
            if shape.len() != 3 {
                return Err(Error::Embedding(format!(
                    "Expected 3D output tensor, got {}D with shape {:?}",
                    shape.len(),
                    &**shape
                )));
            }

            let actual_batch_size = shape[0] as usize;
            let actual_seq_len = shape[1] as usize;
            let hidden_dim = shape[2] as usize;

            if actual_batch_size != batch_size {
                return Err(Error::Embedding(format!(
                    "Batch size mismatch: expected {}, got {}",
                    batch_size, actual_batch_size
                )));
            }

            if actual_seq_len != seq_len {
                return Err(Error::Embedding(format!(
                    "Sequence length mismatch: expected {}, got {}",
                    seq_len, actual_seq_len
                )));
            }

            (hidden_dim, data.to_vec())
        };

        // Apply mean pooling and L2 normalization to each sequence in the batch
        let mut embeddings = Vec::with_capacity(batch_size);
        let sequence_elements = seq_len * hidden_dim;

        for i in 0..batch_size {
            // Extract this sequence's embeddings from the flattened data
            let start_idx = i * sequence_elements;
            let end_idx = start_idx + sequence_elements;
            let sequence_data = &raw_data[start_idx..end_idx];

            // Get this sequence's attention mask
            let mask_start = i * seq_len;
            let mask_end = mask_start + seq_len;
            let sequence_mask = &batch.attention_mask[mask_start..mask_end];

            // Apply mean pooling over the sequence dimension
            let embedding = Self::mean_pooling_from_flat_static(
                sequence_data,
                sequence_mask,
                seq_len,
                hidden_dim,
            )?;

            // L2 normalize the embedding
            let normalized = Self::l2_normalize_static(&embedding);
            embeddings.push(normalized);
        }

        Ok(embeddings)
    }

    /// Generate embeddings for multiple texts with automatic chunking
    ///
    /// This method splits large inputs into optimal batch sizes and processes
    /// them efficiently. Use this for very large batches where memory might
    /// be a concern.
    pub fn embed_batch_chunked(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = self.batch_size;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in chunks of batch_size
        for chunk in texts.chunks(batch_size) {
            let chunk_embeddings = self.embed_batch(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
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

    #[test]
    fn test_embed_batch_empty() {
        let mut engine = EmbeddingEngine::without_model();
        let result = engine.embed_batch(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_embed_batch_optimized_empty() {
        let mut engine = EmbeddingEngine::without_model();
        let result = engine.embed_batch_optimized(&[], 32);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_embed_large_batch_empty() {
        let mut engine = EmbeddingEngine::without_model();
        let result = engine.embed_large_batch(&[], 32);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}

/// Integration tests that require model files
/// These tests verify batch vs sequential equivalence
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::model::ModelManager;

    /// Helper to create an engine with model if available
    fn create_test_engine() -> Option<EmbeddingEngine> {
        let manager = match ModelManager::new() {
            Ok(m) => m,
            Err(_) => return None,
        };

        if !manager.is_model_available() {
            return None;
        }

        let config = manager.get_config();
        EmbeddingEngine::new(config).ok()
    }

    #[test]
    #[ignore = "Requires model files to be downloaded"]
    fn test_batch_vs_sequential_equivalence() {
        let mut engine = match create_test_engine() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: model not available");
                return;
            }
        };

        let texts = vec![
            "hello world",
            "test embedding",
            "rust code",
            "machine learning",
            "natural language processing",
        ];

        // Generate embeddings sequentially
        let sequential: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| engine.embed(t).unwrap())
            .collect();

        // Generate embeddings in batch
        let batched = engine.embed_batch(&texts).unwrap();

        // Verify same number of results
        assert_eq!(sequential.len(), batched.len());

        // Verify vectors are identical within f32 precision
        for (i, (seq, bat)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(seq.len(), bat.len(), "Dimension mismatch at index {}", i);

            for (j, (s, b)) in seq.iter().zip(bat.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-5,
                    "Value mismatch at text {} dim {}: sequential={}, batched={}",
                    i, j, s, b
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires model files to be downloaded"]
    fn test_batch_size_variations() {
        let mut engine = match create_test_engine() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: model not available");
                return;
            }
        };

        let texts: Vec<&str> = (0..10)
            .map(|i| match i % 5 {
                0 => "hello world",
                1 => "test embedding model",
                2 => "rust programming language",
                3 => "machine learning algorithms",
                _ => "natural language processing tasks",
            })
            .collect();

        // Test different batch sizes
        let batch_sizes = [1, 2, 4, 8, 16, 32];

        let baseline = engine.embed_batch_optimized(&texts, 1).unwrap();

        for batch_size in batch_sizes {
            let result = engine.embed_batch_optimized(&texts, batch_size).unwrap();

            assert_eq!(baseline.len(), result.len(), "Length mismatch for batch_size={}", batch_size);

            for (i, (base, res)) in baseline.iter().zip(result.iter()).enumerate() {
                for (j, (b, r)) in base.iter().zip(res.iter()).enumerate() {
                    assert!(
                        (b - r).abs() < 1e-5,
                        "Value mismatch at batch_size={}, text={}, dim={}: base={}, result={}",
                        batch_size, i, j, b, r
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "Requires model files to be downloaded"]
    fn test_large_batch_parallel_equivalence() {
        let mut engine = match create_test_engine() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: model not available");
                return;
            }
        };

        let sample_texts: Vec<String> = (0..100)
            .map(|i| format!("sample text number {} for embedding test", i))
            .collect();

        let texts: Vec<&str> = sample_texts.iter().map(|s| s.as_str()).collect();

        // Get baseline with sequential batch processing
        let sequential = engine.embed_batch_optimized(&texts, 32).unwrap();

        // Get results with large batch processing
        let parallel = engine.embed_large_batch(&texts, 32).unwrap();

        // Verify same number of results
        assert_eq!(sequential.len(), parallel.len());

        // Verify vectors are identical within f32 precision
        for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(seq.len(), par.len(), "Dimension mismatch at index {}", i);

            for (j, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
                assert!(
                    (s - p).abs() < 1e-5,
                    "Value mismatch at text {} dim {}: sequential={}, parallel={}",
                    i, j, s, p
                );
            }
        }
    }

    #[test]
    #[ignore = "Performance test - run manually"]
    fn bench_batch_throughput() {
        let mut engine = match create_test_engine() {
            Some(e) => e,
            None => {
                eprintln!("Skipping benchmark: model not available");
                return;
            }
        };

        let sample_texts: Vec<String> = (0..1000)
            .map(|i| format!("sample text number {} for embedding test", i))
            .collect();

        let texts: Vec<&str> = sample_texts.iter().map(|s| s.as_str()).collect();

        // Time sequential processing (batch_size=1)
        let start_sequential = std::time::Instant::now();
        let _sequential_results = engine.embed_batch_optimized(&texts, 1).unwrap();
        let sequential_duration = start_sequential.elapsed();

        // Time batched processing (batch_size=32)
        let start_batched = std::time::Instant::now();
        let _batched_results = engine.embed_batch_optimized(&texts, 32).unwrap();
        let batched_duration = start_batched.elapsed();

        let speedup = sequential_duration.as_secs_f64() / batched_duration.as_secs_f64();

        println!("\n=== Batch Inference Benchmark ===");
        println!("Texts processed: {}", texts.len());
        println!("Sequential (batch_size=1): {:?}", sequential_duration);
        println!("Batched (batch_size=32): {:?}", batched_duration);
        println!("Speedup: {:.2}x", speedup);

        // Assert >2x improvement
        assert!(
            speedup > 2.0,
            "Expected >2x speedup, got {:.2}x (sequential: {:?}, batched: {:?})",
            speedup,
            sequential_duration,
            batched_duration
        );
    }

    #[test]
    #[ignore = "Performance test - run manually"]
    fn bench_various_batch_sizes() {
        let mut engine = match create_test_engine() {
            Some(e) => e,
            None => {
                eprintln!("Skipping benchmark: model not available");
                return;
            }
        };

        let sample_texts: Vec<String> = (0..500)
            .map(|i| format!("sample text number {} for embedding test", i))
            .collect();

        let texts: Vec<&str> = sample_texts.iter().map(|s| s.as_str()).collect();

        let batch_sizes = [1, 4, 8, 16, 32, 64];

        println!("\n=== Batch Size Comparison ===");
        println!("Texts processed: {}", texts.len());

        let mut baseline_duration = None;

        for batch_size in batch_sizes {
            let start = std::time::Instant::now();
            let _results = engine.embed_batch_optimized(&texts, batch_size).unwrap();
            let duration = start.elapsed();

            let speedup = if let Some(baseline) = baseline_duration {
                baseline / duration.as_secs_f64()
            } else {
                baseline_duration = Some(duration.as_secs_f64());
                1.0
            };

            println!(
                "batch_size={:2}: {:?} ({:.2}x vs batch_size=1)",
                batch_size, duration, speedup
            );
        }
    }
}
