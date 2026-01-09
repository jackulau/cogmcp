//! Embedding inference using ONNX Runtime

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tracing::{debug, info};

use cogmcp_core::{Error, Result};

use crate::model::{ModelConfig, MAX_BATCH_SIZE};
use crate::tokenizer::{BatchTokenizedInput, Tokenizer};

/// Progress information for batch embedding operations
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Total number of items to process
    pub total_items: usize,
    /// Number of items processed so far
    pub processed_items: usize,
    /// Current batch number (1-indexed)
    pub current_batch: usize,
    /// Total number of batches
    pub total_batches: usize,
    /// Time elapsed since start
    pub elapsed_time: Duration,
    /// Processing rate in items per second
    pub items_per_second: f64,
}

impl BatchProgress {
    /// Create a new BatchProgress
    pub fn new(
        total_items: usize,
        processed_items: usize,
        current_batch: usize,
        total_batches: usize,
        elapsed_time: Duration,
    ) -> Self {
        let items_per_second = if elapsed_time.as_secs_f64() > 0.0 {
            processed_items as f64 / elapsed_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            total_items,
            processed_items,
            current_batch,
            total_batches,
            elapsed_time,
            items_per_second,
        }
    }

    /// Get estimated time remaining
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.items_per_second > 0.0 && self.processed_items < self.total_items {
            let remaining_items = self.total_items - self.processed_items;
            let remaining_secs = remaining_items as f64 / self.items_per_second;
            Some(Duration::from_secs_f64(remaining_secs))
        } else {
            None
        }
    }

    /// Get completion percentage
    pub fn percentage(&self) -> f64 {
        if self.total_items > 0 {
            (self.processed_items as f64 / self.total_items as f64) * 100.0
        } else {
            100.0
        }
    }
}

/// Thread-safe metrics for embedding performance tracking
#[derive(Debug, Default)]
pub struct EmbeddingMetrics {
    /// Total number of embeddings generated
    total_embeddings: AtomicU64,
    /// Total inference time in microseconds
    total_inference_time_us: AtomicU64,
    /// Total number of batches processed
    total_batches: AtomicU64,
    /// Total items processed across all batches
    total_items_in_batches: AtomicU64,
}

impl EmbeddingMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed embedding operation
    pub fn record_embedding(&self, inference_time: Duration) {
        self.total_embeddings.fetch_add(1, Ordering::Relaxed);
        self.total_inference_time_us
            .fetch_add(inference_time.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a completed batch operation
    pub fn record_batch(&self, batch_size: usize, inference_time: Duration) {
        self.total_embeddings
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_inference_time_us
            .fetch_add(inference_time.as_micros() as u64, Ordering::Relaxed);
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        self.total_items_in_batches
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    /// Get total embeddings generated
    pub fn total_embeddings_generated(&self) -> u64 {
        self.total_embeddings.load(Ordering::Relaxed)
    }

    /// Get total inference time
    pub fn total_inference_time(&self) -> Duration {
        Duration::from_micros(self.total_inference_time_us.load(Ordering::Relaxed))
    }

    /// Get average batch size
    pub fn average_batch_size(&self) -> f64 {
        let batches = self.total_batches.load(Ordering::Relaxed);
        if batches > 0 {
            self.total_items_in_batches.load(Ordering::Relaxed) as f64 / batches as f64
        } else {
            0.0
        }
    }

    /// Get throughput in embeddings per second
    pub fn throughput_per_second(&self) -> f64 {
        let time = self.total_inference_time();
        if time.as_secs_f64() > 0.0 {
            self.total_embeddings_generated() as f64 / time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.total_embeddings.store(0, Ordering::Relaxed);
        self.total_inference_time_us.store(0, Ordering::Relaxed);
        self.total_batches.store(0, Ordering::Relaxed);
        self.total_items_in_batches.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_embeddings_generated: self.total_embeddings_generated(),
            total_inference_time: self.total_inference_time(),
            average_batch_size: self.average_batch_size(),
            throughput_per_second: self.throughput_per_second(),
        }
    }
}

/// A point-in-time snapshot of embedding metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total number of embeddings generated
    pub total_embeddings_generated: u64,
    /// Total inference time
    pub total_inference_time: Duration,
    /// Average batch size
    pub average_batch_size: f64,
    /// Throughput in embeddings per second
    pub throughput_per_second: f64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Embeddings: {}, Total time: {:.2}s, Avg batch: {:.1}, Throughput: {:.1}/s",
            self.total_embeddings_generated,
            self.total_inference_time.as_secs_f64(),
            self.average_batch_size,
            self.throughput_per_second
        )
    }
}

/// Embedding engine for generating text embeddings using ONNX Runtime
pub struct EmbeddingEngine {
    /// Model configuration
    config: ModelConfig,
    /// ONNX Runtime session
    session: Option<Session>,
    /// Tokenizer for text preprocessing
    tokenizer: Option<Tokenizer>,
    /// Performance metrics
    metrics: EmbeddingMetrics,
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngine")
            .field("config", &self.config)
            .field("session", &self.session.is_some())
            .field("tokenizer", &self.tokenizer.is_some())
            .finish()
    }
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
                metrics: EmbeddingMetrics::new(),
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
            metrics: EmbeddingMetrics::new(),
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
            metrics: EmbeddingMetrics::new(),
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
        let start = Instant::now();
        let result = self.embed_internal(text);
        let elapsed = start.elapsed();

        if result.is_ok() {
            self.metrics.record_embedding(elapsed);
        }

        result
    }

    /// Internal embedding generation without metrics recording
    fn embed_internal(&mut self, text: &str) -> Result<Vec<f32>> {
        // First, tokenize the text using the tokenizer
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Embedding("Tokenizer not loaded".into())
        })?;

        let encoded = tokenizer.encode(text)?;
        let seq_len = encoded.input_ids.len();
        let attention_mask_clone = encoded.attention_mask.clone();

        // Create input tensors with shape [1, seq_len] using (shape, Vec) tuple format
        let shape = [1, seq_len];
        let input_ids_tensor = Tensor::from_array((shape, encoded.input_ids))
            .map_err(|e| Error::Embedding(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array((shape, encoded.attention_mask))
            .map_err(|e| Error::Embedding(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids_tensor = Tensor::from_array((shape, encoded.token_type_ids))
            .map_err(|e| Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Run inference and extract data in a block to limit borrows
        let (hidden_dim, raw_data) = {
            let session = self.session.as_mut().ok_or_else(|| {
                Error::Embedding("Model not loaded. Call ensure_model_available() first.".into())
            })?;

            let inputs = ort::inputs![input_ids_tensor, attention_mask_tensor, token_type_ids_tensor]
                .map_err(|e| Error::Embedding(format!("Failed to create inputs: {}", e)))?;

            let outputs = session
                .run(inputs)
                .map_err(|e| Error::Embedding(format!("ONNX inference failed: {}", e)))?;

            // Extract the sentence embedding from the output
            let output_value = outputs.iter().next()
                .ok_or_else(|| Error::Embedding("No output tensor found".into()))?;

            let tensor_view = output_value.1
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Embedding(format!("Failed to extract output tensor: {}", e)))?;
            let shape = tensor_view.shape().to_vec();

            let shape = tensor_view.shape();

            // Get dimensions: should be [1, seq_len, hidden_dim]
            let shape = tensor_view.shape();
            if shape.len() != 3 {
                return Err(Error::Embedding(format!(
                    "Expected 3D output tensor, got {}D with shape {:?}",
                    shape.len(),
                    shape
                )));
            }

            let hidden_dim = shape[2];
            // Copy the data to owned Vec before dropping outputs
            let raw_data: Vec<f32> = tensor_view.iter().cloned().collect();
            (hidden_dim, raw_data)
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

    /// Generate embeddings for multiple texts using true ONNX batch inference
    ///
    /// This method uses ONNX Runtime's batch inference capabilities for better performance.
    /// Large batches are automatically chunked based on the configured batch size.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_progress(texts, |_| {})
    }

    /// Generate embeddings for multiple texts with progress callback
    ///
    /// The callback receives a `BatchProgress` after each batch is processed,
    /// allowing for progress reporting during long-running operations.
    ///
    /// # Arguments
    /// * `texts` - The texts to generate embeddings for
    /// * `on_progress` - Callback function invoked after each batch
    ///
    /// # Example
    /// ```ignore
    /// engine.embed_batch_with_progress(&texts, |progress| {
    ///     println!("{:.1}% complete ({}/{})",
    ///         progress.percentage(),
    ///         progress.processed_items,
    ///         progress.total_items);
    /// })?;
    /// ```
    pub fn embed_batch_with_progress<F>(
        &mut self,
        texts: &[&str],
        mut on_progress: F,
    ) -> Result<Vec<Vec<f32>>>
    where
        F: FnMut(BatchProgress),
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = self.config.batch_size.min(MAX_BATCH_SIZE);

        // If batch is small enough, process in one go
        if texts.len() <= batch_size {
            return self.embed_batch_chunk(texts);
        }

        // Process in chunks for large batches
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch_size) {
            let chunk_embeddings = self.embed_batch_chunk(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Process a single batch chunk through ONNX inference
    fn embed_batch_chunk(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // For single text, use the optimized single-item path
        if texts.len() == 1 {
            return Ok(vec![self.embed(texts[0])?]);
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Embedding("Tokenizer not loaded".into())
        })?;

        // Tokenize all texts with padding to same length
        let batch_input = tokenizer.encode_batch(texts)?;

        // Prepare batch tensors
        let (input_ids_tensor, attention_mask_tensor, token_type_ids_tensor) =
            self.prepare_batch_inputs(&batch_input)?;

        // Run inference and extract data
        let (hidden_dim, raw_data, batch_size, seq_len) = {
            let session = self.session.as_mut().ok_or_else(|| {
                Error::Embedding("Model not loaded. Call ensure_model_available() first.".into())
            })?;

            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor, token_type_ids_tensor])
                .map_err(|e| Error::Embedding(format!("ONNX inference failed: {}", e)))?;

            let output_value = outputs.iter().next()
                .ok_or_else(|| Error::Embedding("No output tensor found".into()))?;

            let (shape, data) = output_value.1
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Embedding(format!("Failed to extract output tensor: {}", e)))?;

            // Shape should be [batch_size, seq_len, hidden_dim]
            if shape.len() != 3 {
                return Err(Error::Embedding(format!(
                    "Expected 3D output tensor, got {}D with shape {:?}",
                    shape.len(),
                    &**shape
                )));
            }

            let batch_size = shape[0] as usize;
            let seq_len = shape[1] as usize;
            let hidden_dim = shape[2] as usize;

            (hidden_dim, data.to_vec(), batch_size, seq_len)
        };

        // Extract and normalize embeddings for each item in the batch
        let mut embeddings = Vec::with_capacity(batch_size);
        let stride = seq_len * hidden_dim;

        for i in 0..batch_size {
            let start = i * stride;
            let end = start + stride;
            let item_data = &raw_data[start..end];

            // Get attention mask for this item
            let mask_start = i * batch_input.seq_length;
            let mask_end = mask_start + batch_input.seq_length;
            let attention_mask = &batch_input.attention_mask[mask_start..mask_end];

            // Apply mean pooling
            let embedding = Self::mean_pooling_from_flat_static(
                item_data,
                attention_mask,
                seq_len,
                hidden_dim,
            )?;

            // Normalize the embedding
            let normalized = Self::l2_normalize_static(&embedding);
            embeddings.push(normalized);
        }

        Ok(embeddings)
    }

    /// Prepare batch input tensors for ONNX inference
    fn prepare_batch_inputs(
        &self,
        batch_input: &BatchTokenizedInput,
    ) -> Result<(Tensor<i64>, Tensor<i64>, Tensor<i64>)> {
        let batch_size = batch_input.batch_size;
        let seq_len = batch_input.seq_length;

        // Create 2D arrays with shape [batch_size, seq_len]
        let input_ids: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch_input.input_ids.clone())
                .map_err(|e| Error::Embedding(format!("Failed to create input_ids tensor: {}", e)))?;

        let attention_mask: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch_input.attention_mask.clone())
                .map_err(|e| Error::Embedding(format!("Failed to create attention_mask tensor: {}", e)))?;

        let token_type_ids: Array2<i64> =
            Array2::from_shape_vec((batch_size, seq_len), batch_input.token_type_ids.clone())
                .map_err(|e| Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Create ONNX tensors
        let input_ids_tensor = Tensor::from_array(input_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask)
            .map_err(|e| Error::Embedding(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)
            .map_err(|e| Error::Embedding(format!("Failed to create token_type_ids tensor: {}", e)))?;

        Ok((input_ids_tensor, attention_mask_tensor, token_type_ids_tensor))
    }

    /// Get the configured batch size
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
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
    fn test_embed_batch_without_model_returns_error() {
        let mut engine = EmbeddingEngine::without_model();
        let result = engine.embed_batch(&["test text"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_size_getter() {
        let engine = EmbeddingEngine::without_model();
        assert_eq!(engine.batch_size(), crate::model::DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn test_batch_size_respects_config() {
        let config = ModelConfig::default().with_batch_size(16);
        let engine = EmbeddingEngine {
            config,
            session: None,
            tokenizer: None,
        };
        assert_eq!(engine.batch_size(), 16);
    }

    #[test]
    fn test_batch_size_capped_at_max() {
        let config = ModelConfig::default().with_batch_size(999);
        let engine = EmbeddingEngine {
            config,
            session: None,
            tokenizer: None,
        };
        assert_eq!(engine.batch_size(), MAX_BATCH_SIZE);
    }
}
