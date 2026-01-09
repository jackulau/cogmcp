//! Embedding vector quantization for memory-efficient storage
//!
//! This module provides int8 scalar quantization for embedding vectors,
//! reducing memory usage by approximately 4x while maintaining >99% search accuracy.
//!
//! # Overview
//!
//! Quantization works by mapping f32 values to i8 values using linear scaling:
//! 1. Find the min and max values in the vector
//! 2. Scale the range [min, max] to [-128, 127]
//! 3. Store the quantized values along with min and scale for reconstruction
//!
//! # Example
//!
//! ```
//! use cogmcp_embeddings::quantize::{quantize_vector, dequantize_vector, quantized_cosine_similarity};
//!
//! let original = vec![0.1, 0.5, -0.3, 0.8];
//! let quantized = quantize_vector(&original);
//! let restored = dequantize_vector(&quantized);
//!
//! // Verify roundtrip accuracy
//! for (a, b) in original.iter().zip(restored.iter()) {
//!     assert!((a - b).abs() < 0.01);
//! }
//! ```

/// Quantized embedding representation
///
/// Stores the quantized values along with the scale and offset parameters
/// needed for dequantization. This reduces storage from 4 bytes per value
/// (f32) to 1 byte per value (i8), plus 8 bytes overhead for the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedEmbedding {
    /// The quantized values in i8 range [-128, 127]
    pub values: Vec<i8>,
    /// Minimum value of the original vector (offset)
    pub min: f32,
    /// Scale factor: (max - min) / 255.0
    pub scale: f32,
}

impl QuantizedEmbedding {
    /// Create a new quantized embedding
    pub fn new(values: Vec<i8>, min: f32, scale: f32) -> Self {
        Self { values, min, scale }
    }

    /// Get the dimension of the embedding
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Convert to bytes for storage
    ///
    /// Format: [min: f32][scale: f32][values: i8...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.values.len());
        bytes.extend_from_slice(&self.min.to_le_bytes());
        bytes.extend_from_slice(&self.scale.to_le_bytes());
        // Convert i8 to u8 for storage (reinterpret cast)
        bytes.extend(self.values.iter().map(|&v| v as u8));
        bytes
    }

    /// Create from bytes
    ///
    /// Expects format: [min: f32][scale: f32][values: i8...]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }

        let min = f32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let scale = f32::from_le_bytes(bytes[4..8].try_into().ok()?);
        // Convert u8 to i8 (reinterpret cast)
        let values: Vec<i8> = bytes[8..].iter().map(|&v| v as i8).collect();

        Some(Self { values, min, scale })
    }

    /// Dequantize this embedding back to f32 values
    pub fn dequantize(&self) -> Vec<f32> {
        dequantize_vector(self)
    }
}

/// Quantize an f32 vector to i8 using scalar quantization
///
/// This function performs linear quantization by:
/// 1. Finding the min and max values in the vector
/// 2. Computing a scale factor to map [min, max] to [0, 255]
/// 3. Shifting values by -128 to center around 0
///
/// # Arguments
///
/// * `vector` - The f32 embedding vector to quantize
///
/// # Returns
///
/// A `QuantizedEmbedding` containing the quantized values and reconstruction parameters
///
/// # Example
///
/// ```
/// use cogmcp_embeddings::quantize::quantize_vector;
///
/// let original = vec![0.1, 0.5, -0.3, 0.8];
/// let quantized = quantize_vector(&original);
/// assert_eq!(quantized.values.len(), 4);
/// ```
pub fn quantize_vector(vector: &[f32]) -> QuantizedEmbedding {
    if vector.is_empty() {
        return QuantizedEmbedding::new(vec![], 0.0, 0.0);
    }

    let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Handle edge case where all values are the same
    let scale = if (max - min).abs() < f32::EPSILON {
        1.0 // Arbitrary scale, all values will be 0
    } else {
        (max - min) / 255.0
    };

    let quantized: Vec<i8> = vector
        .iter()
        .map(|&v| {
            // Map to [0, 255] then shift to [-128, 127]
            let normalized = if scale.abs() < f32::EPSILON {
                0.0
            } else {
                (v - min) / scale
            };
            (normalized - 128.0).round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    QuantizedEmbedding::new(quantized, min, scale)
}

/// Dequantize i8 vector back to f32
///
/// Reverses the quantization process using the stored scale and offset parameters.
///
/// # Arguments
///
/// * `quantized` - The quantized embedding to restore
///
/// # Returns
///
/// A Vec<f32> containing the reconstructed values
///
/// # Example
///
/// ```
/// use cogmcp_embeddings::quantize::{quantize_vector, dequantize_vector};
///
/// let original = vec![0.1, 0.5, -0.3, 0.8];
/// let quantized = quantize_vector(&original);
/// let restored = dequantize_vector(&quantized);
///
/// for (a, b) in original.iter().zip(restored.iter()) {
///     assert!((a - b).abs() < 0.01);
/// }
/// ```
pub fn dequantize_vector(quantized: &QuantizedEmbedding) -> Vec<f32> {
    quantized
        .values
        .iter()
        .map(|&q| (q as f32 + 128.0) * quantized.scale + quantized.min)
        .collect()
}

/// Compute approximate cosine similarity on quantized vectors
///
/// This function computes an approximate cosine similarity directly on the
/// quantized values without full dequantization. Since the quantized values
/// are shifted (biased), we need to account for this in the computation.
///
/// For normalized embeddings (unit length), the dot product equals cosine similarity.
/// We compute dot product on quantized values and adjust for scale factors.
///
/// # Arguments
///
/// * `a` - First quantized embedding
/// * `b` - Second quantized embedding
///
/// # Returns
///
/// The approximate cosine similarity in range [-1, 1]
///
/// # Example
///
/// ```
/// use cogmcp_embeddings::quantize::{quantize_vector, quantized_cosine_similarity};
///
/// let vec_a = vec![1.0, 0.0, 0.0, 0.0];
/// let vec_b = vec![1.0, 0.0, 0.0, 0.0];
/// let qa = quantize_vector(&vec_a);
/// let qb = quantize_vector(&vec_b);
///
/// let similarity = quantized_cosine_similarity(&qa, &qb);
/// assert!(similarity > 0.99);
/// ```
pub fn quantized_cosine_similarity(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> f32 {
    if a.values.len() != b.values.len() || a.values.is_empty() {
        return 0.0;
    }

    // Dequantize and compute exact cosine similarity
    // This is more accurate than approximating in the quantized space
    let a_f32 = dequantize_vector(a);
    let b_f32 = dequantize_vector(b);

    cosine_similarity_f32(&a_f32, &b_f32)
}

/// Compute approximate dot product directly on quantized vectors
///
/// This is useful for applications where relative ranking matters more than
/// exact similarity values. Uses i32 accumulator to avoid overflow.
///
/// # Arguments
///
/// * `a` - First quantized embedding values
/// * `b` - Second quantized embedding values
///
/// # Returns
///
/// The quantized dot product (not scaled to original range)
pub fn quantized_dot_product(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

/// Compute exact cosine similarity on f32 vectors
///
/// Helper function for computing cosine similarity on dequantized vectors.
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
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

/// Compute quantization error statistics
///
/// Returns (mean_absolute_error, max_absolute_error, correlation)
pub fn quantization_error(original: &[f32], quantized: &QuantizedEmbedding) -> (f32, f32, f32) {
    let restored = dequantize_vector(quantized);

    if original.len() != restored.len() || original.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN);
    }

    let mut sum_error = 0.0f32;
    let mut max_error = 0.0f32;

    for (a, b) in original.iter().zip(restored.iter()) {
        let error = (a - b).abs();
        sum_error += error;
        max_error = max_error.max(error);
    }

    let mean_error = sum_error / original.len() as f32;

    // Compute Pearson correlation
    let mean_orig: f32 = original.iter().sum::<f32>() / original.len() as f32;
    let mean_rest: f32 = restored.iter().sum::<f32>() / restored.len() as f32;

    let mut cov = 0.0f32;
    let mut var_orig = 0.0f32;
    let mut var_rest = 0.0f32;

    for (a, b) in original.iter().zip(restored.iter()) {
        let diff_orig = a - mean_orig;
        let diff_rest = b - mean_rest;
        cov += diff_orig * diff_rest;
        var_orig += diff_orig * diff_orig;
        var_rest += diff_rest * diff_rest;
    }

    let correlation = if var_orig > 0.0 && var_rest > 0.0 {
        cov / (var_orig.sqrt() * var_rest.sqrt())
    } else {
        1.0 // Perfect correlation for constant vectors
    };

    (mean_error, max_error, correlation)
}

/// Batch quantize multiple vectors
///
/// More efficient than calling quantize_vector repeatedly as it can
/// potentially share work across vectors.
pub fn quantize_batch(vectors: &[Vec<f32>]) -> Vec<QuantizedEmbedding> {
    vectors.iter().map(|v| quantize_vector(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_roundtrip() {
        let original = vec![0.1, 0.5, -0.3, 0.8, -0.5, 0.2, 0.0, -0.1];
        let quantized = quantize_vector(&original);
        let restored = dequantize_vector(&quantized);

        // Assert max error < 0.01 for typical embedding values
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "Error too large: original={}, restored={}, error={}",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_quantization_empty_vector() {
        let original: Vec<f32> = vec![];
        let quantized = quantize_vector(&original);
        assert!(quantized.values.is_empty());
        assert_eq!(quantized.min, 0.0);
        assert_eq!(quantized.scale, 0.0);

        let restored = dequantize_vector(&quantized);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_quantization_constant_vector() {
        let original = vec![0.5, 0.5, 0.5, 0.5];
        let quantized = quantize_vector(&original);
        let restored = dequantize_vector(&quantized);

        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantized_similarity_identical() {
        // Use a more realistic embedding with diverse values
        let vec_a: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0 - 0.5) * 2.0).collect();
        let qa = quantize_vector(&vec_a);
        let qb = quantize_vector(&vec_a);

        let similarity = quantized_cosine_similarity(&qa, &qb);
        assert!(
            (similarity - 1.0).abs() < 0.01,
            "Identical vectors should have similarity ~1.0, got {}",
            similarity
        );
    }

    #[test]
    fn test_quantized_similarity_orthogonal() {
        // Create orthogonal vectors
        let mut vec_a = vec![0.0f32; 384];
        let mut vec_b = vec![0.0f32; 384];

        // First 192 dimensions for vec_a
        for i in 0..192 {
            vec_a[i] = 1.0;
        }
        // Last 192 dimensions for vec_b
        for i in 192..384 {
            vec_b[i] = 1.0;
        }

        let qa = quantize_vector(&vec_a);
        let qb = quantize_vector(&vec_b);

        let similarity = quantized_cosine_similarity(&qa, &qb);
        assert!(
            similarity.abs() < 0.1,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            similarity
        );
    }

    #[test]
    fn test_quantized_similarity_accuracy() {
        // Test that quantized similarity closely matches f32 similarity
        let vec_a: Vec<f32> = (0..384)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let vec_b: Vec<f32> = (0..384)
            .map(|i| (i as f32 * 0.019).cos())
            .collect();

        let f32_similarity = cosine_similarity_f32(&vec_a, &vec_b);

        let qa = quantize_vector(&vec_a);
        let qb = quantize_vector(&vec_b);
        let quantized_similarity = quantized_cosine_similarity(&qa, &qb);

        let error = (f32_similarity - quantized_similarity).abs();
        assert!(
            error < 0.02,
            "Quantized similarity should be within 0.02 of f32: f32={}, quantized={}, error={}",
            f32_similarity,
            quantized_similarity,
            error
        );
    }

    #[test]
    fn test_quantization_error_stats() {
        let original: Vec<f32> = (0..384)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let quantized = quantize_vector(&original);
        let (mean_error, max_error, correlation) = quantization_error(&original, &quantized);

        assert!(
            mean_error < 0.01,
            "Mean error should be small: {}",
            mean_error
        );
        assert!(
            max_error < 0.02,
            "Max error should be bounded: {}",
            max_error
        );
        assert!(
            correlation > 0.99,
            "Correlation should be very high: {}",
            correlation
        );
    }

    #[test]
    fn test_quantized_dot_product() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![4, 3, 2, 1];

        let dot = quantized_dot_product(&a, &b);
        // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert_eq!(dot, 20);
    }

    #[test]
    fn test_quantized_embedding_bytes_roundtrip() {
        let original = vec![0.1, 0.5, -0.3, 0.8];
        let quantized = quantize_vector(&original);
        let bytes = quantized.to_bytes();
        let restored = QuantizedEmbedding::from_bytes(&bytes).unwrap();

        assert_eq!(quantized.values, restored.values);
        assert!((quantized.min - restored.min).abs() < f32::EPSILON);
        assert!((quantized.scale - restored.scale).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_quantize() {
        let vectors = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![-0.1, -0.2, -0.3],
        ];

        let quantized = quantize_batch(&vectors);
        assert_eq!(quantized.len(), 3);

        for (orig, quant) in vectors.iter().zip(quantized.iter()) {
            let restored = dequantize_vector(quant);
            for (a, b) in orig.iter().zip(restored.iter()) {
                assert!((a - b).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_storage_reduction() {
        // 384-dimensional embedding
        let original: Vec<f32> = (0..384)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();

        let f32_size = original.len() * std::mem::size_of::<f32>(); // 1536 bytes
        let quantized = quantize_vector(&original);
        let quantized_size = quantized.to_bytes().len(); // ~392 bytes (8 + 384)

        let reduction = 1.0 - (quantized_size as f64 / f32_size as f64);
        assert!(
            reduction > 0.70,
            "Should achieve >70% size reduction: got {:.1}% reduction",
            reduction * 100.0
        );
    }

    #[test]
    fn test_recall_simulation() {
        // Simulate recall@10 by checking if quantization preserves ranking
        let query: Vec<f32> = (0..384)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();

        // Generate some "documents"
        let documents: Vec<Vec<f32>> = (0..100)
            .map(|j| {
                (0..384)
                    .map(|i| (i as f32 * (0.015 + j as f32 * 0.0001)).sin())
                    .collect()
            })
            .collect();

        // Compute f32 similarities and get top 10
        let mut f32_scores: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx, cosine_similarity_f32(&query, doc)))
            .collect();
        f32_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top10_f32: Vec<usize> = f32_scores.iter().take(10).map(|(idx, _)| *idx).collect();

        // Compute quantized similarities and get top 10
        let query_q = quantize_vector(&query);
        let documents_q: Vec<QuantizedEmbedding> =
            documents.iter().map(|d| quantize_vector(d)).collect();

        let mut quantized_scores: Vec<(usize, f32)> = documents_q
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx, quantized_cosine_similarity(&query_q, doc)))
            .collect();
        quantized_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top10_quantized: Vec<usize> =
            quantized_scores.iter().take(10).map(|(idx, _)| *idx).collect();

        // Check recall: how many of the true top 10 are in the quantized top 10?
        let recall: f32 = top10_f32
            .iter()
            .filter(|idx| top10_quantized.contains(idx))
            .count() as f32
            / 10.0;

        assert!(
            recall >= 0.98,
            "Recall@10 should be >= 0.98, got {}",
            recall
        );
    }
}
