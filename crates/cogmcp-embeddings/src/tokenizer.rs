//! Tokenization utilities for text preprocessing

use std::fmt;
use std::path::Path;

use ndarray::Array2;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::debug;

use cogmcp_core::{Error, Result};

/// Tokenizer wrapper for preparing text inputs
pub struct Tokenizer {
    inner: HfTokenizer,
    max_length: usize,
    padding_token_id: i64,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Load a tokenizer from a JSON file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading tokenizer from {:?}", path);

        let inner = HfTokenizer::from_file(path).map_err(|e| {
            Error::Embedding(format!("Failed to load tokenizer: {}", e))
        })?;

        // Get padding token ID from tokenizer config, default to 0 if not found
        let padding_token_id = inner
            .get_padding()
            .and_then(|p| inner.token_to_id(&p.pad_token))
            .map(|id| id as i64)
            .unwrap_or(0);

        debug!("Padding token ID: {}", padding_token_id);

        Ok(Self {
            inner,
            max_length: 512,
            padding_token_id,
        })
    }

    /// Set the maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Get the maximum sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get the padding token ID
    pub fn padding_token_id(&self) -> i64 {
        self.padding_token_id
    }

    /// Tokenize a single text and return input tensors
    pub fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| Error::Embedding(format!("Tokenization failed: {}", e)))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let mut token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        // Truncate if necessary
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
            token_type_ids.truncate(self.max_length);
        }

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    /// Tokenize a batch of texts and return padded input tensors
    pub fn encode_batch(&self, texts: &[&str]) -> Result<BatchTokenizedInput> {
        if texts.is_empty() {
            return Ok(BatchTokenizedInput {
                input_ids: vec![],
                attention_mask: vec![],
                token_type_ids: vec![],
                batch_size: 0,
                max_length: 0,
            });
        }

        // Encode all texts
        let encodings: Vec<TokenizedInput> = texts
            .iter()
            .map(|text| self.encode(text))
            .collect::<Result<Vec<_>>>()?;

        // Find the maximum length in this batch (capped by tokenizer's max_length)
        let padded_length = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let batch_size = texts.len();

        // Pad all sequences to the same length
        let mut input_ids = Vec::with_capacity(batch_size * padded_length);
        let mut attention_mask = Vec::with_capacity(batch_size * padded_length);
        let mut token_type_ids = Vec::with_capacity(batch_size * padded_length);

        for encoding in encodings {
            // Add actual tokens
            let len = encoding.input_ids.len().min(padded_length);
            input_ids.extend_from_slice(&encoding.input_ids[..len]);
            attention_mask.extend_from_slice(&encoding.attention_mask[..len]);
            token_type_ids.extend_from_slice(&encoding.token_type_ids[..len]);

            // Pad to padded_length using padding token ID
            let padding_len = padded_length - len;
            if padding_len > 0 {
                input_ids.extend(std::iter::repeat_n(self.padding_token_id, padding_len));
                attention_mask.extend(std::iter::repeat_n(0i64, padding_len));
                token_type_ids.extend(std::iter::repeat_n(0i64, padding_len));
            }
        }

        Ok(BatchTokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            max_length: padded_length,
        })
    }
}

/// Tokenized input for a single text
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    /// Token IDs
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<i64>,
    /// Token type IDs (for sentence pair tasks)
    pub token_type_ids: Vec<i64>,
}

impl TokenizedInput {
    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if the input is empty
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

/// Batch tokenized input for multiple texts
#[derive(Debug, Clone)]
pub struct BatchTokenizedInput {
    /// Flattened input IDs [batch_size * max_length]
    pub input_ids: Vec<i64>,
    /// Flattened attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<i64>,
    /// Flattened token type IDs
    pub token_type_ids: Vec<i64>,
    /// Number of texts in the batch
    pub batch_size: usize,
    /// Padded sequence length (all sequences are padded to this length)
    pub max_length: usize,
}

impl BatchTokenizedInput {
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Convert to ONNX-compatible input tensors
    ///
    /// Returns a tuple of (input_ids, attention_mask, token_type_ids) as 2D arrays
    /// with shape [batch_size, max_length].
    pub fn to_onnx_inputs(&self) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>)> {
        if self.is_empty() {
            return Ok((
                Array2::from_shape_vec((0, 0), vec![]).map_err(|e| {
                    Error::Embedding(format!("Failed to create empty input_ids array: {}", e))
                })?,
                Array2::from_shape_vec((0, 0), vec![]).map_err(|e| {
                    Error::Embedding(format!("Failed to create empty attention_mask array: {}", e))
                })?,
                Array2::from_shape_vec((0, 0), vec![]).map_err(|e| {
                    Error::Embedding(format!("Failed to create empty token_type_ids array: {}", e))
                })?,
            ));
        }

        let shape = (self.batch_size, self.max_length);

        let input_ids = Array2::from_shape_vec(shape, self.input_ids.clone()).map_err(|e| {
            Error::Embedding(format!(
                "Failed to create input_ids tensor: {} (expected shape {:?}, got {} elements)",
                e,
                shape,
                self.input_ids.len()
            ))
        })?;

        let attention_mask =
            Array2::from_shape_vec(shape, self.attention_mask.clone()).map_err(|e| {
                Error::Embedding(format!(
                    "Failed to create attention_mask tensor: {} (expected shape {:?}, got {} elements)",
                    e,
                    shape,
                    self.attention_mask.len()
                ))
            })?;

        let token_type_ids =
            Array2::from_shape_vec(shape, self.token_type_ids.clone()).map_err(|e| {
                Error::Embedding(format!(
                    "Failed to create token_type_ids tensor: {} (expected shape {:?}, got {} elements)",
                    e,
                    shape,
                    self.token_type_ids.len()
                ))
            })?;

        Ok((input_ids, attention_mask, token_type_ids))
    }

    /// Get individual attention masks for each sequence in the batch
    ///
    /// Returns a Vec of Vec<i64>, where each inner Vec is the attention mask
    /// for a single sequence in the batch.
    pub fn attention_masks(&self) -> Vec<Vec<i64>> {
        if self.is_empty() {
            return vec![];
        }

        self.attention_mask
            .chunks(self.max_length)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the tokenizer file to exist
    // In practice, tests should use a mock tokenizer or skip if file not available

    #[test]
    fn test_tokenized_input_len() {
        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        assert_eq!(input.len(), 3);
        assert!(!input.is_empty());
    }

    #[test]
    fn test_batch_tokenized_input_empty() {
        let batch = BatchTokenizedInput {
            input_ids: vec![],
            attention_mask: vec![],
            token_type_ids: vec![],
            batch_size: 0,
            max_length: 0,
        };
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_to_onnx_inputs() {
        // Create a batch with 2 sequences, each of length 3
        let batch = BatchTokenizedInput {
            input_ids: vec![1, 2, 3, 4, 5, 6],
            attention_mask: vec![1, 1, 1, 1, 1, 0],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
            batch_size: 2,
            max_length: 3,
        };

        let (input_ids, attention_mask, token_type_ids) = batch.to_onnx_inputs().unwrap();

        // Check shapes
        assert_eq!(input_ids.shape(), &[2, 3]);
        assert_eq!(attention_mask.shape(), &[2, 3]);
        assert_eq!(token_type_ids.shape(), &[2, 3]);

        // Check values for first sequence
        assert_eq!(input_ids[[0, 0]], 1);
        assert_eq!(input_ids[[0, 1]], 2);
        assert_eq!(input_ids[[0, 2]], 3);

        // Check values for second sequence
        assert_eq!(input_ids[[1, 0]], 4);
        assert_eq!(input_ids[[1, 1]], 5);
        assert_eq!(input_ids[[1, 2]], 6);

        // Check attention mask shows padding in second sequence
        assert_eq!(attention_mask[[0, 0]], 1);
        assert_eq!(attention_mask[[0, 1]], 1);
        assert_eq!(attention_mask[[0, 2]], 1);
        assert_eq!(attention_mask[[1, 0]], 1);
        assert_eq!(attention_mask[[1, 1]], 1);
        assert_eq!(attention_mask[[1, 2]], 0); // Padding
    }

    #[test]
    fn test_batch_to_onnx_inputs_empty() {
        let batch = BatchTokenizedInput {
            input_ids: vec![],
            attention_mask: vec![],
            token_type_ids: vec![],
            batch_size: 0,
            max_length: 0,
        };

        let (input_ids, attention_mask, token_type_ids) = batch.to_onnx_inputs().unwrap();

        assert_eq!(input_ids.shape(), &[0, 0]);
        assert_eq!(attention_mask.shape(), &[0, 0]);
        assert_eq!(token_type_ids.shape(), &[0, 0]);
    }

    #[test]
    fn test_batch_attention_masks() {
        // Create a batch with 2 sequences, max_length 4
        // First sequence: all real tokens [1, 1, 1, 1]
        // Second sequence: 2 real tokens + 2 padding [1, 1, 0, 0]
        let batch = BatchTokenizedInput {
            input_ids: vec![1, 2, 3, 4, 5, 6, 0, 0],
            attention_mask: vec![1, 1, 1, 1, 1, 1, 0, 0],
            token_type_ids: vec![0, 0, 0, 0, 0, 0, 0, 0],
            batch_size: 2,
            max_length: 4,
        };

        let masks = batch.attention_masks();

        assert_eq!(masks.len(), 2);
        assert_eq!(masks[0], vec![1, 1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);
    }

    #[test]
    fn test_batch_attention_masks_empty() {
        let batch = BatchTokenizedInput {
            input_ids: vec![],
            attention_mask: vec![],
            token_type_ids: vec![],
            batch_size: 0,
            max_length: 0,
        };

        let masks = batch.attention_masks();
        assert!(masks.is_empty());
    }

    #[test]
    fn test_batch_variable_length_sequences() {
        // Simulate a batch where sequences have different original lengths
        // but are padded to the same length
        // Sequence 1: 5 tokens (full)
        // Sequence 2: 3 tokens (2 padding)
        // Sequence 3: 1 token (4 padding)
        let batch = BatchTokenizedInput {
            input_ids: vec![
                1, 2, 3, 4, 5, // seq 1: 5 tokens
                6, 7, 8, 0, 0, // seq 2: 3 tokens + 2 padding
                9, 0, 0, 0, 0, // seq 3: 1 token + 4 padding
            ],
            attention_mask: vec![
                1, 1, 1, 1, 1, // seq 1: all real
                1, 1, 1, 0, 0, // seq 2: 3 real, 2 padding
                1, 0, 0, 0, 0, // seq 3: 1 real, 4 padding
            ],
            token_type_ids: vec![0; 15],
            batch_size: 3,
            max_length: 5,
        };

        // Test to_onnx_inputs
        let (input_ids, attention_mask, _) = batch.to_onnx_inputs().unwrap();

        assert_eq!(input_ids.shape(), &[3, 5]);
        assert_eq!(attention_mask.shape(), &[3, 5]);

        // Verify attention mask correctly identifies real tokens vs padding
        // Sequence 1
        for i in 0..5 {
            assert_eq!(attention_mask[[0, i]], 1, "seq 1, pos {}", i);
        }
        // Sequence 2
        for i in 0..3 {
            assert_eq!(attention_mask[[1, i]], 1, "seq 2, pos {}", i);
        }
        for i in 3..5 {
            assert_eq!(attention_mask[[1, i]], 0, "seq 2, pos {} (padding)", i);
        }
        // Sequence 3
        assert_eq!(attention_mask[[2, 0]], 1, "seq 3, pos 0");
        for i in 1..5 {
            assert_eq!(attention_mask[[2, i]], 0, "seq 3, pos {} (padding)", i);
        }

        // Test attention_masks method
        let masks = batch.attention_masks();
        assert_eq!(masks.len(), 3);
        assert_eq!(masks[0], vec![1, 1, 1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[2], vec![1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_batch_padding_token_used() {
        // Test that the padding token ID is used correctly
        // Using a custom padding token (e.g., 99)
        let padding_token_id = 99;
        let batch = BatchTokenizedInput {
            input_ids: vec![
                1, 2, 3, padding_token_id, padding_token_id, // seq 1: 3 real + 2 padding
                4, 5, padding_token_id, padding_token_id, padding_token_id, // seq 2: 2 real + 3 padding
            ],
            attention_mask: vec![
                1, 1, 1, 0, 0, // seq 1
                1, 1, 0, 0, 0, // seq 2
            ],
            token_type_ids: vec![0; 10],
            batch_size: 2,
            max_length: 5,
        };

        let (input_ids, attention_mask, _) = batch.to_onnx_inputs().unwrap();

        // Verify padding positions have the padding token ID
        assert_eq!(input_ids[[0, 3]], padding_token_id);
        assert_eq!(input_ids[[0, 4]], padding_token_id);
        assert_eq!(input_ids[[1, 2]], padding_token_id);
        assert_eq!(input_ids[[1, 3]], padding_token_id);
        assert_eq!(input_ids[[1, 4]], padding_token_id);

        // Verify attention mask is 0 for padding
        assert_eq!(attention_mask[[0, 3]], 0);
        assert_eq!(attention_mask[[0, 4]], 0);
        assert_eq!(attention_mask[[1, 2]], 0);
        assert_eq!(attention_mask[[1, 3]], 0);
        assert_eq!(attention_mask[[1, 4]], 0);
    }
}
