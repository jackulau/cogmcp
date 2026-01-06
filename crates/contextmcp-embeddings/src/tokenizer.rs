//! Tokenization utilities for text preprocessing

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;
use tracing::debug;

use contextmcp_core::{Error, Result};

/// Tokenizer wrapper for preparing text inputs
pub struct Tokenizer {
    inner: HfTokenizer,
    max_length: usize,
}

impl Tokenizer {
    /// Load a tokenizer from a JSON file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading tokenizer from {:?}", path);

        let inner = HfTokenizer::from_file(path).map_err(|e| {
            Error::Embedding(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self {
            inner,
            max_length: 512,
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
                seq_length: 0,
            });
        }

        // Encode all texts
        let encodings: Vec<TokenizedInput> = texts
            .iter()
            .map(|text| self.encode(text))
            .collect::<Result<Vec<_>>>()?;

        // Find the maximum length in this batch (capped by max_length)
        let seq_length = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let batch_size = texts.len();

        // Pad all sequences to the same length
        let mut input_ids = Vec::with_capacity(batch_size * seq_length);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_length);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_length);

        for encoding in encodings {
            // Add actual tokens
            let len = encoding.input_ids.len().min(seq_length);
            input_ids.extend_from_slice(&encoding.input_ids[..len]);
            attention_mask.extend_from_slice(&encoding.attention_mask[..len]);
            token_type_ids.extend_from_slice(&encoding.token_type_ids[..len]);

            // Pad to seq_length
            let padding_len = seq_length - len;
            if padding_len > 0 {
                input_ids.extend(std::iter::repeat(0i64).take(padding_len));
                attention_mask.extend(std::iter::repeat(0i64).take(padding_len));
                token_type_ids.extend(std::iter::repeat(0i64).take(padding_len));
            }
        }

        Ok(BatchTokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            seq_length,
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
    /// Flattened input IDs [batch_size * seq_length]
    pub input_ids: Vec<i64>,
    /// Flattened attention mask
    pub attention_mask: Vec<i64>,
    /// Flattened token type IDs
    pub token_type_ids: Vec<i64>,
    /// Number of texts in the batch
    pub batch_size: usize,
    /// Maximum sequence length in the batch
    pub seq_length: usize,
}

impl BatchTokenizedInput {
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
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
            seq_length: 0,
        };
        assert!(batch.is_empty());
    }
}
