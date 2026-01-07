//! Streaming types and traits for CogMCP
//!
//! This module provides foundational types for streaming results across
//! all search components, allowing clients to receive results incrementally
//! rather than waiting for complete result sets.

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Type alias for a boxed async stream of results.
///
/// This is the standard return type for streaming operations.
pub type ResultStream<T> = Pin<Box<dyn Stream<Item = Result<StreamChunk<T>>> + Send>>;

/// Configuration for streaming behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Number of results to include per chunk (default: 10)
    pub chunk_size: usize,
    /// Maximum time in milliseconds between yields (default: 100)
    pub yield_interval_ms: u64,
    /// Whether streaming is enabled (default: true)
    pub enabled: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10,
            yield_interval_ms: 100,
            enabled: true,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming config with custom chunk size.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Create a new streaming config with custom yield interval.
    pub fn with_yield_interval_ms(mut self, yield_interval_ms: u64) -> Self {
        self.yield_interval_ms = yield_interval_ms;
        self
    }

    /// Create a disabled streaming config (results collected before return).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// A chunk of streamed results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk<T> {
    /// Batch of results in this chunk
    pub items: Vec<T>,
    /// Whether this is the final chunk
    pub is_last: bool,
    /// Total count of results if known
    pub total_count: Option<usize>,
    /// Completion progress (0.0 to 1.0) if known
    pub progress: Option<f32>,
}

impl<T> StreamChunk<T> {
    /// Create a new chunk with items.
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items,
            is_last: false,
            total_count: None,
            progress: None,
        }
    }

    /// Create a final chunk.
    pub fn last(items: Vec<T>) -> Self {
        Self {
            items,
            is_last: true,
            total_count: None,
            progress: None,
        }
    }

    /// Create an empty final chunk.
    pub fn empty_last() -> Self {
        Self {
            items: Vec::new(),
            is_last: true,
            total_count: Some(0),
            progress: Some(1.0),
        }
    }

    /// Set the total count.
    pub fn with_total_count(mut self, total: usize) -> Self {
        self.total_count = Some(total);
        self
    }

    /// Set the progress.
    pub fn with_progress(mut self, progress: f32) -> Self {
        self.progress = Some(progress.clamp(0.0, 1.0));
        self
    }

    /// Mark this chunk as the last one.
    pub fn mark_last(mut self) -> Self {
        self.is_last = true;
        self
    }

    /// Get the number of items in this chunk.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if this chunk has no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Trait for components that can stream results.
///
/// This trait defines the interface for any search or query component
/// that supports streaming results back to the client incrementally.
#[async_trait::async_trait]
pub trait StreamingResult<T: Send + 'static> {
    /// Stream results with the given configuration.
    ///
    /// Returns a stream of chunks that yields results as they become available.
    async fn stream_results(&self, config: &StreamingConfig) -> ResultStream<T>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 10);
        assert_eq!(config.yield_interval_ms, 100);
        assert!(config.enabled);
    }

    #[test]
    fn test_streaming_config_disabled() {
        let config = StreamingConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.chunk_size, 10);
        assert_eq!(config.yield_interval_ms, 100);
    }

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingConfig::default()
            .with_chunk_size(50)
            .with_yield_interval_ms(200);

        assert_eq!(config.chunk_size, 50);
        assert_eq!(config.yield_interval_ms, 200);
        assert!(config.enabled);
    }

    #[test]
    fn test_stream_chunk_new() {
        let chunk: StreamChunk<i32> = StreamChunk::new(vec![1, 2, 3]);
        assert_eq!(chunk.items, vec![1, 2, 3]);
        assert!(!chunk.is_last);
        assert!(chunk.total_count.is_none());
        assert!(chunk.progress.is_none());
        assert_eq!(chunk.len(), 3);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_stream_chunk_last() {
        let chunk: StreamChunk<i32> = StreamChunk::last(vec![4, 5]);
        assert_eq!(chunk.items, vec![4, 5]);
        assert!(chunk.is_last);
    }

    #[test]
    fn test_stream_chunk_empty_last() {
        let chunk: StreamChunk<i32> = StreamChunk::empty_last();
        assert!(chunk.items.is_empty());
        assert!(chunk.is_last);
        assert_eq!(chunk.total_count, Some(0));
        assert_eq!(chunk.progress, Some(1.0));
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_stream_chunk_builders() {
        let chunk: StreamChunk<String> = StreamChunk::new(vec!["hello".to_string()])
            .with_total_count(100)
            .with_progress(0.5)
            .mark_last();

        assert!(chunk.is_last);
        assert_eq!(chunk.total_count, Some(100));
        assert_eq!(chunk.progress, Some(0.5));
    }

    #[test]
    fn test_stream_chunk_progress_clamping() {
        let chunk: StreamChunk<i32> = StreamChunk::new(vec![]).with_progress(1.5);
        assert_eq!(chunk.progress, Some(1.0));

        let chunk: StreamChunk<i32> = StreamChunk::new(vec![]).with_progress(-0.5);
        assert_eq!(chunk.progress, Some(0.0));
    }

    #[test]
    fn test_streaming_config_serialization() {
        let config = StreamingConfig {
            chunk_size: 25,
            yield_interval_ms: 150,
            enabled: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: StreamingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.chunk_size, deserialized.chunk_size);
        assert_eq!(config.yield_interval_ms, deserialized.yield_interval_ms);
        assert_eq!(config.enabled, deserialized.enabled);
    }

    #[test]
    fn test_stream_chunk_serialization() {
        let chunk: StreamChunk<i32> = StreamChunk::new(vec![1, 2, 3])
            .with_total_count(10)
            .with_progress(0.3)
            .mark_last();

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: StreamChunk<i32> = serde_json::from_str(&json).unwrap();

        assert_eq!(chunk.items, deserialized.items);
        assert_eq!(chunk.is_last, deserialized.is_last);
        assert_eq!(chunk.total_count, deserialized.total_count);
        assert_eq!(chunk.progress, deserialized.progress);
    }
}
