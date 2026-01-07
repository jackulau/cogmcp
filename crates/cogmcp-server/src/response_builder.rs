//! Streaming response builder for MCP protocol
//!
//! This module provides utilities for building MCP responses incrementally,
//! supporting progress notifications and chunked content delivery.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Progress notification for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressNotification {
    /// Progress token (typically the request ID)
    #[serde(rename = "progressToken")]
    pub progress_token: ProgressToken,
    /// Current progress value (0.0 to 1.0 or item count)
    pub progress: f64,
    /// Optional total for absolute progress
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    /// Optional message describing current state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Progress token type (can be string or number)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ProgressToken {
    String(String),
    Number(i64),
}

impl From<String> for ProgressToken {
    fn from(s: String) -> Self {
        ProgressToken::String(s)
    }
}

impl From<&str> for ProgressToken {
    fn from(s: &str) -> Self {
        ProgressToken::String(s.to_string())
    }
}

impl From<i64> for ProgressToken {
    fn from(n: i64) -> Self {
        ProgressToken::Number(n)
    }
}

impl From<i32> for ProgressToken {
    fn from(n: i32) -> Self {
        ProgressToken::Number(n as i64)
    }
}

impl ProgressNotification {
    /// Create a new progress notification
    pub fn new(token: impl Into<ProgressToken>, progress: f64) -> Self {
        Self {
            progress_token: token.into(),
            progress,
            total: None,
            message: None,
        }
    }

    /// Set the total value for absolute progress
    pub fn with_total(mut self, total: f64) -> Self {
        self.total = Some(total);
        self
    }

    /// Set a progress message
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Convert to a JSON-RPC notification
    pub fn to_json_rpc(&self) -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": self
        })
    }
}

/// A text content block for MCP responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl TextContent {
    /// Create a new text content block
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.into(),
        }
    }
}

/// Builder for streaming MCP tool call responses
#[derive(Debug)]
pub struct StreamingResponseBuilder {
    /// Content chunks accumulated so far
    chunks: Vec<TextContent>,
    /// Whether this response represents an error
    is_error: bool,
    /// Progress token for notifications
    progress_token: Option<ProgressToken>,
    /// Total items for progress tracking
    total_items: Option<usize>,
    /// Items processed so far
    items_processed: usize,
}

impl StreamingResponseBuilder {
    /// Create a new streaming response builder
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            is_error: false,
            progress_token: None,
            total_items: None,
            items_processed: 0,
        }
    }

    /// Set the progress token for this response
    pub fn with_progress_token(mut self, token: impl Into<ProgressToken>) -> Self {
        self.progress_token = Some(token.into());
        self
    }

    /// Set the total number of items for progress tracking
    pub fn with_total_items(mut self, total: usize) -> Self {
        self.total_items = Some(total);
        self
    }

    /// Mark this response as an error
    pub fn as_error(mut self) -> Self {
        self.is_error = true;
        self
    }

    /// Add a text chunk to the response
    pub fn add_chunk(&mut self, text: impl Into<String>) {
        self.chunks.push(TextContent::new(text));
    }

    /// Add multiple text chunks
    pub fn add_chunks(&mut self, texts: impl IntoIterator<Item = impl Into<String>>) {
        for text in texts {
            self.add_chunk(text);
        }
    }

    /// Increment the items processed counter
    pub fn increment_processed(&mut self, count: usize) {
        self.items_processed += count;
    }

    /// Get the current progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        match self.total_items {
            Some(total) if total > 0 => self.items_processed as f64 / total as f64,
            _ => 0.0,
        }
    }

    /// Create a progress notification for the current state
    pub fn create_progress_notification(&self) -> Option<ProgressNotification> {
        let token = self.progress_token.clone()?;

        let mut notification = ProgressNotification::new(token, self.progress());

        if let Some(total) = self.total_items {
            notification = notification.with_total(total as f64);
        }

        notification = notification.with_message(format!(
            "Processed {} of {} items",
            self.items_processed,
            self.total_items.unwrap_or(0)
        ));

        Some(notification)
    }

    /// Get the number of chunks accumulated
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get the items processed count
    pub fn items_processed(&self) -> usize {
        self.items_processed
    }

    /// Check if any content has been added
    pub fn has_content(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// Build the final MCP CallToolResult response
    pub fn build(self) -> Value {
        let content: Vec<Value> = self
            .chunks
            .into_iter()
            .map(|c| json!({ "type": c.content_type, "text": c.text }))
            .collect();

        json!({
            "content": content,
            "isError": self.is_error
        })
    }

    /// Build with a single combined text result
    pub fn build_combined(self) -> Value {
        let combined_text: String = self
            .chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        json!({
            "content": [{
                "type": "text",
                "text": combined_text
            }],
            "isError": self.is_error
        })
    }
}

impl Default for StreamingResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for when to use streaming responses
#[derive(Debug, Clone)]
pub struct StreamingThreshold {
    /// Minimum number of results to trigger streaming
    pub min_results: usize,
    /// Minimum total content size (chars) to trigger streaming
    pub min_content_size: usize,
}

impl Default for StreamingThreshold {
    fn default() -> Self {
        Self {
            min_results: 10,
            min_content_size: 8192,
        }
    }
}

impl StreamingThreshold {
    /// Create a new threshold configuration
    pub fn new(min_results: usize, min_content_size: usize) -> Self {
        Self {
            min_results,
            min_content_size,
        }
    }

    /// Check if streaming should be used based on result count
    pub fn should_stream_by_count(&self, count: usize) -> bool {
        count >= self.min_results
    }

    /// Check if streaming should be used based on content size
    pub fn should_stream_by_size(&self, size: usize) -> bool {
        size >= self.min_content_size
    }

    /// Check if streaming should be used (either condition)
    pub fn should_stream(&self, count: usize, size: usize) -> bool {
        self.should_stream_by_count(count) || self.should_stream_by_size(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_token_from_string() {
        let token: ProgressToken = "test-token".into();
        assert_eq!(token, ProgressToken::String("test-token".to_string()));
    }

    #[test]
    fn test_progress_token_from_number() {
        let token: ProgressToken = 42i64.into();
        assert_eq!(token, ProgressToken::Number(42));
    }

    #[test]
    fn test_progress_notification_new() {
        let notification = ProgressNotification::new("token-1", 0.5);

        assert_eq!(notification.progress_token, ProgressToken::String("token-1".to_string()));
        assert_eq!(notification.progress, 0.5);
        assert!(notification.total.is_none());
        assert!(notification.message.is_none());
    }

    #[test]
    fn test_progress_notification_with_total() {
        let notification = ProgressNotification::new(123i64, 50.0)
            .with_total(100.0);

        assert_eq!(notification.total, Some(100.0));
    }

    #[test]
    fn test_progress_notification_with_message() {
        let notification = ProgressNotification::new("token", 0.75)
            .with_message("Processing files...");

        assert_eq!(notification.message, Some("Processing files...".to_string()));
    }

    #[test]
    fn test_progress_notification_to_json_rpc() {
        let notification = ProgressNotification::new("token-1", 0.5)
            .with_total(1.0)
            .with_message("Halfway there");

        let json = notification.to_json_rpc();

        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["method"], "notifications/progress");
        assert_eq!(json["params"]["progressToken"], "token-1");
        assert_eq!(json["params"]["progress"], 0.5);
        assert_eq!(json["params"]["total"], 1.0);
        assert_eq!(json["params"]["message"], "Halfway there");
    }

    #[test]
    fn test_text_content_new() {
        let content = TextContent::new("Hello, world!");

        assert_eq!(content.content_type, "text");
        assert_eq!(content.text, "Hello, world!");
    }

    #[test]
    fn test_streaming_response_builder_new() {
        let builder = StreamingResponseBuilder::new();

        assert_eq!(builder.chunk_count(), 0);
        assert_eq!(builder.items_processed(), 0);
        assert!(!builder.has_content());
        assert_eq!(builder.progress(), 0.0);
    }

    #[test]
    fn test_streaming_response_builder_add_chunk() {
        let mut builder = StreamingResponseBuilder::new();

        builder.add_chunk("First chunk");
        builder.add_chunk("Second chunk");

        assert_eq!(builder.chunk_count(), 2);
        assert!(builder.has_content());
    }

    #[test]
    fn test_streaming_response_builder_add_chunks() {
        let mut builder = StreamingResponseBuilder::new();

        builder.add_chunks(vec!["Chunk 1", "Chunk 2", "Chunk 3"]);

        assert_eq!(builder.chunk_count(), 3);
    }

    #[test]
    fn test_streaming_response_builder_progress_tracking() {
        let mut builder = StreamingResponseBuilder::new()
            .with_total_items(10);

        assert_eq!(builder.progress(), 0.0);

        builder.increment_processed(5);
        assert_eq!(builder.progress(), 0.5);

        builder.increment_processed(5);
        assert_eq!(builder.progress(), 1.0);
    }

    #[test]
    fn test_streaming_response_builder_progress_notification() {
        let mut builder = StreamingResponseBuilder::new()
            .with_progress_token("req-123")
            .with_total_items(20);

        builder.increment_processed(10);

        let notification = builder.create_progress_notification();

        assert!(notification.is_some());
        let notification = notification.unwrap();
        assert_eq!(notification.progress_token, ProgressToken::String("req-123".to_string()));
        assert_eq!(notification.progress, 0.5);
        assert_eq!(notification.total, Some(20.0));
    }

    #[test]
    fn test_streaming_response_builder_no_progress_without_token() {
        let builder = StreamingResponseBuilder::new()
            .with_total_items(10);

        let notification = builder.create_progress_notification();
        assert!(notification.is_none());
    }

    #[test]
    fn test_streaming_response_builder_build() {
        let mut builder = StreamingResponseBuilder::new();
        builder.add_chunk("Result 1");
        builder.add_chunk("Result 2");

        let response = builder.build();

        assert_eq!(response["isError"], false);
        assert!(response["content"].is_array());
        assert_eq!(response["content"].as_array().unwrap().len(), 2);
        assert_eq!(response["content"][0]["type"], "text");
        assert_eq!(response["content"][0]["text"], "Result 1");
    }

    #[test]
    fn test_streaming_response_builder_build_error() {
        let mut builder = StreamingResponseBuilder::new().as_error();
        builder.add_chunk("Error message");

        let response = builder.build();

        assert_eq!(response["isError"], true);
    }

    #[test]
    fn test_streaming_response_builder_build_combined() {
        let mut builder = StreamingResponseBuilder::new();
        builder.add_chunk("Part 1");
        builder.add_chunk("Part 2");
        builder.add_chunk("Part 3");

        let response = builder.build_combined();

        assert_eq!(response["content"].as_array().unwrap().len(), 1);
        let text = response["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Part 1"));
        assert!(text.contains("Part 2"));
        assert!(text.contains("Part 3"));
    }

    #[test]
    fn test_streaming_threshold_default() {
        let threshold = StreamingThreshold::default();

        assert_eq!(threshold.min_results, 10);
        assert_eq!(threshold.min_content_size, 8192);
    }

    #[test]
    fn test_streaming_threshold_should_stream_by_count() {
        let threshold = StreamingThreshold::new(5, 1000);

        assert!(!threshold.should_stream_by_count(4));
        assert!(threshold.should_stream_by_count(5));
        assert!(threshold.should_stream_by_count(10));
    }

    #[test]
    fn test_streaming_threshold_should_stream_by_size() {
        let threshold = StreamingThreshold::new(5, 1000);

        assert!(!threshold.should_stream_by_size(999));
        assert!(threshold.should_stream_by_size(1000));
        assert!(threshold.should_stream_by_size(2000));
    }

    #[test]
    fn test_streaming_threshold_should_stream_combined() {
        let threshold = StreamingThreshold::new(5, 1000);

        // Neither condition met
        assert!(!threshold.should_stream(3, 500));

        // Count condition met
        assert!(threshold.should_stream(5, 500));

        // Size condition met
        assert!(threshold.should_stream(3, 1000));

        // Both conditions met
        assert!(threshold.should_stream(10, 2000));
    }
}
