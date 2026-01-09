//! Streaming formatter for converting results to text chunks
//!
//! This module provides utilities for converting search results into
//! chunked text output suitable for streaming MCP responses.

use std::fmt::Display;

/// Configuration for streaming output
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Target size for each chunk in characters
    pub chunk_size: usize,
    /// Whether to format output as markdown
    pub markdown: bool,
    /// Separator between items
    pub separator: String,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 4096,
            markdown: true,
            separator: "\n\n".to_string(),
        }
    }
}

impl StreamingConfig {
    /// Create a new config with the specified chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set markdown formatting
    pub fn with_markdown(mut self, enabled: bool) -> Self {
        self.markdown = enabled;
        self
    }

    /// Set the separator between items
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
}

/// A streaming formatter that converts results into text chunks
#[derive(Debug)]
pub struct StreamingFormatter {
    config: StreamingConfig,
    buffer: String,
    items_formatted: usize,
    total_items: usize,
}

impl StreamingFormatter {
    /// Create a new streaming formatter with default configuration
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            buffer: String::new(),
            items_formatted: 0,
            total_items: 0,
        }
    }

    /// Create a formatter with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            items_formatted: 0,
            total_items: 0,
        }
    }

    /// Set the total number of items (for progress tracking)
    pub fn set_total(&mut self, total: usize) {
        self.total_items = total;
    }

    /// Get the current progress as a percentage (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_items == 0 {
            0.0
        } else {
            self.items_formatted as f64 / self.total_items as f64
        }
    }

    /// Get the number of items formatted so far
    pub fn items_formatted(&self) -> usize {
        self.items_formatted
    }

    /// Get the total number of items
    pub fn total_items(&self) -> usize {
        self.total_items
    }

    /// Format a single item and add it to the buffer
    ///
    /// Returns any chunks that are ready to be sent (when buffer exceeds chunk_size)
    pub fn format_item<T: Display>(&mut self, item: &T) -> Vec<String> {
        let formatted = item.to_string();

        if !self.buffer.is_empty() {
            self.buffer.push_str(&self.config.separator);
        }
        self.buffer.push_str(&formatted);
        self.items_formatted += 1;

        self.extract_chunks()
    }

    /// Format a batch of items
    ///
    /// Returns any chunks that are ready to be sent
    pub fn format_batch<T: Display>(&mut self, items: &[T]) -> Vec<String> {
        let mut chunks = Vec::new();
        for item in items {
            chunks.extend(self.format_item(item));
        }
        chunks
    }

    /// Extract complete chunks from the buffer
    fn extract_chunks(&mut self) -> Vec<String> {
        let mut chunks = Vec::new();

        while self.buffer.len() >= self.config.chunk_size {
            // Try to find a good break point (end of line or separator)
            let break_point = self.find_break_point(self.config.chunk_size);
            let chunk: String = self.buffer.drain(..break_point).collect();
            chunks.push(chunk);
        }

        chunks
    }

    /// Find a good break point near the target position
    fn find_break_point(&self, target: usize) -> usize {
        // Look for newline near target
        if let Some(pos) = self.buffer[..target].rfind('\n') {
            return pos + 1;
        }

        // Look for separator
        if let Some(pos) = self.buffer[..target].rfind(&self.config.separator) {
            return pos + self.config.separator.len();
        }

        // Fall back to target position
        target
    }

    /// Flush any remaining content in the buffer
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buffer))
        }
    }

    /// Format all items at once and return chunks
    ///
    /// This is a convenience method for non-streaming use cases
    pub fn format_all<T: Display>(&mut self, items: &[T]) -> Vec<String> {
        self.set_total(items.len());
        let mut chunks = self.format_batch(items);
        if let Some(final_chunk) = self.flush() {
            chunks.push(final_chunk);
        }
        chunks
    }

    /// Reset the formatter state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.items_formatted = 0;
        self.total_items = 0;
    }
}

impl Default for StreamingFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// A formatted search result for streaming
#[derive(Debug, Clone)]
pub struct FormattedResult {
    pub path: String,
    pub line_number: Option<u32>,
    pub content: String,
    pub score: Option<f64>,
}

impl Display for FormattedResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line_number {
            write!(f, "{}:{}", self.path, line)?;
        } else {
            write!(f, "{}", self.path)?;
        }

        if let Some(score) = self.score {
            write!(f, " (score: {:.2})", score)?;
        }

        writeln!(f)?;
        writeln!(f, "```")?;
        write!(f, "{}", self.content.trim())?;
        writeln!(f)?;
        write!(f, "```")
    }
}

impl FormattedResult {
    pub fn new(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line_number: None,
            content: content.into(),
            score: None,
        }
    }

    pub fn with_line(mut self, line: u32) -> Self {
        self.line_number = Some(line);
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 4096);
        assert!(config.markdown);
        assert_eq!(config.separator, "\n\n");
    }

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingConfig::default()
            .with_chunk_size(1024)
            .with_markdown(false)
            .with_separator("---");

        assert_eq!(config.chunk_size, 1024);
        assert!(!config.markdown);
        assert_eq!(config.separator, "---");
    }

    #[test]
    fn test_formatter_new() {
        let formatter = StreamingFormatter::new();
        assert_eq!(formatter.items_formatted(), 0);
        assert_eq!(formatter.total_items(), 0);
        assert_eq!(formatter.progress(), 0.0);
    }

    #[test]
    fn test_formatter_format_item() {
        let mut formatter = StreamingFormatter::with_config(
            StreamingConfig::default().with_chunk_size(1000)
        );

        let item = "Test item";
        let chunks = formatter.format_item(&item);

        // Item is small, should not produce chunks yet
        assert!(chunks.is_empty());
        assert_eq!(formatter.items_formatted(), 1);
    }

    #[test]
    fn test_formatter_flush() {
        let mut formatter = StreamingFormatter::new();

        formatter.format_item(&"Test item");
        let remaining = formatter.flush();

        assert!(remaining.is_some());
        assert_eq!(remaining.unwrap(), "Test item");
    }

    #[test]
    fn test_formatter_format_all() {
        let mut formatter = StreamingFormatter::with_config(
            StreamingConfig::default().with_chunk_size(50)
        );

        let items: Vec<String> = (0..10)
            .map(|i| format!("Item number {} with some content", i))
            .collect();

        let chunks = formatter.format_all(&items);

        // Should have produced multiple chunks
        assert!(!chunks.is_empty());
        assert_eq!(formatter.items_formatted(), 10);
        assert_eq!(formatter.progress(), 1.0);
    }

    #[test]
    fn test_formatter_progress_tracking() {
        let mut formatter = StreamingFormatter::new();
        formatter.set_total(4);

        formatter.format_item(&"Item 1");
        assert_eq!(formatter.progress(), 0.25);

        formatter.format_item(&"Item 2");
        assert_eq!(formatter.progress(), 0.5);

        formatter.format_item(&"Item 3");
        assert_eq!(formatter.progress(), 0.75);

        formatter.format_item(&"Item 4");
        assert_eq!(formatter.progress(), 1.0);
    }

    #[test]
    fn test_formatter_reset() {
        let mut formatter = StreamingFormatter::new();
        formatter.set_total(10);
        formatter.format_item(&"Test");

        formatter.reset();

        assert_eq!(formatter.items_formatted(), 0);
        assert_eq!(formatter.total_items(), 0);
        assert!(formatter.flush().is_none());
    }

    #[test]
    fn test_formatted_result_display() {
        let result = FormattedResult::new("src/main.rs", "fn main() {}")
            .with_line(10)
            .with_score(0.95);

        let formatted = result.to_string();

        assert!(formatted.contains("src/main.rs:10"));
        assert!(formatted.contains("(score: 0.95)"));
        assert!(formatted.contains("fn main()"));
        assert!(formatted.contains("```"));
    }

    #[test]
    fn test_formatted_result_without_line() {
        let result = FormattedResult::new("src/lib.rs", "pub mod foo;");

        let formatted = result.to_string();

        assert!(formatted.contains("src/lib.rs"));
        assert!(!formatted.contains(":"));
        assert!(formatted.contains("pub mod foo;"));
    }

    #[test]
    fn test_chunking_at_newlines() {
        let mut formatter = StreamingFormatter::with_config(
            StreamingConfig::default().with_chunk_size(30)
        );

        // Format items that will exceed chunk size
        let items = vec![
            "Line one content",
            "Line two content",
            "Line three content",
        ];

        for item in &items {
            formatter.format_item(item);
        }

        let remaining = formatter.flush();
        assert!(remaining.is_some());
    }

    #[test]
    fn test_format_batch() {
        let mut formatter = StreamingFormatter::with_config(
            StreamingConfig::default().with_chunk_size(100)
        );

        let items = vec!["Item A", "Item B", "Item C"];
        let chunks = formatter.format_batch(&items);

        assert_eq!(formatter.items_formatted(), 3);

        // With small items and larger chunk size, no chunks yet
        assert!(chunks.is_empty());

        // But we should have content in buffer
        let remaining = formatter.flush();
        assert!(remaining.is_some());
        let content = remaining.unwrap();
        assert!(content.contains("Item A"));
        assert!(content.contains("Item B"));
        assert!(content.contains("Item C"));
    }

    #[test]
    fn test_large_batch_produces_chunks() {
        let mut formatter = StreamingFormatter::with_config(
            StreamingConfig::default().with_chunk_size(100)
        );

        // Create items that will definitely exceed chunk size
        let items: Vec<String> = (0..20)
            .map(|i| format!("This is item number {} with significant content to ensure chunks", i))
            .collect();

        let chunks = formatter.format_all(&items);

        // Should have multiple chunks
        assert!(chunks.len() > 1);

        // Total content should be preserved
        let total: String = chunks.join("");
        for i in 0..20 {
            assert!(total.contains(&format!("item number {}", i)));
        }
    }
}
