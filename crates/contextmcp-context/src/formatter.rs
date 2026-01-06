//! Context output formatting

use crate::prioritizer::ContextItem;

/// Format for context output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text with file headers
    Plain,
    /// Markdown with code blocks
    Markdown,
    /// JSON structured output
    Json,
}

/// Formats context for output
pub struct ContextFormatter {
    format: OutputFormat,
}

impl ContextFormatter {
    pub fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    /// Format a list of context items
    pub fn format(&self, items: &[ContextItem]) -> String {
        match self.format {
            OutputFormat::Plain => self.format_plain(items),
            OutputFormat::Markdown => self.format_markdown(items),
            OutputFormat::Json => self.format_json(items),
        }
    }

    fn format_plain(&self, items: &[ContextItem]) -> String {
        let mut output = String::new();

        for item in items {
            output.push_str(&format!("=== {} ===\n", item.path));
            if let (Some(start), Some(end)) = (item.line_start, item.line_end) {
                output.push_str(&format!("Lines {}-{}\n", start, end));
            }
            output.push('\n');
            output.push_str(&item.content);
            output.push_str("\n\n");
        }

        output
    }

    fn format_markdown(&self, items: &[ContextItem]) -> String {
        let mut output = String::new();

        for item in items {
            // Detect language from file extension
            let lang = item
                .path
                .rsplit('.')
                .next()
                .unwrap_or("");

            output.push_str(&format!("### `{}`", item.path));
            if let (Some(start), Some(end)) = (item.line_start, item.line_end) {
                output.push_str(&format!(" (lines {}-{})", start, end));
            }
            output.push_str("\n\n");

            output.push_str(&format!("```{}\n", lang));
            output.push_str(&item.content);
            if !item.content.ends_with('\n') {
                output.push('\n');
            }
            output.push_str("```\n\n");
        }

        output
    }

    fn format_json(&self, items: &[ContextItem]) -> String {
        #[derive(serde::Serialize)]
        struct JsonItem<'a> {
            path: &'a str,
            line_start: Option<u32>,
            line_end: Option<u32>,
            content: &'a str,
            priority_score: f32,
        }

        let json_items: Vec<JsonItem> = items
            .iter()
            .map(|item| JsonItem {
                path: &item.path,
                line_start: item.line_start,
                line_end: item.line_end,
                content: &item.content,
                priority_score: item.priority_score,
            })
            .collect();

        serde_json::to_string_pretty(&json_items).unwrap_or_default()
    }
}

impl Default for ContextFormatter {
    fn default() -> Self {
        Self::new(OutputFormat::Markdown)
    }
}
