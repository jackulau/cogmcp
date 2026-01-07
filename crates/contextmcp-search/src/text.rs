//! Full-text search using Tantivy

use cogmcp_core::Result;
use cogmcp_storage::FullTextIndex;

/// Full-text search engine wrapper
pub struct TextSearch<'a> {
    index: &'a FullTextIndex,
}

impl<'a> TextSearch<'a> {
    pub fn new(index: &'a FullTextIndex) -> Self {
        Self { index }
    }

    /// Search for content matching a query
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        let hits = self.index.search(query, limit)?;

        Ok(hits
            .into_iter()
            .map(|hit| TextSearchResult {
                path: hit.path,
                line_number: hit.line_number,
                content: hit.content,
                score: hit.score,
            })
            .collect())
    }
}

/// Result from text search
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    pub path: String,
    pub line_number: u32,
    pub content: String,
    pub score: f32,
}
