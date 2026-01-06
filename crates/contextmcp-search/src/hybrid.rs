//! Hybrid search combining text and semantic search

use contextmcp_core::Result;
use contextmcp_storage::FullTextIndex;

use crate::text::{TextSearch, TextSearchResult};

/// Search mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Full-text keyword search
    Keyword,
    /// Semantic similarity search
    Semantic,
    /// Combined keyword + semantic (default)
    Hybrid,
}

/// Hybrid search combining multiple search strategies
pub struct HybridSearch<'a> {
    text_search: TextSearch<'a>,
    // semantic_search: SemanticSearch, // TODO: Add when embeddings are ready
}

impl<'a> HybridSearch<'a> {
    pub fn new(text_index: &'a FullTextIndex) -> Self {
        Self {
            text_search: TextSearch::new(text_index),
        }
    }

    /// Search with specified mode
    pub fn search(
        &self,
        query: &str,
        mode: SearchMode,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        match mode {
            SearchMode::Keyword => self.keyword_search(query, limit),
            SearchMode::Semantic => self.semantic_search(query, limit),
            SearchMode::Hybrid => self.hybrid_search(query, limit),
        }
    }

    fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<HybridSearchResult>> {
        let results = self.text_search.search(query, limit)?;
        Ok(results
            .into_iter()
            .map(|r| HybridSearchResult {
                path: r.path,
                line_number: Some(r.line_number),
                content: r.content,
                score: r.score,
                match_type: MatchType::Keyword,
            })
            .collect())
    }

    fn semantic_search(&self, _query: &str, _limit: usize) -> Result<Vec<HybridSearchResult>> {
        // TODO: Implement when embeddings are ready
        Ok(Vec::new())
    }

    fn hybrid_search(&self, query: &str, limit: usize) -> Result<Vec<HybridSearchResult>> {
        // For now, just use keyword search
        // TODO: Combine with semantic search using Reciprocal Rank Fusion
        self.keyword_search(query, limit)
    }
}

/// Result from hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub path: String,
    pub line_number: Option<u32>,
    pub content: String,
    pub score: f32,
    pub match_type: MatchType,
}

/// Type of match
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Keyword,
    Semantic,
    Hybrid,
}
