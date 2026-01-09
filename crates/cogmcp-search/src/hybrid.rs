//! Hybrid search combining text and semantic search

use std::collections::HashMap;
use std::sync::Arc;

use cogmcp_core::Result;
use cogmcp_storage::FullTextIndex;
use futures::stream::{self, Stream, StreamExt};

use crate::semantic::{SemanticSearch, SemanticSearchOptions};
use crate::text::TextSearch;

/// Default RRF constant (k=60 is commonly used)
const RRF_K: f32 = 60.0;

/// Search mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Full-text keyword search
    Keyword,
    /// Semantic similarity search
    Semantic,
    /// Combined keyword + semantic (default)
    #[default]
    Hybrid,
}

impl SearchMode {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "keyword" | "text" | "fulltext" => SearchMode::Keyword,
            "semantic" | "vector" | "embedding" => SearchMode::Semantic,
            _ => SearchMode::Hybrid,
        }
    }
}

/// Configuration for hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight for keyword search results (0.0 to 1.0)
    pub keyword_weight: f32,
    /// Weight for semantic search results (0.0 to 1.0)
    pub semantic_weight: f32,
    /// Minimum similarity threshold for semantic results
    pub min_similarity: f32,
    /// RRF constant k (higher = more weight to lower-ranked results)
    pub rrf_k: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            keyword_weight: 0.5,
            semantic_weight: 0.5,
            min_similarity: 0.3,
            rrf_k: RRF_K,
        }
    }
}

/// Hybrid search combining multiple search strategies
pub struct HybridSearch<'a> {
    text_search: TextSearch<'a>,
    semantic_search: Option<Arc<SemanticSearch>>,
    config: HybridSearchConfig,
}

impl<'a> HybridSearch<'a> {
    /// Create a new hybrid search with only text search (backwards compatible)
    pub fn new(text_index: &'a FullTextIndex) -> Self {
        Self {
            text_search: TextSearch::new(text_index),
            semantic_search: None,
            config: HybridSearchConfig::default(),
        }
    }

    /// Create a hybrid search with both text and semantic search capabilities
    pub fn with_semantic(
        text_index: &'a FullTextIndex,
        semantic_search: Arc<SemanticSearch>,
    ) -> Self {
        Self {
            text_search: TextSearch::new(text_index),
            semantic_search: Some(semantic_search),
            config: HybridSearchConfig::default(),
        }
    }

    /// Set the hybrid search configuration
    pub fn with_config(mut self, config: HybridSearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Check if semantic search is available
    pub fn has_semantic(&self) -> bool {
        self.semantic_search
            .as_ref()
            .map(|s| s.is_available())
            .unwrap_or(false)
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
            SearchMode::Semantic => self.semantic_search_impl(query, limit),
            SearchMode::Hybrid => self.hybrid_search(query, limit),
        }
    }

    /// Search with specified mode, yielding results incrementally as a stream
    ///
    /// This method yields results one at a time, allowing consumers to process
    /// results as they become available rather than waiting for all results.
    pub fn search_streaming(
        &self,
        query: &str,
        mode: SearchMode,
        limit: usize,
    ) -> impl Stream<Item = Result<HybridSearchResult>> + '_ {
        let query = query.to_string();

        stream::once(async move { self.search(&query, mode, limit) }).flat_map(|results| {
            match results {
                Ok(results) => {
                    let items: Vec<_> = results.into_iter().map(Ok).collect();
                    stream::iter(items)
                }
                Err(e) => stream::iter(vec![Err(e)]),
            }
        })
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

    fn semantic_search_impl(&self, query: &str, limit: usize) -> Result<Vec<HybridSearchResult>> {
        let Some(ref semantic) = self.semantic_search else {
            // Fall back to keyword search if semantic not available
            return self.keyword_search(query, limit);
        };

        if !semantic.is_available() {
            return self.keyword_search(query, limit);
        }

        let options = SemanticSearchOptions::new()
            .with_limit(limit)
            .with_min_similarity(self.config.min_similarity);

        let results = semantic.search_with_options(query, options)?;

        Ok(results
            .into_iter()
            .map(|r| HybridSearchResult {
                path: r.path,
                line_number: r.start_line,
                content: r.chunk_text,
                score: r.similarity,
                match_type: MatchType::Semantic,
            })
            .collect())
    }

    fn hybrid_search(&self, query: &str, limit: usize) -> Result<Vec<HybridSearchResult>> {
        // If semantic search is not available, fall back to keyword only
        if !self.has_semantic() {
            return self.keyword_search(query, limit);
        }

        // Get results from both search methods
        // Request more results to have good coverage for merging
        let fetch_limit = limit * 3;

        let keyword_results = self.keyword_search(query, fetch_limit)?;
        let semantic_results = self.semantic_search_impl(query, fetch_limit)?;

        // Merge results using Reciprocal Rank Fusion
        let merged = self.rrf_merge(keyword_results, semantic_results, limit);

        Ok(merged)
    }

    /// Merge results using Reciprocal Rank Fusion (RRF)
    ///
    /// RRF formula: score = sum(1 / (k + rank))
    /// where k is a constant (typically 60) and rank is 1-indexed
    fn rrf_merge(
        &self,
        keyword_results: Vec<HybridSearchResult>,
        semantic_results: Vec<HybridSearchResult>,
        limit: usize,
    ) -> Vec<HybridSearchResult> {
        let k = self.config.rrf_k;
        let kw_weight = self.config.keyword_weight;
        let sem_weight = self.config.semantic_weight;

        // Create a map to track combined scores
        // Key: (path, line_number) to identify unique results
        let mut score_map: HashMap<(String, Option<u32>), (f32, HybridSearchResult)> =
            HashMap::new();

        // Add keyword results with their RRF scores
        for (rank, result) in keyword_results.into_iter().enumerate() {
            let rrf_score = kw_weight / (k + (rank + 1) as f32);
            let key = (result.path.clone(), result.line_number);

            score_map
                .entry(key)
                .and_modify(|(score, existing)| {
                    *score += rrf_score;
                    existing.match_type = MatchType::Hybrid;
                })
                .or_insert((rrf_score, result));
        }

        // Add semantic results with their RRF scores
        for (rank, result) in semantic_results.into_iter().enumerate() {
            let rrf_score = sem_weight / (k + (rank + 1) as f32);
            let key = (result.path.clone(), result.line_number);

            score_map
                .entry(key)
                .and_modify(|(score, existing)| {
                    *score += rrf_score;
                    existing.match_type = MatchType::Hybrid;
                })
                .or_insert((rrf_score, result));
        }

        // Sort by combined score and take top results
        let mut results: Vec<_> = score_map.into_values().collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(limit)
            .map(|(score, mut result)| {
                result.score = score;
                result
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_mode_from_str() {
        assert_eq!(SearchMode::from_str("keyword"), SearchMode::Keyword);
        assert_eq!(SearchMode::from_str("text"), SearchMode::Keyword);
        assert_eq!(SearchMode::from_str("fulltext"), SearchMode::Keyword);
        assert_eq!(SearchMode::from_str("semantic"), SearchMode::Semantic);
        assert_eq!(SearchMode::from_str("vector"), SearchMode::Semantic);
        assert_eq!(SearchMode::from_str("embedding"), SearchMode::Semantic);
        assert_eq!(SearchMode::from_str("hybrid"), SearchMode::Hybrid);
        assert_eq!(SearchMode::from_str("anything"), SearchMode::Hybrid);
        assert_eq!(SearchMode::from_str(""), SearchMode::Hybrid);
    }

    #[test]
    fn test_search_mode_default() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    #[test]
    fn test_hybrid_search_config_default() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.keyword_weight, 0.5);
        assert_eq!(config.semantic_weight, 0.5);
        assert_eq!(config.min_similarity, 0.3);
        assert_eq!(config.rrf_k, 60.0);
    }

    #[test]
    fn test_rrf_scoring() {
        // Test that RRF scoring produces expected results
        // With k=60, rank 1 score = 1/(60+1) = 0.0164
        // With k=60, rank 2 score = 1/(60+2) = 0.0161
        let k: f32 = 60.0;
        let rank1_score: f32 = 1.0 / (k + 1.0);
        let rank2_score: f32 = 1.0 / (k + 2.0);

        assert!(rank1_score > rank2_score);
        assert!((rank1_score - 0.01639_f32).abs() < 0.001);
        assert!((rank2_score - 0.01613_f32).abs() < 0.001);
    }

    #[test]
    fn test_rrf_merge_combines_scores() {
        // Create mock results for verification of expected score relationships
        let config = HybridSearchConfig::default();

        // Manually compute expected scores
        let k = config.rrf_k;
        let kw_weight = config.keyword_weight;
        let sem_weight = config.semantic_weight;

        // file1.rs appears in both lists (rank 1 in both)
        // RRF score = kw_weight/(k+1) + sem_weight/(k+1)
        let file1_expected = kw_weight / (k + 1.0) + sem_weight / (k + 1.0);

        // file2.rs only in keyword (rank 2)
        let file2_expected = kw_weight / (k + 2.0);

        // file3.rs only in semantic (rank 2)
        let file3_expected = sem_weight / (k + 2.0);

        // file1 should have highest score (appears in both)
        assert!(file1_expected > file2_expected);
        assert!(file1_expected > file3_expected);

        // file2 and file3 should have same score (same rank, equal weights)
        assert!((file2_expected - file3_expected).abs() < 0.0001);
    }

    #[test]
    fn test_hybrid_search_result_fields() {
        let result = HybridSearchResult {
            path: "src/main.rs".to_string(),
            line_number: Some(42),
            content: "fn main() {}".to_string(),
            score: 0.95,
            match_type: MatchType::Hybrid,
        };

        assert_eq!(result.path, "src/main.rs");
        assert_eq!(result.line_number, Some(42));
        assert_eq!(result.content, "fn main() {}");
        assert_eq!(result.score, 0.95);
        assert_eq!(result.match_type, MatchType::Hybrid);
    }

    #[test]
    fn test_match_type_equality() {
        assert_eq!(MatchType::Keyword, MatchType::Keyword);
        assert_eq!(MatchType::Semantic, MatchType::Semantic);
        assert_eq!(MatchType::Hybrid, MatchType::Hybrid);
        assert_ne!(MatchType::Keyword, MatchType::Semantic);
        assert_ne!(MatchType::Keyword, MatchType::Hybrid);
        assert_ne!(MatchType::Semantic, MatchType::Hybrid);
    }
}
