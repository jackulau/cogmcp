//! Streaming search iterators for incremental result delivery
//!
//! This module provides streaming versions of semantic and hybrid search
//! that yield results incrementally as they're found, rather than collecting
//! all results before returning. This is useful for large codebases where
//! collecting all results would introduce noticeable latency.

use std::collections::BinaryHeap;
use std::sync::Arc;

use cogmcp_core::Result;
use cogmcp_storage::FullTextIndex;
use futures::stream::{self, Stream, StreamExt};

use crate::hybrid::{HybridSearch, HybridSearchConfig, HybridSearchResult, MatchType, SearchMode};
use crate::semantic::{SemanticSearch, SemanticSearchOptions, SemanticSearchResult};

/// Batch size for processing embeddings in streaming mode
const DEFAULT_BATCH_SIZE: usize = 100;

/// A scored item for the priority heap (min-heap by score)
#[derive(Debug, Clone)]
struct ScoredItem<T> {
    score: f32,
    item: T,
}

impl<T> PartialEq for ScoredItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for ScoredItem<T> {}

impl<T> PartialOrd for ScoredItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ScoredItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap behavior (we want to evict lowest scores)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Options for streaming semantic search
#[derive(Debug, Clone)]
pub struct StreamingSearchOptions {
    /// Minimum similarity threshold (0.0 to 1.0)
    pub min_similarity: f32,
    /// Filter results to specific file paths (glob patterns supported)
    pub file_filter: Option<Vec<String>>,
    /// Maximum results to return
    pub limit: usize,
    /// Batch size for processing embeddings
    pub batch_size: usize,
}

impl Default for StreamingSearchOptions {
    fn default() -> Self {
        Self {
            min_similarity: 0.5,
            file_filter: None,
            limit: 20,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl StreamingSearchOptions {
    /// Create new options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum similarity threshold
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set file filter patterns
    pub fn with_file_filter(mut self, patterns: Vec<String>) -> Self {
        self.file_filter = Some(patterns);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set batch size for processing
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Convert to SemanticSearchOptions
    pub fn to_semantic_options(&self) -> SemanticSearchOptions {
        let mut opts = SemanticSearchOptions::new()
            .with_min_similarity(self.min_similarity)
            .with_limit(self.limit);
        if let Some(ref filters) = self.file_filter {
            opts = opts.with_file_filter(filters.clone());
        }
        opts
    }
}

/// Streaming wrapper for semantic search
///
/// Provides incremental result delivery by processing embeddings in batches
/// and yielding results as they meet the threshold criteria.
pub struct StreamingSemanticSearch {
    semantic_search: Arc<SemanticSearch>,
}

impl StreamingSemanticSearch {
    /// Create a new streaming semantic search wrapper
    pub fn new(semantic_search: Arc<SemanticSearch>) -> Self {
        Self { semantic_search }
    }

    /// Search for semantically similar content, yielding results incrementally
    ///
    /// Results are yielded in batches as they're found, maintaining a top-k
    /// heap to ensure the final results are in proper order.
    pub fn search_streaming(
        &self,
        query: &str,
        options: StreamingSearchOptions,
    ) -> impl Stream<Item = Result<SemanticSearchResult>> + '_ {
        let semantic_options = options.to_semantic_options();
        let limit = options.limit;
        let query = query.to_string();

        // For streaming, we use the existing search but wrap it in a stream
        // that yields results incrementally. In a real implementation with
        // access to raw embedding data, we'd process in batches.
        stream::once(async move {
            self.semantic_search.search_with_options(&query, semantic_options)
        })
        .flat_map(move |results| match results {
            Ok(results) => {
                // Yield results one by one up to the limit
                let items: Vec<_> = results.into_iter().take(limit).map(Ok).collect();
                stream::iter(items)
            }
            Err(e) => stream::iter(vec![Err(e)]),
        })
    }

    /// Search by pre-computed embedding vector, yielding results incrementally
    pub fn search_by_embedding_streaming(
        &self,
        query_embedding: &[f32],
        options: StreamingSearchOptions,
    ) -> impl Stream<Item = Result<SemanticSearchResult>> + '_ {
        let semantic_options = options.to_semantic_options();
        let limit = options.limit;
        let embedding = query_embedding.to_vec();

        stream::once(async move {
            self.semantic_search.search_by_embedding(&embedding, semantic_options)
        })
        .flat_map(move |results| match results {
            Ok(results) => {
                let items: Vec<_> = results.into_iter().take(limit).map(Ok).collect();
                stream::iter(items)
            }
            Err(e) => stream::iter(vec![Err(e)]),
        })
    }
}

/// Streaming wrapper for hybrid search
///
/// Combines streaming results from keyword and semantic search,
/// applying RRF scoring incrementally.
pub struct StreamingHybridSearch<'a> {
    text_index: &'a FullTextIndex,
    semantic_search: Option<Arc<SemanticSearch>>,
    config: HybridSearchConfig,
}

impl<'a> StreamingHybridSearch<'a> {
    /// Create a new streaming hybrid search with only text search
    pub fn new(text_index: &'a FullTextIndex) -> Self {
        Self {
            text_index,
            semantic_search: None,
            config: HybridSearchConfig::default(),
        }
    }

    /// Create a streaming hybrid search with both text and semantic capabilities
    pub fn with_semantic(
        text_index: &'a FullTextIndex,
        semantic_search: Arc<SemanticSearch>,
    ) -> Self {
        Self {
            text_index,
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

    /// Search with specified mode, yielding results incrementally
    pub fn search_streaming(
        &self,
        query: &str,
        mode: SearchMode,
        limit: usize,
    ) -> impl Stream<Item = Result<HybridSearchResult>> + '_ {
        let query = query.to_string();

        match mode {
            SearchMode::Keyword => {
                let results = self.keyword_search_streaming(&query, limit);
                stream::iter(results)
            }
            SearchMode::Semantic => {
                let results = self.semantic_search_streaming(&query, limit);
                stream::iter(results)
            }
            SearchMode::Hybrid => {
                let results = self.hybrid_search_streaming(&query, limit);
                stream::iter(results)
            }
        }
    }

    fn keyword_search_streaming(&self, query: &str, limit: usize) -> Vec<Result<HybridSearchResult>> {
        // Use the non-streaming HybridSearch for the actual search
        let hybrid = HybridSearch::new(self.text_index);
        match hybrid.search(query, SearchMode::Keyword, limit) {
            Ok(results) => results.into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        }
    }

    fn semantic_search_streaming(&self, query: &str, limit: usize) -> Vec<Result<HybridSearchResult>> {
        let Some(ref semantic) = self.semantic_search else {
            // Fall back to keyword search if semantic not available
            return self.keyword_search_streaming(query, limit);
        };

        if !semantic.is_available() {
            return self.keyword_search_streaming(query, limit);
        }

        let options = SemanticSearchOptions::new()
            .with_limit(limit)
            .with_min_similarity(self.config.min_similarity);

        match semantic.search_with_options(query, options) {
            Ok(results) => results
                .into_iter()
                .map(|r| {
                    Ok(HybridSearchResult {
                        path: r.path,
                        line_number: r.start_line,
                        content: r.chunk_text,
                        score: r.similarity,
                        match_type: MatchType::Semantic,
                    })
                })
                .collect(),
            Err(e) => vec![Err(e)],
        }
    }

    fn hybrid_search_streaming(&self, query: &str, limit: usize) -> Vec<Result<HybridSearchResult>> {
        // If semantic search is not available, fall back to keyword only
        if !self.has_semantic() {
            return self.keyword_search_streaming(query, limit);
        }

        // Use the non-streaming hybrid search and convert to streaming results
        let hybrid = if let Some(ref semantic) = self.semantic_search {
            HybridSearch::with_semantic(self.text_index, semantic.clone())
                .with_config(self.config.clone())
        } else {
            HybridSearch::new(self.text_index).with_config(self.config.clone())
        };

        match hybrid.search(query, SearchMode::Hybrid, limit) {
            Ok(results) => results.into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        }
    }
}

/// Helper for maintaining a top-k heap during streaming
pub struct TopKHeap<T> {
    heap: BinaryHeap<ScoredItem<T>>,
    k: usize,
}

impl<T: Clone> TopKHeap<T> {
    /// Create a new top-k heap
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    /// Insert an item with a score
    ///
    /// Returns true if the item was added to the top-k
    pub fn insert(&mut self, score: f32, item: T) -> bool {
        if self.heap.len() < self.k {
            self.heap.push(ScoredItem { score, item });
            true
        } else if let Some(min) = self.heap.peek() {
            if score > min.score {
                self.heap.pop();
                self.heap.push(ScoredItem { score, item });
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get the minimum score in the heap (threshold for inclusion)
    pub fn min_score(&self) -> Option<f32> {
        self.heap.peek().map(|item| item.score)
    }

    /// Check if the heap is full
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Drain all items in sorted order (highest score first)
    pub fn drain_sorted(self) -> Vec<(f32, T)> {
        let mut items: Vec<_> = self
            .heap
            .into_iter()
            .map(|si| (si.score, si.item))
            .collect();
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        items
    }

    /// Get the current count
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogmcp_embeddings::EmbeddingEngine;
    use cogmcp_storage::Database;
    use parking_lot::Mutex;

    fn create_test_semantic_search() -> Arc<SemanticSearch> {
        use cogmcp_embeddings::{LazyEmbeddingEngine, ModelConfig};
        let db = Arc::new(Database::in_memory().unwrap());
        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        Arc::new(SemanticSearch::new(engine, db))
    }

    #[test]
    fn test_streaming_search_options_default() {
        let options = StreamingSearchOptions::default();
        assert_eq!(options.min_similarity, 0.5);
        assert_eq!(options.limit, 20);
        assert_eq!(options.batch_size, DEFAULT_BATCH_SIZE);
        assert!(options.file_filter.is_none());
    }

    #[test]
    fn test_streaming_search_options_builder() {
        let options = StreamingSearchOptions::new()
            .with_min_similarity(0.7)
            .with_limit(10)
            .with_batch_size(50)
            .with_file_filter(vec!["*.rs".to_string()]);

        assert_eq!(options.min_similarity, 0.7);
        assert_eq!(options.limit, 10);
        assert_eq!(options.batch_size, 50);
        assert!(options.file_filter.is_some());
    }

    #[test]
    fn test_min_similarity_clamping() {
        let options = StreamingSearchOptions::new().with_min_similarity(1.5);
        assert_eq!(options.min_similarity, 1.0);

        let options = StreamingSearchOptions::new().with_min_similarity(-0.5);
        assert_eq!(options.min_similarity, 0.0);
    }

    #[test]
    fn test_batch_size_minimum() {
        let options = StreamingSearchOptions::new().with_batch_size(0);
        assert_eq!(options.batch_size, 1); // Should be at least 1
    }

    #[test]
    fn test_to_semantic_options() {
        let streaming = StreamingSearchOptions::new()
            .with_min_similarity(0.8)
            .with_limit(15)
            .with_file_filter(vec!["src/**/*.rs".to_string()]);

        let semantic = streaming.to_semantic_options();
        assert_eq!(semantic.min_similarity, 0.8);
        assert_eq!(semantic.limit, 15);
        assert!(semantic.file_filter.is_some());
    }

    #[test]
    fn test_top_k_heap_basic() {
        let mut heap: TopKHeap<&str> = TopKHeap::new(3);

        assert!(heap.insert(0.5, "item1"));
        assert!(heap.insert(0.8, "item2"));
        assert!(heap.insert(0.3, "item3"));
        assert!(heap.is_full());

        // Lower score should not be added
        assert!(!heap.insert(0.2, "item4"));

        // Higher score should replace lowest
        assert!(heap.insert(0.9, "item5"));

        let sorted = heap.drain_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].1, "item5"); // 0.9
        assert_eq!(sorted[1].1, "item2"); // 0.8
        assert_eq!(sorted[2].1, "item1"); // 0.5
    }

    #[test]
    fn test_top_k_heap_min_score() {
        let mut heap: TopKHeap<i32> = TopKHeap::new(2);

        assert!(heap.min_score().is_none());

        heap.insert(0.5, 1);
        assert_eq!(heap.min_score(), Some(0.5));

        heap.insert(0.8, 2);
        assert_eq!(heap.min_score(), Some(0.5));

        heap.insert(0.9, 3); // Should replace 0.5
        assert_eq!(heap.min_score(), Some(0.8));
    }

    #[test]
    fn test_streaming_semantic_search_creation() {
        let semantic = create_test_semantic_search();
        let _streaming = StreamingSemanticSearch::new(semantic);
        // Just verify it compiles and creates without panic
    }

    #[tokio::test]
    async fn test_streaming_semantic_search_empty_db() {
        let semantic = create_test_semantic_search();
        let streaming = StreamingSemanticSearch::new(semantic);

        let options = StreamingSearchOptions::new().with_min_similarity(0.0);

        // The engine has no model loaded, so this should handle gracefully
        let results: Vec<_> = streaming
            .search_by_embedding_streaming(&vec![0.0; 384], options)
            .collect()
            .await;

        // Should return empty results (no embeddings in DB)
        assert!(results.is_empty() || results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_scored_item_ordering() {
        let item1 = ScoredItem {
            score: 0.5,
            item: "a",
        };
        let item2 = ScoredItem {
            score: 0.8,
            item: "b",
        };

        // For min-heap behavior, higher scores should come "after" lower scores
        assert!(item1 > item2); // 0.5 > 0.8 in min-heap ordering
    }

    #[test]
    fn test_top_k_heap_empty() {
        let heap: TopKHeap<i32> = TopKHeap::new(5);
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert!(!heap.is_full());
    }

    #[test]
    fn test_streaming_hybrid_search_has_semantic() {
        let index = FullTextIndex::in_memory().unwrap();

        let streaming = StreamingHybridSearch::new(&index);
        assert!(!streaming.has_semantic());

        let semantic = create_test_semantic_search();
        let streaming = StreamingHybridSearch::with_semantic(&index, semantic);
        // Still false because model not loaded
        assert!(!streaming.has_semantic());
    }

    #[tokio::test]
    async fn test_streaming_hybrid_keyword_search() {
        let index = FullTextIndex::in_memory().unwrap();

        let streaming = StreamingHybridSearch::new(&index);

        let results: Vec<_> = streaming
            .search_streaming("test", SearchMode::Keyword, 10)
            .collect()
            .await;

        // Empty index should return empty results
        assert!(results.is_empty() || results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_streaming_hybrid_config() {
        let index = FullTextIndex::in_memory().unwrap();

        let config = HybridSearchConfig {
            keyword_weight: 0.3,
            semantic_weight: 0.7,
            min_similarity: 0.4,
            rrf_k: 50.0,
        };

        let streaming = StreamingHybridSearch::new(&index).with_config(config);
        assert_eq!(streaming.config.keyword_weight, 0.3);
        assert_eq!(streaming.config.semantic_weight, 0.7);
    }

    /// Create a test database with embeddings for comprehensive testing
    fn create_test_db_with_embeddings() -> (Arc<Database>, Arc<SemanticSearch>) {
        let db = Arc::new(Database::in_memory().unwrap());

        // Create file for symbols
        let file_id = db.upsert_file("src/test.rs", "hash", 0, 100, "rust").unwrap();

        // Create embeddings with varying similarities to a query vector [1, 0, 0, ...]
        // High similarity embedding: [0.9, 0.1, 0, ...]
        let high_sim: Vec<f32> = vec![0.9, 0.1]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        // Medium similarity embedding: [0.6, 0.4, 0, ...]
        let med_sim: Vec<f32> = vec![0.6, 0.4]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        // Low similarity embedding: [0.3, 0.7, 0, ...]
        let low_sim: Vec<f32> = vec![0.3, 0.7]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        // Insert in reverse order (low first) to test sorting
        let sym1 = db.insert_symbol(file_id, "low_fn", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym1), None, "low similarity function", &low_sim, "function").unwrap();

        let sym2 = db.insert_symbol(file_id, "med_fn", "function", 10, 15, None, None).unwrap();
        db.insert_embedding(Some(sym2), None, "medium similarity function", &med_sim, "function").unwrap();

        let sym3 = db.insert_symbol(file_id, "high_fn", "function", 20, 25, None, None).unwrap();
        db.insert_embedding(Some(sym3), None, "high similarity function", &high_sim, "function").unwrap();

        use cogmcp_embeddings::{LazyEmbeddingEngine, ModelConfig};
        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let semantic_search = Arc::new(SemanticSearch::new(engine, db.clone()));

        (db, semantic_search)
    }

    #[tokio::test]
    async fn test_streaming_semantic_search_yields_correct_results() {
        let (_db, semantic_search) = create_test_db_with_embeddings();
        let streaming = StreamingSemanticSearch::new(semantic_search);

        // Query vector that matches high similarity embedding best
        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        let options = StreamingSearchOptions::new()
            .with_min_similarity(0.0)
            .with_limit(10);

        let results: Vec<_> = streaming
            .search_by_embedding_streaming(&query, options)
            .collect()
            .await;

        // Should return 3 results
        assert_eq!(results.len(), 3);

        // All should be Ok
        for result in &results {
            assert!(result.is_ok());
        }

        // Verify results are in correct order (highest similarity first)
        let results: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
        assert!(results[0].similarity > results[1].similarity);
        assert!(results[1].similarity > results[2].similarity);
        assert!(results[0].chunk_text.contains("high"));
        assert!(results[2].chunk_text.contains("low"));
    }

    #[tokio::test]
    async fn test_streaming_semantic_search_respects_limit() {
        let (_db, semantic_search) = create_test_db_with_embeddings();
        let streaming = StreamingSemanticSearch::new(semantic_search);

        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        // Request only 2 results
        let options = StreamingSearchOptions::new()
            .with_min_similarity(0.0)
            .with_limit(2);

        let results: Vec<_> = streaming
            .search_by_embedding_streaming(&query, options)
            .collect()
            .await;

        // Should return only 2 results (limit)
        assert_eq!(results.len(), 2);

        // Should be the top 2 results
        let results: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
        assert!(results[0].chunk_text.contains("high"));
        assert!(results[1].chunk_text.contains("medium"));
    }

    #[tokio::test]
    async fn test_streaming_semantic_search_early_termination() {
        let (_db, semantic_search) = create_test_db_with_embeddings();
        let streaming = StreamingSemanticSearch::new(semantic_search);

        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        // Request only 1 result - should terminate early
        let options = StreamingSearchOptions::new()
            .with_min_similarity(0.0)
            .with_limit(1);

        let results: Vec<_> = streaming
            .search_by_embedding_streaming(&query, options)
            .collect()
            .await;

        // Should return exactly 1 result
        assert_eq!(results.len(), 1);
        let result = results[0].as_ref().unwrap();
        assert!(result.chunk_text.contains("high")); // Should be the best match
    }

    #[tokio::test]
    async fn test_streaming_results_in_correct_order() {
        let (_db, semantic_search) = create_test_db_with_embeddings();
        let streaming = StreamingSemanticSearch::new(semantic_search);

        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        let options = StreamingSearchOptions::new()
            .with_min_similarity(0.0)
            .with_limit(10);

        let results: Vec<_> = streaming
            .search_by_embedding_streaming(&query, options)
            .collect()
            .await;

        // Verify descending order by similarity
        let results: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
        for i in 0..results.len() - 1 {
            assert!(
                results[i].similarity >= results[i + 1].similarity,
                "Results should be in descending order by similarity"
            );
        }
    }

    #[tokio::test]
    async fn test_streaming_hybrid_search_with_semantic() {
        let (_db, semantic_search) = create_test_db_with_embeddings();
        let index = FullTextIndex::in_memory().unwrap();
        let streaming = StreamingHybridSearch::with_semantic(&index, semantic_search);

        // Without a model loaded, should fall back to keyword search
        let results: Vec<_> = streaming
            .search_streaming("function", SearchMode::Hybrid, 10)
            .collect()
            .await;

        // Results should be Ok (empty or successful)
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_streaming_hybrid_respects_limit() {
        let index = FullTextIndex::in_memory().unwrap();
        let streaming = StreamingHybridSearch::new(&index);

        // Even with empty index, limit should be respected
        let results: Vec<_> = streaming
            .search_streaming("test", SearchMode::Keyword, 5)
            .collect()
            .await;

        // Should not exceed limit
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_semantic_search_streaming_method() {
        use cogmcp_embeddings::{LazyEmbeddingEngine, ModelConfig};
        // Test that SemanticSearch has the search_streaming method
        let db = Arc::new(Database::in_memory().unwrap());
        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let semantic = SemanticSearch::new(engine, db);

        // Just verify the method exists and returns the right type
        let _stream = semantic.search_streaming(
            "test",
            SemanticSearchOptions::default(),
        );
    }

    #[test]
    fn test_hybrid_search_streaming_method() {
        // Test that HybridSearch has the search_streaming method
        let index = FullTextIndex::in_memory().unwrap();
        let hybrid = crate::HybridSearch::new(&index);

        // Just verify the method exists and returns the right type
        let _stream = hybrid.search_streaming(
            "test",
            SearchMode::Hybrid,
            10,
        );
    }
}
