//! Search result caching layer
//!
//! This module provides multi-level caching for search operations to reduce
//! redundant computation and improve response latency for repeated queries.

use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

use crate::semantic::SemanticSearchResult;

/// Configuration for the search cache
#[derive(Debug, Clone)]
pub struct SearchCacheConfig {
    /// Maximum number of cached search results
    pub max_entries: usize,
    /// TTL for cached results
    pub result_ttl: Duration,
    /// TTL for cached embeddings
    pub embedding_ttl: Duration,
    /// Whether caching is enabled
    pub enabled: bool,
}

impl Default for SearchCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            result_ttl: Duration::from_secs(300),     // 5 minutes
            embedding_ttl: Duration::from_secs(3600), // 1 hour
            enabled: true,
        }
    }
}

/// Cached search result with metadata
#[derive(Clone)]
struct CachedResult {
    results: Vec<SemanticSearchResult>,
    created_at: Instant,
    index_version: u64,
}

/// Cached embedding with metadata
#[derive(Clone)]
struct CachedEmbedding {
    embedding: Vec<f32>,
    created_at: Instant,
}

/// Statistics for cache monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Current number of entries in result cache
    pub result_cache_size: usize,
    /// Current number of entries in embedding cache
    pub embedding_cache_size: usize,
    /// Current index version
    pub index_version: u64,
}

impl CacheStats {
    /// Calculate the cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64) / (total as f64)
        }
    }
}

/// Multi-level cache for search operations
///
/// Provides:
/// 1. LRU cache for full search results
/// 2. LRU cache for query embeddings
/// 3. Index version tracking for cache invalidation
/// 4. Statistics for monitoring
pub struct SearchCache {
    /// Cache of query -> search results
    result_cache: RwLock<LruCache<String, CachedResult>>,
    /// Cache of query -> embedding
    embedding_cache: RwLock<LruCache<String, CachedEmbedding>>,
    /// TTL for cached results
    result_ttl: Duration,
    /// TTL for cached embeddings
    embedding_ttl: Duration,
    /// Index version for invalidation
    index_version: AtomicU64,
    /// Cache hit counter
    hit_count: AtomicU64,
    /// Cache miss counter
    miss_count: AtomicU64,
    /// Whether caching is enabled
    enabled: bool,
}

impl SearchCache {
    /// Create a new search cache with the given configuration
    pub fn new(config: SearchCacheConfig) -> Self {
        let max_entries = NonZeroUsize::new(config.max_entries.max(1)).unwrap();

        Self {
            result_cache: RwLock::new(LruCache::new(max_entries)),
            embedding_cache: RwLock::new(LruCache::new(max_entries)),
            result_ttl: config.result_ttl,
            embedding_ttl: config.embedding_ttl,
            index_version: AtomicU64::new(0),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
            enabled: config.enabled,
        }
    }

    /// Create a new search cache with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SearchCacheConfig::default())
    }

    /// Get or compute search results for a query
    ///
    /// If the results are cached and valid, returns them immediately.
    /// Otherwise, calls the compute function and caches the results.
    pub fn get_or_compute_results<F>(
        &self,
        cache_key: &str,
        compute: F,
    ) -> cogmcp_core::Result<Vec<SemanticSearchResult>>
    where
        F: FnOnce() -> cogmcp_core::Result<Vec<SemanticSearchResult>>,
    {
        if !self.enabled {
            return compute();
        }

        let current_version = self.index_version.load(Ordering::SeqCst);

        // Try to get from cache
        {
            let mut cache = self.result_cache.write();
            if let Some(cached) = cache.get(cache_key) {
                // Check if still valid (not expired and same index version)
                if cached.created_at.elapsed() < self.result_ttl
                    && cached.index_version == current_version
                {
                    self.hit_count.fetch_add(1, Ordering::Relaxed);
                    debug!("Search cache hit for key: {}", cache_key);
                    return Ok(cached.results.clone());
                }
            }
        }

        // Cache miss - compute results
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        debug!("Search cache miss for key: {}", cache_key);

        let results = compute()?;

        // Store in cache
        {
            let mut cache = self.result_cache.write();
            cache.put(
                cache_key.to_string(),
                CachedResult {
                    results: results.clone(),
                    created_at: Instant::now(),
                    index_version: current_version,
                },
            );
        }

        Ok(results)
    }

    /// Get or compute embedding for a query
    pub fn get_or_compute_embedding<F>(
        &self,
        query: &str,
        compute: F,
    ) -> cogmcp_core::Result<Vec<f32>>
    where
        F: FnOnce() -> cogmcp_core::Result<Vec<f32>>,
    {
        if !self.enabled {
            return compute();
        }

        let normalized_query = normalize_query(query);

        // Try to get from cache
        {
            let mut cache = self.embedding_cache.write();
            if let Some(cached) = cache.get(&normalized_query) {
                if cached.created_at.elapsed() < self.embedding_ttl {
                    self.hit_count.fetch_add(1, Ordering::Relaxed);
                    debug!("Embedding cache hit for query: {}", query);
                    return Ok(cached.embedding.clone());
                }
            }
        }

        // Cache miss - compute embedding
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        debug!("Embedding cache miss for query: {}", query);

        let embedding = compute()?;

        // Store in cache
        {
            let mut cache = self.embedding_cache.write();
            cache.put(
                normalized_query,
                CachedEmbedding {
                    embedding: embedding.clone(),
                    created_at: Instant::now(),
                },
            );
        }

        Ok(embedding)
    }

    /// Invalidate cache on index changes
    ///
    /// This increments the index version, causing all result cache entries
    /// to be considered stale on next access.
    pub fn invalidate_on_index_change(&self) {
        let old_version = self.index_version.fetch_add(1, Ordering::SeqCst);
        debug!(
            "Index version incremented from {} to {}",
            old_version,
            old_version + 1
        );
    }

    /// Clear all cached data
    pub fn clear(&self) {
        {
            let mut cache = self.result_cache.write();
            cache.clear();
        }
        {
            let mut cache = self.embedding_cache.write();
            cache.clear();
        }
        debug!("Search cache cleared");
    }

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        let result_cache_size = {
            let cache = self.result_cache.read();
            cache.len()
        };
        let embedding_cache_size = {
            let cache = self.embedding_cache.read();
            cache.len()
        };

        CacheStats {
            hits: self.hit_count.load(Ordering::Relaxed),
            misses: self.miss_count.load(Ordering::Relaxed),
            result_cache_size,
            embedding_cache_size,
            index_version: self.index_version.load(Ordering::SeqCst),
        }
    }

    /// Get current index version
    pub fn index_version(&self) -> u64 {
        self.index_version.load(Ordering::SeqCst)
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for SearchCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Normalize a query string for better cache hit rates
///
/// Normalization includes:
/// - Converting to lowercase
/// - Collapsing whitespace
/// - Trimming leading/trailing whitespace
pub fn normalize_query(query: &str) -> String {
    query
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate a cache key for search operations
///
/// The key includes the normalized query and search parameters.
pub fn make_cache_key(query: &str, limit: usize, min_similarity: f32) -> String {
    format!(
        "{}:{}:{:.2}",
        normalize_query(query),
        limit,
        min_similarity
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::ChunkType;

    fn make_test_results() -> Vec<SemanticSearchResult> {
        vec![
            SemanticSearchResult {
                path: "test.rs".to_string(),
                chunk_text: "fn test() {}".to_string(),
                similarity: 0.95,
                chunk_type: ChunkType::Function,
                start_line: Some(1),
                end_line: Some(3),
                symbol_id: Some(1),
                context: None,
            },
        ]
    }

    #[test]
    fn test_normalize_query() {
        assert_eq!(normalize_query("  HELLO   World  "), "hello world");
        assert_eq!(normalize_query("Test"), "test");
        assert_eq!(normalize_query("  multiple   spaces  "), "multiple spaces");
        assert_eq!(normalize_query("already normalized"), "already normalized");
        assert_eq!(normalize_query(""), "");
    }

    #[test]
    fn test_make_cache_key() {
        let key1 = make_cache_key("test query", 10, 0.5);
        let key2 = make_cache_key("  TEST   QUERY  ", 10, 0.5);
        assert_eq!(key1, key2); // Same normalized query

        let key3 = make_cache_key("test query", 20, 0.5);
        assert_ne!(key1, key3); // Different limit

        let key4 = make_cache_key("test query", 10, 0.7);
        assert_ne!(key1, key4); // Different similarity
    }

    #[test]
    fn test_cache_hit() {
        let cache = SearchCache::with_defaults();
        let results = make_test_results();
        let results_clone = results.clone();

        // First call - should compute
        let result1 = cache
            .get_or_compute_results("test_key", || Ok(results))
            .unwrap();
        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].path, "test.rs");

        // Second call - should hit cache (compute should not be called)
        let result2 = cache
            .get_or_compute_results("test_key", || {
                panic!("Should not compute - cache hit expected");
            })
            .unwrap();
        assert_eq!(result1, result2);

        // Verify stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        // Different key - should miss
        let _ = cache
            .get_or_compute_results("different_key", || Ok(results_clone))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = SearchCache::with_defaults();
        let results = make_test_results();
        let results_clone = results.clone();

        // Store results
        let _ = cache
            .get_or_compute_results("test_key", || Ok(results))
            .unwrap();

        // Invalidate cache
        cache.invalidate_on_index_change();

        // Should miss now due to version mismatch
        let mut computed = false;
        let _ = cache
            .get_or_compute_results("test_key", || {
                computed = true;
                Ok(results_clone)
            })
            .unwrap();

        assert!(computed, "Should have computed due to invalidation");
    }

    #[test]
    fn test_embedding_cache() {
        let cache = SearchCache::with_defaults();
        let embedding = vec![0.1, 0.2, 0.3];
        let embedding_clone = embedding.clone();

        // First call - should compute
        let result1 = cache
            .get_or_compute_embedding("test query", || Ok(embedding))
            .unwrap();
        assert_eq!(result1, vec![0.1, 0.2, 0.3]);

        // Second call with normalized equivalent - should hit cache
        let result2 = cache
            .get_or_compute_embedding("  TEST   QUERY  ", || {
                panic!("Should not compute - cache hit expected");
            })
            .unwrap();
        assert_eq!(result1, result2);

        // Verify stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        // Different query - should miss
        let _ = cache
            .get_or_compute_embedding("different query", || Ok(embedding_clone))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_cache_disabled() {
        let config = SearchCacheConfig {
            enabled: false,
            ..Default::default()
        };
        let cache = SearchCache::new(config);

        let mut compute_count = 0;
        let _ = cache
            .get_or_compute_results("test_key", || {
                compute_count += 1;
                Ok(make_test_results())
            })
            .unwrap();

        let _ = cache
            .get_or_compute_results("test_key", || {
                compute_count += 1;
                Ok(make_test_results())
            })
            .unwrap();

        // Both calls should have computed (cache disabled)
        assert_eq!(compute_count, 2);
    }

    #[test]
    fn test_cache_clear() {
        let cache = SearchCache::with_defaults();

        // Add some entries
        let _ = cache
            .get_or_compute_results("key1", || Ok(make_test_results()))
            .unwrap();
        let _ = cache
            .get_or_compute_embedding("query1", || Ok(vec![0.1, 0.2]))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.result_cache_size, 1);
        assert_eq!(stats.embedding_cache_size, 1);

        // Clear cache
        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.result_cache_size, 0);
        assert_eq!(stats.embedding_cache_size, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = SearchCache::with_defaults();

        // Initial stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.result_cache_size, 0);
        assert_eq!(stats.embedding_cache_size, 0);
        assert_eq!(stats.index_version, 0);
        assert_eq!(stats.hit_rate(), 0.0);

        // Add entries and verify
        let _ = cache
            .get_or_compute_results("key1", || Ok(make_test_results()))
            .unwrap();
        let _ = cache
            .get_or_compute_results("key1", || Ok(make_test_results()))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_lru_eviction() {
        let config = SearchCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let cache = SearchCache::new(config);

        // Fill cache to capacity
        let _ = cache
            .get_or_compute_results("key1", || Ok(make_test_results()))
            .unwrap();
        let _ = cache
            .get_or_compute_results("key2", || Ok(make_test_results()))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.result_cache_size, 2);

        // Add another entry - should evict oldest
        let _ = cache
            .get_or_compute_results("key3", || Ok(make_test_results()))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.result_cache_size, 2); // Still 2 due to LRU eviction
    }

    #[test]
    fn test_search_result_equality() {
        let result1 = SemanticSearchResult {
            path: "test.rs".to_string(),
            chunk_text: "fn test() {}".to_string(),
            similarity: 0.95,
            chunk_type: ChunkType::Function,
            start_line: Some(1),
            end_line: Some(3),
            symbol_id: Some(1),
            context: None,
        };

        let result2 = SemanticSearchResult {
            path: "test.rs".to_string(),
            chunk_text: "fn test() {}".to_string(),
            similarity: 0.95,
            chunk_type: ChunkType::Function,
            start_line: Some(1),
            end_line: Some(3),
            symbol_id: Some(1),
            context: None,
        };

        // Both should have same path and content
        assert_eq!(result1.path, result2.path);
        assert_eq!(result1.chunk_text, result2.chunk_text);
        assert_eq!(result1.similarity, result2.similarity);
    }
}
