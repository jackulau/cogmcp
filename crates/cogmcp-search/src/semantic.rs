//! Semantic search using embeddings
//!
//! This module provides semantic search functionality that uses vector embeddings
//! to find semantically similar code chunks in the codebase. When HNSW indexing is
//! enabled and the number of embeddings exceeds the threshold, HNSW approximate
//! nearest neighbor search is used for O(log n) performance. Otherwise, brute-force
//! search is used.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use cogmcp_core::{Error, Result};
use cogmcp_embeddings::{EmbeddingEngine, LazyEmbeddingEngine};
use cogmcp_storage::{cache::Cache, Database};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::hnsw::{HnswConfig, HnswIndex};

use crate::cache::{CacheStats, SearchCache, SearchCacheConfig, make_cache_key};

/// Chunk type for categorizing search results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkType {
    /// A function or method
    Function,
    /// A struct, class, or type definition
    Type,
    /// A module or namespace
    Module,
    /// A constant or static variable
    Constant,
    /// A doc comment or documentation block
    Documentation,
    /// Generic code block
    Code,
}

impl ChunkType {
    /// Convert from string representation
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "function" | "method" | "fn" => ChunkType::Function,
            "type" | "struct" | "class" | "enum" | "interface" | "trait" => ChunkType::Type,
            "module" | "namespace" | "mod" => ChunkType::Module,
            "constant" | "const" | "static" => ChunkType::Constant,
            "doc" | "documentation" | "comment" => ChunkType::Documentation,
            _ => ChunkType::Code,
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkType::Function => "function",
            ChunkType::Type => "type",
            ChunkType::Module => "module",
            ChunkType::Constant => "constant",
            ChunkType::Documentation => "documentation",
            ChunkType::Code => "code",
        }
    }
}

/// Options for semantic search
#[derive(Debug, Clone)]
pub struct SemanticSearchOptions {
    /// Minimum similarity threshold (0.0 to 1.0)
    pub min_similarity: f32,
    /// Filter results to specific file paths (glob patterns supported)
    pub file_filter: Option<Vec<String>>,
    /// Filter by chunk types
    pub chunk_types: Option<Vec<ChunkType>>,
    /// Maximum results to return
    pub limit: usize,
}

impl Default for SemanticSearchOptions {
    fn default() -> Self {
        Self {
            min_similarity: 0.5,
            file_filter: None,
            chunk_types: None,
            limit: 20,
        }
    }
}

impl SemanticSearchOptions {
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

    /// Set chunk type filter
    pub fn with_chunk_types(mut self, types: Vec<ChunkType>) -> Self {
        self.chunk_types = Some(types);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Semantic search result
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticSearchResult {
    /// File path containing the match
    pub path: String,
    /// Matched chunk text
    pub chunk_text: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Type of chunk
    pub chunk_type: ChunkType,
    /// Start line number (1-indexed)
    pub start_line: Option<u32>,
    /// End line number (1-indexed)
    pub end_line: Option<u32>,
    /// Symbol ID if this is associated with a symbol
    pub symbol_id: Option<i64>,
    /// Context snippet around the match
    pub context: Option<String>,
}

/// Input for batch indexing a chunk of text
#[derive(Debug, Clone)]
pub struct ChunkInput {
    /// The text content to index
    pub chunk_text: String,
    /// Optional symbol ID if this chunk is associated with a symbol
    pub symbol_id: Option<i64>,
    /// Optional file ID if this chunk is associated with a file
    pub file_id: Option<i64>,
    /// Type of chunk
    pub chunk_type: ChunkType,
}

impl ChunkInput {
    /// Create a new chunk input
    pub fn new(chunk_text: impl Into<String>, chunk_type: ChunkType) -> Self {
        Self {
            chunk_text: chunk_text.into(),
            symbol_id: None,
            file_id: None,
            chunk_type,
        }
    }

    /// Set the symbol ID
    pub fn with_symbol_id(mut self, id: i64) -> Self {
        self.symbol_id = Some(id);
        self
    }

    /// Set the file ID
    pub fn with_file_id(mut self, id: i64) -> Self {
        self.file_id = Some(id);
        self
    }
}

/// Embedding with associated metadata for vector search
#[derive(Debug, Clone)]
struct EmbeddingRecord {
    #[allow(dead_code)]
    id: i64,
    symbol_id: Option<i64>,
    chunk_text: String,
    embedding: Vec<f32>,
    chunk_type: String,
    file_path: Option<String>,
    start_line: Option<u32>,
    end_line: Option<u32>,
}

/// Semantic search engine
///
/// Provides semantic similarity search over code embeddings by:
/// 1. Generating embeddings for query text
/// 2. Searching the vector storage for similar embeddings (using HNSW when available)
/// 3. Enriching results with file path and line number information
///
/// Uses `LazyEmbeddingEngine` to defer ONNX model loading until the first
/// actual search query, improving server startup time.
pub struct SemanticSearch {
    /// Lazy embedding engine for generating query embeddings (defers model loading)
    engine: Arc<LazyEmbeddingEngine>,
    /// Database for storing and retrieving embeddings
    db: Arc<Database>,
    /// LRU cache for query embeddings with TTL-based expiration
    query_cache: LruCacheWithTtl<String, Vec<f32>>,
    /// Cached embeddings from database for in-memory search
    embeddings_cache: RwLock<Option<Vec<EmbeddingRecord>>>,
    /// Multi-level search cache for results and embeddings
    search_cache: Arc<SearchCache>,
}

impl SemanticSearch {
    /// Create a new semantic search instance
    ///
    /// The `LazyEmbeddingEngine` defers model loading until the first search query,
    /// which can significantly improve server startup time.
    pub fn new(engine: Arc<LazyEmbeddingEngine>, db: Arc<Database>) -> Self {
        Self {
            engine,
            db,
            // Cache query embeddings for 5 minutes (legacy cache, kept for backward compat)
            query_cache: Cache::new(Duration::from_secs(300)),
            embeddings_cache: RwLock::new(None),
            search_cache: Arc::new(SearchCache::new(cache_config)),
        }
    }

    /// Check if the embedding model files are available for loading
    ///
    /// This returns true if the model files exist, even if they haven't
    /// been loaded yet (due to lazy loading). Use `is_loaded()` to check
    /// if the model has actually been loaded.
    pub fn is_available(&self) -> bool {
        self.engine.is_available()
    }

    /// Check if the embedding model is currently loaded in memory
    ///
    /// Returns true only if the model has been loaded (after the first
    /// embedding request). Returns false if the model is still in its
    /// lazy/unloaded state.
    pub fn is_loaded(&self) -> bool {
        self.engine.is_loaded()
    }

    /// Get the embedding dimension
    ///
    /// This returns the configured dimension without loading the model.
    pub fn embedding_dim(&self) -> usize {
        self.engine.embedding_dim()
    }

    /// Search for semantically similar content using a text query
    #[instrument(skip(self), level = "debug")]
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SemanticSearchResult>> {
        self.search_with_options(query, SemanticSearchOptions::default().with_limit(limit))
    }

    /// Search for semantically similar content with custom options
    #[instrument(skip(self), level = "debug")]
    pub fn search_with_options(
        &self,
        query: &str,
        options: SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        // Create cache key from query and options
        let cache_key = make_cache_key(query, options.limit, options.min_similarity);

        // Use search cache for results
        self.search_cache.get_or_compute_results(&cache_key, || {
            self.search_uncached(query, &options)
        })
    }

    /// Perform search without caching (internal implementation)
    fn search_uncached(&self, query: &str, options: &SemanticSearchOptions) -> Result<Vec<SemanticSearchResult>> {
        // Get query embedding using the search cache for embeddings
        let query_embedding = self.search_cache.get_or_compute_embedding(query, || {
            self.engine.lock().embed(query)
        })?;

        // Search with the embedding
        self.search_by_embedding_internal(&query_embedding, options)
    }

    /// Search using a pre-computed embedding vector
    #[instrument(skip(self, query_embedding), level = "debug")]
    pub fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        options: SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        self.search_by_embedding_internal(query_embedding, &options)
    }

    /// Internal search using a pre-computed embedding vector
    fn search_by_embedding_internal(
        &self,
        query_embedding: &[f32],
        options: &SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        // Load embeddings from database if not cached
        self.ensure_embeddings_loaded()?;

        // Decide whether to use HNSW or brute-force search
        if self.should_use_hnsw() {
            debug!("Using HNSW approximate nearest neighbor search");
            self.search_by_embedding_hnsw(query_embedding, options)
        } else {
            debug!("Using brute-force cosine similarity search");
            self.search_by_embedding_brute_force(query_embedding, options)
        }
    }

    /// Search using HNSW approximate nearest neighbor index
    fn search_by_embedding_hnsw(
        &self,
        query_embedding: &[f32],
        options: SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        let embeddings = self.embeddings_cache.read();
        let records = embeddings.as_ref().ok_or_else(|| {
            Error::Search("Embeddings cache not loaded".into())
        })?;

        let hnsw_guard = self.hnsw_index.read();
        let hnsw = hnsw_guard.as_ref().ok_or_else(|| {
            Error::Search("HNSW index not available".into())
        })?;

        // Request more results from HNSW to account for filtering
        // We request 3x the limit to handle filtering, then take the top N
        let hnsw_limit = options.limit * 3;
        let hnsw_results = hnsw.search_with_threshold(
            query_embedding,
            hnsw_limit,
            options.min_similarity,
        )?;

        // Map HNSW results back to records and apply additional filters
        let results: Vec<SemanticSearchResult> = hnsw_results
            .into_iter()
            .filter_map(|hnsw_result| {
                let record = records.get(hnsw_result.index)?;

                // Apply file filter if specified
                if let Some(ref patterns) = options.file_filter {
                    if let Some(ref path) = record.file_path {
                        let matches = patterns.iter().any(|pattern| {
                            path.contains(pattern) || glob_match(pattern, path)
                        });
                        if !matches {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }

                // Apply chunk type filter if specified
                if let Some(ref types) = options.chunk_types {
                    let chunk_type = ChunkType::from_str(&record.chunk_type);
                    if !types.contains(&chunk_type) {
                        return None;
                    }
                }

                Some(SemanticSearchResult {
                    path: record.file_path.clone().unwrap_or_default(),
                    chunk_text: record.chunk_text.clone(),
                    similarity: hnsw_result.similarity,
                    chunk_type: ChunkType::from_str(&record.chunk_type),
                    start_line: record.start_line,
                    end_line: record.end_line,
                    symbol_id: record.symbol_id,
                    context: None,
                })
            })
            .take(options.limit)
            .collect();

        // Results from HNSW are already sorted by similarity
        debug!("HNSW search returned {} results", results.len());
        Ok(results)
    }

    /// Search using brute-force cosine similarity
    fn search_by_embedding_brute_force(
        &self,
        query_embedding: &[f32],
        options: SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        let embeddings = self.embeddings_cache.read();
        let embeddings = embeddings.as_ref().ok_or_else(|| {
            Error::Search("Embeddings cache not loaded".into())
        })?;

        // Compute similarities and filter
        let mut results: Vec<(f32, &EmbeddingRecord)> = embeddings
            .iter()
            .filter_map(|record| {
                let similarity = EmbeddingEngine::cosine_similarity(query_embedding, &record.embedding);

                // Apply minimum similarity filter
                if similarity < options.min_similarity {
                    return None;
                }

                // Apply file filter if specified
                if let Some(ref patterns) = options.file_filter {
                    if let Some(ref path) = record.file_path {
                        let matches = patterns.iter().any(|pattern| {
                            path.contains(pattern) || glob_match(pattern, path)
                        });
                        if !matches {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }

                // Apply chunk type filter if specified
                if let Some(ref types) = options.chunk_types {
                    let chunk_type = ChunkType::from_str(&record.chunk_type);
                    if !types.contains(&chunk_type) {
                        return None;
                    }
                }

                Some((similarity, record))
            })
            .collect();

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results
        let results: Vec<SemanticSearchResult> = results
            .into_iter()
            .take(options.limit)
            .map(|(similarity, record)| SemanticSearchResult {
                path: record.file_path.clone().unwrap_or_default(),
                chunk_text: record.chunk_text.clone(),
                similarity,
                chunk_type: ChunkType::from_str(&record.chunk_type),
                start_line: record.start_line,
                end_line: record.end_line,
                symbol_id: record.symbol_id,
                context: None,
            })
            .collect();

        debug!("Brute-force search returned {} results", results.len());
        Ok(results)
    }

    /// Search for semantically similar content, yielding results incrementally as a stream
    ///
    /// This method processes embeddings and yields results as they meet the threshold,
    /// enabling early consumption of results without waiting for all comparisons to complete.
    #[instrument(skip(self), level = "debug")]
    pub fn search_streaming(
        &self,
        query: &str,
        options: SemanticSearchOptions,
    ) -> impl Stream<Item = Result<SemanticSearchResult>> + '_ {
        let query = query.to_string();

        stream::once(async move {
            // Get query embedding
            let query_embedding = self.get_or_compute_embedding(&query)?;

            // Perform the search
            self.search_by_embedding(&query_embedding, options)
        })
        .flat_map(|results| match results {
            Ok(results) => {
                // Yield results one by one
                let items: Vec<_> = results.into_iter().map(Ok).collect();
                stream::iter(items)
            }
            Err(e) => stream::iter(vec![Err(e)]),
        })
    }

    /// Search by pre-computed embedding vector, yielding results incrementally as a stream
    #[instrument(skip(self, query_embedding), level = "debug")]
    pub fn search_by_embedding_streaming(
        &self,
        query_embedding: &[f32],
        options: SemanticSearchOptions,
    ) -> impl Stream<Item = Result<SemanticSearchResult>> + '_ {
        let embedding = query_embedding.to_vec();
        let limit = options.limit;

        stream::once(async move { self.search_by_embedding(&embedding, options) }).flat_map(
            move |results| match results {
                Ok(results) => {
                    let items: Vec<_> = results.into_iter().take(limit).map(Ok).collect();
                    stream::iter(items)
                }
                Err(e) => stream::iter(vec![Err(e)]),
            },
        )
    }

    /// Get or compute embedding for a query string
    ///
    /// On the first call, this will trigger lazy loading of the embedding model
    /// if it hasn't been loaded yet.
    fn get_or_compute_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.query_cache.get(&query.to_string()) {
            debug!("Using cached embedding for query");
            return Ok(cached);
        }

        // Compute embedding (LazyEmbeddingEngine handles locking internally)
        let embedding = self.engine.embed(query)?;

        // Cache the result
        self.query_cache.insert(query.to_string(), embedding.clone());

        Ok(embedding)
    }

    /// Ensure embeddings are loaded from the database
    fn ensure_embeddings_loaded(&self) -> Result<()> {
        // Check if already loaded
        {
            let cache = self.embeddings_cache.read();
            if cache.is_some() {
                return Ok(());
            }
        }

        // Load from database
        let embeddings = self.load_embeddings_from_db()?;

        // Store in cache
        let mut cache = self.embeddings_cache.write();
        *cache = Some(embeddings);

        Ok(())
    }

    /// Load embeddings from the database
    fn load_embeddings_from_db(&self) -> Result<Vec<EmbeddingRecord>> {
        self.db.with_connection(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT e.id, e.symbol_id, e.chunk_text, e.embedding, e.chunk_type,
                           f.path, s.start_line, s.end_line
                    FROM embeddings e
                    LEFT JOIN symbols s ON e.symbol_id = s.id
                    LEFT JOIN files f ON s.file_id = f.id
                    "#,
                )
                .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

            let rows = stmt
                .query_map([], |row| {
                    let embedding_bytes: Vec<u8> = row.get(3)?;
                    let embedding: Vec<f32> = embedding_bytes
                        .chunks(4)
                        .map(|chunk| {
                            let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                            f32::from_le_bytes(arr)
                        })
                        .collect();

                    Ok(EmbeddingRecord {
                        id: row.get(0)?,
                        symbol_id: row.get(1)?,
                        chunk_text: row.get(2)?,
                        embedding,
                        chunk_type: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        file_path: row.get(5)?,
                        start_line: row.get(6)?,
                        end_line: row.get(7)?,
                    })
                })
                .map_err(|e| Error::Storage(format!("Failed to query embeddings: {}", e)))?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
            }

            debug!("Loaded {} embeddings from database", results.len());
            Ok(results)
        })
    }

    /// Invalidate the embeddings cache (call after indexing new content)
    ///
    /// This also invalidates the HNSW index since the embeddings have changed.
    /// Call `rebuild_hnsw_index()` after adding new embeddings to rebuild the index.
    pub fn invalidate_cache(&self) {
        let mut cache = self.embeddings_cache.write();
        *cache = None;
        self.query_cache.clear();
        // Also invalidate search cache on index change
        self.search_cache.invalidate_on_index_change();
    }

    /// Get cache statistics for monitoring
    pub fn cache_stats(&self) -> CacheStats {
        self.search_cache.stats()
    }

    /// Get the search cache for external use
    pub fn search_cache(&self) -> &Arc<SearchCache> {
        &self.search_cache
    }

    /// Index text and store its embedding
    ///
    /// On the first call, this will trigger lazy loading of the embedding model
    /// if it hasn't been loaded yet.
    pub fn index_chunk(
        &self,
        chunk_text: &str,
        symbol_id: Option<i64>,
        chunk_type: ChunkType,
    ) -> Result<i64> {
        // LazyEmbeddingEngine handles locking internally
        let embedding = self.engine.embed(chunk_text)?;
        let id = self.db.insert_embedding(
            symbol_id,
            None, // file_id
            chunk_text,
            &embedding,
            chunk_type.as_str(),
        )?;

        // Invalidate cache since we added new data
        self.invalidate_cache();

        Ok(id)
    }

    /// Index multiple text chunks in a single batch operation
    ///
    /// This is significantly more efficient than calling `index_chunk` multiple times
    /// as it uses batched ONNX inference and a single database transaction.
    ///
    /// # Arguments
    /// * `chunks` - Vector of chunk inputs to index
    /// * `batch_size` - Number of texts to process in each ONNX forward pass
    ///
    /// # Returns
    /// Vector of embedding IDs for the inserted chunks
    pub fn index_chunks_batch(
        &self,
        chunks: &[ChunkInput],
        batch_size: usize,
    ) -> Result<Vec<i64>> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        // Extract texts for batch embedding
        let texts: Vec<&str> = chunks.iter().map(|c| c.chunk_text.as_str()).collect();

        // Generate embeddings in batch
        let embeddings = self.engine.lock().embed_batch_optimized(&texts, batch_size)?;

        // Prepare inputs for batch database insertion
        let inputs: Vec<cogmcp_storage::EmbeddingInput> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, embedding)| cogmcp_storage::EmbeddingInput {
                symbol_id: chunk.symbol_id,
                file_id: chunk.file_id,
                chunk_text: chunk.chunk_text.clone(),
                embedding: embedding.clone(),
                chunk_type: chunk.chunk_type.as_str().to_string(),
            })
            .collect();

        // Insert all embeddings in a single transaction
        let ids = self.db.insert_embeddings_batch(&inputs)?;

        // Invalidate cache since we added new data
        self.invalidate_cache();

        Ok(ids)
    }

    /// Index multiple text chunks using parallel processing for very large batches
    ///
    /// Uses Rayon to process embedding generation across multiple CPU cores,
    /// then inserts all results in a single database transaction.
    ///
    /// # Arguments
    /// * `chunks` - Vector of chunk inputs to index
    /// * `batch_size` - Number of texts to process in each ONNX forward pass
    ///
    /// # Returns
    /// Vector of embedding IDs for the inserted chunks
    pub fn index_chunks_parallel(
        &self,
        chunks: &[ChunkInput],
        batch_size: usize,
    ) -> Result<Vec<i64>> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        // Extract texts for batch embedding
        let texts: Vec<&str> = chunks.iter().map(|c| c.chunk_text.as_str()).collect();

        // Generate embeddings using parallel batch processing
        let embeddings = self.engine.lock().embed_large_batch(&texts, batch_size)?;

        // Prepare inputs for batch database insertion
        let inputs: Vec<cogmcp_storage::EmbeddingInput> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, embedding)| cogmcp_storage::EmbeddingInput {
                symbol_id: chunk.symbol_id,
                file_id: chunk.file_id,
                chunk_text: chunk.chunk_text.clone(),
                embedding: embedding.clone(),
                chunk_type: chunk.chunk_type.as_str().to_string(),
            })
            .collect();

        // Insert all embeddings in a single transaction
        let ids = self.db.insert_embeddings_batch(&inputs)?;

        // Invalidate cache since we added new data
        self.invalidate_cache();

        Ok(ids)
    }
}

impl Default for SemanticSearch {
    fn default() -> Self {
        use cogmcp_embeddings::ModelConfig;
        Self::new(
            Arc::new(LazyEmbeddingEngine::new(ModelConfig::default())),
            Arc::new(Database::in_memory().expect("Failed to create in-memory database")),
            SearchCacheConfig::default(),
        )
    }
}

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }

    // Handle ** for any path segment (recursive matching)
    if pattern.contains("**") {
        let parts: Vec<&str> = pattern.split("**").collect();
        if parts.len() == 2 {
            let prefix = parts[0].trim_end_matches('/');
            let suffix = parts[1].trim_start_matches('/');

            // Check prefix matches
            let prefix_ok = prefix.is_empty() || text.starts_with(prefix);
            if !prefix_ok {
                return false;
            }

            // For suffix, we need to check if the remaining text ends with the suffix pattern
            // Handle suffix patterns like "*.rs"
            if !suffix.is_empty() {
                if suffix.contains('*') {
                    // Handle wildcard in suffix (e.g., "*.rs")
                    let suffix_parts: Vec<&str> = suffix.split('*').collect();
                    if suffix_parts.len() == 2 {
                        // For patterns like "*.rs", the text should end with ".rs"
                        return text.ends_with(suffix_parts[1]);
                    }
                }
                return text.ends_with(suffix);
            }
            return true;
        }
    }

    // Handle * for single segment wildcard
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return text.starts_with(parts[0]) && text.ends_with(parts[1]);
        }
    }

    // Simple substring match
    text.contains(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogmcp_embeddings::ModelConfig;

    fn create_test_search() -> SemanticSearch {
        // Create an in-memory database
        let db = Arc::new(Database::in_memory().unwrap());

        // Create a lazy embedding engine with default config (no model files)
        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));

        SemanticSearch::with_cache_config(engine, db, SearchCacheConfig::default())
    }

    #[test]
    fn test_semantic_search_creation() {
        let search = create_test_search();
        // With default config (empty paths), model files are not available
        assert!(!search.is_available());
        // Model is not loaded yet (lazy loading)
        assert!(!search.is_loaded());
    }

    #[test]
    fn test_search_options_builder() {
        let options = SemanticSearchOptions::new()
            .with_min_similarity(0.7)
            .with_limit(10)
            .with_file_filter(vec!["*.rs".to_string()])
            .with_chunk_types(vec![ChunkType::Function, ChunkType::Type]);

        assert_eq!(options.min_similarity, 0.7);
        assert_eq!(options.limit, 10);
        assert!(options.file_filter.is_some());
        assert!(options.chunk_types.is_some());
    }

    #[test]
    fn test_min_similarity_clamping() {
        let options = SemanticSearchOptions::new().with_min_similarity(1.5);
        assert_eq!(options.min_similarity, 1.0);

        let options = SemanticSearchOptions::new().with_min_similarity(-0.5);
        assert_eq!(options.min_similarity, 0.0);
    }

    #[test]
    fn test_chunk_type_from_str() {
        assert_eq!(ChunkType::from_str("function"), ChunkType::Function);
        assert_eq!(ChunkType::from_str("method"), ChunkType::Function);
        assert_eq!(ChunkType::from_str("struct"), ChunkType::Type);
        assert_eq!(ChunkType::from_str("class"), ChunkType::Type);
        assert_eq!(ChunkType::from_str("module"), ChunkType::Module);
        assert_eq!(ChunkType::from_str("const"), ChunkType::Constant);
        assert_eq!(ChunkType::from_str("doc"), ChunkType::Documentation);
        assert_eq!(ChunkType::from_str("unknown"), ChunkType::Code);
    }

    #[test]
    fn test_chunk_type_as_str() {
        assert_eq!(ChunkType::Function.as_str(), "function");
        assert_eq!(ChunkType::Type.as_str(), "type");
        assert_eq!(ChunkType::Module.as_str(), "module");
        assert_eq!(ChunkType::Constant.as_str(), "constant");
        assert_eq!(ChunkType::Documentation.as_str(), "documentation");
        assert_eq!(ChunkType::Code.as_str(), "code");
    }

    #[test]
    fn test_glob_match() {
        // Simple substring
        assert!(glob_match("main", "src/main.rs"));

        // Wildcard matching
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("src/*.rs", "src/main.rs"));

        // Double wildcard
        assert!(glob_match("src/**/*.rs", "src/foo/bar/main.rs"));
        assert!(glob_match("**/*.rs", "any/path/file.rs"));
    }

    #[test]
    fn test_search_returns_ordered_by_similarity() {
        let search = create_test_search();

        // Search with empty database should return empty results
        let results = search.search_by_embedding(
            &vec![0.0; 384],
            SemanticSearchOptions::default(),
        ).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_cache_invalidation() {
        let search = create_test_search();

        // Load embeddings (empty)
        search.ensure_embeddings_loaded().unwrap();

        // Verify cache is populated
        {
            let cache = search.embeddings_cache.read();
            assert!(cache.is_some());
        }

        // Invalidate
        search.invalidate_cache();

        // Verify cache is cleared
        {
            let cache = search.embeddings_cache.read();
            assert!(cache.is_none());
        }
    }

    #[test]
    fn test_semantic_search_result_fields() {
        let result = SemanticSearchResult {
            path: "src/main.rs".to_string(),
            chunk_text: "fn main() {}".to_string(),
            similarity: 0.95,
            chunk_type: ChunkType::Function,
            start_line: Some(1),
            end_line: Some(5),
            symbol_id: Some(42),
            context: Some("surrounding code".to_string()),
        };

        assert_eq!(result.path, "src/main.rs");
        assert_eq!(result.similarity, 0.95);
        assert_eq!(result.chunk_type, ChunkType::Function);
        assert_eq!(result.start_line, Some(1));
        assert_eq!(result.end_line, Some(5));
    }

    #[test]
    fn test_search_with_mock_embeddings() {
        // Create test database with embeddings
        let db = Arc::new(Database::in_memory().unwrap());

        // Insert a file and symbol
        let file_id = db.upsert_file("src/lib.rs", "hash1", 1234567890, 1000, "rust").unwrap();
        let symbol_id = db.insert_symbol(
            file_id,
            "test_function",
            "function",
            10,
            20,
            Some("fn test_function()"),
            None,
        ).unwrap();

        // Insert embedding for the symbol
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        db.insert_embedding(
            Some(symbol_id),
            None,
            "fn test_function() { /* test code */ }",
            &embedding,
            "function",
        ).unwrap();

        // Create search with lazy engine (default config, no model files)
        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        // Load embeddings
        search.ensure_embeddings_loaded().unwrap();

        // Verify embeddings are loaded
        {
            let cache = search.embeddings_cache.read();
            let embeddings = cache.as_ref().unwrap();
            assert_eq!(embeddings.len(), 1);
        }
    }

    #[test]
    fn test_search_results_ordered_by_similarity() {
        let db = Arc::new(Database::in_memory().unwrap());

        // Create file for symbols
        let file_id = db.upsert_file("src/test.rs", "hash", 0, 100, "rust").unwrap();

        // Insert multiple embeddings with different similarity to a query
        // Query vector: [1, 0, 0, ...]
        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        // High similarity embedding: [0.9, 0.1, 0, ...]
        let high_sim: Vec<f32> = vec![0.9, 0.1]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        // Low similarity embedding: [0.1, 0.9, 0, ...]
        let low_sim: Vec<f32> = vec![0.1, 0.9]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        // Insert in reverse order (low first)
        let sym1 = db.insert_symbol(file_id, "low_fn", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym1), None, "low similarity function", &low_sim, "function").unwrap();

        let sym2 = db.insert_symbol(file_id, "high_fn", "function", 10, 15, None, None).unwrap();
        db.insert_embedding(Some(sym2), None, "high similarity function", &high_sim, "function").unwrap();

        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new().with_min_similarity(0.0),
        ).unwrap();

        // Should return 2 results, highest similarity first
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity > results[1].similarity);
        assert!(results[0].chunk_text.contains("high"));
        assert!(results[1].chunk_text.contains("low"));
    }

    #[test]
    fn test_min_similarity_filtering() {
        let db = Arc::new(Database::in_memory().unwrap());
        let file_id = db.upsert_file("src/test.rs", "hash", 0, 100, "rust").unwrap();

        // Create an embedding that will have ~0.5 similarity with query [1, 0, ...]
        let embedding: Vec<f32> = vec![0.5, 0.5]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(382))
            .collect();

        let sym = db.insert_symbol(file_id, "test_fn", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym), None, "test function", &embedding, "function").unwrap();

        let query: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat(0.0).take(383))
            .collect();

        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        // With low threshold, should return result
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new().with_min_similarity(0.0),
        ).unwrap();
        assert_eq!(results.len(), 1);

        // With high threshold, should filter it out
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new().with_min_similarity(0.99),
        ).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_file_filter() {
        let db = Arc::new(Database::in_memory().unwrap());

        // Create two files
        let file1_id = db.upsert_file("src/main.rs", "hash1", 0, 100, "rust").unwrap();
        let file2_id = db.upsert_file("tests/test.rs", "hash2", 0, 100, "rust").unwrap();

        let embedding: Vec<f32> = vec![1.0; 384];

        let sym1 = db.insert_symbol(file1_id, "main_fn", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym1), None, "main function", &embedding, "function").unwrap();

        let sym2 = db.insert_symbol(file2_id, "test_fn", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym2), None, "test function", &embedding, "function").unwrap();

        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        let query: Vec<f32> = vec![1.0; 384];

        // Filter to only src files
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new()
                .with_min_similarity(0.0)
                .with_file_filter(vec!["src/".to_string()]),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].path.starts_with("src/"));
    }

    #[test]
    fn test_chunk_type_filter() {
        let db = Arc::new(Database::in_memory().unwrap());
        let file_id = db.upsert_file("src/lib.rs", "hash", 0, 100, "rust").unwrap();

        let embedding: Vec<f32> = vec![1.0; 384];

        // Insert function
        let sym1 = db.insert_symbol(file_id, "my_function", "function", 1, 5, None, None).unwrap();
        db.insert_embedding(Some(sym1), None, "my function", &embedding, "function").unwrap();

        // Insert type
        let sym2 = db.insert_symbol(file_id, "MyStruct", "struct", 10, 15, None, None).unwrap();
        db.insert_embedding(Some(sym2), None, "my struct", &embedding, "type").unwrap();

        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        let query: Vec<f32> = vec![1.0; 384];

        // Filter to only functions
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new()
                .with_min_similarity(0.0)
                .with_chunk_types(vec![ChunkType::Function]),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_type, ChunkType::Function);

        // Filter to only types
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new()
                .with_min_similarity(0.0)
                .with_chunk_types(vec![ChunkType::Type]),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_type, ChunkType::Type);
    }

    #[test]
    fn test_limit_results() {
        let db = Arc::new(Database::in_memory().unwrap());
        let file_id = db.upsert_file("src/lib.rs", "hash", 0, 100, "rust").unwrap();

        let embedding: Vec<f32> = vec![1.0; 384];

        // Insert 5 embeddings
        for i in 0..5 {
            let sym = db.insert_symbol(
                file_id,
                &format!("fn_{}", i),
                "function",
                i * 10,
                i * 10 + 5,
                None,
                None,
            ).unwrap();
            db.insert_embedding(Some(sym), None, &format!("function {}", i), &embedding, "function").unwrap();
        }

        let engine = Arc::new(LazyEmbeddingEngine::new(ModelConfig::default()));
        let search = SemanticSearch::new(engine, db);

        let query: Vec<f32> = vec![1.0; 384];

        // Limit to 3 results
        let results = search.search_by_embedding(
            &query,
            SemanticSearchOptions::new()
                .with_min_similarity(0.0)
                .with_limit(3),
        ).unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_default_semantic_search() {
        // Test the Default implementation
        let search = SemanticSearch::default();
        assert!(!search.is_available());
        assert!(!search.is_loaded());
        assert_eq!(search.embedding_dim(), 384);
    }

    #[test]
    fn test_lazy_loading_state() {
        let search = create_test_search();

        // Before any embedding calls, model should not be loaded
        assert!(!search.is_loaded());

        // With default config (empty paths), model files are not available
        assert!(!search.is_available());

        // embedding_dim should work without loading the model
        assert_eq!(search.embedding_dim(), 384);

        // Model still not loaded after getting dimension
        assert!(!search.is_loaded());
    }

    #[test]
    fn test_is_available_vs_is_loaded() {
        // These test the semantic difference between is_available and is_loaded:
        // - is_available: checks if model FILES exist on disk
        // - is_loaded: checks if model is LOADED into memory

        let search = create_test_search();

        // With default config (empty paths), neither condition is true
        assert!(!search.is_available());
        assert!(!search.is_loaded());

        // In a real scenario with valid model paths but before first query:
        // - is_available() would be true (files exist)
        // - is_loaded() would be false (not loaded yet)
        // After first query:
        // - Both would be true
    }
}
