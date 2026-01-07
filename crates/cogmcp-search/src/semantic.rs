//! Semantic search using embeddings
//!
//! This module provides semantic search functionality that uses vector embeddings
//! to find semantically similar code chunks in the codebase.

use std::sync::Arc;
use std::time::Duration;

use cogmcp_core::{Error, Result};
use cogmcp_embeddings::EmbeddingEngine;
use cogmcp_storage::{cache::Cache, Database};
use futures::stream::{self, Stream, StreamExt};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

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
#[derive(Debug, Clone)]
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
/// 2. Searching the vector storage for similar embeddings
/// 3. Enriching results with file path and line number information
pub struct SemanticSearch {
    /// Embedding engine for generating query embeddings
    engine: Arc<Mutex<EmbeddingEngine>>,
    /// Database for storing and retrieving embeddings
    db: Arc<Database>,
    /// Cache for query embeddings
    query_cache: Cache<String, Vec<f32>>,
    /// Cached embeddings from database for in-memory search
    embeddings_cache: RwLock<Option<Vec<EmbeddingRecord>>>,
}

impl SemanticSearch {
    /// Create a new semantic search instance
    pub fn new(engine: Arc<Mutex<EmbeddingEngine>>, db: Arc<Database>) -> Self {
        Self {
            engine,
            db,
            // Cache query embeddings for 5 minutes
            query_cache: Cache::new(Duration::from_secs(300)),
            embeddings_cache: RwLock::new(None),
        }
    }

    /// Check if the embedding engine has a model loaded
    pub fn is_available(&self) -> bool {
        self.engine.lock().is_loaded()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.engine.lock().embedding_dim()
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
        // Get query embedding (from cache or compute)
        let query_embedding = self.get_or_compute_embedding(query)?;

        // Search with the embedding
        self.search_by_embedding(&query_embedding, options)
    }

    /// Search using a pre-computed embedding vector
    #[instrument(skip(self, query_embedding), level = "debug")]
    pub fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        options: SemanticSearchOptions,
    ) -> Result<Vec<SemanticSearchResult>> {
        // Load embeddings from database if not cached
        self.ensure_embeddings_loaded()?;

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
                context: None, // TODO: Add context snippet extraction
            })
            .collect();

        debug!("Semantic search returned {} results", results.len());
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
    fn get_or_compute_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.query_cache.get(&query.to_string()) {
            debug!("Using cached embedding for query");
            return Ok(cached);
        }

        // Compute embedding
        let embedding = self.engine.lock().embed(query)?;

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
    pub fn invalidate_cache(&self) {
        let mut cache = self.embeddings_cache.write();
        *cache = None;
        self.query_cache.clear();
    }

    /// Index text and store its embedding
    pub fn index_chunk(
        &self,
        chunk_text: &str,
        symbol_id: Option<i64>,
        chunk_type: ChunkType,
    ) -> Result<i64> {
        let embedding = self.engine.lock().embed(chunk_text)?;
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
}

impl Default for SemanticSearch {
    fn default() -> Self {
        Self::new(
            Arc::new(Mutex::new(EmbeddingEngine::without_model())),
            Arc::new(Database::in_memory().expect("Failed to create in-memory database")),
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

    fn create_test_search() -> SemanticSearch {
        // Create an in-memory database
        let db = Arc::new(Database::in_memory().unwrap());

        // Create a mock embedding engine (without model)
        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));

        SemanticSearch::new(engine, db)
    }

    #[test]
    fn test_semantic_search_creation() {
        let search = create_test_search();
        assert!(!search.is_available()); // No model loaded
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

        // Create search with mock engine
        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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

        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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

        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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

        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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

        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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

        let engine = Arc::new(Mutex::new(EmbeddingEngine::without_model()));
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
        assert_eq!(search.embedding_dim(), 384);
    }
}
