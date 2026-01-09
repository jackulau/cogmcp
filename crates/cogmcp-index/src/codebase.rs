//! Codebase file indexing

use crate::parser::{CodeParser, ExtractedSymbol};
use cogmcp_core::types::Language;
use cogmcp_core::{Config, Error, Result};
use cogmcp_embeddings::inference::MetricsSnapshot;
use cogmcp_embeddings::LazyEmbeddingEngine;
use cogmcp_storage::{Database, EmbeddingInput, FullTextIndex};
use globset::{Glob, GlobSet, GlobSetBuilder};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tracing::info;
use walkdir::WalkDir;

/// Progress information for indexing operations
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Total number of files to process
    pub total_files: usize,
    /// Number of files processed so far
    pub processed_files: usize,
    /// Current file being processed
    pub current_file: Option<String>,
    /// Time elapsed since start
    pub elapsed_time: Duration,
    /// Processing rate in files per second
    pub files_per_second: f64,
}

impl IndexProgress {
    /// Get completion percentage
    pub fn percentage(&self) -> f64 {
        if self.total_files > 0 {
            (self.processed_files as f64 / self.total_files as f64) * 100.0
        } else {
            100.0
        }
    }

    /// Get estimated time remaining
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.files_per_second > 0.0 && self.processed_files < self.total_files {
            let remaining_files = self.total_files - self.processed_files;
            let remaining_secs = remaining_files as f64 / self.files_per_second;
            Some(Duration::from_secs_f64(remaining_secs))
        } else {
            None
        }
    }
}

/// Indexes files in a codebase
pub struct CodebaseIndexer {
    root: PathBuf,
    config: Config,
    ignore_patterns: GlobSet,
    include_extensions: HashSet<String>,
    parser: Arc<CodeParser>,
    embedding_engine: Option<Arc<LazyEmbeddingEngine>>,
}

impl CodebaseIndexer {
    /// Create a new indexer for the given root directory
    pub fn new(root: PathBuf, config: Config, parser: Arc<CodeParser>) -> Result<Self> {
        let ignore_patterns = Self::build_ignore_patterns(&config)?;
        let include_extensions: HashSet<String> = config
            .indexing
            .include_types
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        Ok(Self {
            root,
            config,
            ignore_patterns,
            include_extensions,
            parser,
            embedding_engine: None,
        })
    }

    /// Create an indexer with embedding support
    pub fn with_embedding_engine(
        root: PathBuf,
        config: Config,
        parser: Arc<CodeParser>,
        engine: Arc<LazyEmbeddingEngine>,
    ) -> Result<Self> {
        let mut indexer = Self::new(root, config, parser)?;
        indexer.embedding_engine = Some(engine);
        Ok(indexer)
    }

    /// Set the embedding engine
    pub fn set_embedding_engine(&mut self, engine: Arc<LazyEmbeddingEngine>) {
        self.embedding_engine = Some(engine);
    }

    /// Check if embeddings are enabled and available
    ///
    /// Returns true if embeddings are enabled in config and model files exist.
    /// This does NOT require the model to be loaded.
    pub fn embeddings_enabled(&self) -> bool {
        self.config.indexing.enable_embeddings
            && self.embedding_engine.as_ref().map_or(false, |e| e.is_available())
    }

    fn build_ignore_patterns(config: &Config) -> Result<GlobSet> {
        let mut builder = GlobSetBuilder::new();

        for pattern in &config.indexing.ignore_patterns {
            let glob = Glob::new(pattern)
                .map_err(|e| Error::Config(format!("Invalid glob pattern '{}': {}", pattern, e)))?;
            builder.add(glob);
        }

        builder
            .build()
            .map_err(|e| Error::Config(format!("Failed to build glob set: {}", e)))
    }

    /// Index all files in the codebase
    pub fn index_all(&self, db: &Database, text_index: &FullTextIndex) -> Result<IndexResult> {
        if self.config.indexing.enable_parallel_indexing {
            self.index_all_parallel(db, text_index)
        } else {
            self.index_all_sequential(db, text_index)
        }
    }

    /// Index all files sequentially (original implementation)
    fn index_all_sequential(&self, db: &Database, text_index: &FullTextIndex) -> Result<IndexResult> {
        let mut result = IndexResult::default();
        let gitignore = self.load_gitignore()?;

        // First pass: collect all files to process for accurate progress tracking
        let files_to_process: Vec<_> = WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path(), &gitignore))
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                let ext = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                self.include_extensions.contains(&ext)
            })
            .filter(|entry| {
                fs::metadata(entry.path())
                    .map(|m| m.len() / 1024 <= self.config.indexing.max_file_size)
                    .unwrap_or(false)
            })
            .collect();

        let total_files = files_to_process.len();
        info!("Indexing {} files in {}", total_files, self.root.display());
        let start_time = Instant::now();

        // Reset embedding metrics if available
        if let Some(ref engine) = self.embedding_engine {
            engine.lock().reset_metrics();
        }

        // Second pass: process files with progress tracking
        for (idx, entry) in files_to_process.iter().enumerate() {
            let path = entry.path();
            let relative_path = path
                .strip_prefix(&self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            // Index the file and track symbol statistics
            match self.index_file_with_stats(path, db, text_index, &mut result) {
                Ok(_) => result.indexed += 1,
                Err(e) => {
                    tracing::warn!("Failed to index {:?}: {}", path, e);
                    result.errors += 1;
                }
            }

            // Report progress
            let elapsed = start_time.elapsed();
            let processed = idx + 1;
            let files_per_second = if elapsed.as_secs_f64() > 0.0 {
                processed as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            let progress = IndexProgress {
                total_files,
                processed_files: processed,
                current_file: Some(relative_path),
                elapsed_time: elapsed,
                files_per_second,
            };
            on_progress(progress);
        }

        // Commit the text index
        text_index.commit()?;

        // Record final timing and metrics
        result.indexing_time = start_time.elapsed();

        // Capture embedding metrics if available
        if let Some(ref engine) = self.embedding_engine {
            result.embedding_metrics = Some(engine.lock().get_metrics());
        }

        // Log final summary
        info!(
            "Indexing complete: {} files in {:.2}s ({:.1} files/s)",
            result.indexed,
            result.indexing_time.as_secs_f64(),
            result.indexed as f64 / result.indexing_time.as_secs_f64().max(0.001)
        );

        if let Some(ref metrics) = result.embedding_metrics {
            info!(
                "Embedding metrics: {} embeddings, {:.2}s inference, {:.1} emb/s",
                metrics.total_embeddings_generated,
                metrics.total_inference_time.as_secs_f64(),
                metrics.throughput_per_second
            );
        }

        Ok(result)
    }

    /// Index all files in parallel using rayon
    fn index_all_parallel(&self, db: &Database, text_index: &FullTextIndex) -> Result<IndexResult> {
        let gitignore = self.load_gitignore()?;

        // Configure rayon thread pool if parallel_workers is specified
        if self.config.indexing.parallel_workers > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.indexing.parallel_workers)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        // Phase 1: Collect all valid file paths
        let mut file_paths: Vec<PathBuf> = Vec::new();
        let mut skipped = 0u64;
        let mut walk_errors = 0u64;

        for entry in WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path(), &gitignore))
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Failed to read directory entry: {}", e);
                    walk_errors += 1;
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();

            // Check file extension
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if !self.include_extensions.contains(&ext) {
                skipped += 1;
                continue;
            }

            // Check file size
            let metadata = match fs::metadata(path) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("Failed to get metadata for {:?}: {}", path, e);
                    walk_errors += 1;
                    continue;
                }
            };

            let size_kb = metadata.len() / 1024;
            if size_kb > self.config.indexing.max_file_size {
                skipped += 1;
                continue;
            }

            file_paths.push(path.to_path_buf());
        }

        tracing::info!(
            "Parallel indexing: collected {} files to process",
            file_paths.len()
        );

        // Phase 2: Process files in parallel
        let indexed = AtomicU64::new(0);
        let errors = AtomicU64::new(walk_errors);
        let symbols_extracted = AtomicU64::new(0);
        let symbols_with_visibility = AtomicU64::new(0);
        let symbols_by_kind: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));

        // Collect parsed file data in parallel
        let parsed_files: Vec<_> = file_paths
            .par_iter()
            .filter_map(|path| {
                match self.parse_file_data(path) {
                    Ok(data) => Some(data),
                    Err(e) => {
                        tracing::warn!("Failed to parse {:?}: {}", path, e);
                        errors.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                }
            })
            .collect();

        // Get file IDs that already have embeddings for incremental indexing
        let existing_embeddings = db.get_file_ids_with_embeddings().unwrap_or_default();

        // Phase 3: Insert data sequentially and collect chunks for embedding
        let mut chunks_for_embedding: Vec<ChunkData> = Vec::new();

        for file_data in &parsed_files {
            match self.insert_file_data(db, text_index, file_data) {
                Ok(file_stats) => {
                    indexed.fetch_add(1, Ordering::Relaxed);
                    symbols_extracted.fetch_add(file_stats.symbols_extracted, Ordering::Relaxed);
                    symbols_with_visibility.fetch_add(file_stats.symbols_with_visibility, Ordering::Relaxed);

                    // Merge symbols_by_kind
                    let mut kind_map = symbols_by_kind.lock();
                    for (kind, count) in file_stats.symbols_by_kind {
                        *kind_map.entry(kind).or_insert(0) += count;
                    }

                    // Collect chunks for embedding if embeddings are enabled
                    if self.embeddings_enabled() {
                        // Get the file_id for this file
                        if let Ok(Some(file_row)) = db.get_file_by_path(&file_data.relative_path) {
                            // Check for incremental: skip if file already has embeddings
                            if !existing_embeddings.contains(&file_row.id) {
                                // Create chunks from file content
                                let chunk = ChunkData {
                                    file_id: file_row.id,
                                    symbol_id: None,
                                    text: file_data.content.clone(),
                                    chunk_type: "file".to_string(),
                                };
                                chunks_for_embedding.push(chunk);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to insert file data for {:?}: {}", file_data.relative_path, e);
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Commit the text index
        text_index.commit()?;

        // Phase 4: Generate embeddings in batches
        if self.embeddings_enabled() && !chunks_for_embedding.is_empty() {
            let batch_size = self.config.indexing.embedding_batch_size;
            let total_chunks = chunks_for_embedding.len();
            tracing::info!(
                "Generating embeddings for {} chunks in batches of {}",
                total_chunks,
                batch_size
            );

            self.generate_embeddings_batch(db, chunks_for_embedding, batch_size)?;
        }

        Ok(IndexResult {
            indexed: indexed.load(Ordering::Relaxed),
            skipped,
            errors: errors.load(Ordering::Relaxed),
            symbols_extracted: symbols_extracted.load(Ordering::Relaxed),
            symbols_by_kind: Arc::try_unwrap(symbols_by_kind)
                .map(|mutex| mutex.into_inner())
                .unwrap_or_else(|arc| arc.lock().clone()),
            symbols_with_visibility: symbols_with_visibility.load(Ordering::Relaxed),
        })
    }

    /// Generate embeddings for chunks in batches
    fn generate_embeddings_batch(
        &self,
        db: &Database,
        chunks: Vec<ChunkData>,
        batch_size: usize,
    ) -> Result<u64> {
        let engine = match &self.embedding_engine {
            Some(e) => e,
            None => return Ok(0),
        };

        let mut embeddings_generated = 0u64;
        let mut embedding_errors = 0u64;

        for batch in chunks.chunks(batch_size) {
            // Collect texts for batch embedding
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();

            // Generate embeddings in batch
            let embeddings = {
                let mut engine_guard = engine.lock();
                match engine_guard.embed_batch(&texts) {
                    Ok(embs) => embs,
                    Err(e) => {
                        tracing::warn!("Failed to generate batch embeddings: {}", e);
                        embedding_errors += batch.len() as u64;
                        continue;
                    }
                }
            };

            // Prepare inputs for batch insert
            let embedding_inputs: Vec<EmbeddingInput> = batch
                .iter()
                .zip(embeddings.iter())
                .map(|(chunk, emb)| EmbeddingInput {
                    symbol_id: chunk.symbol_id,
                    file_id: Some(chunk.file_id),
                    chunk_text: chunk.text.clone(),
                    embedding: emb.clone(),
                    chunk_type: chunk.chunk_type.clone(),
                })
                .collect();

            // Batch insert embeddings
            match db.insert_embeddings_batch(&embedding_inputs) {
                Ok(ids) => {
                    embeddings_generated += ids.len() as u64;
                    tracing::debug!("Inserted {} embeddings", ids.len());
                }
                Err(e) => {
                    tracing::warn!("Failed to insert batch embeddings: {}", e);
                    embedding_errors += batch.len() as u64;
                }
            }
        }

        if embedding_errors > 0 {
            tracing::warn!(
                "Embedding generation completed with {} errors",
                embedding_errors
            );
        }

        tracing::info!(
            "Generated {} embeddings successfully",
            embeddings_generated
        );

        Ok(embeddings_generated)
    }

    /// Parse file data without database operations (thread-safe)
    fn parse_file_data(&self, path: &Path) -> Result<ParsedFileData> {
        let content = fs::read_to_string(path)?;
        let relative_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Compute file hash
        let hash = Self::hash_content(&content);

        // Get file metadata
        let metadata = fs::metadata(path)?;
        let modified_at = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let size = metadata.len();

        // Detect language
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);
        let language_str = format!("{:?}", language).to_lowercase();

        // Parse and extract symbols
        let symbols = self.parser.parse(&content, language).unwrap_or_default();

        Ok(ParsedFileData {
            relative_path,
            hash,
            modified_at,
            size,
            language_str,
            content,
            symbols,
        })
    }

    /// Insert parsed file data into database (must be called sequentially)
    fn insert_file_data(
        &self,
        db: &Database,
        text_index: &FullTextIndex,
        data: &ParsedFileData,
    ) -> Result<FileIndexStats> {
        let mut stats = FileIndexStats::default();

        // Insert into database
        let file_id = db.upsert_file(
            &data.relative_path,
            &data.hash,
            data.modified_at,
            data.size,
            &data.language_str,
        )?;

        // Track statistics
        for sym in &data.symbols {
            stats.symbols_extracted += 1;
            let kind_str = format!("{:?}", sym.kind).to_lowercase();
            *stats.symbols_by_kind.entry(kind_str).or_insert(0) += 1;

            if sym.visibility != cogmcp_core::types::SymbolVisibility::Unknown {
                stats.symbols_with_visibility += 1;
            }
        }

        // Delete old symbols for this file
        let _ = db.delete_symbols_for_file(file_id);

        // Insert new symbols with parent relationship handling
        self.insert_symbols_with_relationships(db, file_id, &data.symbols)?;

        // Index in full-text search
        text_index.index_file(&data.relative_path, &data.content)?;

        Ok(stats)
    }

    /// Index a single file and update statistics
    fn index_file_with_stats(
        &self,
        path: &Path,
        db: &Database,
        text_index: &FullTextIndex,
        result: &mut IndexResult,
    ) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let relative_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Compute file hash
        let hash = Self::hash_content(&content);

        // Get file metadata
        let metadata = fs::metadata(path)?;
        let modified_at = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let size = metadata.len();

        // Detect language
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);
        let language_str = format!("{:?}", language).to_lowercase();

        // Insert into database
        let file_id = db.upsert_file(&relative_path, &hash, modified_at, size, &language_str)?;

        // Parse and extract symbols with enhanced metadata
        if let Ok(symbols) = self.parser.parse(&content, language) {
            // Delete old symbols for this file
            let _ = db.delete_symbols_for_file(file_id);

            // Track statistics
            for sym in &symbols {
                result.symbols_extracted += 1;
                let kind_str = format!("{:?}", sym.kind).to_lowercase();
                *result.symbols_by_kind.entry(kind_str).or_insert(0) += 1;

                if sym.visibility != cogmcp_core::types::SymbolVisibility::Unknown {
                    result.symbols_with_visibility += 1;
                }
            }

            // Insert new symbols with parent relationship handling
            self.insert_symbols_with_relationships(db, file_id, &symbols)?;
        }

        // Index in full-text search
        text_index.index_file(&relative_path, &content)?;

        Ok(())
    }

    /// Index a single file
    pub fn index_file(
        &self,
        path: &Path,
        db: &Database,
        text_index: &FullTextIndex,
    ) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let relative_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Compute file hash
        let hash = Self::hash_content(&content);

        // Get file metadata
        let metadata = fs::metadata(path)?;
        let modified_at = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let size = metadata.len();

        // Detect language
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);
        let language_str = format!("{:?}", language).to_lowercase();

        // Insert into database
        let file_id = db.upsert_file(&relative_path, &hash, modified_at, size, &language_str)?;

        // Parse and extract symbols with enhanced metadata
        if let Ok(symbols) = self.parser.parse(&content, language) {
            // Delete old symbols for this file
            let _ = db.delete_symbols_for_file(file_id);

            // Insert new symbols with parent relationship handling
            self.insert_symbols_with_relationships(db, file_id, &symbols)?;
        }

        // Index in full-text search
        text_index.index_file(&relative_path, &content)?;

        Ok(())
    }

    /// Insert symbols and establish parent-child relationships
    fn insert_symbols_with_relationships(
        &self,
        db: &Database,
        file_id: i64,
        symbols: &[ExtractedSymbol],
    ) -> Result<()> {
        // First pass: insert all symbols without parent references and track name -> id mapping
        let mut name_to_id: HashMap<String, i64> = HashMap::new();

        for sym in symbols {
            let kind_str = format!("{:?}", sym.kind).to_lowercase();
            let visibility_str = sym.visibility.as_str();

            // Serialize type parameters and parameters as JSON
            let type_params_json = if sym.type_parameters.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&sym.type_parameters).unwrap_or_default())
            };

            let params_json = if sym.parameters.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&sym.parameters).unwrap_or_default())
            };

            let symbol_id = db.insert_symbol_extended(
                file_id,
                &sym.name,
                &kind_str,
                sym.start_line,
                sym.end_line,
                sym.signature.as_deref(),
                sym.doc_comment.as_deref(),
                Some(visibility_str),
                sym.modifiers.is_async,
                sym.modifiers.is_static,
                sym.modifiers.is_abstract,
                sym.modifiers.is_exported,
                sym.modifiers.is_const,
                sym.modifiers.is_unsafe,
                None, // parent_symbol_id will be set in second pass
                type_params_json.as_deref(),
                params_json.as_deref(),
                sym.return_type.as_deref(),
            )?;

            name_to_id.insert(sym.name.clone(), symbol_id);
        }

        // Second pass: update parent references
        for sym in symbols {
            if let Some(parent_name) = &sym.parent_name {
                if let Some(&parent_id) = name_to_id.get(parent_name) {
                    if let Some(&symbol_id) = name_to_id.get(&sym.name) {
                        // Update the parent reference using the database method
                        db.update_symbol_parent(symbol_id, parent_id)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn should_ignore(&self, path: &Path, gitignore: &GlobSet) -> bool {
        let relative = path.strip_prefix(&self.root).unwrap_or(path);
        let path_str = relative.to_string_lossy();

        // Check gitignore patterns
        if gitignore.is_match(relative) {
            return true;
        }

        // Check config ignore patterns
        if self.ignore_patterns.is_match(relative) {
            return true;
        }

        // Always ignore .git directory
        if path_str.contains(".git") {
            return true;
        }

        false
    }

    fn load_gitignore(&self) -> Result<GlobSet> {
        let gitignore_path = self.root.join(".gitignore");
        let mut builder = GlobSetBuilder::new();

        if gitignore_path.exists() {
            let content = fs::read_to_string(&gitignore_path)?;
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                // Convert gitignore pattern to glob
                let pattern = if let Some(stripped) = line.strip_prefix('/') {
                    stripped.to_string()
                } else if let Some(stripped) = line.strip_suffix('/') {
                    format!("**/{}", stripped)
                } else {
                    format!("**/{}", line)
                };

                if let Ok(glob) = Glob::new(&pattern) {
                    builder.add(glob);
                }
            }
        }

        builder
            .build()
            .map_err(|e| Error::Config(format!("Failed to build gitignore: {}", e)))
    }

    fn hash_content(content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// Result of an indexing operation
#[derive(Debug, Default)]
pub struct IndexResult {
    pub indexed: u64,
    pub skipped: u64,
    pub errors: u64,
    pub symbols_extracted: u64,
    pub symbols_by_kind: HashMap<String, u64>,
    pub symbols_with_visibility: u64,
    /// Total indexing time
    pub indexing_time: Duration,
    /// Embedding performance metrics (if embeddings were enabled)
    pub embedding_metrics: Option<MetricsSnapshot>,
}

/// Parsed file data for parallel processing
#[derive(Debug)]
struct ParsedFileData {
    relative_path: String,
    hash: String,
    modified_at: i64,
    size: u64,
    language_str: String,
    content: String,
    symbols: Vec<ExtractedSymbol>,
}

/// Statistics for a single file's indexing
#[derive(Debug, Default)]
struct FileIndexStats {
    symbols_extracted: u64,
    symbols_by_kind: HashMap<String, u64>,
    symbols_with_visibility: u64,
}

/// Chunk data for batch embedding generation
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub file_id: i64,
    pub symbol_id: Option<i64>,
    pub text: String,
    pub chunk_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn setup_test_files(temp_dir: &TempDir) {
        // Create a Rust file with various symbols
        let rust_file = temp_dir.path().join("lib.rs");
        let mut file = fs::File::create(&rust_file).unwrap();
        writeln!(
            file,
            r#"//! Test module
pub struct Container<T> {{
    data: Vec<T>,
}}

impl<T> Container<T> {{
    pub fn new() -> Self {{
        Self {{ data: Vec::new() }}
    }}

    pub async fn async_method(&self) -> bool {{
        true
    }}

    fn private_method(&self) {{}}
}}

pub(crate) fn crate_visible_fn() {{}}

const MY_CONST: i32 = 42;
"#
        )
        .unwrap();

        // Create a TypeScript file
        let ts_file = temp_dir.path().join("index.ts");
        let mut file = fs::File::create(&ts_file).unwrap();
        writeln!(
            file,
            r#"export class MyClass {{
    private count: number;

    public static create(): MyClass {{
        return new MyClass();
    }}

    public async fetchData(): Promise<void> {{}}
}}

export function exportedFunction(): string {{
    return "hello";
}}
"#
        )
        .unwrap();

        // Create a Python file
        let py_file = temp_dir.path().join("module.py");
        let mut file = fs::File::create(&py_file).unwrap();
        writeln!(
            file,
            r#"class MyClass:
    """A sample class."""

    def public_method(self):
        pass

    def _protected_method(self):
        pass

    def __private_method(self):
        pass

def public_function():
    """A public function."""
    pass

async def async_function():
    pass
"#
        )
        .unwrap();
    }

    fn create_test_indexer(root: PathBuf) -> CodebaseIndexer {
        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        let parser = Arc::new(CodeParser::new());
        CodebaseIndexer::new(root, config, parser).unwrap()
    }

    #[test]
    fn test_index_rust_file_extracts_enhanced_metadata() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Verify files were indexed
        assert!(result.indexed > 0);
        assert!(result.symbols_extracted > 0);

        // Verify symbols by kind
        assert!(result.symbols_by_kind.contains_key("struct"));
        assert!(result.symbols_by_kind.contains_key("function"));

        // Verify visibility was extracted
        assert!(result.symbols_with_visibility > 0);

        // Query the database for specific symbols
        let container = db.find_symbols_by_name("Container", false).unwrap();
        assert!(!container.is_empty());
        assert_eq!(container[0].visibility, Some("public".to_string()));

        let async_method = db.find_symbols_by_name("async_method", false).unwrap();
        assert!(!async_method.is_empty());
        assert!(async_method[0].is_async);

        let private_method = db.find_symbols_by_name("private_method", false).unwrap();
        assert!(!private_method.is_empty());
        assert_eq!(private_method[0].visibility, Some("private".to_string()));

        let crate_fn = db.find_symbols_by_name("crate_visible_fn", false).unwrap();
        assert!(!crate_fn.is_empty());
        assert_eq!(crate_fn[0].visibility, Some("crate".to_string()));
    }

    #[test]
    fn test_index_typescript_file_extracts_exports() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        indexer.index_all(&db, &text_index).unwrap();

        // Verify TypeScript class was extracted
        let my_class = db.find_symbols_by_name("MyClass", false).unwrap();
        assert!(!my_class.is_empty());

        // Verify exported function
        let exported_fn = db.find_symbols_by_name("exportedFunction", false).unwrap();
        assert!(!exported_fn.is_empty());
    }

    #[test]
    fn test_index_python_file_with_visibility_convention() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        indexer.index_all(&db, &text_index).unwrap();

        // Verify Python class
        let py_class = db.find_symbols_by_name("MyClass", false).unwrap();
        // MyClass might appear in TS and Python, check we have at least one
        assert!(!py_class.is_empty());

        // Verify public function
        let public_fn = db.find_symbols_by_name("public_function", false).unwrap();
        assert!(!public_fn.is_empty());
        assert_eq!(public_fn[0].visibility, Some("public".to_string()));

        // Verify protected method
        let protected_method = db.find_symbols_by_name("_protected_method", false).unwrap();
        assert!(!protected_method.is_empty());
        assert_eq!(protected_method[0].visibility, Some("protected".to_string()));

        // Verify private method (Python __ prefix)
        let private_method = db.find_symbols_by_name("__private_method", false).unwrap();
        assert!(!private_method.is_empty());
        assert_eq!(private_method[0].visibility, Some("private".to_string()));

        // Verify async function
        let async_fn = db.find_symbols_by_name("async_function", false).unwrap();
        assert!(!async_fn.is_empty());
        assert!(async_fn[0].is_async);
    }

    #[test]
    fn test_nested_symbols_have_parent_references() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        indexer.index_all(&db, &text_index).unwrap();

        // Find the Python class methods that should have parent references
        let public_method = db.find_symbols_by_name("public_method", false).unwrap();
        assert!(!public_method.is_empty());
        // The method should have a parent reference (though parent_symbol_id depends on name match)
        // Since we use name matching, we verify parent relationships work

        // Get extended stats to verify parent relationships are tracked
        let stats = db.get_extended_stats().unwrap();
        assert!(stats.symbols_with_parent > 0, "Some symbols should have parent references");
    }

    #[test]
    fn test_reindexing_clears_old_symbols() {
        let temp_dir = TempDir::new().unwrap();

        // Create initial file
        let rust_file = temp_dir.path().join("test.rs");
        {
            let mut file = fs::File::create(&rust_file).unwrap();
            writeln!(file, "pub fn old_function() {{}}").unwrap();
        }

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        // First index
        indexer.index_all(&db, &text_index).unwrap();

        let old_fn = db.find_symbols_by_name("old_function", false).unwrap();
        assert!(!old_fn.is_empty());

        // Modify the file
        {
            let mut file = fs::File::create(&rust_file).unwrap();
            writeln!(file, "pub fn new_function() {{}}").unwrap();
        }

        // Re-index
        indexer.index_all(&db, &text_index).unwrap();

        // Old function should be gone
        let old_fn = db.find_symbols_by_name("old_function", false).unwrap();
        assert!(old_fn.is_empty(), "Old symbols should be cleared on reindex");

        // New function should exist
        let new_fn = db.find_symbols_by_name("new_function", false).unwrap();
        assert!(!new_fn.is_empty(), "New symbols should be indexed");
    }

    #[test]
    fn test_extended_stats_after_indexing() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        indexer.index_all(&db, &text_index).unwrap();

        let stats = db.get_extended_stats().unwrap();

        // Verify file count
        assert_eq!(stats.file_count, 3, "Should have indexed 3 files");

        // Verify symbol breakdown by kind
        assert!(stats.symbols_by_kind.len() > 0);

        // Verify visibility breakdown
        assert!(stats.symbols_by_visibility.len() > 0);
        assert!(stats.symbols_by_visibility.contains_key("public"));

        // Verify visibility extraction coverage
        assert!(stats.symbols_with_visibility > 0);
    }

    fn create_parallel_test_indexer(root: PathBuf) -> CodebaseIndexer {
        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        config.indexing.enable_parallel_indexing = true;
        config.indexing.parallel_workers = 2; // Use 2 workers for testing
        config.indexing.embedding_batch_size = 2;
        let parser = Arc::new(CodeParser::new());
        CodebaseIndexer::new(root, config, parser).unwrap()
    }

    fn create_sequential_test_indexer(root: PathBuf) -> CodebaseIndexer {
        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        config.indexing.enable_parallel_indexing = false;
        let parser = Arc::new(CodeParser::new());
        CodebaseIndexer::new(root, config, parser).unwrap()
    }

    #[test]
    fn test_parallel_indexing_produces_same_results_as_sequential() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        // Index with parallel indexer
        let parallel_indexer = create_parallel_test_indexer(temp_dir.path().to_path_buf());
        let parallel_db = Database::in_memory().unwrap();
        let parallel_text_index = FullTextIndex::in_memory().unwrap();

        let parallel_result = parallel_indexer.index_all(&parallel_db, &parallel_text_index).unwrap();

        // Index with sequential indexer
        let sequential_indexer = create_sequential_test_indexer(temp_dir.path().to_path_buf());
        let sequential_db = Database::in_memory().unwrap();
        let sequential_text_index = FullTextIndex::in_memory().unwrap();

        let sequential_result = sequential_indexer.index_all(&sequential_db, &sequential_text_index).unwrap();

        // Verify same number of indexed files
        assert_eq!(
            parallel_result.indexed, sequential_result.indexed,
            "Parallel and sequential should index same number of files"
        );

        // Verify same number of symbols extracted
        assert_eq!(
            parallel_result.symbols_extracted, sequential_result.symbols_extracted,
            "Parallel and sequential should extract same number of symbols"
        );

        // Verify same symbols by kind
        assert_eq!(
            parallel_result.symbols_by_kind, sequential_result.symbols_by_kind,
            "Parallel and sequential should have same symbols by kind"
        );

        // Verify no errors
        assert_eq!(parallel_result.errors, 0, "Parallel indexing should have no errors");
        assert_eq!(sequential_result.errors, 0, "Sequential indexing should have no errors");
    }

    #[test]
    fn test_parallel_indexing_handles_many_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create many small files to test parallelism
        for i in 0..20 {
            let rust_file = temp_dir.path().join(format!("file_{}.rs", i));
            let mut file = fs::File::create(&rust_file).unwrap();
            writeln!(file, "pub fn function_{}() {{}}", i).unwrap();
        }

        let indexer = create_parallel_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Verify all files were indexed
        assert_eq!(result.indexed, 20, "Should have indexed all 20 files");
        assert_eq!(result.errors, 0, "Should have no errors");

        // Verify symbols were extracted from each file
        assert_eq!(result.symbols_by_kind.get("function"), Some(&20), "Should have 20 functions");
    }

    #[test]
    fn test_parallel_indexing_with_custom_workers() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        config.indexing.enable_parallel_indexing = true;
        config.indexing.parallel_workers = 4; // Use 4 workers
        let parser = Arc::new(CodeParser::new());

        let indexer = CodebaseIndexer::new(temp_dir.path().to_path_buf(), config, parser).unwrap();
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Should successfully index files with custom worker count
        assert_eq!(result.indexed, 3, "Should have indexed 3 files");
        assert_eq!(result.errors, 0, "Should have no errors");
    }

    #[test]
    fn test_parallel_indexing_graceful_error_handling() {
        let temp_dir = TempDir::new().unwrap();

        // Create a valid file
        let valid_file = temp_dir.path().join("valid.rs");
        let mut file = fs::File::create(&valid_file).unwrap();
        writeln!(file, "pub fn valid_fn() {{}}").unwrap();

        // Create a directory that looks like a file (edge case)
        let subdir = temp_dir.path().join("subdir.rs");
        fs::create_dir(&subdir).unwrap();

        let indexer = create_parallel_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Should have indexed the valid file
        assert_eq!(result.indexed, 1, "Should have indexed 1 valid file");
    }

    #[test]
    fn test_parallel_indexing_config_defaults() {
        let config = Config::default();

        // Verify parallel indexing defaults
        assert!(config.indexing.enable_parallel_indexing, "Parallel indexing should be enabled by default");
        assert_eq!(config.indexing.parallel_workers, 0, "Workers should default to 0 (auto-detect)");
        assert_eq!(config.indexing.embedding_batch_size, 64, "Batch size should default to 64");
    }

    #[test]
    fn test_chunk_data_creation() {
        let chunk = ChunkData {
            file_id: 1,
            symbol_id: Some(2),
            text: "test content".to_string(),
            chunk_type: "file".to_string(),
        };

        assert_eq!(chunk.file_id, 1);
        assert_eq!(chunk.symbol_id, Some(2));
        assert_eq!(chunk.text, "test content");
        assert_eq!(chunk.chunk_type, "file");
    }

    #[test]
    fn test_incremental_embedding_file_ids_tracking() {
        let db = Database::in_memory().unwrap();

        // Initially no files have embeddings
        let file_ids = db.get_file_ids_with_embeddings().unwrap();
        assert!(file_ids.is_empty(), "Initially no files should have embeddings");

        // Insert a file
        let file_id = db.upsert_file("test.rs", "hash123", 0, 100, "rust").unwrap();

        // Insert an embedding for this file
        db.insert_embedding(None, Some(file_id), "test content", &vec![0.1, 0.2, 0.3], "file").unwrap();

        // Now the file should be in the set
        let file_ids = db.get_file_ids_with_embeddings().unwrap();
        assert!(file_ids.contains(&file_id), "File with embedding should be in the set");

        // Check individual file
        assert!(db.file_has_embeddings(file_id).unwrap(), "File should have embeddings");
    }
}
