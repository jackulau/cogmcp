//! Codebase file indexing

use crate::parallel_indexer::{ParallelIndexConfig, ParallelIndexer, ProgressReport};
use crate::parser::{CodeParser, ExtractedSymbol};
use cogmcp_core::types::Language;
use cogmcp_core::{Config, Error, Result};
use cogmcp_embeddings::{EmbeddingEngine, MetricsSnapshot};
use cogmcp_storage::{Database, FullTextIndex};
use globset::{Glob, GlobSet, GlobSetBuilder};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
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
    embedding_engine: Option<Arc<Mutex<EmbeddingEngine>>>,
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
        engine: Arc<Mutex<EmbeddingEngine>>,
    ) -> Result<Self> {
        let mut indexer = Self::new(root, config, parser)?;
        indexer.embedding_engine = Some(engine);
        Ok(indexer)
    }

    /// Set the embedding engine
    pub fn set_embedding_engine(&mut self, engine: Arc<Mutex<EmbeddingEngine>>) {
        self.embedding_engine = Some(engine);
    }

    /// Check if embeddings are enabled and available
    pub fn embeddings_enabled(&self) -> bool {
        self.config.indexing.enable_embeddings
            && self.embedding_engine.as_ref().map_or(false, |e| e.lock().is_loaded())
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
        self.index_all_with_progress(db, text_index, |_| {})
    }

    /// Index all files in the codebase with progress callback
    ///
    /// The callback receives an `IndexProgress` after each file is processed,
    /// allowing for progress reporting during long-running operations.
    ///
    /// # Arguments
    /// * `db` - The database to store file and symbol information
    /// * `text_index` - The full-text index for content search
    /// * `on_progress` - Callback function invoked after each file
    ///
    /// # Example
    /// ```ignore
    /// indexer.index_all_with_progress(&db, &text_index, |progress| {
    ///     info!("{:.1}% complete ({}/{} files)",
    ///         progress.percentage(),
    ///         progress.processed_files,
    ///         progress.total_files);
    /// })?;
    /// ```
    pub fn index_all_with_progress<F>(
        &self,
        db: &Database,
        text_index: &FullTextIndex,
        mut on_progress: F,
    ) -> Result<IndexResult>
    where
        F: FnMut(IndexProgress),
    {
        let start_time = Instant::now();
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

    /// Index all files, automatically choosing parallel or sequential indexing based on config
    ///
    /// When `config.indexing.enable_parallel` is true, this uses the `ParallelIndexer` for
    /// significantly improved performance (typically 3-5x faster). Otherwise, it falls back
    /// to the sequential `index_all` method.
    pub fn index_all_smart(&self, db: &Database, text_index: &FullTextIndex) -> Result<IndexResult> {
        if self.config.indexing.enable_parallel {
            self.index_all_parallel(db, text_index, None::<fn(ProgressReport)>)
        } else {
            self.index_all(db, text_index)
        }
    }

    /// Index all files using the parallel indexer with optional progress callback
    ///
    /// This method provides high-performance parallel indexing:
    /// - Phase 1: Parallel file discovery and reading
    /// - Phase 2: Parallel parsing with batching
    /// - Phase 3: Batch database writes
    /// - Phase 4: Parallel embedding generation (if enabled)
    /// - Phase 5: Full-text index updates
    pub fn index_all_parallel<F>(
        &self,
        db: &Database,
        text_index: &FullTextIndex,
        progress_callback: Option<F>,
    ) -> Result<IndexResult>
    where
        F: Fn(crate::parallel_indexer::ProgressReport) + Send + Sync + 'static,
    {
        let parallel_config = ParallelIndexConfig {
            num_threads: if self.config.indexing.parallel_threads > 0 {
                self.config.indexing.parallel_threads
            } else {
                num_cpus::get()
            },
            batch_size: self.config.indexing.batch_size,
            embedding_batch_size: self.config.indexing.embedding_batch_size,
            parallel_walking: true,
        };

        let mut parallel_indexer = if let Some(engine) = &self.embedding_engine {
            ParallelIndexer::with_embedding_engine(
                self.root.clone(),
                self.config.clone(),
                parallel_config,
                self.parser.clone(),
                engine.clone(),
            )?
        } else {
            ParallelIndexer::new(
                self.root.clone(),
                self.config.clone(),
                parallel_config,
                self.parser.clone(),
            )?
        };

        if let Some(callback) = progress_callback {
            parallel_indexer.set_progress_callback(callback);
        }

        let parallel_result = parallel_indexer.index_all(db, text_index)?;

        // Convert ParallelIndexResult to IndexResult
        Ok(IndexResult {
            indexed: parallel_result.files_indexed,
            skipped: parallel_result.files_skipped,
            errors: parallel_result.files_errored,
            symbols_extracted: parallel_result.symbols_extracted,
            symbols_by_kind: parallel_result.symbols_by_kind,
            symbols_with_visibility: parallel_result.symbols_with_visibility,
        })
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
                let pattern = if line.starts_with('/') {
                    line[1..].to_string()
                } else if line.ends_with('/') {
                    format!("**/{}", &line[..line.len() - 1])
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

    #[test]
    fn test_index_progress_percentage() {
        let progress = IndexProgress {
            total_files: 100,
            processed_files: 25,
            current_file: Some("test.rs".to_string()),
            elapsed_time: Duration::from_secs(5),
            files_per_second: 5.0,
        };

        assert!((progress.percentage() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_progress_percentage_empty() {
        let progress = IndexProgress {
            total_files: 0,
            processed_files: 0,
            current_file: None,
            elapsed_time: Duration::ZERO,
            files_per_second: 0.0,
        };

        assert!((progress.percentage() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_progress_estimated_remaining() {
        let progress = IndexProgress {
            total_files: 100,
            processed_files: 50,
            current_file: Some("test.rs".to_string()),
            elapsed_time: Duration::from_secs(10),
            files_per_second: 5.0,
        };

        // 50 remaining files / 5 files per second = 10 seconds
        let remaining = progress.estimated_remaining().unwrap();
        assert!((remaining.as_secs_f64() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_progress_estimated_remaining_complete() {
        let progress = IndexProgress {
            total_files: 100,
            processed_files: 100,
            current_file: None,
            elapsed_time: Duration::from_secs(20),
            files_per_second: 5.0,
        };

        assert!(progress.estimated_remaining().is_none());
    }

    #[test]
    fn test_index_all_with_progress_callback() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let mut progress_updates = Vec::new();

        let result = indexer
            .index_all_with_progress(&db, &text_index, |progress| {
                progress_updates.push((progress.processed_files, progress.total_files));
            })
            .unwrap();

        // Should have received progress updates for each file
        assert_eq!(progress_updates.len(), 3, "Should have 3 progress updates");

        // Final progress should be complete
        let (processed, total) = progress_updates.last().unwrap();
        assert_eq!(*processed, *total);
        assert_eq!(*total, 3);

        // Result should have indexing time
        assert!(result.indexing_time.as_millis() > 0);

        // Verify files were indexed
        assert_eq!(result.indexed, 3);
    }

    #[test]
    fn test_index_result_includes_timing() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Indexing time should be recorded
        assert!(result.indexing_time > Duration::ZERO);
    }

    #[test]
    fn test_index_result_default() {
        let result = IndexResult::default();

        assert_eq!(result.indexed, 0);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.errors, 0);
        assert_eq!(result.symbols_extracted, 0);
        assert!(result.symbols_by_kind.is_empty());
        assert_eq!(result.symbols_with_visibility, 0);
        assert_eq!(result.indexing_time, Duration::ZERO);
        assert!(result.embedding_metrics.is_none());
    }
}
