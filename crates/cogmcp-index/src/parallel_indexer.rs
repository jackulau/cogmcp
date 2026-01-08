//! Parallel indexing orchestrator
//!
//! This module provides a high-performance parallel indexing pipeline that
//! orchestrates file discovery, parsing, database storage, and embedding
//! generation using concurrent processing.

use crate::parser::{CodeParser, ExtractedSymbol};
use cogmcp_core::types::Language;
use cogmcp_core::{Config, Error, Result};
use cogmcp_embeddings::EmbeddingEngine;
use cogmcp_storage::{Database, EmbeddingInput, FullTextIndex};
use globset::{Glob, GlobSet, GlobSetBuilder};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use walkdir::WalkDir;

/// Configuration for parallel indexing
#[derive(Debug, Clone)]
pub struct ParallelIndexConfig {
    /// Number of threads for parallel operations (default: num_cpus)
    pub num_threads: usize,
    /// Batch size for database operations (default: 100)
    pub batch_size: usize,
    /// Batch size for embedding generation (default: 32)
    pub embedding_batch_size: usize,
    /// Enable parallel file walking
    pub parallel_walking: bool,
}

impl Default for ParallelIndexConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            batch_size: 100,
            embedding_batch_size: 32,
            parallel_walking: true,
        }
    }
}

/// Progress callback for reporting indexing progress
pub type ProgressCallback = Box<dyn Fn(ProgressReport) + Send + Sync>;

/// Progress report during indexing
#[derive(Debug, Clone)]
pub struct ProgressReport {
    /// Current phase of indexing
    pub phase: IndexPhase,
    /// Number of items processed so far
    pub processed: u64,
    /// Total number of items (if known)
    pub total: Option<u64>,
    /// Description of current operation
    pub message: String,
}

/// Phases of the indexing pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPhase {
    /// Discovering files in the codebase
    FileDiscovery,
    /// Reading file contents
    FileReading,
    /// Parsing files and extracting symbols
    Parsing,
    /// Writing to database
    DatabaseWrite,
    /// Generating embeddings
    EmbeddingGeneration,
    /// Updating full-text index
    FullTextIndex,
    /// Completed
    Complete,
}

/// File data collected during discovery
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub relative_path: String,
    pub content: String,
    pub hash: String,
    pub modified_at: i64,
    pub size: u64,
    pub language: Language,
}

/// Parsed file with extracted symbols
#[derive(Debug)]
pub struct ParsedFile {
    pub file: DiscoveredFile,
    pub symbols: Vec<ExtractedSymbol>,
}

/// Result of a parallel indexing operation
#[derive(Debug, Default)]
pub struct ParallelIndexResult {
    pub files_indexed: u64,
    pub files_skipped: u64,
    pub files_errored: u64,
    pub symbols_extracted: u64,
    pub symbols_by_kind: HashMap<String, u64>,
    pub symbols_with_visibility: u64,
    pub embeddings_generated: u64,
}

/// Parallel indexer that orchestrates all parallel components
pub struct ParallelIndexer {
    root: PathBuf,
    config: Config,
    parallel_config: ParallelIndexConfig,
    ignore_patterns: GlobSet,
    include_extensions: HashSet<String>,
    parser: Arc<CodeParser>,
    embedding_engine: Option<Arc<Mutex<EmbeddingEngine>>>,
    progress_callback: Option<Arc<ProgressCallback>>,
}

impl ParallelIndexer {
    /// Create a new parallel indexer
    pub fn new(
        root: PathBuf,
        config: Config,
        parallel_config: ParallelIndexConfig,
        parser: Arc<CodeParser>,
    ) -> Result<Self> {
        let ignore_patterns = Self::build_ignore_patterns(&config)?;
        let include_extensions: HashSet<String> = config
            .indexing
            .include_types
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        // Configure rayon thread pool
        if parallel_config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(parallel_config.num_threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        Ok(Self {
            root,
            config,
            parallel_config,
            ignore_patterns,
            include_extensions,
            parser,
            embedding_engine: None,
            progress_callback: None,
        })
    }

    /// Create a parallel indexer with embedding support
    pub fn with_embedding_engine(
        root: PathBuf,
        config: Config,
        parallel_config: ParallelIndexConfig,
        parser: Arc<CodeParser>,
        engine: Arc<Mutex<EmbeddingEngine>>,
    ) -> Result<Self> {
        let mut indexer = Self::new(root, config, parallel_config, parser)?;
        indexer.embedding_engine = Some(engine);
        Ok(indexer)
    }

    /// Set the embedding engine
    pub fn set_embedding_engine(&mut self, engine: Arc<Mutex<EmbeddingEngine>>) {
        self.embedding_engine = Some(engine);
    }

    /// Set a progress callback for reporting indexing progress
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(ProgressReport) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(Box::new(callback)));
    }

    /// Check if embeddings are enabled and available
    pub fn embeddings_enabled(&self) -> bool {
        self.config.indexing.enable_embeddings
            && self
                .embedding_engine
                .as_ref()
                .map_or(false, |e| e.lock().is_loaded())
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

    /// Report progress if a callback is set
    fn report_progress(&self, phase: IndexPhase, processed: u64, total: Option<u64>, message: &str) {
        if let Some(callback) = &self.progress_callback {
            callback(ProgressReport {
                phase,
                processed,
                total,
                message: message.to_string(),
            });
        }
    }

    /// Index all files in the codebase using the parallel pipeline
    pub fn index_all(&self, db: &Database, text_index: &FullTextIndex) -> Result<ParallelIndexResult> {
        let mut result = ParallelIndexResult::default();

        // Phase 1: Parallel file discovery and reading
        self.report_progress(IndexPhase::FileDiscovery, 0, None, "Discovering files...");
        let files = self.discover_and_read_files()?;
        let total_files = files.len() as u64;
        self.report_progress(
            IndexPhase::FileDiscovery,
            total_files,
            Some(total_files),
            &format!("Discovered {} files", total_files),
        );

        if files.is_empty() {
            self.report_progress(IndexPhase::Complete, 0, Some(0), "No files to index");
            return Ok(result);
        }

        // Phase 2: Parallel parsing
        self.report_progress(IndexPhase::Parsing, 0, Some(total_files), "Parsing files...");
        let parsed_files = self.parse_files_parallel(&files, &mut result)?;
        self.report_progress(
            IndexPhase::Parsing,
            parsed_files.len() as u64,
            Some(total_files),
            &format!("Parsed {} files", parsed_files.len()),
        );

        // Phase 3: Batch database writes
        self.report_progress(
            IndexPhase::DatabaseWrite,
            0,
            Some(parsed_files.len() as u64),
            "Writing to database...",
        );
        self.write_to_database_batch(db, &parsed_files, &mut result)?;
        self.report_progress(
            IndexPhase::DatabaseWrite,
            result.files_indexed,
            Some(parsed_files.len() as u64),
            &format!("Indexed {} files to database", result.files_indexed),
        );

        // Phase 4: Parallel embedding generation (if enabled)
        if self.embeddings_enabled() {
            self.report_progress(
                IndexPhase::EmbeddingGeneration,
                0,
                None,
                "Generating embeddings...",
            );
            self.generate_embeddings_batch(db, &parsed_files, &mut result)?;
            self.report_progress(
                IndexPhase::EmbeddingGeneration,
                result.embeddings_generated,
                None,
                &format!("Generated {} embeddings", result.embeddings_generated),
            );
        }

        // Phase 5: Update full-text index
        self.report_progress(
            IndexPhase::FullTextIndex,
            0,
            Some(parsed_files.len() as u64),
            "Updating full-text index...",
        );
        self.update_text_index(text_index, &parsed_files)?;
        text_index.commit()?;
        self.report_progress(
            IndexPhase::FullTextIndex,
            parsed_files.len() as u64,
            Some(parsed_files.len() as u64),
            "Full-text index updated",
        );

        self.report_progress(
            IndexPhase::Complete,
            result.files_indexed,
            Some(result.files_indexed),
            &format!(
                "Indexing complete: {} files, {} symbols",
                result.files_indexed, result.symbols_extracted
            ),
        );

        Ok(result)
    }

    /// Discover and read all files in parallel
    fn discover_and_read_files(&self) -> Result<Vec<DiscoveredFile>> {
        let gitignore = self.load_gitignore()?;

        // Collect file paths first
        let paths: Vec<PathBuf> = WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path(), &gitignore))
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                let ext = e
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                self.include_extensions.contains(&ext)
            })
            .map(|e| e.path().to_path_buf())
            .collect();

        // Read files in parallel
        let files: Vec<DiscoveredFile> = paths
            .par_iter()
            .filter_map(|path| self.read_file(path).ok())
            .collect();

        Ok(files)
    }

    /// Read a single file and extract metadata
    fn read_file(&self, path: &Path) -> Result<DiscoveredFile> {
        let metadata = fs::metadata(path)?;

        // Check file size
        let size_kb = metadata.len() / 1024;
        if size_kb > self.config.indexing.max_file_size {
            return Err(Error::Config(format!(
                "File too large: {} KB > {} KB",
                size_kb, self.config.indexing.max_file_size
            )));
        }

        let content = fs::read_to_string(path)?;
        let relative_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let hash = Self::hash_content(&content);
        let modified_at = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);

        Ok(DiscoveredFile {
            path: path.to_path_buf(),
            relative_path,
            content,
            hash,
            modified_at,
            size: metadata.len(),
            language,
        })
    }

    /// Parse files in parallel
    fn parse_files_parallel(
        &self,
        files: &[DiscoveredFile],
        result: &mut ParallelIndexResult,
    ) -> Result<Vec<ParsedFile>> {
        let error_count = AtomicU64::new(0);
        let skip_count = AtomicU64::new(0);

        let parsed: Vec<ParsedFile> = files
            .par_iter()
            .filter_map(|file| {
                match self.parser.parse(&file.content, file.language) {
                    Ok(symbols) => Some(ParsedFile {
                        file: file.clone(),
                        symbols,
                    }),
                    Err(e) => {
                        tracing::warn!("Failed to parse {:?}: {}", file.path, e);
                        error_count.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                }
            })
            .collect();

        result.files_errored = error_count.load(Ordering::Relaxed);
        result.files_skipped = skip_count.load(Ordering::Relaxed);

        Ok(parsed)
    }

    /// Write parsed files to database in batches
    fn write_to_database_batch(
        &self,
        db: &Database,
        parsed_files: &[ParsedFile],
        result: &mut ParallelIndexResult,
    ) -> Result<()> {
        for chunk in parsed_files.chunks(self.parallel_config.batch_size) {
            for parsed in chunk {
                let language_str = format!("{:?}", parsed.file.language).to_lowercase();

                // Upsert file
                let file_id = db.upsert_file(
                    &parsed.file.relative_path,
                    &parsed.file.hash,
                    parsed.file.modified_at,
                    parsed.file.size,
                    &language_str,
                )?;

                // Delete old symbols for this file
                let _ = db.delete_symbols_for_file(file_id);

                // Insert symbols
                self.insert_symbols_with_relationships(db, file_id, &parsed.symbols, result)?;

                result.files_indexed += 1;
            }
        }

        Ok(())
    }

    /// Insert symbols and establish parent-child relationships
    fn insert_symbols_with_relationships(
        &self,
        db: &Database,
        file_id: i64,
        symbols: &[ExtractedSymbol],
        result: &mut ParallelIndexResult,
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

            // Track statistics
            result.symbols_extracted += 1;
            *result.symbols_by_kind.entry(kind_str).or_insert(0) += 1;

            if sym.visibility != cogmcp_core::types::SymbolVisibility::Unknown {
                result.symbols_with_visibility += 1;
            }
        }

        // Second pass: update parent references
        for sym in symbols {
            if let Some(parent_name) = &sym.parent_name {
                if let Some(&parent_id) = name_to_id.get(parent_name) {
                    if let Some(&symbol_id) = name_to_id.get(&sym.name) {
                        db.update_symbol_parent(symbol_id, parent_id)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate embeddings for symbols in batches
    fn generate_embeddings_batch(
        &self,
        db: &Database,
        parsed_files: &[ParsedFile],
        result: &mut ParallelIndexResult,
    ) -> Result<()> {
        let engine = match &self.embedding_engine {
            Some(e) => e,
            None => return Ok(()),
        };

        // Collect all symbols with their text for embedding
        let mut embedding_inputs: Vec<(String, Option<i64>, Option<i64>)> = Vec::new();

        for parsed in parsed_files {
            // Get file ID from database
            if let Ok(Some(file_row)) = db.get_file_by_path(&parsed.file.relative_path) {
                for sym in &parsed.symbols {
                    // Create embedding text from symbol info
                    let embed_text = format!(
                        "{} {} {}",
                        sym.name,
                        sym.signature.as_deref().unwrap_or(""),
                        sym.doc_comment.as_deref().unwrap_or("")
                    );

                    if !embed_text.trim().is_empty() {
                        // Get symbol ID
                        if let Ok(symbols) = db.find_symbols_by_name(&sym.name, false) {
                            if let Some(symbol_row) = symbols.iter().find(|s| s.file_id == file_row.id) {
                                embedding_inputs.push((
                                    embed_text,
                                    Some(symbol_row.id),
                                    Some(file_row.id),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Process in batches
        for chunk in embedding_inputs.chunks(self.parallel_config.embedding_batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|(t, _, _)| t.as_str()).collect();

            let embeddings = {
                let mut engine_guard = engine.lock();
                engine_guard.embed_batch(&texts)?
            };

            // Prepare batch input for database
            let mut batch: Vec<EmbeddingInput> = Vec::with_capacity(chunk.len());
            for ((text, symbol_id, file_id), embedding) in chunk.iter().zip(embeddings.into_iter()) {
                batch.push(EmbeddingInput {
                    symbol_id: *symbol_id,
                    file_id: *file_id,
                    chunk_text: text.clone(),
                    embedding,
                    chunk_type: "symbol".to_string(),
                });
            }

            // Insert batch
            db.insert_embeddings_batch(&batch)?;
            result.embeddings_generated += batch.len() as u64;
        }

        Ok(())
    }

    /// Update the full-text index
    fn update_text_index(&self, text_index: &FullTextIndex, parsed_files: &[ParsedFile]) -> Result<()> {
        for parsed in parsed_files {
            text_index.index_file(&parsed.file.relative_path, &parsed.file.content)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn setup_test_files(temp_dir: &TempDir) {
        // Create a Rust file
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

    fn create_test_indexer(root: PathBuf) -> ParallelIndexer {
        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        config.indexing.enable_embeddings = false;
        let parser = Arc::new(CodeParser::new());
        let parallel_config = ParallelIndexConfig::default();
        ParallelIndexer::new(root, config, parallel_config, parser).unwrap()
    }

    #[test]
    fn test_parallel_index_files() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Verify files were indexed
        assert!(result.files_indexed > 0, "Should have indexed some files");
        assert!(result.symbols_extracted > 0, "Should have extracted some symbols");

        // Verify symbols by kind
        assert!(result.symbols_by_kind.contains_key("struct") || result.symbols_by_kind.contains_key("class"));
        assert!(result.symbols_by_kind.contains_key("function") || result.symbols_by_kind.contains_key("method"));

        // Verify visibility was extracted
        assert!(result.symbols_with_visibility > 0);
    }

    #[test]
    fn test_parallel_indexer_produces_same_results_as_sequential() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        // Index with parallel indexer
        let parallel_indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let parallel_db = Database::in_memory().unwrap();
        let parallel_text_index = FullTextIndex::in_memory().unwrap();
        let parallel_result = parallel_indexer.index_all(&parallel_db, &parallel_text_index).unwrap();

        // Index with sequential indexer
        let mut config = Config::default();
        config.indexing.include_types = vec!["rs".to_string(), "ts".to_string(), "py".to_string()];
        let parser = Arc::new(CodeParser::new());
        let sequential_indexer = crate::codebase::CodebaseIndexer::new(
            temp_dir.path().to_path_buf(),
            config,
            parser,
        )
        .unwrap();
        let sequential_db = Database::in_memory().unwrap();
        let sequential_text_index = FullTextIndex::in_memory().unwrap();
        let sequential_result = sequential_indexer.index_all(&sequential_db, &sequential_text_index).unwrap();

        // Compare results
        assert_eq!(
            parallel_result.files_indexed, sequential_result.indexed,
            "Should index same number of files"
        );
        assert_eq!(
            parallel_result.symbols_extracted, sequential_result.symbols_extracted,
            "Should extract same number of symbols"
        );
    }

    #[test]
    fn test_progress_callback() {
        let temp_dir = TempDir::new().unwrap();
        setup_test_files(&temp_dir);

        let mut indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let progress_reports = Arc::new(Mutex::new(Vec::new()));
        let reports_clone = progress_reports.clone();

        indexer.set_progress_callback(move |report| {
            reports_clone.lock().push(report);
        });

        indexer.index_all(&db, &text_index).unwrap();

        let reports = progress_reports.lock();
        assert!(!reports.is_empty(), "Should have received progress reports");

        // Verify we got reports for different phases
        let phases: Vec<IndexPhase> = reports.iter().map(|r| r.phase).collect();
        assert!(phases.contains(&IndexPhase::FileDiscovery));
        assert!(phases.contains(&IndexPhase::Parsing));
        assert!(phases.contains(&IndexPhase::DatabaseWrite));
        assert!(phases.contains(&IndexPhase::Complete));
    }

    #[test]
    fn test_parallel_config_defaults() {
        let config = ParallelIndexConfig::default();
        assert!(config.num_threads > 0);
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.embedding_batch_size, 32);
        assert!(config.parallel_walking);
    }

    #[test]
    fn test_empty_directory() {
        let temp_dir = TempDir::new().unwrap();

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        assert_eq!(result.files_indexed, 0);
        assert_eq!(result.symbols_extracted, 0);
    }

    #[test]
    fn test_file_size_limit() {
        let temp_dir = TempDir::new().unwrap();

        // Create a file that exceeds the default size limit
        let large_file = temp_dir.path().join("large.rs");
        let large_content = "x".repeat(600 * 1024); // 600KB, exceeds 500KB default
        fs::write(&large_file, large_content).unwrap();

        let indexer = create_test_indexer(temp_dir.path().to_path_buf());
        let db = Database::in_memory().unwrap();
        let text_index = FullTextIndex::in_memory().unwrap();

        let result = indexer.index_all(&db, &text_index).unwrap();

        // Large file should be skipped
        assert_eq!(result.files_indexed, 0);
    }
}
