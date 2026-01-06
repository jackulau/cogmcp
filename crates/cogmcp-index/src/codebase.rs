//! Codebase file indexing

use crate::parser::{CodeParser, ExtractedSymbol};
use cogmcp_core::types::Language;
use cogmcp_core::{Config, Error, Result};
use cogmcp_embeddings::EmbeddingEngine;
use cogmcp_storage::{Database, FullTextIndex};
use globset::{Glob, GlobSet, GlobSetBuilder};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use walkdir::WalkDir;

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
        let mut result = IndexResult::default();
        let gitignore = self.load_gitignore()?;

        for entry in WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path(), &gitignore))
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Failed to read directory entry: {}", e);
                    result.errors += 1;
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
                result.skipped += 1;
                continue;
            }

            // Check file size
            let metadata = match fs::metadata(path) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("Failed to get metadata for {:?}: {}", path, e);
                    result.errors += 1;
                    continue;
                }
            };

            let size_kb = metadata.len() / 1024;
            if size_kb > self.config.indexing.max_file_size {
                result.skipped += 1;
                continue;
            }

            // Index the file and track symbol statistics
            match self.index_file_with_stats(path, db, text_index, &mut result) {
                Ok(_) => result.indexed += 1,
                Err(e) => {
                    tracing::warn!("Failed to index {:?}: {}", path, e);
                    result.errors += 1;
                }
            }
        }

        // Commit the text index
        text_index.commit()?;

        Ok(result)
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
}
