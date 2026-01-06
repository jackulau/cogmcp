//! Codebase file indexing

use crate::parser::CodeParser;
use contextmcp_core::types::Language;
use contextmcp_core::{Config, Error, Result};
use contextmcp_storage::{Database, FullTextIndex};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::HashSet;
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
        })
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

            // Index the file
            match self.index_file(path, db, text_index) {
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

        // Parse and extract symbols
        if let Ok(symbols) = self.parser.parse(&content, language) {
            // Delete old symbols for this file
            let _ = db.delete_symbols_for_file(file_id);

            // Insert new symbols
            for sym in symbols {
                let kind_str = format!("{:?}", sym.kind).to_lowercase();
                let _ = db.insert_symbol(
                    file_id,
                    &sym.name,
                    &kind_str,
                    sym.start_line,
                    sym.end_line,
                    sym.signature.as_deref(),
                    sym.doc_comment.as_deref(),
                );
            }
        }

        // Index in full-text search
        text_index.index_file(&relative_path, &content)?;

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
}
