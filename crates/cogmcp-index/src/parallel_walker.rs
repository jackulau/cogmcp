//! Parallel file discovery and content reading using rayon.
//!
//! This module provides a parallel file walker that discovers and reads files
//! concurrently for improved indexing performance on large codebases.

use cogmcp_core::{Error, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Configuration for the parallel file walker.
#[derive(Debug, Clone)]
pub struct WalkerConfig {
    /// Maximum file size in bytes to include.
    pub max_file_size: u64,
    /// Glob patterns to ignore.
    pub ignore_patterns: GlobSet,
    /// File extensions to include (lowercase, without dot).
    pub include_extensions: HashSet<String>,
}

impl WalkerConfig {
    /// Create a new walker configuration.
    pub fn new(
        max_file_size: u64,
        ignore_patterns: Vec<String>,
        include_extensions: Vec<String>,
    ) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in &ignore_patterns {
            let glob = Glob::new(pattern)
                .map_err(|e| Error::Config(format!("Invalid glob pattern '{}': {}", pattern, e)))?;
            builder.add(glob);
        }
        let ignore_patterns = builder
            .build()
            .map_err(|e| Error::Config(format!("Failed to build glob set: {}", e)))?;

        let include_extensions = include_extensions
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();

        Ok(Self {
            max_file_size,
            ignore_patterns,
            include_extensions,
        })
    }

    /// Create a default configuration suitable for code indexing.
    pub fn default_config() -> Result<Self> {
        Self::new(
            500 * 1024, // 500 KB
            vec![
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "dist/**".to_string(),
                ".git/**".to_string(),
                "*.lock".to_string(),
                "__pycache__/**".to_string(),
                ".venv/**".to_string(),
                "venv/**".to_string(),
            ],
            vec![
                "rs".to_string(),
                "ts".to_string(),
                "tsx".to_string(),
                "js".to_string(),
                "jsx".to_string(),
                "py".to_string(),
                "go".to_string(),
                "java".to_string(),
                "md".to_string(),
                "json".to_string(),
                "toml".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
            ],
        )
    }
}

/// A discovered file entry with its content and metadata.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// File content as a string.
    pub content: String,
    /// SHA256 hash of the content.
    pub hash: String,
    /// File size in bytes.
    pub size: u64,
}

/// Parallel file walker for discovering and reading files concurrently.
#[derive(Debug)]
pub struct ParallelFileWalker {
    /// Root directory to walk.
    root: PathBuf,
    /// Configuration for the walker.
    config: WalkerConfig,
    /// Gitignore patterns loaded from .gitignore file.
    gitignore_patterns: GlobSet,
}

impl ParallelFileWalker {
    /// Create a new parallel file walker.
    pub fn new(root: PathBuf, config: WalkerConfig) -> Result<Self> {
        let gitignore_patterns = Self::load_gitignore(&root)?;
        Ok(Self {
            root,
            config,
            gitignore_patterns,
        })
    }

    /// Load gitignore patterns from the .gitignore file in the root directory.
    fn load_gitignore(root: &Path) -> Result<GlobSet> {
        let gitignore_path = root.join(".gitignore");
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

    /// Check if a path should be ignored based on patterns.
    fn should_ignore(&self, path: &Path) -> bool {
        let relative = path.strip_prefix(&self.root).unwrap_or(path);
        let path_str = relative.to_string_lossy();

        // Check gitignore patterns
        if self.gitignore_patterns.is_match(relative) {
            return true;
        }

        // Check config ignore patterns
        if self.config.ignore_patterns.is_match(relative) {
            return true;
        }

        // Always ignore .git directory
        if path_str.contains(".git") {
            return true;
        }

        false
    }

    /// Check if a file should be included based on extension.
    fn should_include_extension(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        self.config.include_extensions.contains(&ext)
    }

    /// Discover all files in the directory that match the configuration.
    /// Returns a list of file paths that should be processed.
    pub fn discover_files(&self) -> Result<Vec<PathBuf>> {
        let files: Vec<PathBuf> = WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !self.should_ignore(e.path()))
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| self.should_include_extension(entry.path()))
            .filter(|entry| {
                entry
                    .metadata()
                    .map(|m| m.len() <= self.config.max_file_size)
                    .unwrap_or(false)
            })
            .map(|entry| entry.into_path())
            .collect();

        Ok(files)
    }

    /// Read file content and compute hash.
    fn read_file_entry(path: &Path) -> Option<FileEntry> {
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to read file {:?}: {}", path, e);
                return None;
            }
        };

        let size = match fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                tracing::warn!("Failed to get metadata for {:?}: {}", path, e);
                return None;
            }
        };

        let hash = Self::hash_content(&content);

        Some(FileEntry {
            path: path.to_path_buf(),
            content,
            hash,
            size,
        })
    }

    /// Compute SHA256 hash of content.
    fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    /// Read files in parallel and return their entries.
    /// This method first discovers files, then reads them in parallel using rayon.
    pub fn read_files_parallel(&self) -> Result<Vec<FileEntry>> {
        let files = self.discover_files()?;

        let entries: Vec<FileEntry> = files
            .par_iter()
            .filter_map(|path| Self::read_file_entry(path))
            .collect();

        Ok(entries)
    }

    /// Read files in parallel with chunking for memory efficiency.
    /// Processes files in chunks of the specified size to limit memory usage.
    pub fn read_files_chunked(&self, chunk_size: usize) -> Result<Vec<FileEntry>> {
        let files = self.discover_files()?;
        let mut all_entries = Vec::with_capacity(files.len());

        for chunk in files.chunks(chunk_size) {
            let entries: Vec<FileEntry> = chunk
                .par_iter()
                .filter_map(|path| Self::read_file_entry(path))
                .collect();
            all_entries.extend(entries);
        }

        Ok(all_entries)
    }

    /// Get the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn setup_test_directory() -> TempDir {
        let temp_dir = TempDir::new().unwrap();

        // Create a Rust file
        let rust_file = temp_dir.path().join("main.rs");
        let mut file = fs::File::create(&rust_file).unwrap();
        writeln!(file, "fn main() {{ println!(\"Hello\"); }}").unwrap();

        // Create a TypeScript file
        let ts_file = temp_dir.path().join("index.ts");
        let mut file = fs::File::create(&ts_file).unwrap();
        writeln!(file, "export function hello(): string {{ return 'hello'; }}").unwrap();

        // Create a Python file
        let py_file = temp_dir.path().join("script.py");
        let mut file = fs::File::create(&py_file).unwrap();
        writeln!(file, "def hello():\n    print('Hello')").unwrap();

        // Create a file that should be ignored (no extension match)
        let txt_file = temp_dir.path().join("readme.txt");
        let mut file = fs::File::create(&txt_file).unwrap();
        writeln!(file, "This is a readme").unwrap();

        // Create a .gitignore file
        let gitignore = temp_dir.path().join(".gitignore");
        let mut file = fs::File::create(&gitignore).unwrap();
        writeln!(file, "ignored_file.rs").unwrap();

        // Create a file that should be gitignored
        let ignored_file = temp_dir.path().join("ignored_file.rs");
        let mut file = fs::File::create(&ignored_file).unwrap();
        writeln!(file, "// This should be ignored").unwrap();

        // Create a subdirectory with a file
        let subdir = temp_dir.path().join("src");
        fs::create_dir(&subdir).unwrap();
        let lib_file = subdir.join("lib.rs");
        let mut file = fs::File::create(&lib_file).unwrap();
        writeln!(file, "pub mod tests;").unwrap();

        // Create node_modules directory (should be ignored)
        let node_modules = temp_dir.path().join("node_modules");
        fs::create_dir(&node_modules).unwrap();
        let ignored_js = node_modules.join("package.js");
        let mut file = fs::File::create(&ignored_js).unwrap();
        writeln!(file, "// This should be ignored").unwrap();

        temp_dir
    }

    #[test]
    fn test_walker_config_new() {
        let config = WalkerConfig::new(
            1024 * 1024,
            vec!["*.log".to_string()],
            vec!["rs".to_string(), "py".to_string()],
        )
        .unwrap();

        assert_eq!(config.max_file_size, 1024 * 1024);
        assert!(config.include_extensions.contains("rs"));
        assert!(config.include_extensions.contains("py"));
    }

    #[test]
    fn test_walker_config_default() {
        let config = WalkerConfig::default_config().unwrap();
        assert!(config.include_extensions.contains("rs"));
        assert!(config.include_extensions.contains("ts"));
        assert!(config.include_extensions.contains("py"));
    }

    #[test]
    fn test_discover_files() {
        let temp_dir = setup_test_directory();
        let config = WalkerConfig::default_config().unwrap();
        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();

        let files = walker.discover_files().unwrap();

        // Should find: main.rs, index.ts, script.py, src/lib.rs
        // Should NOT find: readme.txt, ignored_file.rs, node_modules/package.js
        assert_eq!(files.len(), 4, "Should discover exactly 4 files");

        let file_names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert!(file_names.contains(&"main.rs".to_string()));
        assert!(file_names.contains(&"index.ts".to_string()));
        assert!(file_names.contains(&"script.py".to_string()));
        assert!(file_names.contains(&"lib.rs".to_string()));
        assert!(!file_names.contains(&"readme.txt".to_string()));
        assert!(!file_names.contains(&"ignored_file.rs".to_string()));
        assert!(!file_names.contains(&"package.js".to_string()));
    }

    #[test]
    fn test_read_files_parallel() {
        let temp_dir = setup_test_directory();
        let config = WalkerConfig::default_config().unwrap();
        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();

        let entries = walker.read_files_parallel().unwrap();

        assert_eq!(entries.len(), 4, "Should read exactly 4 files");

        // Verify content was read correctly
        for entry in &entries {
            assert!(!entry.content.is_empty(), "Content should not be empty");
            assert!(!entry.hash.is_empty(), "Hash should not be empty");
            assert!(entry.size > 0, "Size should be greater than 0");
        }

        // Find the main.rs entry and verify its content
        let main_entry = entries
            .iter()
            .find(|e| e.path.file_name().unwrap().to_string_lossy() == "main.rs")
            .expect("Should find main.rs");
        assert!(main_entry.content.contains("fn main()"));
    }

    #[test]
    fn test_read_files_chunked() {
        let temp_dir = setup_test_directory();
        let config = WalkerConfig::default_config().unwrap();
        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();

        let entries = walker.read_files_chunked(2).unwrap();

        assert_eq!(entries.len(), 4, "Should read exactly 4 files with chunking");
    }

    #[test]
    fn test_hash_content() {
        let content = "Hello, World!";
        let hash = ParallelFileWalker::hash_content(content);

        // SHA256 hash should be 64 hex characters
        assert_eq!(hash.len(), 64);

        // Same content should produce same hash
        let hash2 = ParallelFileWalker::hash_content(content);
        assert_eq!(hash, hash2);

        // Different content should produce different hash
        let hash3 = ParallelFileWalker::hash_content("Different content");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_file_size_filtering() {
        let temp_dir = TempDir::new().unwrap();

        // Create a small file
        let small_file = temp_dir.path().join("small.rs");
        let mut file = fs::File::create(&small_file).unwrap();
        writeln!(file, "fn small() {{}}").unwrap();

        // Create a large file (larger than 100 bytes)
        let large_file = temp_dir.path().join("large.rs");
        let mut file = fs::File::create(&large_file).unwrap();
        let large_content = "x".repeat(200);
        writeln!(file, "{}", large_content).unwrap();

        let config = WalkerConfig::new(
            100, // 100 bytes max
            vec![],
            vec!["rs".to_string()],
        )
        .unwrap();

        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();
        let files = walker.discover_files().unwrap();

        assert_eq!(files.len(), 1, "Should only find the small file");
        assert!(
            files[0].file_name().unwrap().to_string_lossy() == "small.rs",
            "Should find small.rs"
        );
    }

    #[test]
    fn test_gitignore_patterns() {
        let temp_dir = TempDir::new().unwrap();

        // Create .gitignore
        let gitignore = temp_dir.path().join(".gitignore");
        let mut file = fs::File::create(&gitignore).unwrap();
        writeln!(file, "ignored/").unwrap();
        writeln!(file, "*.generated.rs").unwrap();

        // Create files
        let normal_file = temp_dir.path().join("normal.rs");
        let mut file = fs::File::create(&normal_file).unwrap();
        writeln!(file, "fn normal() {{}}").unwrap();

        let generated_file = temp_dir.path().join("code.generated.rs");
        let mut file = fs::File::create(&generated_file).unwrap();
        writeln!(file, "fn generated() {{}}").unwrap();

        let ignored_dir = temp_dir.path().join("ignored");
        fs::create_dir(&ignored_dir).unwrap();
        let ignored_file = ignored_dir.join("file.rs");
        let mut file = fs::File::create(&ignored_file).unwrap();
        writeln!(file, "fn ignored() {{}}").unwrap();

        let config = WalkerConfig::new(1024 * 1024, vec![], vec!["rs".to_string()]).unwrap();

        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();
        let files = walker.discover_files().unwrap();

        let file_names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert!(file_names.contains(&"normal.rs".to_string()));
        assert!(!file_names.contains(&"code.generated.rs".to_string()));
        assert!(!file_names.contains(&"file.rs".to_string()));
    }

    #[test]
    fn test_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalkerConfig::default_config().unwrap();
        let walker = ParallelFileWalker::new(temp_dir.path().to_path_buf(), config).unwrap();

        let files = walker.discover_files().unwrap();
        assert!(files.is_empty(), "Should find no files in empty directory");

        let entries = walker.read_files_parallel().unwrap();
        assert!(
            entries.is_empty(),
            "Should have no entries in empty directory"
        );
    }

    #[test]
    fn test_file_entry_struct() {
        let entry = FileEntry {
            path: PathBuf::from("/test/file.rs"),
            content: "fn test() {}".to_string(),
            hash: "abc123".to_string(),
            size: 12,
        };

        assert_eq!(entry.path, PathBuf::from("/test/file.rs"));
        assert_eq!(entry.content, "fn test() {}");
        assert_eq!(entry.hash, "abc123");
        assert_eq!(entry.size, 12);
    }
}
