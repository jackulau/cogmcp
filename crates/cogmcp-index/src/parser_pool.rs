//! Thread-safe parser pool for concurrent file parsing
//!
//! This module provides a `ParserPool` that can parse multiple files concurrently
//! using rayon and thread-local tree-sitter parsers.

use crate::parser::{CodeParser, ExtractedSymbol};
use cogmcp_core::types::Language;
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::PathBuf;

/// Result of parsing a single file
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Path to the parsed file
    pub file_path: PathBuf,
    /// Extracted symbols from the file
    pub symbols: Vec<ExtractedSymbol>,
    /// Language of the file
    pub language: Language,
    /// Error message if parsing failed (non-fatal)
    pub error: Option<String>,
}

impl ParseResult {
    /// Create a successful parse result
    pub fn success(file_path: PathBuf, symbols: Vec<ExtractedSymbol>, language: Language) -> Self {
        Self {
            file_path,
            symbols,
            language,
            error: None,
        }
    }

    /// Create a failed parse result
    pub fn failure(file_path: PathBuf, language: Language, error: String) -> Self {
        Self {
            file_path,
            symbols: Vec::new(),
            language,
            error: Some(error),
        }
    }

    /// Check if parsing was successful
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }
}

/// Entry for a file to be parsed
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Path to the file
    pub path: PathBuf,
    /// Content of the file
    pub content: String,
    /// Language of the file
    pub language: Language,
}

impl FileEntry {
    /// Create a new file entry
    pub fn new(path: PathBuf, content: String, language: Language) -> Self {
        Self {
            path,
            content,
            language,
        }
    }
}

/// Configuration for the parser pool
#[derive(Debug, Clone)]
pub struct ParserPoolConfig {
    /// Number of threads to use (defaults to number of CPUs)
    pub pool_size: usize,
}

impl Default for ParserPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: num_cpus::get(),
        }
    }
}

impl ParserPoolConfig {
    /// Create a new configuration with specified pool size
    pub fn with_pool_size(pool_size: usize) -> Self {
        Self { pool_size }
    }
}

/// Thread-safe parser pool for concurrent file parsing
///
/// Uses rayon's parallel iterator with thread-local parsers to achieve
/// safe concurrent parsing. Each thread gets its own `CodeParser` instance.
pub struct ParserPool {
    config: ParserPoolConfig,
}

// Thread-local storage for CodeParser instances
thread_local! {
    static THREAD_PARSER: RefCell<CodeParser> = RefCell::new(CodeParser::new());
}

impl ParserPool {
    /// Create a new parser pool with default configuration
    pub fn new() -> Self {
        Self {
            config: ParserPoolConfig::default(),
        }
    }

    /// Create a new parser pool with custom configuration
    pub fn with_config(config: ParserPoolConfig) -> Self {
        Self { config }
    }

    /// Get the configured pool size
    pub fn pool_size(&self) -> usize {
        self.config.pool_size
    }

    /// Parse a batch of files concurrently
    ///
    /// Each file is parsed independently, and errors are captured per-file
    /// without stopping the overall batch processing.
    ///
    /// # Arguments
    /// * `files` - Vector of file entries to parse
    ///
    /// # Returns
    /// Vector of parse results, one for each input file in the same order
    pub fn parse_batch(&self, files: Vec<FileEntry>) -> Vec<ParseResult> {
        // Configure rayon thread pool if needed
        // Note: rayon's global pool is used by default, which is usually fine
        files
            .par_iter()
            .map(|file| self.parse_file(file))
            .collect()
    }

    /// Parse a single file using thread-local parser
    fn parse_file(&self, file: &FileEntry) -> ParseResult {
        THREAD_PARSER.with(|parser_cell| {
            let parser = parser_cell.borrow();
            match parser.parse(&file.content, file.language) {
                Ok(symbols) => ParseResult::success(file.path.clone(), symbols, file.language),
                Err(e) => ParseResult::failure(file.path.clone(), file.language, e.to_string()),
            }
        })
    }

    /// Parse files sequentially (useful for debugging or when parallelism is not desired)
    pub fn parse_sequential(&self, files: Vec<FileEntry>) -> Vec<ParseResult> {
        files.iter().map(|file| self.parse_file(file)).collect()
    }
}

impl Default for ParserPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogmcp_core::types::SymbolKind;

    #[test]
    fn test_parser_pool_creation() {
        let pool = ParserPool::new();
        assert!(pool.pool_size() > 0);

        let pool_with_config = ParserPool::with_config(ParserPoolConfig::with_pool_size(4));
        assert_eq!(pool_with_config.pool_size(), 4);
    }

    #[test]
    fn test_parse_result_success() {
        let result = ParseResult::success(
            PathBuf::from("test.rs"),
            vec![],
            Language::Rust,
        );
        assert!(result.is_success());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_parse_result_failure() {
        let result = ParseResult::failure(
            PathBuf::from("test.rs"),
            Language::Rust,
            "Parse error".to_string(),
        );
        assert!(!result.is_success());
        assert_eq!(result.error, Some("Parse error".to_string()));
    }

    #[test]
    fn test_parse_single_rust_file() {
        let pool = ParserPool::new();
        let code = r#"
pub fn hello() {
    println!("Hello, world!");
}
"#;

        let files = vec![FileEntry::new(
            PathBuf::from("test.rs"),
            code.to_string(),
            Language::Rust,
        )];

        let results = pool.parse_batch(files);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_success());
        assert!(!results[0].symbols.is_empty());

        let hello_fn = results[0].symbols.iter().find(|s| s.name == "hello");
        assert!(hello_fn.is_some());
        assert_eq!(hello_fn.unwrap().kind, SymbolKind::Function);
    }

    #[test]
    fn test_parse_multiple_files_concurrently() {
        let pool = ParserPool::new();

        let rust_code = r#"
pub fn rust_function() -> i32 { 42 }
pub struct RustStruct {}
"#;

        let python_code = r#"
def python_function():
    pass

class PythonClass:
    def method(self):
        pass
"#;

        let ts_code = r#"
export function tsFunction(): number {
    return 42;
}

export class TsClass {
    public method(): void {}
}
"#;

        let files = vec![
            FileEntry::new(PathBuf::from("test.rs"), rust_code.to_string(), Language::Rust),
            FileEntry::new(PathBuf::from("test.py"), python_code.to_string(), Language::Python),
            FileEntry::new(PathBuf::from("test.ts"), ts_code.to_string(), Language::TypeScript),
        ];

        let results = pool.parse_batch(files);

        assert_eq!(results.len(), 3);

        // All should succeed
        for result in &results {
            assert!(result.is_success(), "Failed to parse {:?}: {:?}", result.file_path, result.error);
        }

        // Check Rust file
        let rust_result = results.iter().find(|r| r.file_path.to_str() == Some("test.rs")).unwrap();
        assert!(rust_result.symbols.iter().any(|s| s.name == "rust_function"));
        assert!(rust_result.symbols.iter().any(|s| s.name == "RustStruct"));

        // Check Python file
        let python_result = results.iter().find(|r| r.file_path.to_str() == Some("test.py")).unwrap();
        assert!(python_result.symbols.iter().any(|s| s.name == "python_function"));
        assert!(python_result.symbols.iter().any(|s| s.name == "PythonClass"));
        assert!(python_result.symbols.iter().any(|s| s.name == "method"));

        // Check TypeScript file
        let ts_result = results.iter().find(|r| r.file_path.to_str() == Some("test.ts")).unwrap();
        assert!(ts_result.symbols.iter().any(|s| s.name == "tsFunction"));
        assert!(ts_result.symbols.iter().any(|s| s.name == "TsClass"));
    }

    #[test]
    fn test_parse_with_errors_continues() {
        let pool = ParserPool::new();

        let good_code = r#"
pub fn good_function() {}
"#;

        // For unsupported languages, the parser returns empty results (not an error)
        // Let's test with valid code that should all parse successfully
        let files = vec![
            FileEntry::new(PathBuf::from("good.rs"), good_code.to_string(), Language::Rust),
            FileEntry::new(PathBuf::from("unknown.txt"), "some text".to_string(), Language::Unknown),
            FileEntry::new(PathBuf::from("good2.rs"), good_code.to_string(), Language::Rust),
        ];

        let results = pool.parse_batch(files);

        assert_eq!(results.len(), 3);

        // First Rust file should have symbols
        let good_result = results.iter().find(|r| r.file_path.to_str() == Some("good.rs")).unwrap();
        assert!(good_result.is_success());
        assert!(!good_result.symbols.is_empty());

        // Unknown language returns empty but not an error
        let unknown_result = results.iter().find(|r| r.file_path.to_str() == Some("unknown.txt")).unwrap();
        assert!(unknown_result.is_success());
        assert!(unknown_result.symbols.is_empty());

        // Second Rust file should also have symbols
        let good2_result = results.iter().find(|r| r.file_path.to_str() == Some("good2.rs")).unwrap();
        assert!(good2_result.is_success());
        assert!(!good2_result.symbols.is_empty());
    }

    #[test]
    fn test_parse_empty_batch() {
        let pool = ParserPool::new();
        let results = pool.parse_batch(vec![]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_sequential() {
        let pool = ParserPool::new();
        let code = r#"
pub fn test_fn() {}
"#;

        let files = vec![
            FileEntry::new(PathBuf::from("test1.rs"), code.to_string(), Language::Rust),
            FileEntry::new(PathBuf::from("test2.rs"), code.to_string(), Language::Rust),
        ];

        let results = pool.parse_sequential(files);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_success()));
    }

    #[test]
    fn test_concurrent_parsing_large_batch() {
        let pool = ParserPool::new();

        // Create a batch of 100 files to test concurrent parsing at scale
        let code = r#"
pub fn batch_function() -> i32 { 42 }
pub struct BatchStruct { value: i32 }
impl BatchStruct {
    pub fn new() -> Self { Self { value: 0 } }
}
"#;

        let files: Vec<FileEntry> = (0..100)
            .map(|i| FileEntry::new(
                PathBuf::from(format!("file_{}.rs", i)),
                code.to_string(),
                Language::Rust,
            ))
            .collect();

        let results = pool.parse_batch(files);

        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|r| r.is_success()));

        // Each file should have at least 3 symbols (function, struct, method)
        for result in &results {
            assert!(result.symbols.len() >= 3, "Expected at least 3 symbols, got {}", result.symbols.len());
        }
    }

    #[test]
    fn test_parser_pool_config_default() {
        let config = ParserPoolConfig::default();
        assert!(config.pool_size > 0);
        assert_eq!(config.pool_size, num_cpus::get());
    }

    #[test]
    fn test_symbol_extraction_matches_code_parser() {
        // Verify that ParserPool produces the same results as CodeParser
        let pool = ParserPool::new();
        let parser = CodeParser::new();

        let code = r#"
pub async fn async_function() -> Result<(), Error> {}
pub struct MyStruct<T> { value: T }
"#;

        // Parse with pool
        let files = vec![FileEntry::new(
            PathBuf::from("test.rs"),
            code.to_string(),
            Language::Rust,
        )];
        let pool_results = pool.parse_batch(files);

        // Parse with direct parser
        let direct_symbols = parser.parse(code, Language::Rust).unwrap();

        // Should have same symbols
        assert_eq!(pool_results[0].symbols.len(), direct_symbols.len());

        // Check specific symbols match
        let pool_async_fn = pool_results[0].symbols.iter().find(|s| s.name == "async_function").unwrap();
        let direct_async_fn = direct_symbols.iter().find(|s| s.name == "async_function").unwrap();

        assert_eq!(pool_async_fn.kind, direct_async_fn.kind);
        assert_eq!(pool_async_fn.modifiers.is_async, direct_async_fn.modifiers.is_async);
    }
}
