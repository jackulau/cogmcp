//! Actionable error types that provide user-friendly messages and recovery suggestions.

use std::fmt;

/// Error codes for categorizing actionable errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Index-related errors (1xx)
    IndexEmpty = 100,
    IndexStale = 101,
    IndexCorrupted = 102,
    IndexInProgress = 103,

    // Search-related errors (2xx)
    SearchNoResults = 200,
    SearchInvalidPattern = 201,
    SearchTimeout = 202,

    // File-related errors (3xx)
    FileNotFound = 300,
    FileUnreadable = 301,
    FileNotIndexed = 302,

    // Symbol-related errors (4xx)
    SymbolNotFound = 400,
    SymbolAmbiguous = 401,

    // Semantic search errors (5xx)
    SemanticDisabled = 500,
    SemanticModelNotLoaded = 501,
    SemanticNoEmbeddings = 502,

    // Configuration errors (6xx)
    ConfigInvalid = 600,
    ConfigMissing = 601,

    // General errors (9xx)
    InternalError = 900,
    Unknown = 999,
}

/// An actionable error that provides guidance to the user
#[derive(Debug, Clone)]
pub struct ActionableError {
    /// Error code for programmatic handling
    pub code: ErrorCode,
    /// User-friendly error message
    pub message: String,
    /// Suggested actions to resolve the error
    pub suggestions: Vec<String>,
    /// The underlying error message (for debugging)
    pub cause: Option<String>,
}

impl ActionableError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            suggestions: Vec::new(),
            cause: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }

    /// Format the error for display to the user
    pub fn to_user_message(&self) -> String {
        let mut output = self.message.clone();

        if !self.suggestions.is_empty() {
            output.push_str("\n\nSuggested actions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("\n  {}. {}", i + 1, suggestion));
            }
        }

        output
    }
}

impl fmt::Display for ActionableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_user_message())
    }
}

impl std::error::Error for ActionableError {}

// Pre-built common errors with suggestions

impl ActionableError {
    /// Error when index is empty
    pub fn index_empty() -> Self {
        Self::new(ErrorCode::IndexEmpty, "The code index is empty.")
            .with_suggestion("Run `reindex` tool to index the codebase")
            .with_suggestion("Check that the root directory contains source files")
    }

    /// Error when no search results found
    pub fn no_search_results(query: &str) -> Self {
        Self::new(
            ErrorCode::SearchNoResults,
            format!("No results found for '{}'.", query),
        )
        .with_suggestion("Try a broader search term")
        .with_suggestion("Check spelling and try alternative terms")
        .with_suggestion("Use `index_status` to verify files are indexed")
    }

    /// Error when no symbols found
    pub fn symbol_not_found(name: &str) -> Self {
        Self::new(
            ErrorCode::SymbolNotFound,
            format!("No symbols found matching '{}'.", name),
        )
        .with_suggestion("Try a partial match with fuzzy=true")
        .with_suggestion("Check symbol name spelling and case")
        .with_suggestion("Run `reindex` if the symbol was recently added")
    }

    /// Error when file not found
    pub fn file_not_found(path: &str) -> Self {
        Self::new(ErrorCode::FileNotFound, format!("File not found: {}", path))
            .with_suggestion("Check the file path is relative to the project root")
            .with_suggestion("Verify the file exists in the repository")
    }

    /// Error when file cannot be read
    pub fn file_unreadable(path: &str, cause: &str) -> Self {
        Self::new(
            ErrorCode::FileUnreadable,
            format!("Cannot read file: {}", path),
        )
        .with_cause(cause)
        .with_suggestion("Check file permissions")
        .with_suggestion("Ensure the file is not open in another process")
    }

    /// Error when semantic search is disabled
    pub fn semantic_disabled() -> Self {
        Self::new(ErrorCode::SemanticDisabled, "Semantic search is disabled.")
            .with_suggestion(
                "Enable embeddings in configuration: set indexing.enable_embeddings = true",
            )
            .with_suggestion("Use `context_search` with mode='keyword' for text-based search")
    }

    /// Error when embedding model is not loaded
    pub fn semantic_model_not_loaded() -> Self {
        Self::new(
            ErrorCode::SemanticModelNotLoaded,
            "Semantic search model is not loaded.",
        )
        .with_suggestion("Check that the embedding model downloaded successfully")
        .with_suggestion("Verify network connectivity for model download")
        .with_suggestion("Run `reindex` to regenerate embeddings")
    }

    /// Error when search fails
    pub fn search_failed(cause: &str) -> Self {
        Self::new(ErrorCode::InternalError, "Search operation failed.")
            .with_cause(cause)
            .with_suggestion("Try a simpler search pattern")
            .with_suggestion("Run `reindex` if the index may be corrupted")
    }

    /// Error when indexing fails
    pub fn indexing_failed(cause: &str) -> Self {
        Self::new(ErrorCode::InternalError, "Indexing operation failed.")
            .with_cause(cause)
            .with_suggestion("Check disk space availability")
            .with_suggestion("Verify write permissions to the data directory")
    }

    /// Error when stats retrieval fails
    pub fn stats_failed(cause: &str) -> Self {
        Self::new(
            ErrorCode::InternalError,
            "Failed to retrieve index statistics.",
        )
        .with_cause(cause)
        .with_suggestion("Run `reindex` to rebuild the index")
    }

    /// Error for invalid search pattern
    pub fn invalid_pattern(pattern: &str, cause: &str) -> Self {
        Self::new(
            ErrorCode::SearchInvalidPattern,
            format!("Invalid search pattern: {}", pattern),
        )
        .with_cause(cause)
        .with_suggestion("Check regex syntax if using regular expressions")
        .with_suggestion("Escape special characters with backslash")
    }

    /// Error when parse fails
    pub fn parse_failed(file_path: &str, cause: &str) -> Self {
        Self::new(
            ErrorCode::InternalError,
            format!("Failed to parse file: {}", file_path),
        )
        .with_cause(cause)
        .with_suggestion("Check that the file has a supported extension")
        .with_suggestion("Verify the file contains valid syntax")
    }

    /// Error when file has no symbols
    pub fn no_symbols_in_file(file_path: &str) -> Self {
        Self::new(
            ErrorCode::SymbolNotFound,
            format!("No symbols found in '{}'.", file_path),
        )
        .with_suggestion("Check that the file type is supported for symbol extraction")
        .with_suggestion("Verify the file contains function/class definitions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actionable_error_formatting() {
        let err = ActionableError::index_empty();
        let msg = err.to_user_message();

        assert!(msg.contains("index is empty"));
        assert!(msg.contains("reindex"));
        assert!(msg.contains("Suggested actions"));
    }

    #[test]
    fn test_error_with_cause() {
        let err = ActionableError::search_failed("connection reset");

        assert!(err.cause.is_some());
        assert_eq!(err.cause.as_deref(), Some("connection reset"));
    }

    #[test]
    fn test_custom_suggestions() {
        let err = ActionableError::new(ErrorCode::Unknown, "Custom error")
            .with_suggestion("Try this")
            .with_suggestion("Or try that");

        assert_eq!(err.suggestions.len(), 2);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(ErrorCode::IndexEmpty as i32, 100);
        assert_eq!(ErrorCode::SemanticDisabled as i32, 500);
    }

    #[test]
    fn test_display_impl() {
        let err = ActionableError::no_search_results("foo");
        let display = format!("{}", err);

        assert!(display.contains("No results found for 'foo'"));
        assert!(display.contains("1."));
        assert!(display.contains("2."));
    }

    #[test]
    fn test_with_suggestions_replaces() {
        let err = ActionableError::new(ErrorCode::Unknown, "Test")
            .with_suggestion("First")
            .with_suggestions(vec!["Replaced".to_string()]);

        assert_eq!(err.suggestions.len(), 1);
        assert_eq!(err.suggestions[0], "Replaced");
    }

    #[test]
    fn test_all_prebuilt_errors() {
        // Verify all pre-built errors have at least one suggestion
        let errors = vec![
            ActionableError::index_empty(),
            ActionableError::no_search_results("query"),
            ActionableError::symbol_not_found("name"),
            ActionableError::file_not_found("/path"),
            ActionableError::file_unreadable("/path", "permission denied"),
            ActionableError::semantic_disabled(),
            ActionableError::semantic_model_not_loaded(),
            ActionableError::search_failed("error"),
            ActionableError::indexing_failed("error"),
            ActionableError::stats_failed("error"),
            ActionableError::invalid_pattern("*[", "invalid regex"),
            ActionableError::parse_failed("/file.rs", "syntax error"),
            ActionableError::no_symbols_in_file("/file.txt"),
        ];

        for err in errors {
            assert!(
                !err.suggestions.is_empty(),
                "Error {:?} should have suggestions",
                err.code
            );
        }
    }
}
