//! Error types for CogMCP

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main error type for CogMCP operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Indexing error: {0}")]
    Index(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Git error: {0}")]
    Git(String),

    #[error("File system error: {0}")]
    FileSystem(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("{0}")]
    Other(String),
}

/// Result type alias for CogMCP operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error codes for categorizing actionable errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum ErrorCode {
    /// Missing required parameter
    MissingParameter = 1001,
    /// Invalid parameter value
    InvalidParameter = 1002,
    /// Configuration error
    ConfigInvalid = 2001,
    /// Index is empty
    IndexEmpty = 3001,
    /// Index operation failed
    IndexFailed = 3002,
    /// Search returned no results
    NoSearchResults = 4001,
    /// Search operation failed
    SearchFailed = 4002,
    /// Semantic search not available
    SemanticUnavailable = 4003,
    /// File not found
    FileNotFound = 5001,
    /// File read failed
    FileReadFailed = 5002,
    /// Parse failed
    ParseFailed = 5003,
    /// Stats retrieval failed
    StatsFailed = 6001,
    /// Unknown tool
    UnknownTool = 7001,
    /// Generic internal error
    Internal = 9999,
}

impl ErrorCode {
    /// Get the numeric code value
    pub fn code(&self) -> i32 {
        *self as i32
    }
}

/// An actionable error with suggestions for resolution
///
/// This error type is designed to provide helpful, actionable feedback
/// to users (typically AI assistants) when tool operations fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableError {
    /// Error code for programmatic handling
    pub code: ErrorCode,
    /// Human-readable error message
    pub message: String,
    /// Actionable suggestions for resolving the error
    pub suggestions: Vec<String>,
    /// Optional underlying cause
    pub cause: Option<String>,
}

impl ActionableError {
    /// Create a new actionable error
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            suggestions: Vec::new(),
            cause: None,
        }
    }

    /// Add a suggestion for resolving the error
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add an underlying cause
    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }

    /// Format the error as a user-friendly message
    pub fn to_user_message(&self) -> String {
        let mut output = format!("❌ Error: {}\n", self.message);

        if !self.suggestions.is_empty() {
            output.push_str("\n💡 Suggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("   {}. {}\n", i + 1, suggestion));
            }
        }

        if let Some(ref cause) = self.cause {
            output.push_str(&format!("\n🔍 Cause: {}\n", cause));
        }

        output
    }

    // Factory methods for common error scenarios

    /// Missing required parameter
    pub fn missing_parameter(param_name: &str) -> Self {
        Self::new(
            ErrorCode::MissingParameter,
            format!("Missing required parameter: '{}'", param_name),
        )
        .with_suggestion(format!("Provide a value for the '{}' parameter", param_name))
    }

    /// Invalid parameter value
    pub fn invalid_parameter(param_name: &str, reason: &str) -> Self {
        Self::new(
            ErrorCode::InvalidParameter,
            format!("Invalid value for parameter '{}': {}", param_name, reason),
        )
    }

    /// Index is empty
    pub fn index_empty() -> Self {
        Self::new(
            ErrorCode::IndexEmpty,
            "The codebase index is empty",
        )
        .with_suggestion("Run the 'reindex' tool to index your codebase")
        .with_suggestion("Verify that the root directory contains source files")
    }

    /// No search results
    pub fn no_search_results(query: &str) -> Self {
        Self::new(
            ErrorCode::NoSearchResults,
            format!("No results found for: '{}'", query),
        )
        .with_suggestion("Try a broader search term")
        .with_suggestion("Check if the index contains the files you're searching")
        .with_suggestion("Use 'index_status' to verify the index state")
    }

    /// Search operation failed
    pub fn search_failed(reason: &str) -> Self {
        Self::new(
            ErrorCode::SearchFailed,
            "Search operation failed",
        )
        .with_cause(reason.to_string())
        .with_suggestion("Try again with a simpler query")
        .with_suggestion("Check 'index_status' to verify index health")
    }

    /// Semantic search not available
    pub fn semantic_unavailable() -> Self {
        Self::new(
            ErrorCode::SemanticUnavailable,
            "Semantic search is not available",
        )
        .with_suggestion("Enable embeddings in configuration")
        .with_suggestion("Use 'context_search' with mode='keyword' instead")
    }

    /// File not found
    pub fn file_not_found(path: &str) -> Self {
        Self::new(
            ErrorCode::FileNotFound,
            format!("File not found: '{}'", path),
        )
        .with_suggestion("Verify the file path is correct")
        .with_suggestion("Use a path relative to the repository root")
    }

    /// File read failed
    pub fn file_read_failed(path: &str, reason: &str) -> Self {
        Self::new(
            ErrorCode::FileReadFailed,
            format!("Failed to read file: '{}'", path),
        )
        .with_cause(reason.to_string())
        .with_suggestion("Check file permissions")
        .with_suggestion("Verify the file exists and is readable")
    }

    /// Parse failed
    pub fn parse_failed(reason: &str) -> Self {
        Self::new(
            ErrorCode::ParseFailed,
            "Failed to parse file content",
        )
        .with_cause(reason.to_string())
    }

    /// Stats retrieval failed
    pub fn stats_failed(reason: &str) -> Self {
        Self::new(
            ErrorCode::StatsFailed,
            "Failed to retrieve index statistics",
        )
        .with_cause(reason.to_string())
        .with_suggestion("The index may be corrupted; try running 'reindex' with force=true")
    }

    /// Unknown tool
    pub fn unknown_tool(name: &str) -> Self {
        Self::new(
            ErrorCode::UnknownTool,
            format!("Unknown tool: '{}'", name),
        )
        .with_suggestion("Use 'tools/list' to see available tools")
    }

    /// Symbol not found
    pub fn symbol_not_found(name: &str) -> Self {
        Self::new(
            ErrorCode::NoSearchResults,
            format!("No symbols found matching: '{}'", name),
        )
        .with_suggestion("Try enabling fuzzy matching with fuzzy=true")
        .with_suggestion("Check if the symbol exists in the indexed codebase")
        .with_suggestion("Use 'reindex' to ensure the index is up to date")
    }

    /// Index operation failed
    pub fn index_failed(reason: &str) -> Self {
        Self::new(
            ErrorCode::IndexFailed,
            "Indexing operation failed",
        )
        .with_cause(reason.to_string())
        .with_suggestion("Check disk space and file permissions")
        .with_suggestion("Review the error cause for more details")
    }
}

impl std::fmt::Display for ActionableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ActionableError {}

impl From<Error> for ActionableError {
    fn from(err: Error) -> Self {
        match &err {
            Error::Config(msg) => ActionableError::new(ErrorCode::ConfigInvalid, msg.clone()),
            Error::Storage(msg) => ActionableError::new(ErrorCode::Internal, msg.clone())
                .with_cause("Storage error"),
            Error::Index(msg) => ActionableError::index_failed(msg),
            Error::Search(msg) => ActionableError::search_failed(msg),
            Error::NotFound(msg) => ActionableError::new(ErrorCode::FileNotFound, msg.clone()),
            Error::InvalidArgument(msg) => {
                ActionableError::new(ErrorCode::InvalidParameter, msg.clone())
            }
            _ => ActionableError::new(ErrorCode::Internal, err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_values() {
        assert_eq!(ErrorCode::MissingParameter.code(), 1001);
        assert_eq!(ErrorCode::IndexEmpty.code(), 3001);
        assert_eq!(ErrorCode::Internal.code(), 9999);
    }

    #[test]
    fn test_actionable_error_creation() {
        let err = ActionableError::new(ErrorCode::MissingParameter, "Missing pattern")
            .with_suggestion("Provide a pattern")
            .with_cause("Pattern is required");

        assert_eq!(err.code, ErrorCode::MissingParameter);
        assert_eq!(err.message, "Missing pattern");
        assert_eq!(err.suggestions.len(), 1);
        assert!(err.cause.is_some());
    }

    #[test]
    fn test_actionable_error_user_message() {
        let err = ActionableError::missing_parameter("pattern");
        let msg = err.to_user_message();

        assert!(msg.contains("Error:"));
        assert!(msg.contains("pattern"));
        assert!(msg.contains("Suggestions:"));
    }

    #[test]
    fn test_factory_methods() {
        let err = ActionableError::index_empty();
        assert_eq!(err.code, ErrorCode::IndexEmpty);
        assert!(!err.suggestions.is_empty());

        let err = ActionableError::no_search_results("test");
        assert_eq!(err.code, ErrorCode::NoSearchResults);
        assert!(err.message.contains("test"));

        let err = ActionableError::unknown_tool("badtool");
        assert_eq!(err.code, ErrorCode::UnknownTool);
        assert!(err.message.contains("badtool"));
    }

    #[test]
    fn test_from_error() {
        let err = Error::Search("search failed".to_string());
        let actionable: ActionableError = err.into();
        assert_eq!(actionable.code, ErrorCode::SearchFailed);
    }

    #[test]
    fn test_serialization() {
        let err = ActionableError::missing_parameter("query");
        let json = serde_json::to_string(&err).unwrap();
        let parsed: ActionableError = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.code, err.code);
        assert_eq!(parsed.message, err.message);
        assert_eq!(parsed.suggestions, err.suggestions);
    }
}
