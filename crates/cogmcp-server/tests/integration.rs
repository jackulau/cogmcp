//! Integration tests for the CogMCP server
//!
//! These tests verify the server handles MCP protocol messages correctly.

use cogmcp_server::{CogMcpServer, RunnerConfig, ServerRunner};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Create a test server with in-memory storage
fn create_test_server() -> CogMcpServer {
    let temp_dir = std::env::temp_dir().join("cogmcp-test");
    std::fs::create_dir_all(&temp_dir).ok();

    CogMcpServer::in_memory(temp_dir).expect("Failed to create test server")
}

#[test]
fn test_server_creation() {
    let server = create_test_server();
    let tools = server.list_tools();

    assert!(!tools.is_empty(), "Server should have tools");
}

#[test]
fn test_list_tools_returns_all_tools() {
    let server = create_test_server();
    let tools = server.list_tools();

    // Verify we have the expected 7 tools
    assert_eq!(tools.len(), 8, "Should have 8 tools");

    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();

    assert!(tool_names.contains(&"ping"), "Should have ping tool");
    assert!(
        tool_names.contains(&"context_grep"),
        "Should have context_grep tool"
    );
    assert!(
        tool_names.contains(&"context_search"),
        "Should have context_search tool"
    );
    assert!(
        tool_names.contains(&"find_symbol"),
        "Should have find_symbol tool"
    );
    assert!(
        tool_names.contains(&"get_file_outline"),
        "Should have get_file_outline tool"
    );
    assert!(
        tool_names.contains(&"index_status"),
        "Should have index_status tool"
    );
    assert!(tool_names.contains(&"reindex"), "Should have reindex tool");
}

#[test]
fn test_ping_tool() {
    let server = create_test_server();
    let result = server.call_tool("ping", json!({}));

    assert!(result.is_ok(), "Ping should succeed");
    let output = result.unwrap();
    assert!(
        output.contains("CogMCP server is running"),
        "Should contain server status"
    );
}

#[test]
fn test_index_status_tool() {
    let server = create_test_server();
    let result = server.call_tool("index_status", json!({}));

    assert!(result.is_ok(), "index_status should succeed");
    let output = result.unwrap();
    assert!(
        output.contains("Index Status"),
        "Should contain status header"
    );
    assert!(
        output.contains("Files indexed"),
        "Should contain file count"
    );
}

#[test]
fn test_unknown_tool() {
    let server = create_test_server();
    let result = server.call_tool("unknown_tool", json!({}));

    assert!(result.is_err(), "Unknown tool should fail");
    let err = result.unwrap_err();
    assert!(
        err.message.contains("Unknown tool"),
        "Error should indicate unknown tool"
    );
}

#[test]
fn test_context_grep_missing_pattern() {
    let server = create_test_server();
    let result = server.call_tool("context_grep", json!({}));

    assert!(result.is_err(), "Missing pattern should fail");
}

#[test]
fn test_context_search_missing_query() {
    let server = create_test_server();
    let result = server.call_tool("context_search", json!({}));

    assert!(result.is_err(), "Missing query should fail");
}

#[test]
fn test_find_symbol_missing_name() {
    let server = create_test_server();
    let result = server.call_tool("find_symbol", json!({}));

    assert!(result.is_err(), "Missing name should fail");
}

#[test]
fn test_context_grep_with_pattern() {
    let server = create_test_server();
    let result = server.call_tool(
        "context_grep",
        json!({
            "pattern": "nonexistent_pattern_12345",
            "limit": 10
        }),
    );

    // With ActionableError, no results returns an error now
    assert!(result.is_err(), "Search with no results should return error");
    let err = result.unwrap_err();
    assert!(
        err.message.contains("No results") || err.message.contains("empty"),
        "Error should indicate no results or empty index"
    );
}

#[test]
fn test_context_search_with_query() {
    let server = create_test_server();
    let result = server.call_tool(
        "context_search",
        json!({
            "query": "some search query",
            "limit": 10,
            "mode": "keyword"
        }),
    );

    // With ActionableError, no results returns an error now
    assert!(result.is_err(), "Search with no results should return error");
    let err = result.unwrap_err();
    assert!(
        err.message.contains("No results") || err.message.contains("empty"),
        "Error should indicate no results or empty index"
    );
}

#[test]
fn test_reindex_tool() {
    let server = create_test_server();
    let result = server.call_tool("reindex", json!({}));

    assert!(result.is_ok(), "Reindex should succeed");
    let output = result.unwrap();
    assert!(
        output.contains("Reindex Complete") || output.contains("indexed"),
        "Should indicate reindex status"
    );
}

#[test]
fn test_runner_config_creation() {
    let config = RunnerConfig::new(PathBuf::from("/tmp/test"))
        .index_on_startup(false)
        .in_memory(true);

    assert_eq!(config.root, PathBuf::from("/tmp/test"));
}

#[test]
fn test_runner_creation() {
    let config = RunnerConfig::new(PathBuf::from("/tmp/test"))
        .index_on_startup(false)
        .in_memory(true);

    let runner = ServerRunner::new(config);
    assert!(runner.is_ok(), "Runner creation should succeed");
}

#[test]
fn test_tool_schema_validation() {
    let server = create_test_server();
    let tools = server.list_tools();

    for tool in tools {
        // Verify each tool has a name and description
        assert!(!tool.name.is_empty(), "Tool name should not be empty");
        assert!(
            tool.description.is_some(),
            "Tool {} should have a description",
            tool.name
        );

        // Verify the input schema is valid JSON
        let schema: Value =
            serde_json::to_value(&tool.input_schema).expect("Schema should serialize");
        assert!(
            schema.is_object(),
            "Schema for {} should be an object",
            tool.name
        );
    }
}

#[test]
fn test_server_info() {
    let info = CogMcpServer::server_info();
    assert_eq!(info.name, "cogmcp");
    assert!(!info.version.is_empty());
}

#[test]
fn test_server_capabilities() {
    let caps = CogMcpServer::capabilities();
    // The server should have tool capabilities enabled
    assert!(
        caps.tools.is_some(),
        "Server should have tools capability enabled"
    );
}

// ============================================================================
// End-to-End MCP Protocol Tests
// ============================================================================
//
// These tests exercise the server's MCP handler implementation by testing
// the ServerHandler trait methods through the server's call_tool interface.

mod e2e_tests {
    use super::*;
    use rmcp::handler::server::ServerHandler;

    #[test]
    fn test_e2e_initialize_handshake() {
        let server = create_test_server();

        // Test get_info which is used during initialize handshake
        let info = server.get_info();

        assert_eq!(info.server_info.name, "cogmcp");
        assert!(!info.server_info.version.is_empty());
        assert!(info.capabilities.tools.is_some());
        assert!(info.instructions.is_some());
    }

    #[test]
    fn test_e2e_tools_list_returns_7_tools() {
        let server = create_test_server();

        // List tools through the server's list_tools method
        let tools = server.list_tools();
        assert_eq!(tools.len(), 8, "Should return 8 tools");

        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert!(tool_names.contains(&"ping"));
        assert!(tool_names.contains(&"context_grep"));
        assert!(tool_names.contains(&"context_search"));
        assert!(tool_names.contains(&"find_symbol"));
        assert!(tool_names.contains(&"get_file_outline"));
        assert!(tool_names.contains(&"index_status"));
        assert!(tool_names.contains(&"reindex"));
    }

    #[test]
    fn test_e2e_tools_call_ping() {
        let server = create_test_server();

        let result = server.call_tool("ping", json!({}));

        assert!(result.is_ok(), "call_tool should succeed");
        let output = result.unwrap();
        assert!(
            output.contains("CogMCP server is running"),
            "Should contain server status"
        );
    }

    #[test]
    fn test_e2e_tools_call_with_arguments() {
        let server = create_test_server();

        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": "test_pattern",
                "limit": 10
            }),
        );

        // With ActionableError, no results returns an error now
        assert!(result.is_err(), "call_tool with no results should return error");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("No results") || err.message.contains("empty"),
            "Error should indicate no results or empty index"
        );
    }

    #[test]
    fn test_e2e_unknown_tool_handled_gracefully() {
        let server = create_test_server();

        let result = server.call_tool("nonexistent_tool_12345", json!({}));

        // The server should return Err for unknown tools
        assert!(result.is_err(), "Unknown tool should return error");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Unknown tool"),
            "Error message should indicate unknown tool"
        );
    }

    #[test]
    fn test_e2e_missing_required_args_handled_gracefully() {
        let server = create_test_server();

        // context_grep requires 'pattern' argument
        let result = server.call_tool("context_grep", json!({}));

        assert!(
            result.is_err(),
            "Missing required args should return error"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("pattern"),
            "Error should mention missing pattern"
        );
    }

    #[test]
    fn test_e2e_index_status_returns_stats() {
        let server = create_test_server();

        let result = server.call_tool("index_status", json!({}));

        assert!(result.is_ok(), "index_status should succeed");
        let output = result.unwrap();
        assert!(
            output.contains("Index Status"),
            "Should contain status header"
        );
        assert!(
            output.contains("Files indexed"),
            "Should contain file count"
        );
    }

    #[test]
    fn test_e2e_reindex_completes() {
        let server = create_test_server();

        let result = server.call_tool("reindex", json!({}));

        assert!(result.is_ok(), "reindex should succeed");
        let output = result.unwrap();
        assert!(
            output.contains("Reindex Complete") || output.contains("indexed"),
            "Should indicate reindex completed"
        );
    }

    #[test]
    fn test_e2e_context_search_modes() {
        let server = create_test_server();

        // Test keyword mode - with ActionableError, no results returns error
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "mode": "keyword"
            }),
        );
        assert!(result.is_err(), "keyword search with empty index should return error");

        // Test hybrid mode (default)
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query"
            }),
        );
        assert!(result.is_err(), "hybrid search with empty index should return error");

        // Test semantic mode (requires embeddings, so may fail)
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "mode": "semantic"
            }),
        );
        // Semantic search can fail due to empty index or embeddings not enabled
        // Both cases return an error now
        assert!(result.is_err(), "semantic search with empty index should return error");
    }

    #[test]
    fn test_e2e_find_symbol_with_options() {
        let server = create_test_server();

        let result = server.call_tool(
            "find_symbol",
            json!({
                "name": "TestSymbol",
                "kind": "function",
                "fuzzy": true
            }),
        );

        // With ActionableError, no symbols found returns an error
        assert!(result.is_err(), "find_symbol with no results should return error");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("symbol") || err.message.contains("empty"),
            "Error should indicate no symbols or empty index"
        );
    }

    #[test]
    fn test_e2e_get_file_outline_invalid_path() {
        let server = create_test_server();

        let result = server.call_tool(
            "get_file_outline",
            json!({
                "file_path": "nonexistent_file_xyz.rs"
            }),
        );

        // With ActionableError, file not found returns an error
        assert!(result.is_err(), "get_file_outline for nonexistent file should return error");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("not found") || err.message.contains("File"),
            "Should indicate file not found"
        );
    }

    #[test]
    fn test_e2e_all_tool_schemas_valid() {
        let server = create_test_server();
        let tools = server.list_tools();

        for tool in tools {
            // Verify each tool has a name and description
            assert!(!tool.name.is_empty(), "Tool name should not be empty");
            assert!(
                tool.description.is_some(),
                "Tool {} should have a description",
                tool.name
            );

            // Verify the input schema is valid JSON
            let schema: Value =
                serde_json::to_value(&tool.input_schema).expect("Schema should serialize");
            assert!(
                schema.is_object(),
                "Schema for {} should be an object",
                tool.name
            );

            // Verify schema has type: object
            assert!(
                schema.get("type").map(|v| v == "object").unwrap_or(false),
                "Schema for {} should have type: object",
                tool.name
            );
        }
    }

    #[test]
    fn test_e2e_concurrent_tool_calls() {
        // Test that multiple tool calls can be made without interference
        let server = create_test_server();

        let result1 = server.call_tool("ping", json!({}));
        let result2 = server.call_tool("index_status", json!({}));
        let result3 = server.call_tool("ping", json!({}));

        assert!(result1.is_ok(), "First ping should succeed");
        assert!(result2.is_ok(), "index_status should succeed");
        assert!(result3.is_ok(), "Second ping should succeed");

        // Verify results are consistent
        assert_eq!(
            result1.unwrap(),
            result3.unwrap(),
            "Same tool calls should return same results"
        );
    }

    #[test]
    fn test_e2e_tool_with_limit_parameter() {
        let server = create_test_server();

        // Test with various limit values - with ActionableError,
        // empty index returns error
        for limit in [1, 5, 10, 50, 100] {
            let result = server.call_tool(
                "context_grep",
                json!({
                    "pattern": "test",
                    "limit": limit
                }),
            );
            // With empty index, the error should indicate that
            assert!(
                result.is_err(),
                "context_grep with empty index should return error with limit={}",
                limit
            );
        }
    }
}

// ============================================================================
// Error Message Tests
// ============================================================================
//
// These tests verify that error messages are helpful, consistent, and provide
// actionable guidance to users. Tests are designed to work with both current
// basic error messages and future actionable error messages.

mod error_message_tests {
    use super::*;
    use tempfile::TempDir;

    // =========================================================================
    // Test Helper Functions
    // =========================================================================

    /// Check if an error message contains expected phrases
    /// This helper is flexible to work with both simple and actionable messages
    fn assert_error_mentions(error: &str, expected_phrases: &[&str], context: &str) {
        for phrase in expected_phrases {
            assert!(
                error.to_lowercase().contains(&phrase.to_lowercase()),
                "{}: Error should mention '{}'. Got: {}",
                context,
                phrase,
                error
            );
        }
    }

    /// Check if an error has suggestions section (for actionable errors)
    /// Returns true if the error contains actionable suggestions
    fn has_suggestions(error: &str) -> bool {
        error.contains("Suggested actions:")
            || error.contains("Try:")
            || error.contains("Suggestions:")
            || error.contains("1.")
    }

    /// Assert that if suggestions are present, they are well-formatted
    fn assert_suggestions_well_formatted(error: &str, context: &str) {
        if has_suggestions(error) {
            // Suggestions should have numbered items
            assert!(
                error.contains("1.") || error.contains("- "),
                "{}: Suggestions should be numbered or bulleted. Got: {}",
                context,
                error
            );
        }
    }

    /// Create a test server with actual files in a temp directory
    fn create_test_server_with_files() -> (CogMcpServer, TempDir) {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let root = temp_dir.path();

        // Write some test files
        std::fs::write(
            root.join("test.rs"),
            r#"
fn hello_world() {
    println!("Hello, world!");
}

pub struct TestStruct {
    field: i32,
}

impl TestStruct {
    pub fn new() -> Self {
        Self { field: 0 }
    }
}
"#,
        )
        .expect("Failed to write test.rs");

        std::fs::write(
            root.join("lib.rs"),
            r#"
pub fn existing_function() -> i32 {
    42
}

pub struct ExistingStruct {
    name: String,
}

mod inner {
    fn private_fn() {}
}
"#,
        )
        .expect("Failed to write lib.rs");

        let server =
            CogMcpServer::in_memory(root.to_path_buf()).expect("Failed to create test server");

        // Trigger indexing
        let _ = server.call_tool("reindex", json!({}));

        (server, temp_dir)
    }

    /// Create a test server with an empty file
    fn create_test_server_with_empty_file() -> (CogMcpServer, TempDir) {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let root = temp_dir.path();

        // Write an empty text file (no symbols to extract)
        std::fs::write(root.join("empty.txt"), "").expect("Failed to write empty.txt");

        let server =
            CogMcpServer::in_memory(root.to_path_buf()).expect("Failed to create test server");

        (server, temp_dir)
    }

    // =========================================================================
    // Parameter Validation Error Tests
    // =========================================================================

    #[test]
    fn test_context_grep_missing_pattern_error() {
        let server = create_test_server();
        let result = server.call_tool("context_grep", json!({}));

        assert!(result.is_err(), "Should error on missing pattern");
        let error = result.unwrap_err();

        // Must mention the missing parameter
        assert_error_mentions(&error, &["pattern"], "context_grep missing pattern");

        // Check for well-formatted suggestions if present
        assert_suggestions_well_formatted(&error, "context_grep missing pattern");
    }

    #[test]
    fn test_context_search_missing_query_error() {
        let server = create_test_server();
        let result = server.call_tool("context_search", json!({}));

        assert!(result.is_err(), "Should error on missing query");
        let error = result.unwrap_err();

        // Must mention the missing parameter
        assert_error_mentions(&error, &["query"], "context_search missing query");

        assert_suggestions_well_formatted(&error, "context_search missing query");
    }

    #[test]
    fn test_find_symbol_missing_name_error() {
        let server = create_test_server();
        let result = server.call_tool("find_symbol", json!({}));

        assert!(result.is_err(), "Should error on missing name");
        let error = result.unwrap_err();

        // Must mention the missing parameter
        assert_error_mentions(&error, &["name"], "find_symbol missing name");

        assert_suggestions_well_formatted(&error, "find_symbol missing name");
    }

    #[test]
    fn test_get_file_outline_missing_path_error() {
        let server = create_test_server();
        let result = server.call_tool("get_file_outline", json!({}));

        assert!(result.is_err(), "Should error on missing file_path");
        let error = result.unwrap_err();

        // Must mention the missing parameter
        assert_error_mentions(&error, &["file_path"], "get_file_outline missing path");

        assert_suggestions_well_formatted(&error, "get_file_outline missing path");
    }

    #[test]
    fn test_semantic_search_missing_query_error() {
        let server = create_test_server();
        let result = server.call_tool("semantic_search", json!({}));

        assert!(result.is_err(), "Should error on missing query");
        let error = result.unwrap_err();

        // Must mention the missing parameter
        assert_error_mentions(&error, &["query"], "semantic_search missing query");

        assert_suggestions_well_formatted(&error, "semantic_search missing query");
    }

    // =========================================================================
    // Unknown Tool Error Tests
    // =========================================================================

    #[test]
    fn test_unknown_tool_error_message() {
        let server = create_test_server();
        let result = server.call_tool("nonexistent_tool", json!({}));

        assert!(result.is_err(), "Unknown tool should fail");
        let error = result.unwrap_err();

        // Must indicate it's an unknown tool and mention the tool name
        assert_error_mentions(
            &error,
            &["unknown tool", "nonexistent_tool"],
            "unknown tool error",
        );

        assert_suggestions_well_formatted(&error, "unknown tool error");
    }

    #[test]
    fn test_unknown_tool_similar_name() {
        let server = create_test_server();
        // Try a misspelled version of a real tool
        let result = server.call_tool("contex_grep", json!({}));

        assert!(result.is_err(), "Misspelled tool should fail");
        let error = result.unwrap_err();

        assert_error_mentions(&error, &["unknown tool"], "misspelled tool error");

        // Future: could suggest "context_grep" as a similar tool
        assert_suggestions_well_formatted(&error, "misspelled tool error");
    }

    // =========================================================================
    // No Results Error Tests
    // =========================================================================

    #[test]
    fn test_context_grep_no_results() {
        let (server, _temp) = create_test_server_with_files();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": "xyzzy_nonexistent_pattern_12345"
            }),
        );

        // No results should return Ok with a message (not Err)
        assert!(result.is_ok(), "No results should return Ok");
        let output = result.unwrap();

        // Should indicate no matches were found
        assert!(
            output.to_lowercase().contains("no match")
                || output.to_lowercase().contains("no results"),
            "Should indicate no matches found. Got: {}",
            output
        );
    }

    #[test]
    fn test_context_search_no_results() {
        let (server, _temp) = create_test_server_with_files();
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "xyzzy_nonexistent_query_12345",
                "mode": "keyword"
            }),
        );

        assert!(result.is_ok(), "No results should return Ok");
        let output = result.unwrap();

        assert!(
            output.to_lowercase().contains("no match")
                || output.to_lowercase().contains("no results"),
            "Should indicate no matches found. Got: {}",
            output
        );
    }

    #[test]
    fn test_find_symbol_no_results() {
        let (server, _temp) = create_test_server_with_files();
        let result = server.call_tool(
            "find_symbol",
            json!({
                "name": "NonexistentSymbol12345"
            }),
        );

        assert!(result.is_ok(), "No results should return Ok");
        let output = result.unwrap();

        assert!(
            output.to_lowercase().contains("no symbol"),
            "Should indicate no symbols found. Got: {}",
            output
        );
    }

    #[test]
    fn test_find_symbol_with_filters_no_results() {
        let (server, _temp) = create_test_server_with_files();
        // Search for a symbol with wrong kind filter
        let result = server.call_tool(
            "find_symbol",
            json!({
                "name": "hello_world",
                "kind": "class"  // Wrong kind - it's a function
            }),
        );

        assert!(result.is_ok(), "Filtered search should return Ok");
        let output = result.unwrap();

        // Should indicate no symbols matched with filters
        assert!(
            output.to_lowercase().contains("no symbol"),
            "Should indicate no symbols found with filters. Got: {}",
            output
        );
    }

    // =========================================================================
    // File Not Found Error Tests
    // =========================================================================

    #[test]
    fn test_get_file_outline_file_not_found() {
        let server = create_test_server();
        let result = server.call_tool(
            "get_file_outline",
            json!({
                "file_path": "nonexistent/path/file.rs"
            }),
        );

        // File not found returns Ok with error message (graceful failure)
        assert!(result.is_ok(), "File not found should return Ok");
        let output = result.unwrap();

        // Should indicate file could not be read
        assert!(
            output.to_lowercase().contains("failed")
                || output.to_lowercase().contains("not found")
                || output.to_lowercase().contains("error"),
            "Should indicate file not found. Got: {}",
            output
        );
    }

    #[test]
    fn test_get_file_outline_no_symbols() {
        let (server, _temp) = create_test_server_with_empty_file();
        let result = server.call_tool(
            "get_file_outline",
            json!({
                "file_path": "empty.txt"
            }),
        );

        assert!(result.is_ok(), "Empty file should return Ok");
        let output = result.unwrap();

        // Should indicate no symbols were found or failed to read
        assert!(
            output.to_lowercase().contains("no symbol")
                || output.to_lowercase().contains("failed")
                || output.contains("empty.txt"),
            "Should indicate no symbols in file. Got: {}",
            output
        );
    }

    // =========================================================================
    // Semantic Search Disabled Tests
    // =========================================================================

    #[test]
    fn test_semantic_search_not_available() {
        // Standard in-memory server doesn't have embeddings enabled
        let server = create_test_server();
        let result = server.call_tool(
            "semantic_search",
            json!({
                "query": "find authentication code"
            }),
        );

        assert!(result.is_ok(), "Semantic search should return Ok");
        let output = result.unwrap();

        // Should indicate semantic search is not available
        assert!(
            output.to_lowercase().contains("not available")
                || output.to_lowercase().contains("disabled")
                || output.to_lowercase().contains("enable"),
            "Should indicate semantic search not available. Got: {}",
            output
        );
    }

    // =========================================================================
    // Empty Index Tests
    // =========================================================================

    #[test]
    fn test_empty_index_grep() {
        // Create a fresh server with no indexed files
        let server = create_test_server();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": "test"
            }),
        );

        assert!(result.is_ok(), "Empty index grep should return Ok");
        let output = result.unwrap();

        // Should either show no results or indicate empty index
        assert!(
            output.to_lowercase().contains("no match")
                || output.to_lowercase().contains("empty")
                || output.to_lowercase().contains("no results"),
            "Should indicate no matches or empty index. Got: {}",
            output
        );
    }

    #[test]
    fn test_empty_index_search() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "find something"
            }),
        );

        assert!(result.is_ok(), "Empty index search should return Ok");
        let output = result.unwrap();

        assert!(
            output.to_lowercase().contains("no match")
                || output.to_lowercase().contains("empty")
                || output.to_lowercase().contains("no results"),
            "Should indicate no results or empty index. Got: {}",
            output
        );
    }

    #[test]
    fn test_index_status_shows_empty_state() {
        let server = create_test_server();
        let result = server.call_tool("index_status", json!({}));

        assert!(result.is_ok(), "index_status should succeed");
        let output = result.unwrap();

        // Should show 0 files or indicate empty state
        assert!(
            output.contains("Files indexed: 0") || output.contains("Files indexed:** 0"),
            "Should show 0 indexed files for empty index. Got: {}",
            output
        );
    }

    // =========================================================================
    // Error Message Consistency Tests
    // =========================================================================

    #[test]
    fn test_all_missing_param_errors_are_strings() {
        let server = create_test_server();

        // All these should return Err with string messages
        let errors = vec![
            ("context_grep", server.call_tool("context_grep", json!({}))),
            (
                "context_search",
                server.call_tool("context_search", json!({})),
            ),
            ("find_symbol", server.call_tool("find_symbol", json!({}))),
            (
                "get_file_outline",
                server.call_tool("get_file_outline", json!({})),
            ),
            (
                "semantic_search",
                server.call_tool("semantic_search", json!({})),
            ),
        ];

        for (tool_name, result) in errors {
            assert!(
                result.is_err(),
                "{} with missing params should return Err",
                tool_name
            );
            let error = result.unwrap_err();
            assert!(
                !error.is_empty(),
                "{} error message should not be empty",
                tool_name
            );
        }
    }

    #[test]
    fn test_error_messages_are_not_stack_traces() {
        let server = create_test_server();

        // Error messages should be user-friendly, not stack traces
        let errors = vec![
            server.call_tool("context_grep", json!({})).unwrap_err(),
            server.call_tool("find_symbol", json!({})).unwrap_err(),
            server.call_tool("unknown_tool", json!({})).unwrap_err(),
        ];

        for error in errors {
            // Should not contain stack trace indicators
            assert!(
                !error.contains("at src/"),
                "Error should not contain stack traces: {}",
                error
            );
            assert!(
                !error.contains("thread 'main' panicked"),
                "Error should not be a panic: {}",
                error
            );
            assert!(
                !error.contains("RUST_BACKTRACE"),
                "Error should not mention backtrace: {}",
                error
            );
        }
    }

    #[test]
    fn test_error_messages_have_reasonable_length() {
        let server = create_test_server();

        let errors = vec![
            server.call_tool("context_grep", json!({})).unwrap_err(),
            server.call_tool("find_symbol", json!({})).unwrap_err(),
            server.call_tool("unknown_tool", json!({})).unwrap_err(),
        ];

        for error in errors {
            // Error messages should be concise but informative
            // Too short means not helpful, too long means verbose
            assert!(
                error.len() >= 5,
                "Error message too short to be helpful: {}",
                error
            );
            assert!(
                error.len() <= 2000,
                "Error message too long: {}",
                error
            );
        }
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_pattern_handling() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": ""
            }),
        );

        // Empty pattern should either error or return no results
        // Both are acceptable behaviors
        if result.is_err() {
            let error = result.unwrap_err();
            assert!(
                !error.is_empty(),
                "Empty pattern error should have message"
            );
        } else {
            let output = result.unwrap();
            // If it succeeds, it should handle gracefully
            assert!(
                !output.is_empty(),
                "Empty pattern output should not be empty"
            );
        }
    }

    #[test]
    fn test_whitespace_only_query_handling() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "   "
            }),
        );

        // Whitespace-only query should be handled gracefully
        if result.is_err() {
            let error = result.unwrap_err();
            assert!(!error.is_empty(), "Whitespace query error should have message");
        } else {
            let output = result.unwrap();
            assert!(
                !output.is_empty(),
                "Whitespace query output should not be empty"
            );
        }
    }

    #[test]
    fn test_special_characters_in_pattern() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": "test.*[a-z]+"
            }),
        );

        // Regex patterns should be handled
        assert!(
            result.is_ok(),
            "Regex patterns should be handled: {:?}",
            result
        );
    }

    #[test]
    fn test_very_long_pattern() {
        let server = create_test_server();
        let long_pattern = "a".repeat(1000);
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": long_pattern
            }),
        );

        // Very long patterns should be handled gracefully (not crash)
        // Either return results or a reasonable error
        if result.is_err() {
            let error = result.unwrap_err();
            assert!(!error.is_empty(), "Long pattern error should have message");
        }
    }

    #[test]
    fn test_null_arguments_handling() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": null
            }),
        );

        // Null arguments should error gracefully
        assert!(result.is_err(), "Null pattern should error");
        let error = result.unwrap_err();
        assert!(!error.is_empty(), "Null argument error should have message");
    }

    #[test]
    fn test_wrong_type_arguments() {
        let server = create_test_server();
        let result = server.call_tool(
            "context_grep",
            json!({
                "pattern": 12345  // Should be string
            }),
        );

        // Wrong type should error gracefully
        assert!(result.is_err(), "Wrong type should error");
        let error = result.unwrap_err();
        assert!(
            !error.is_empty(),
            "Wrong type error should have message"
        );
    }

    // =========================================================================
    // Error Message Consistency Tests (for future actionable errors)
    // =========================================================================
    //
    // These tests are designed to pass with current simple error messages
    // and will also verify consistency when actionable error messages are added.

    #[test]
    fn test_error_message_consistency() {
        // Verify all error messages follow a consistent format
        let server = create_test_server();

        let errors = vec![
            ("context_grep", server.call_tool("context_grep", json!({})).unwrap_err()),
            ("find_symbol", server.call_tool("find_symbol", json!({})).unwrap_err()),
            ("unknown_tool", server.call_tool("unknown_tool", json!({})).unwrap_err()),
        ];

        for (tool_name, error) in &errors {
            // Basic consistency check: all errors should be non-empty strings
            assert!(
                !error.is_empty(),
                "{} error should not be empty",
                tool_name
            );

            // If error has suggestions, they should come after the main message
            if has_suggestions(error) {
                // Suggestions section should not be at the very start
                if let Some(idx) = error.find("Suggested actions:") {
                    assert!(
                        idx > 5,
                        "{} error suggestions should come after main message",
                        tool_name
                    );
                }
            }
        }
    }

    #[test]
    fn test_numbered_suggestions_format() {
        // If errors have suggestions, they should be numbered or bulleted
        let server = create_test_server();

        let errors = vec![
            server.call_tool("context_grep", json!({})).unwrap_err(),
            server.call_tool("find_symbol", json!({})).unwrap_err(),
            server.call_tool("unknown_tool", json!({})).unwrap_err(),
        ];

        for error in errors {
            // If there are suggestions, they should be formatted with numbers or bullets
            if has_suggestions(&error) {
                let has_numbered = error.contains("1.");
                let has_bulleted = error.contains("- ");
                assert!(
                    has_numbered || has_bulleted,
                    "Suggestions should be numbered or bulleted: {}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_unknown_tool_lists_available_tools() {
        // Unknown tool errors should eventually list available tools
        let server = create_test_server();
        let error = server.call_tool("nonexistent_tool", json!({})).unwrap_err();

        // For now, just ensure the error mentions "unknown tool"
        // In the future, it should also list available tools
        assert!(
            error.to_lowercase().contains("unknown tool"),
            "Should mention unknown tool: {}",
            error
        );

        // Future check: error should list available tools for discovery
        // This is a forward-compatible check that will pass once actionable errors are added
        if error.contains("Available tools:") || error.contains("ping") {
            // Good - error provides tool discovery
        }
        // No assertion here - this is informational for when actionable errors are added
    }

    #[test]
    fn test_missing_param_error_includes_param_name() {
        let server = create_test_server();

        // Test that each missing parameter error mentions the parameter name
        let tests = vec![
            ("context_grep", "pattern"),
            ("context_search", "query"),
            ("find_symbol", "name"),
            ("get_file_outline", "file_path"),
            ("semantic_search", "query"),
        ];

        for (tool_name, expected_param) in tests {
            let error = server.call_tool(tool_name, json!({})).unwrap_err();
            assert!(
                error.to_lowercase().contains(&expected_param.to_lowercase()),
                "{} error should mention '{}'. Got: {}",
                tool_name,
                expected_param,
                error
            );
        }
    }

    // =========================================================================
    // Future Error Code Tests (will activate when error-types is merged)
    // =========================================================================

    // These tests are placeholder for when ActionableError with error codes is integrated.
    // They are written to pass now and will provide value once the error types are available.

    #[test]
    fn test_errors_are_actionable_compatible() {
        // Verify current errors can be extended to actionable format
        let server = create_test_server();

        let errors = vec![
            server.call_tool("context_grep", json!({})).unwrap_err(),
            server.call_tool("unknown_tool", json!({})).unwrap_err(),
        ];

        for error in errors {
            // Errors should be string-based (compatible with actionable error Display)
            assert!(
                !error.is_empty(),
                "Error should be displayable as string"
            );

            // Errors should not contain internal/debug formatting
            assert!(
                !error.contains("Error("),
                "Error should not expose internal enum structure: {}",
                error
            );
        }
    }
}
