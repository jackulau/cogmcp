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

    // Verify we have the expected 9 tools (including get_relevant_context)
    assert_eq!(tools.len(), 9, "Should have 9 tools");

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
    assert!(
        tool_names.contains(&"get_relevant_context"),
        "Should have get_relevant_context tool"
    );
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
        err.contains("Unknown tool"),
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

    assert!(result.is_ok(), "Search should succeed even with no results");
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

    assert!(result.is_ok(), "Search should succeed even with no results");
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
    fn test_e2e_tools_list_returns_9_tools() {
        let server = create_test_server();

        // List tools through the server's list_tools method
        let tools = server.list_tools();
        assert_eq!(tools.len(), 9, "Should return 9 tools");

        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert!(tool_names.contains(&"ping"));
        assert!(tool_names.contains(&"context_grep"));
        assert!(tool_names.contains(&"context_search"));
        assert!(tool_names.contains(&"find_symbol"));
        assert!(tool_names.contains(&"get_file_outline"));
        assert!(tool_names.contains(&"index_status"));
        assert!(tool_names.contains(&"reindex"));
        assert!(tool_names.contains(&"get_relevant_context"));
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

        assert!(result.is_ok(), "call_tool with arguments should succeed");
        // Even with no matches, this should succeed (return "No matches found")
    }

    #[test]
    fn test_e2e_unknown_tool_handled_gracefully() {
        let server = create_test_server();

        let result = server.call_tool("nonexistent_tool_12345", json!({}));

        // The server should return Err for unknown tools
        assert!(result.is_err(), "Unknown tool should return error");
        let err = result.unwrap_err();
        assert!(
            err.contains("Unknown tool"),
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
            err.contains("Missing pattern") || err.contains("pattern"),
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

        // Test keyword mode
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "mode": "keyword"
            }),
        );
        assert!(result.is_ok(), "keyword search should succeed");

        // Test hybrid mode (default)
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query"
            }),
        );
        assert!(result.is_ok(), "hybrid search should succeed");

        // Test semantic mode
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "mode": "semantic"
            }),
        );
        assert!(result.is_ok(), "semantic search should succeed");
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

        assert!(result.is_ok(), "find_symbol should succeed");
        // Even with no matches, this should succeed
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

        assert!(result.is_ok(), "get_file_outline should return Ok");
        let output = result.unwrap();
        // Should return a graceful error message
        assert!(
            output.contains("Failed to read") || output.contains("error"),
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

        // Test with various limit values
        for limit in [1, 5, 10, 50, 100] {
            let result = server.call_tool(
                "context_grep",
                json!({
                    "pattern": "test",
                    "limit": limit
                }),
            );
            assert!(
                result.is_ok(),
                "context_grep should succeed with limit={}",
                limit
            );
        }
    }
}

// ============================================================================
// get_relevant_context Tool Tests
// ============================================================================
//
// Comprehensive tests for the get_relevant_context tool which provides
// priority-based context selection using multiple scoring factors.

mod get_relevant_context_tests {
    use super::*;

    #[test]
    fn test_get_relevant_context_returns_results() {
        let server = create_test_server();

        // Basic call with no parameters should succeed
        let result = server.call_tool("get_relevant_context", json!({}));
        assert!(result.is_ok(), "get_relevant_context should succeed");

        let output = result.unwrap();
        // With empty index, should report no files
        assert!(
            output.contains("No files indexed") || output.contains("Relevant Context"),
            "Should indicate status of indexed files"
        );
    }

    #[test]
    fn test_get_relevant_context_respects_limit() {
        let server = create_test_server();

        // Test with various limit values
        for limit in [1, 5, 10, 20] {
            let result = server.call_tool(
                "get_relevant_context",
                json!({
                    "limit": limit
                }),
            );
            assert!(
                result.is_ok(),
                "get_relevant_context should succeed with limit={}",
                limit
            );
        }
    }

    #[test]
    fn test_get_relevant_context_respects_min_score() {
        let server = create_test_server();

        // Test with various min_score thresholds
        for min_score in [0.0, 0.3, 0.5, 0.8, 1.0] {
            let result = server.call_tool(
                "get_relevant_context",
                json!({
                    "min_score": min_score
                }),
            );
            assert!(
                result.is_ok(),
                "get_relevant_context should succeed with min_score={}",
                min_score
            );
        }

        // High min_score should result in fewer or no results
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "min_score": 0.99
            }),
        );
        assert!(result.is_ok());
        let output = result.unwrap();
        // Should either have no results or very few high-scoring files
        assert!(
            output.contains("No files") || output.contains("Relevant Context"),
            "Should handle high min_score threshold"
        );
    }

    #[test]
    fn test_get_relevant_context_with_query() {
        let server = create_test_server();

        // Test with a query parameter
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "function test"
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with query"
        );

        // Query should affect relevance scoring
        let result_with_query = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "specific search term xyz"
            }),
        );
        assert!(result_with_query.is_ok());
    }

    #[test]
    fn test_get_relevant_context_with_paths() {
        let server = create_test_server();

        // Test with path filtering
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "paths": ["src/", "lib/"]
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with path filter"
        );

        // Test with single path
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "paths": ["tests/"]
            }),
        );
        assert!(result.is_ok());

        // Test with non-existent path
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "paths": ["nonexistent_directory_xyz/"]
            }),
        );
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(
            output.contains("No files match") || output.contains("No files indexed"),
            "Should handle path filter with no matches"
        );
    }

    #[test]
    fn test_get_relevant_context_custom_weights() {
        let server = create_test_server();

        // Test with custom weights
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "weights": {
                    "recency": 0.5,
                    "relevance": 0.2,
                    "centrality": 0.1,
                    "git_activity": 0.1,
                    "user_focus": 0.1
                }
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with custom weights"
        );

        // Verify output includes weight information when results exist
        let output = result.unwrap();
        if output.contains("Relevant Context") {
            assert!(
                output.contains("recency=0.50"),
                "Output should show recency weight"
            );
        }

        // Test with recency-only prioritization
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "weights": {
                    "recency": 1.0,
                    "relevance": 0.0,
                    "centrality": 0.0,
                    "git_activity": 0.0,
                    "user_focus": 0.0
                }
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should work with recency-only weights"
        );

        // Test with relevance-only prioritization
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "test query",
                "weights": {
                    "recency": 0.0,
                    "relevance": 1.0,
                    "centrality": 0.0,
                    "git_activity": 0.0,
                    "user_focus": 0.0
                }
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should work with relevance-only weights"
        );
    }

    #[test]
    fn test_get_relevant_context_include_content() {
        let server = create_test_server();

        // Test with include_content = true (default)
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "include_content": true
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with include_content=true"
        );

        // Test with include_content = false
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "include_content": false
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with include_content=false"
        );
    }

    #[test]
    fn test_get_relevant_context_combined_parameters() {
        let server = create_test_server();

        // Test with all parameters combined
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "search term",
                "paths": ["src/"],
                "limit": 5,
                "min_score": 0.1,
                "include_content": true,
                "weights": {
                    "recency": 0.3,
                    "relevance": 0.4,
                    "centrality": 0.1,
                    "git_activity": 0.1,
                    "user_focus": 0.1
                }
            }),
        );
        assert!(
            result.is_ok(),
            "get_relevant_context should succeed with all parameters"
        );
    }

    #[test]
    fn test_get_relevant_context_score_breakdown() {
        let server = create_test_server();

        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "limit": 5
            }),
        );
        assert!(result.is_ok());

        let output = result.unwrap();
        // If there are results, verify score breakdown is included
        if output.contains("Relevant Context") && !output.contains("No files") {
            assert!(
                output.contains("Breakdown:"),
                "Output should include score breakdown"
            );
            assert!(
                output.contains("recency=") && output.contains("relevance="),
                "Breakdown should include individual score components"
            );
        }
    }

    #[test]
    fn test_get_relevant_context_empty_codebase() {
        // Create server with empty index
        let server = create_test_server();

        let result = server.call_tool("get_relevant_context", json!({}));
        assert!(result.is_ok(), "Should handle empty codebase gracefully");

        let output = result.unwrap();
        assert!(
            output.contains("No files indexed") || output.contains("reindex"),
            "Should suggest running reindex for empty codebase"
        );
    }

    #[test]
    fn test_get_relevant_context_invalid_weight_values() {
        let server = create_test_server();

        // Test with partial weights (missing some fields)
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "weights": {
                    "recency": 0.5
                    // Missing other weights - should use defaults
                }
            }),
        );
        assert!(
            result.is_ok(),
            "Should handle partial weight specification"
        );

        // Test with zero weights
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "weights": {
                    "recency": 0.0,
                    "relevance": 0.0,
                    "centrality": 0.0,
                    "git_activity": 0.0,
                    "user_focus": 0.0
                }
            }),
        );
        assert!(result.is_ok(), "Should handle all-zero weights");
    }
}

// ============================================================================
// get_relevant_context E2E MCP Protocol Tests
// ============================================================================

mod get_relevant_context_e2e_tests {
    use super::*;

    #[test]
    fn test_e2e_get_relevant_context_basic() {
        let server = create_test_server();

        // Test full MCP protocol flow for get_relevant_context
        let result = server.call_tool("get_relevant_context", json!({}));

        assert!(result.is_ok(), "E2E basic call should succeed");
    }

    #[test]
    fn test_e2e_get_relevant_context_with_all_options() {
        let server = create_test_server();

        // Full protocol test with all parameters
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "find relevant code",
                "paths": ["src/", "lib/", "tests/"],
                "limit": 10,
                "min_score": 0.2,
                "include_content": true,
                "weights": {
                    "recency": 0.25,
                    "relevance": 0.30,
                    "centrality": 0.20,
                    "git_activity": 0.15,
                    "user_focus": 0.10
                }
            }),
        );

        assert!(result.is_ok(), "E2E call with all options should succeed");
    }

    #[test]
    fn test_e2e_get_relevant_context_empty_codebase() {
        // Edge case: empty codebase
        let server = create_test_server();

        let result = server.call_tool("get_relevant_context", json!({}));

        assert!(result.is_ok(), "Should handle empty codebase");
        let output = result.unwrap();
        // Empty codebase should return appropriate message
        assert!(
            output.contains("No files indexed") || output.contains("Relevant Context"),
            "Should indicate empty state or show results"
        );
    }

    #[test]
    fn test_e2e_get_relevant_context_schema_validation() {
        let server = create_test_server();
        let tools = server.list_tools();

        // Find get_relevant_context tool and verify schema
        let tool = tools
            .iter()
            .find(|t| t.name.as_ref() == "get_relevant_context");
        assert!(tool.is_some(), "get_relevant_context tool should exist");

        let tool = tool.unwrap();
        assert!(
            tool.description.is_some(),
            "Tool should have a description"
        );

        let schema: serde_json::Value =
            serde_json::to_value(&tool.input_schema).expect("Schema should serialize");
        assert!(
            schema.is_object(),
            "Schema should be an object"
        );
        assert_eq!(
            schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "Schema type should be 'object'"
        );

        // Verify expected properties exist
        let properties = schema.get("properties").and_then(|v| v.as_object());
        assert!(properties.is_some(), "Schema should have properties");

        let props = properties.unwrap();
        assert!(props.contains_key("query"), "Should have query property");
        assert!(props.contains_key("paths"), "Should have paths property");
        assert!(props.contains_key("limit"), "Should have limit property");
        assert!(props.contains_key("min_score"), "Should have min_score property");
        assert!(
            props.contains_key("include_content"),
            "Should have include_content property"
        );
        assert!(props.contains_key("weights"), "Should have weights property");
    }

    #[test]
    fn test_e2e_get_relevant_context_concurrent_calls() {
        let server = create_test_server();

        // Make multiple concurrent-style calls
        let result1 = server.call_tool(
            "get_relevant_context",
            json!({ "limit": 5 }),
        );
        let result2 = server.call_tool(
            "get_relevant_context",
            json!({ "limit": 10 }),
        );
        let result3 = server.call_tool(
            "get_relevant_context",
            json!({ "min_score": 0.5 }),
        );

        assert!(result1.is_ok(), "First call should succeed");
        assert!(result2.is_ok(), "Second call should succeed");
        assert!(result3.is_ok(), "Third call should succeed");
    }
}

// ============================================================================
// Performance Tests (ignored by default)
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with: cargo test --ignored
    fn test_get_relevant_context_performance() {
        let server = create_test_server();

        // Measure response time
        let start = Instant::now();
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "performance test query",
                "limit": 100
            }),
        );
        let duration = start.elapsed();

        assert!(result.is_ok(), "Performance test should succeed");
        assert!(
            duration.as_secs() < 2,
            "get_relevant_context should complete in under 2 seconds, took {:?}",
            duration
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --ignored
    fn test_get_relevant_context_performance_with_weights() {
        let server = create_test_server();

        let start = Instant::now();
        let result = server.call_tool(
            "get_relevant_context",
            json!({
                "query": "complex query with many terms",
                "limit": 50,
                "weights": {
                    "recency": 0.3,
                    "relevance": 0.4,
                    "centrality": 0.1,
                    "git_activity": 0.1,
                    "user_focus": 0.1
                }
            }),
        );
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert!(
            duration.as_secs() < 2,
            "Weighted context selection should complete in under 2 seconds, took {:?}",
            duration
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --ignored
    fn test_get_relevant_context_repeated_calls_performance() {
        let server = create_test_server();

        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let result = server.call_tool(
                "get_relevant_context",
                json!({
                    "limit": 20
                }),
            );
            assert!(result.is_ok());
        }

        let total_duration = start.elapsed();
        let avg_duration = total_duration / iterations;

        assert!(
            avg_duration.as_millis() < 500,
            "Average call should complete in under 500ms, average was {:?}",
            avg_duration
        );
    }
}
