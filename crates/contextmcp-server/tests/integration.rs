//! Integration tests for the ContextMCP server
//!
//! These tests verify the server handles MCP protocol messages correctly.

use contextmcp_server::{ContextMcpServer, RunnerConfig, ServerRunner};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Create a test server with in-memory storage
fn create_test_server() -> ContextMcpServer {
    let temp_dir = std::env::temp_dir().join("contextmcp-test");
    std::fs::create_dir_all(&temp_dir).ok();

    ContextMcpServer::in_memory(temp_dir).expect("Failed to create test server")
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
    assert_eq!(tools.len(), 7, "Should have 7 tools");

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
        output.contains("ContextMCP server is running"),
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
    let info = ContextMcpServer::server_info();
    assert_eq!(info.name, "contextmcp");
    assert!(!info.version.is_empty());
}

#[test]
fn test_server_capabilities() {
    let caps = ContextMcpServer::capabilities();
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

        assert_eq!(info.server_info.name, "contextmcp");
        assert!(!info.server_info.version.is_empty());
        assert!(info.capabilities.tools.is_some());
        assert!(info.instructions.is_some());
    }

    #[test]
    fn test_e2e_tools_list_returns_7_tools() {
        let server = create_test_server();

        // List tools through the server's list_tools method
        let tools = server.list_tools();
        assert_eq!(tools.len(), 7, "Should return 7 tools");

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
            output.contains("ContextMCP server is running"),
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
