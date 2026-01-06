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
