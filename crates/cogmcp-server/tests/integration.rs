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

    // Verify we have the expected 9 tools
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
        tool_names.contains(&"reload_config"),
        "Should have reload_config tool"
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
        assert!(tool_names.contains(&"reload_config"));
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
// Streaming Integration Tests
// ============================================================================
//
// These tests verify streaming functionality for large result sets.

mod streaming_tests {
    use super::*;
    use cogmcp_core::config::StreamingConfigOptions;

    #[test]
    fn test_streaming_config_default() {
        let server = create_test_server();
        let config = server.get_streaming_config();

        assert!(config.enabled, "Streaming should be enabled by default");
        assert_eq!(config.auto_stream_threshold, 50, "Default threshold should be 50");
        assert_eq!(config.chunk_size, 10, "Default chunk size should be 10");
    }

    #[test]
    fn test_context_search_with_streaming_disabled() {
        let server = create_test_server();

        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "limit": 10,
                "streaming": false
            }),
        );

        assert!(result.is_ok(), "Search with streaming disabled should succeed");
        let output = result.unwrap();
        // Non-streaming output should not contain streaming markers
        assert!(!output.contains("Streaming"), "Should not contain streaming markers");
    }

    #[test]
    fn test_context_search_with_streaming_enabled() {
        let server = create_test_server();

        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "limit": 10,
                "streaming": true
            }),
        );

        assert!(result.is_ok(), "Search with streaming enabled should succeed");
        // Note: If there are no results, streaming markers won't appear
    }

    #[test]
    fn test_context_search_with_custom_chunk_size() {
        let server = create_test_server();

        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test query",
                "limit": 20,
                "streaming": true,
                "chunk_size": 5
            }),
        );

        assert!(result.is_ok(), "Search with custom chunk size should succeed");
    }

    #[test]
    fn test_semantic_search_with_streaming_disabled() {
        let server = create_test_server();

        let result = server.call_tool(
            "semantic_search",
            json!({
                "query": "test query",
                "limit": 10,
                "streaming": false
            }),
        );

        assert!(result.is_ok(), "Semantic search with streaming disabled should succeed");
    }

    #[test]
    fn test_semantic_search_with_streaming_enabled() {
        let server = create_test_server();

        let result = server.call_tool(
            "semantic_search",
            json!({
                "query": "test query",
                "limit": 10,
                "streaming": true
            }),
        );

        assert!(result.is_ok(), "Semantic search with streaming enabled should succeed");
    }

    #[test]
    fn test_semantic_search_with_custom_chunk_size() {
        let server = create_test_server();

        let result = server.call_tool(
            "semantic_search",
            json!({
                "query": "test query",
                "limit": 20,
                "streaming": true,
                "chunk_size": 5
            }),
        );

        assert!(result.is_ok(), "Semantic search with custom chunk size should succeed");
    }

    #[test]
    fn test_streaming_config_modification() {
        let mut server = create_test_server();

        // Verify default
        assert!(server.is_streaming_enabled());

        // Disable streaming
        let new_config = StreamingConfigOptions {
            enabled: false,
            auto_stream_threshold: 100,
            chunk_size: 20,
            yield_interval_ms: 200,
        };
        server.set_streaming_config(new_config);

        // Verify change
        assert!(!server.is_streaming_enabled());
        assert_eq!(server.get_streaming_config().auto_stream_threshold, 100);
        assert_eq!(server.get_streaming_config().chunk_size, 20);
    }

    #[test]
    fn test_auto_stream_threshold() {
        let server = create_test_server();
        let config = server.get_streaming_config();

        // Test threshold behavior
        assert!(!config.should_auto_stream(49), "Should not auto-stream with 49 results");
        assert!(config.should_auto_stream(50), "Should auto-stream with 50 results");
        assert!(config.should_auto_stream(100), "Should auto-stream with 100 results");
    }

    #[test]
    fn test_streaming_results_in_different_modes() {
        let server = create_test_server();

        // Test streaming with keyword mode
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test",
                "mode": "keyword",
                "streaming": true
            }),
        );
        assert!(result.is_ok(), "Keyword search with streaming should succeed");

        // Test streaming with hybrid mode
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test",
                "mode": "hybrid",
                "streaming": true
            }),
        );
        assert!(result.is_ok(), "Hybrid search with streaming should succeed");

        // Test streaming with semantic mode
        let result = server.call_tool(
            "context_search",
            json!({
                "query": "test",
                "mode": "semantic",
                "streaming": true
            }),
        );
        assert!(result.is_ok(), "Semantic search with streaming should succeed");
    }

    #[test]
    fn test_context_search_tool_schema_includes_streaming() {
        let server = create_test_server();
        let tools = server.list_tools();

        let context_search = tools
            .iter()
            .find(|t| t.name.as_ref() == "context_search")
            .expect("context_search tool should exist");

        let schema: Value = serde_json::to_value(&context_search.input_schema)
            .expect("Schema should serialize");

        let properties = schema.get("properties").expect("Schema should have properties");
        assert!(
            properties.get("streaming").is_some(),
            "context_search should have streaming parameter"
        );
        assert!(
            properties.get("chunk_size").is_some(),
            "context_search should have chunk_size parameter"
        );
    }

    #[test]
    fn test_semantic_search_tool_schema_includes_streaming() {
        let server = create_test_server();
        let tools = server.list_tools();

        let semantic_search = tools
            .iter()
            .find(|t| t.name.as_ref() == "semantic_search")
            .expect("semantic_search tool should exist");

        let schema: Value = serde_json::to_value(&semantic_search.input_schema)
            .expect("Schema should serialize");

        let properties = schema.get("properties").expect("Schema should have properties");
        assert!(
            properties.get("streaming").is_some(),
            "semantic_search should have streaming parameter"
        );
        assert!(
            properties.get("chunk_size").is_some(),
            "semantic_search should have chunk_size parameter"
        );
    }

    #[test]
    fn test_streaming_behavior_consistency() {
        let server = create_test_server();

        // Make the same query with and without streaming
        let result_no_stream = server.call_tool(
            "context_search",
            json!({
                "query": "test",
                "streaming": false
            }),
        );

        let result_stream = server.call_tool(
            "context_search",
            json!({
                "query": "test",
                "streaming": true
            }),
        );

        // Both should succeed
        assert!(result_no_stream.is_ok(), "Non-streaming search should succeed");
        assert!(result_stream.is_ok(), "Streaming search should succeed");
    }
}
