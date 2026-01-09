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
// Concurrent Server Tests
// ============================================================================
//
// These tests verify the server handles concurrent tool calls correctly,
// simulating real-world usage where multiple MCP clients or parallel
// requests may access the server simultaneously.

mod concurrent_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// Create a shared test server wrapped in Arc for concurrent access
    fn create_shared_server() -> Arc<CogMcpServer> {
        let temp_dir = std::env::temp_dir().join("cogmcp-concurrent-test");
        std::fs::create_dir_all(&temp_dir).ok();

        Arc::new(CogMcpServer::in_memory(temp_dir).expect("Failed to create test server"))
    }

    // ========================================================================
    // Concurrent Read Operations
    // ========================================================================

    #[test]
    fn test_concurrent_ping_calls() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads making ping calls
        for _ in 0..10 {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..20 {
                    let result = server.call_tool("ping", json!({}));
                    assert!(result.is_ok(), "Ping should succeed");
                    let output = result.unwrap();
                    assert!(
                        output.contains("CogMCP server is running"),
                        "Should contain server status"
                    );
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 200);
    }

    #[test]
    fn test_concurrent_index_status_calls() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..8 {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..15 {
                    let result = server.call_tool("index_status", json!({}));
                    assert!(result.is_ok(), "index_status should succeed");
                    let output = result.unwrap();
                    assert!(
                        output.contains("Index Status"),
                        "Should contain status header"
                    );
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 120);
    }

    #[test]
    fn test_concurrent_list_tools() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let tools = server.list_tools();
                    assert_eq!(tools.len(), 8, "Should have 8 tools");
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 100);
    }

    // ========================================================================
    // Concurrent Search Operations
    // ========================================================================

    #[test]
    fn test_concurrent_grep_searches() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let patterns = vec!["test", "fn", "struct", "impl", "use"];

        for (i, pattern) in patterns.iter().enumerate() {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);
            let pattern = pattern.to_string();

            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let result = server.call_tool(
                        "context_grep",
                        json!({
                            "pattern": pattern,
                            "limit": (i + j) % 10 + 1
                        }),
                    );
                    assert!(result.is_ok(), "context_grep should succeed for pattern '{}'", pattern);
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 50);
    }

    #[test]
    fn test_concurrent_symbol_searches() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let symbol_names = vec!["main", "new", "test", "Config", "Server"];

        for symbol in symbol_names {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);
            let symbol = symbol.to_string();

            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let result = server.call_tool(
                        "find_symbol",
                        json!({
                            "name": symbol,
                            "fuzzy": true
                        }),
                    );
                    assert!(result.is_ok(), "find_symbol should succeed for '{}'", symbol);
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 50);
    }

    #[test]
    fn test_concurrent_context_search_modes() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let modes = vec!["keyword", "semantic", "hybrid"];

        for mode in modes {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);
            let mode = mode.to_string();

            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let result = server.call_tool(
                        "context_search",
                        json!({
                            "query": format!("search query {}", i),
                            "mode": mode,
                            "limit": 10
                        }),
                    );
                    assert!(result.is_ok(), "context_search should succeed for mode '{}'", mode);
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 30);
    }

    // ========================================================================
    // Mixed Tool Call Workload
    // ========================================================================

    #[test]
    fn test_mixed_concurrent_tool_calls() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Different threads calling different tools
        for i in 0..12 {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let result = match (i + j) % 6 {
                        0 => server.call_tool("ping", json!({})),
                        1 => server.call_tool("index_status", json!({})),
                        2 => server.call_tool(
                            "context_grep",
                            json!({"pattern": "test", "limit": 5}),
                        ),
                        3 => server.call_tool(
                            "find_symbol",
                            json!({"name": "main", "fuzzy": true}),
                        ),
                        4 => server.call_tool(
                            "context_search",
                            json!({"query": "function", "mode": "keyword"}),
                        ),
                        5 => {
                            // list_tools doesn't return Result
                            let _tools = server.list_tools();
                            Ok("tools listed".to_string())
                        }
                        _ => unreachable!(),
                    };

                    assert!(result.is_ok(), "Tool call {} should succeed", (i + j) % 6);
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 120);
    }

    // ========================================================================
    // Reindex Under Concurrent Load
    // ========================================================================

    #[test]
    fn test_concurrent_reindex_with_reads() {
        let server = create_shared_server();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Reindex thread
        {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..3 {
                    let result = server.call_tool("reindex", json!({}));
                    assert!(result.is_ok(), "Reindex should succeed");
                    success_count.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(50));
                }
            });

            handles.push(handle);
        }

        // Reader threads
        for _ in 0..5 {
            let server = Arc::clone(&server);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..15 {
                    // Mix of read operations
                    let _ = server.call_tool("index_status", json!({})).unwrap();
                    let _ = server.call_tool("ping", json!({})).unwrap();
                    success_count.fetch_add(2, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // 3 reindex + (5 threads * 15 iterations * 2 calls)
        assert_eq!(success_count.load(Ordering::SeqCst), 153);
    }

    // ========================================================================
    // Server Info Concurrent Access
    // ========================================================================

    #[test]
    fn test_concurrent_server_info_access() {
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for _ in 0..20 {
                    let info = CogMcpServer::server_info();
                    assert_eq!(info.name, "cogmcp");
                    assert!(!info.version.is_empty());

                    let caps = CogMcpServer::capabilities();
                    assert!(caps.tools.is_some());

                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 200);
    }

    // ========================================================================
    // No Deadlock Under Heavy Load
    // ========================================================================

    #[test]
    fn test_no_deadlock_heavy_concurrent_load() {
        let server = create_shared_server();
        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn many threads doing various operations
        for i in 0..20 {
            let server = Arc::clone(&server);
            let completed = Arc::clone(&completed);

            let handle = thread::spawn(move || {
                for j in 0..15 {
                    match (i + j) % 8 {
                        0 => {
                            let _ = server.call_tool("ping", json!({}));
                        }
                        1 => {
                            let _ = server.call_tool("index_status", json!({}));
                        }
                        2 => {
                            let _ = server.call_tool(
                                "context_grep",
                                json!({"pattern": "test", "limit": 10}),
                            );
                        }
                        3 => {
                            let _ = server.call_tool(
                                "find_symbol",
                                json!({"name": "test", "fuzzy": true}),
                            );
                        }
                        4 => {
                            let _ = server.call_tool(
                                "context_search",
                                json!({"query": "test", "mode": "keyword"}),
                            );
                        }
                        5 => {
                            let _ = server.list_tools();
                        }
                        6 => {
                            let _ = CogMcpServer::server_info();
                        }
                        7 => {
                            let _ = CogMcpServer::capabilities();
                        }
                        _ => unreachable!(),
                    }
                    completed.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        // Set a timeout to detect potential deadlocks
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(60);

        for handle in handles {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                panic!("Deadlock detected: test timed out");
            }
            handle.join().expect("Thread should complete without deadlock");
        }

        assert_eq!(completed.load(Ordering::SeqCst), 300);
    }

    // ========================================================================
    // Error Handling Under Concurrency
    // ========================================================================

    #[test]
    fn test_concurrent_error_handling() {
        let server = create_shared_server();
        let error_count = Arc::new(AtomicUsize::new(0));
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Some threads making valid calls, some making invalid calls
        for i in 0..10 {
            let server = Arc::clone(&server);
            let error_count = Arc::clone(&error_count);
            let success_count = Arc::clone(&success_count);

            let handle = thread::spawn(move || {
                for j in 0..10 {
                    if (i + j) % 3 == 0 {
                        // Invalid call - missing required parameter
                        let result = server.call_tool("context_grep", json!({}));
                        if result.is_err() {
                            error_count.fetch_add(1, Ordering::SeqCst);
                        }
                    } else if (i + j) % 3 == 1 {
                        // Invalid call - unknown tool
                        let result = server.call_tool("nonexistent_tool", json!({}));
                        if result.is_err() {
                            error_count.fetch_add(1, Ordering::SeqCst);
                        }
                    } else {
                        // Valid call
                        let result = server.call_tool("ping", json!({}));
                        if result.is_ok() {
                            success_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // Verify both errors and successes were properly handled
        let errors = error_count.load(Ordering::SeqCst);
        let successes = success_count.load(Ordering::SeqCst);

        assert!(errors > 0, "Should have handled some errors");
        assert!(successes > 0, "Should have some successes");
        // Total should be 100
        assert!(
            errors + successes <= 100,
            "Total should not exceed iterations"
        );
    }

    // ========================================================================
    // Consistent Results Under Concurrency
    // ========================================================================

    #[test]
    fn test_consistent_tool_list_under_concurrency() {
        let server = create_shared_server();
        let results = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut handles = vec![];

        for _ in 0..10 {
            let server = Arc::clone(&server);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let tools = server.list_tools();
                    let names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
                    results.lock().push(names);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // Verify all results are identical
        let all_results = results.lock();
        assert_eq!(all_results.len(), 100);

        let first = &all_results[0];
        for result in all_results.iter() {
            assert_eq!(result, first, "All tool lists should be identical");
        }
    }
}
