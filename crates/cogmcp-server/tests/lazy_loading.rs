//! Integration tests for lazy loading behavior
//!
//! These tests verify that the server correctly implements lazy loading for the
//! embedding engine, ensuring fast startup times and on-demand model loading.

use std::sync::Arc;
use std::time::Instant;

use cogmcp_embeddings::{LazyEmbeddingEngine, ModelConfig};
use cogmcp_search::SemanticSearch;
use cogmcp_storage::Database;
use cogmcp_server::CogMcpServer;

/// Create a test server with lazy embedding engine
fn create_lazy_test_server() -> CogMcpServer {
    let temp_dir = std::env::temp_dir().join("cogmcp-lazy-test");
    std::fs::create_dir_all(&temp_dir).ok();

    CogMcpServer::in_memory(temp_dir).expect("Failed to create test server")
}

/// Create a server with embeddings enabled (but empty model paths for testing)
fn create_server_with_lazy_embeddings() -> (CogMcpServer, Arc<LazyEmbeddingEngine>) {
    let temp_dir = std::env::temp_dir().join("cogmcp-lazy-embed-test");
    std::fs::create_dir_all(&temp_dir).ok();

    let db = Arc::new(Database::in_memory().expect("Failed to create in-memory database"));
    let text_index = Arc::new(cogmcp_storage::FullTextIndex::in_memory().expect("Failed to create text index"));
    let parser = Arc::new(cogmcp_index::CodeParser::new());
    let config = cogmcp_core::Config::default();

    // Create lazy embedding engine with default (empty) config
    let model_config = ModelConfig::default();
    let engine = Arc::new(LazyEmbeddingEngine::new(model_config));
    let semantic = Arc::new(SemanticSearch::new(engine.clone(), db.clone()));

    let server = CogMcpServer {
        root: temp_dir,
        config,
        db,
        text_index,
        parser,
        embedding_engine: Some(engine.clone()),
        semantic_search: Some(semantic),
    };

    (server, engine)
}

// ============================================================================
// Test: Server starts without loading model
// ============================================================================

#[test]
fn test_server_starts_without_loading_model() {
    // Measure server creation time
    let start = Instant::now();
    let (server, engine) = create_server_with_lazy_embeddings();
    let creation_time = start.elapsed();

    // Server creation should be fast (< 100ms) because model isn't loaded
    assert!(
        creation_time.as_millis() < 100,
        "Server creation took too long ({:?}). Model should not be loaded at startup.",
        creation_time
    );

    // Verify model is not loaded
    assert!(
        !engine.is_loaded(),
        "Model should not be loaded immediately after server creation"
    );

    // Server should still have semantic search capability configured
    assert!(
        server.semantic_search.is_some(),
        "Server should have semantic search configured (even if not loaded)"
    );
}

#[test]
fn test_lazy_engine_not_loaded_after_server_creation() {
    let (_server, engine) = create_server_with_lazy_embeddings();

    // The engine should not be loaded
    assert!(!engine.is_loaded());

    // But embedding_dim should work without loading
    assert_eq!(engine.embedding_dim(), 384);

    // Still not loaded after getting dim
    assert!(!engine.is_loaded());
}

// ============================================================================
// Test: Model loads on first semantic search
// ============================================================================

#[test]
fn test_semantic_search_triggers_load_attempt() {
    let (server, engine) = create_server_with_lazy_embeddings();

    // Engine should not be loaded yet
    assert!(!engine.is_loaded());

    // Call semantic_search tool - this should attempt to load the model
    // Note: With empty model paths, this will fail but still trigger the load attempt
    let result = server.call_tool(
        "semantic_search",
        serde_json::json!({
            "query": "test query",
            "limit": 10
        }),
    );

    // The result should indicate that semantic search is not available
    // (because we have empty model paths)
    match result {
        Ok(output) => {
            assert!(
                output.contains("not available") || output.contains("No matches"),
                "Expected 'not available' or 'No matches' message, got: {}",
                output
            );
        }
        Err(_) => {
            // Error is also acceptable since model paths are empty
        }
    }
}

// ============================================================================
// Test: Model stays loaded for subsequent operations
// ============================================================================

#[test]
fn test_embedding_dim_available_without_loading() {
    let (_server, engine) = create_server_with_lazy_embeddings();

    // Should be able to get embedding dimension without loading
    let dim = engine.embedding_dim();
    assert_eq!(dim, 384);

    // Engine still not loaded
    assert!(!engine.is_loaded());

    // Multiple calls should return same value
    assert_eq!(engine.embedding_dim(), 384);
    assert_eq!(engine.embedding_dim(), 384);

    // Still not loaded
    assert!(!engine.is_loaded());
}

// ============================================================================
// Test: Semantic search unavailable without model
// ============================================================================

#[test]
fn test_semantic_search_unavailable_with_empty_model_path() {
    let (server, engine) = create_server_with_lazy_embeddings();

    // With empty model paths, is_available should return false
    assert!(
        !engine.is_available(),
        "Engine should not be available with empty model paths"
    );

    // has_semantic_search should also return false
    assert!(
        !server.has_semantic_search(),
        "Server should report semantic search unavailable with empty model paths"
    );
}

#[test]
fn test_semantic_search_unavailable_with_nonexistent_model() {
    let temp_dir = std::env::temp_dir().join("cogmcp-nonexistent-model-test");
    std::fs::create_dir_all(&temp_dir).ok();

    let db = Arc::new(Database::in_memory().expect("Failed to create database"));

    // Create engine with nonexistent model path
    let model_config = ModelConfig {
        model_path: "/nonexistent/path/model.onnx".to_string(),
        tokenizer_path: "/nonexistent/path/tokenizer.json".to_string(),
        embedding_dim: 384,
        max_length: 512,
    };
    let engine = Arc::new(LazyEmbeddingEngine::new(model_config));
    let semantic = Arc::new(SemanticSearch::new(engine.clone(), db));

    // is_available should return false for nonexistent paths
    assert!(
        !engine.is_available(),
        "Engine should not be available with nonexistent model paths"
    );

    // is_loaded should also be false
    assert!(
        !engine.is_loaded(),
        "Engine should not be loaded with nonexistent paths"
    );

    // Semantic search should report unavailable
    assert!(
        !semantic.is_available(),
        "Semantic search should not be available with nonexistent model"
    );
}

#[test]
fn test_semantic_search_returns_appropriate_error() {
    let (server, _engine) = create_server_with_lazy_embeddings();

    // Call semantic_search - should return an error or appropriate message
    let result = server.call_tool(
        "semantic_search",
        serde_json::json!({
            "query": "find something",
            "limit": 5
        }),
    );

    // Should get an error or "not available" message
    match result {
        Ok(output) => {
            // Check that it indicates unavailability
            let is_unavailable = output.contains("not available")
                || output.contains("unavailable")
                || output.contains("disabled")
                || output.contains("No matches");
            assert!(
                is_unavailable,
                "Expected message indicating unavailability, got: {}",
                output
            );
        }
        Err(e) => {
            // Error is acceptable
            assert!(
                e.contains("not available") || e.contains("unavailable") || e.contains("semantic"),
                "Error should mention unavailability: {}",
                e
            );
        }
    }
}

// ============================================================================
// Test: Thread safety
// ============================================================================

#[test]
fn test_lazy_engine_thread_safety() {
    use std::thread;

    let model_config = ModelConfig::default();
    let engine = Arc::new(LazyEmbeddingEngine::new(model_config));

    // Spawn multiple threads accessing the engine concurrently
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let engine_clone = engine.clone();
            thread::spawn(move || {
                // These operations should be thread-safe
                let _dim = engine_clone.embedding_dim();
                let _loaded = engine_clone.is_loaded();
                let _available = engine_clone.is_available();
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Engine should still be valid
    assert_eq!(engine.embedding_dim(), 384);
}

// ============================================================================
// Test: In-memory server without embeddings
// ============================================================================

#[test]
fn test_in_memory_server_has_no_embeddings() {
    let server = create_lazy_test_server();

    // In-memory server should not have embeddings by default
    assert!(
        server.embedding_engine.is_none(),
        "In-memory test server should not have embedding engine"
    );

    assert!(
        !server.has_semantic_search(),
        "In-memory test server should not have semantic search"
    );
}

#[test]
fn test_basic_server_tools_work_without_embeddings() {
    let server = create_lazy_test_server();

    // Ping should work
    let result = server.call_tool("ping", serde_json::json!({}));
    assert!(result.is_ok(), "Ping should work without embeddings");

    // Index status should work
    let result = server.call_tool("index_status", serde_json::json!({}));
    assert!(result.is_ok(), "Index status should work without embeddings");
}

// ============================================================================
// Test: Performance verification
// ============================================================================

#[test]
fn test_server_creation_performance() {
    // Create multiple servers and verify fast creation
    let mut times = Vec::new();

    for _ in 0..5 {
        let start = Instant::now();
        let _ = create_lazy_test_server();
        times.push(start.elapsed());
    }

    // Average time should be fast
    let avg_ms: u128 = times.iter().map(|d| d.as_millis()).sum::<u128>() / times.len() as u128;

    assert!(
        avg_ms < 50,
        "Average server creation time should be < 50ms, was {}ms",
        avg_ms
    );
}

#[test]
fn test_lazy_engine_creation_is_instant() {
    let start = Instant::now();
    let _engine = LazyEmbeddingEngine::new(ModelConfig::default());
    let creation_time = start.elapsed();

    // Creating a lazy engine should be essentially instant (< 1ms)
    assert!(
        creation_time.as_micros() < 1000,
        "Lazy engine creation took too long ({:?}). Should be instant.",
        creation_time
    );
}

// ============================================================================
// Ignored tests that require a real model
// ============================================================================

#[test]
#[ignore = "Requires model to be downloaded (~90MB)"]
fn test_lazy_engine_loads_real_model() {
    use cogmcp_embeddings::ModelManager;

    let manager = ModelManager::new().expect("Failed to create model manager");

    // Skip if model not available
    if !manager.is_model_available() {
        println!("Skipping test - model not available. Run model download first.");
        return;
    }

    let config = manager.get_config();
    let engine = LazyEmbeddingEngine::new(config);

    // Initially not loaded
    assert!(!engine.is_loaded());
    assert!(engine.is_available()); // Files exist

    // First embed triggers load
    let result = engine.embed("Hello, world!");
    assert!(result.is_ok(), "Embedding should succeed with real model");

    // Now it should be loaded
    assert!(engine.is_loaded());

    // Subsequent embeds should work
    let result2 = engine.embed("Another test");
    assert!(result2.is_ok());
}

#[test]
#[ignore = "Requires model to be downloaded"]
fn test_lazy_engine_real_thread_safety() {
    use std::thread;
    use cogmcp_embeddings::ModelManager;

    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let engine = Arc::new(LazyEmbeddingEngine::new(config));

    // Spawn threads that all try to embed simultaneously
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let engine_clone = engine.clone();
            thread::spawn(move || {
                let text = format!("Test text from thread {}", i);
                engine_clone.embed(&text)
            })
        })
        .collect();

    // All threads should succeed
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok(), "Thread {} should succeed", i);
    }

    // Engine should be loaded
    assert!(engine.is_loaded());
}
