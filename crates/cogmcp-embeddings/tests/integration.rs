//! Integration tests for the embedding engine
//!
//! These tests require the model to be downloaded. They are ignored by default
//! and can be run with: cargo test -p cogmcp-embeddings --test integration -- --ignored
//!
//! To download the model, use ModelManager::ensure_model_available() first.

use std::sync::Arc;
use std::thread;

use cogmcp_embeddings::{EmbeddingEngine, LazyEmbeddingEngine, ModelConfig, ModelManager};

/// Test that model can be downloaded and loaded successfully
#[test]
#[ignore = "Requires downloading model (~90MB)"]
fn test_model_download_and_load() {
    let manager = ModelManager::new().expect("Failed to create model manager");
    let config = manager
        .ensure_model_available()
        .expect("Failed to download model");

    assert!(config.model_exists(), "Model file should exist after download");
    assert!(
        config.tokenizer_exists(),
        "Tokenizer file should exist after download"
    );

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    assert!(engine.is_loaded(), "Engine should be loaded");
    assert_eq!(engine.embedding_dim(), 384);
}

/// Test embedding generation produces correct dimensions
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_embed_produces_384_dimensions() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    // Skip if model not available
    if !manager.is_model_available() {
        println!("Skipping test - model not available. Run with model download first.");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let embedding = engine.embed("Hello, world!").expect("Failed to embed text");
    assert_eq!(embedding.len(), 384, "Embedding should be 384-dimensional");

    // Verify it's normalized (L2 norm should be ~1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be L2 normalized, got norm: {}",
        norm
    );
}

/// Test batch embedding consistency
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_embed_batch_consistency() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let text = "The quick brown fox jumps over the lazy dog.";

    // Generate single embedding
    let single = engine.embed(text).expect("Failed to embed single");

    // Generate batch embedding
    let batch = engine
        .embed_batch(&[text])
        .expect("Failed to embed batch");

    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].len(), 384);

    // Single and batch should produce identical results
    let similarity = EmbeddingEngine::cosine_similarity(&single, &batch[0]);
    assert!(
        similarity > 0.999,
        "Single and batch embedding should be nearly identical, got similarity: {}",
        similarity
    );
}

/// Test semantic similarity works correctly
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_semantic_similarity() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Similar sentences
    let a = engine.embed("The cat sits on the mat.").unwrap();
    let b = engine.embed("A feline rests on a rug.").unwrap();

    // Unrelated sentence
    let c = engine.embed("The stock market crashed yesterday.").unwrap();

    let sim_ab = EmbeddingEngine::cosine_similarity(&a, &b);
    let sim_ac = EmbeddingEngine::cosine_similarity(&a, &c);
    let sim_bc = EmbeddingEngine::cosine_similarity(&b, &c);

    println!("Similarity (cat/feline): {}", sim_ab);
    println!("Similarity (cat/stock): {}", sim_ac);
    println!("Similarity (feline/stock): {}", sim_bc);

    // Similar sentences should have higher similarity than unrelated ones
    assert!(
        sim_ab > sim_ac,
        "Similar sentences should have higher similarity: {} vs {}",
        sim_ab,
        sim_ac
    );
    assert!(
        sim_ab > sim_bc,
        "Similar sentences should have higher similarity: {} vs {}",
        sim_ab,
        sim_bc
    );
}

/// Test handling of empty and whitespace text
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_edge_cases() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Empty string should still produce an embedding (from special tokens)
    let empty = engine.embed("").expect("Failed to embed empty string");
    assert_eq!(empty.len(), 384);

    // Whitespace only
    let whitespace = engine.embed("   ").expect("Failed to embed whitespace");
    assert_eq!(whitespace.len(), 384);

    // Long text (should be truncated but still work)
    let long_text = "word ".repeat(1000);
    let long_embedding = engine
        .embed(&long_text)
        .expect("Failed to embed long text");
    assert_eq!(long_embedding.len(), 384);
}

/// Test batch embedding with multiple texts
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_embed_batch_multiple() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let texts = vec![
        "First sentence about programming.",
        "Second sentence about cooking.",
        "Third sentence about sports.",
    ];

    let embeddings = engine.embed_batch(&texts).expect("Failed to batch embed");

    assert_eq!(embeddings.len(), 3, "Should have 3 embeddings");

    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "Embedding {} should be 384-dim", i);

        // Check normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding {} should be normalized",
            i
        );
    }
}

/// Test missing model file returns clear error
#[test]
fn test_missing_model_error() {
    let config = ModelConfig::with_path("/nonexistent/path/model.onnx");
    let result = EmbeddingEngine::new(config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("Model file"),
        "Error should mention missing model: {}",
        err_msg
    );
}

// ============================================================================
// LazyEmbeddingEngine Tests
// ============================================================================

/// Test that LazyEmbeddingEngine is not loaded initially
#[test]
fn test_lazy_engine_not_loaded_initially() {
    let config = ModelConfig::default();
    let engine = LazyEmbeddingEngine::new(config);

    // Engine should not be loaded before any embed calls
    assert!(!engine.is_loaded(), "Engine should not be loaded initially");
}

/// Test embedding_dim works without loading the model
#[test]
fn test_lazy_engine_embedding_dim_without_loading() {
    let mut config = ModelConfig::default();
    config.embedding_dim = 512; // Custom dimension

    let engine = LazyEmbeddingEngine::new(config);

    // embedding_dim should work without loading the model
    assert_eq!(engine.embedding_dim(), 512);
    assert!(!engine.is_loaded(), "Engine should not be loaded after getting dim");
}

/// Test is_available returns false with empty config
#[test]
fn test_lazy_engine_is_available_empty_config() {
    let config = ModelConfig::default();
    let engine = LazyEmbeddingEngine::new(config);

    // With empty paths, should return false
    assert!(!engine.is_available());
    assert!(!engine.is_loaded());
}

/// Test is_available returns false with nonexistent paths
#[test]
fn test_lazy_engine_is_available_nonexistent_paths() {
    let config = ModelConfig {
        model_path: "/nonexistent/path/model.onnx".to_string(),
        tokenizer_path: "/nonexistent/path/tokenizer.json".to_string(),
        embedding_dim: 384,
        max_length: 512,
        batch_size: 32,
    };
    let engine = LazyEmbeddingEngine::new(config);

    // With nonexistent paths, should return false
    assert!(!engine.is_available());
    assert!(!engine.is_loaded());
}

/// Test cosine_similarity static method
#[test]
fn test_lazy_engine_cosine_similarity() {
    // Static method should work without any engine instance
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = LazyEmbeddingEngine::cosine_similarity(&a, &b);
    assert!((similarity - 1.0).abs() < 1e-6);

    // Orthogonal vectors
    let c = vec![0.0, 1.0, 0.0];
    let sim_ac = LazyEmbeddingEngine::cosine_similarity(&a, &c);
    assert!(sim_ac.abs() < 1e-6);
}

/// Test Debug implementation
#[test]
fn test_lazy_engine_debug() {
    let config = ModelConfig::default();
    let engine = LazyEmbeddingEngine::new(config);

    // Debug should work and show is_loaded status
    let debug_str = format!("{:?}", engine);
    assert!(debug_str.contains("LazyEmbeddingEngine"));
    assert!(debug_str.contains("is_loaded"));
}

/// Test embed returns error without model
#[test]
fn test_lazy_engine_embed_without_model_returns_error() {
    let config = ModelConfig::default();
    let engine = LazyEmbeddingEngine::new(config);

    // Attempting to embed with empty config should fail
    let result = engine.embed("test");
    assert!(result.is_err());
}

/// Test thread safety of lazy engine
#[test]
fn test_lazy_engine_thread_safety() {
    let config = ModelConfig::default();
    let engine = Arc::new(LazyEmbeddingEngine::new(config));

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

/// Test lazy engine loads on first real embed call (requires model)
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_lazy_engine_loads_real_model() {
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

    // Verify embedding is correct
    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 384);

    // Verify it's normalized
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be L2 normalized, got norm: {}",
        norm
    );
}

/// Test lazy engine thread safety with real model
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_lazy_engine_real_thread_safety() {
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
        assert!(result.is_ok(), "Thread {} should succeed: {:?}", i, result.err());
    }

    // Engine should be loaded
    assert!(engine.is_loaded());
}

/// Test that subsequent embeds don't reload the model
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_lazy_engine_reuses_loaded_engine() {
    use std::time::Instant;

    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let engine = LazyEmbeddingEngine::new(config);

    // First embed - should be slower (includes model loading)
    let start = Instant::now();
    let _first = engine.embed("First embedding");
    let first_time = start.elapsed();

    assert!(engine.is_loaded());

    // Second embed - should be faster (no model loading)
    let start = Instant::now();
    let _second = engine.embed("Second embedding");
    let second_time = start.elapsed();

    // Third embed - should also be fast
    let start = Instant::now();
    let _third = engine.embed("Third embedding");
    let third_time = start.elapsed();

    println!("First embed (with load): {:?}", first_time);
    println!("Second embed (no load): {:?}", second_time);
    println!("Third embed (no load): {:?}", third_time);

    // Subsequent embeds should be significantly faster
    // (at least 5x faster since model loading takes ~1-2 seconds)
    assert!(
        second_time < first_time,
        "Second embed should be faster than first"
    );
    assert!(
        third_time < first_time,
        "Third embed should be faster than first"
    );
}
