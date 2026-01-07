//! Integration tests for the embedding engine
//!
//! These tests require the model to be downloaded. They are ignored by default
//! and can be run with: cargo test -p cogmcp-embeddings --test integration -- --ignored
//!
//! To download the model, use ModelManager::ensure_model_available() first.

use cogmcp_embeddings::{EmbeddingEngine, ModelManager};

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
    use cogmcp_embeddings::ModelConfig;

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
