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

/// Test that batch inference produces identical results to sequential processing
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_batch_vs_sequential_consistency() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let texts = vec![
        "Machine learning is transforming technology.",
        "Deep neural networks learn complex patterns.",
        "Natural language processing understands human text.",
        "Computer vision recognizes objects in images.",
        "Reinforcement learning trains agents through rewards.",
        "Transfer learning leverages pre-trained models.",
        "Attention mechanisms help models focus on relevant input.",
        "The quick brown fox jumps over the lazy dog.",
    ];

    // Generate embeddings sequentially
    let sequential_embeddings: Vec<Vec<f32>> = texts
        .iter()
        .map(|text| engine.embed(text).expect("Sequential embed failed"))
        .collect();

    // Generate embeddings in a single batch
    let batch_embeddings = engine
        .embed_batch(&texts)
        .expect("Batch embed failed");

    assert_eq!(sequential_embeddings.len(), batch_embeddings.len());

    // Check that batch results match sequential results within floating point tolerance
    for (i, (seq, batch)) in sequential_embeddings.iter().zip(batch_embeddings.iter()).enumerate() {
        assert_eq!(seq.len(), batch.len(), "Embedding {} dimension mismatch", i);

        // Compute cosine similarity - should be nearly identical
        let similarity = EmbeddingEngine::cosine_similarity(seq, batch);
        assert!(
            similarity > 0.9999,
            "Text {}: batch and sequential embeddings should be nearly identical, got similarity: {}",
            i,
            similarity
        );

        // Also check element-wise for small differences
        let max_diff: f32 = seq
            .iter()
            .zip(batch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, diff| max.max(diff));
        assert!(
            max_diff < 1e-5,
            "Text {}: max element difference should be tiny, got: {}",
            i,
            max_diff
        );
    }
}

/// Test embed_batch_chunked with varying batch sizes
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_embed_batch_chunked() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Set a small batch size to force chunking
    engine.set_batch_size(3);
    assert_eq!(engine.batch_size(), 3);

    let texts = vec![
        "Text one about science.",
        "Text two about history.",
        "Text three about art.",
        "Text four about music.",
        "Text five about sports.",
        "Text six about food.",
        "Text seven about travel.",
        "Text eight about technology.",
    ];

    // This should process in 3 chunks: [3, 3, 2]
    let chunked_embeddings = engine
        .embed_batch_chunked(&texts)
        .expect("Chunked embed failed");

    assert_eq!(chunked_embeddings.len(), texts.len());

    // Each embedding should be properly normalized
    for (i, emb) in chunked_embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "Embedding {} should be 384-dim", i);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding {} should be L2 normalized, got norm: {}",
            i,
            norm
        );
    }

    // Verify consistency: chunked should match a fresh single-batch run
    let config2 = manager.get_config();
    let mut engine2 = EmbeddingEngine::new(config2).expect("Failed to create engine2");
    engine2.set_batch_size(100); // Large batch size

    let single_batch = engine2.embed_batch(&texts).expect("Single batch failed");

    for (i, (chunked, single)) in chunked_embeddings.iter().zip(single_batch.iter()).enumerate() {
        let similarity = EmbeddingEngine::cosine_similarity(chunked, single);
        assert!(
            similarity > 0.9999,
            "Text {}: chunked and single-batch should match, got similarity: {}",
            i,
            similarity
        );
    }
}

/// Test batch_size configuration
#[test]
fn test_batch_size_configuration() {
    use cogmcp_embeddings::DEFAULT_BATCH_SIZE;

    let engine = EmbeddingEngine::without_model();

    // Default batch size
    assert_eq!(engine.batch_size(), DEFAULT_BATCH_SIZE);
    assert_eq!(engine.batch_size(), 32);

    // With custom batch size
    let engine2 = EmbeddingEngine::without_model().with_batch_size(16);
    assert_eq!(engine2.batch_size(), 16);

    // Minimum batch size is 1
    let engine3 = EmbeddingEngine::without_model().with_batch_size(0);
    assert_eq!(engine3.batch_size(), 1);

    // Set batch size via setter
    let mut engine4 = EmbeddingEngine::without_model();
    engine4.set_batch_size(64);
    assert_eq!(engine4.batch_size(), 64);
}

/// Test batch inference performance (informational)
#[test]
#[ignore = "Requires model to be downloaded - performance benchmark"]
fn test_batch_performance() {
    use std::time::Instant;

    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Create test texts
    let texts: Vec<&str> = vec![
        "Artificial intelligence is revolutionizing many industries.",
        "Machine learning models can learn from data without explicit programming.",
        "Neural networks are inspired by the human brain's structure.",
        "Deep learning has achieved remarkable results in image recognition.",
        "Natural language processing enables computers to understand human language.",
        "Reinforcement learning trains agents through trial and error.",
        "Computer vision allows machines to interpret visual information.",
        "Transfer learning reduces the need for large training datasets.",
    ];

    // Warm up
    let _ = engine.embed(texts[0]);
    let _ = engine.embed_batch(&texts[..2]);

    // Benchmark sequential processing
    let start = Instant::now();
    for text in &texts {
        let _ = engine.embed(text).expect("Sequential embed failed");
    }
    let sequential_time = start.elapsed();

    // Benchmark batch processing
    let start = Instant::now();
    let _ = engine.embed_batch(&texts).expect("Batch embed failed");
    let batch_time = start.elapsed();

    println!("\nPerformance comparison for {} texts:", texts.len());
    println!("  Sequential: {:?}", sequential_time);
    println!("  Batch:      {:?}", batch_time);
    println!(
        "  Speedup:    {:.2}x",
        sequential_time.as_secs_f64() / batch_time.as_secs_f64()
    );

    // Batch should be faster (at least 1.5x for 8 texts)
    // Note: This may not hold in all environments, so we use a generous threshold
    assert!(
        batch_time < sequential_time,
        "Batch processing should generally be faster than sequential"
    );
}

/// Test batch inference with empty input
#[test]
fn test_batch_empty_input() {
    let mut engine = EmbeddingEngine::without_model();

    let empty: &[&str] = &[];
    let result = engine.embed_batch(empty);

    // This should return Ok with empty vec, not an error
    // (even though model isn't loaded, empty input is handled early)
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

/// Test batch inference with texts of varying lengths
#[test]
#[ignore = "Requires model to be downloaded"]
fn test_batch_varying_lengths() {
    let manager = ModelManager::new().expect("Failed to create model manager");

    if !manager.is_model_available() {
        println!("Skipping test - model not available");
        return;
    }

    let config = manager.get_config();
    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Texts with very different lengths
    let texts = vec![
        "Hi.",
        "A slightly longer sentence about something interesting.",
        "This is an even longer piece of text that contains multiple sentences. It discusses various topics and should require more tokens to encode than the shorter texts in this batch. We want to ensure that padding works correctly.",
        "Short.",
    ];

    let embeddings = engine.embed_batch(&texts).expect("Batch embed failed");

    assert_eq!(embeddings.len(), 4);

    // All embeddings should still be valid (normalized 384-dim vectors)
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "Embedding {} should be 384-dim", i);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding {} should be normalized, got norm: {}",
            i,
            norm
        );
    }

    // Verify against sequential for consistency
    for (i, text) in texts.iter().enumerate() {
        let sequential = engine.embed(text).expect("Sequential embed failed");
        let similarity = EmbeddingEngine::cosine_similarity(&sequential, &embeddings[i]);
        assert!(
            similarity > 0.9999,
            "Text {} ('{}...'): batch should match sequential, got similarity: {}",
            i,
            text.chars().take(20).collect::<String>(),
            similarity
        );
    }
}
