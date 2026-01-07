---
id: onnx-embedding-engine
name: Complete ONNX Embedding Engine
priority: 1
dependencies: []
estimated_hours: 4
tags: [ml, embeddings, onnx]
---

## Objective

Complete the ONNX Runtime embedding engine to generate 384-dimensional vectors from text using the all-MiniLM-L6-v2 model.

## Context

The `contextmcp-embeddings` crate has a skeleton implementation for embedding generation. The model configuration exists but actual ONNX inference is marked as TODO. This subtask completes the embedding pipeline that other subtasks depend on.

## Implementation

1. **Model Download & Management** (`crates/contextmcp-embeddings/src/model.rs`)
   - Implement `ModelManager::ensure_model_available()` to download model if missing
   - Add model integrity verification (hash check)
   - Handle model storage at `~/.local/share/contextmcp/models/`

2. **ONNX Inference Engine** (`crates/contextmcp-embeddings/src/inference.rs`)
   - Complete `EmbeddingEngine::new()` to load ONNX model via `ort` crate
   - Implement `embed()` for single text embedding with tokenization
   - Implement `embed_batch()` for efficient batch processing
   - Add proper error handling for model loading failures

3. **Tokenization**
   - Implement simple WordPiece or use a tokenizer library
   - Handle max sequence length (512 tokens) with truncation
   - Prepare input tensors for ONNX Runtime

4. **Tests** (`crates/contextmcp-embeddings/src/inference.rs`)
   - Test embedding generation produces 384-dim vectors
   - Test batch embedding consistency
   - Test cosine_similarity function (already exists, verify)
   - Add mock/fallback for tests without model file

## Acceptance Criteria

- [ ] `EmbeddingEngine::new()` successfully loads ONNX model
- [ ] `embed(text)` returns `Vec<f32>` with 384 dimensions
- [ ] `embed_batch(texts)` returns consistent embeddings
- [ ] `cosine_similarity()` works correctly with generated embeddings
- [ ] Tests pass with `cargo test -p contextmcp-embeddings`
- [ ] Handles missing model file gracefully with clear error message

## Files to Create/Modify

- `crates/contextmcp-embeddings/src/model.rs` - Model download/management
- `crates/contextmcp-embeddings/src/inference.rs` - ONNX inference implementation
- `crates/contextmcp-embeddings/src/lib.rs` - Re-exports if needed
- `crates/contextmcp-embeddings/Cargo.toml` - Add tokenizer dependency if needed

## Integration Points

- **Provides**: `EmbeddingEngine` with `embed()` and `embed_batch()` methods
- **Consumes**: None (foundational component)
- **Conflicts**: None - isolated to `contextmcp-embeddings` crate

## Technical Notes

- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Uses `ort` crate v2.0.0-rc.9 for ONNX Runtime
- Use `ndarray` for tensor operations
- Consider `tokenizers` crate for proper tokenization
