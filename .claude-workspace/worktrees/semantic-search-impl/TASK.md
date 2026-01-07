---
id: semantic-search-impl
name: Implement Semantic Search Module
priority: 2
dependencies: [onnx-embedding-engine, vector-storage-layer]
estimated_hours: 4
tags: [search, semantic, embeddings]
---

## Objective

Complete the semantic search implementation that uses embeddings to find semantically similar code chunks.

## Context

The `contextmcp-search` crate has a placeholder `SemanticSearch` struct. This subtask implements the actual semantic search logic by connecting the embedding engine with the vector storage layer.

## Implementation

1. **SemanticSearch Implementation** (`crates/contextmcp-search/src/semantic.rs`)
   - Add `EmbeddingEngine` and `Database` as dependencies
   - Implement `SemanticSearch::new(engine, db)` constructor
   - Implement `search(query: &str, limit: usize)` method:
     - Generate embedding for query text
     - Search vector storage for similar embeddings
     - Return ranked `SemanticSearchResult` list

2. **Result Enrichment**
   - Add file path resolution from symbol_id
   - Include line number range from symbol data
   - Add context snippet around the match

3. **Search Options**
   - Add `SemanticSearchOptions` struct with:
     - `min_similarity: f32` - threshold for results
     - `file_filter: Option<Vec<String>>` - limit to specific files
     - `chunk_types: Option<Vec<ChunkType>>` - filter by chunk type

4. **Caching**
   - Cache query embeddings to avoid re-computing for repeated queries
   - Use existing `Cache` from contextmcp-storage

5. **Tests** (`crates/contextmcp-search/src/semantic.rs`)
   - Test search returns results ordered by similarity
   - Test min_similarity threshold filtering
   - Test file filter works correctly
   - Test with mock embeddings for CI without model

## Acceptance Criteria

- [ ] `SemanticSearch::search()` returns relevant code chunks
- [ ] Results are ordered by similarity score (highest first)
- [ ] `min_similarity` threshold filters low-quality matches
- [ ] File filtering works correctly
- [ ] Tests pass with `cargo test -p contextmcp-search`
- [ ] Search completes within reasonable time (<500ms for typical queries)

## Files to Create/Modify

- `crates/contextmcp-search/src/semantic.rs` - Main implementation
- `crates/contextmcp-search/src/lib.rs` - Update exports
- `crates/contextmcp-search/Cargo.toml` - Add dependencies on embeddings, storage

## Integration Points

- **Provides**: `SemanticSearch` API for finding similar code
- **Consumes**: `EmbeddingEngine` (from onnx-embedding-engine), `Database` (from vector-storage-layer)
- **Conflicts**: Do not modify `hybrid.rs` (handled by hybrid-search-integration)

## Technical Notes

- Depends on both Wave 1 subtasks completing first
- Use `EmbeddingEngine::embed()` for query embedding
- Use `Database::search_similar_vectors()` for storage search
- Consider async implementation for larger codebases
