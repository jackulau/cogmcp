---
id: vector-storage-layer
name: Implement Vector Storage in SQLite
priority: 1
dependencies: []
estimated_hours: 3
tags: [storage, sqlite, vectors]
---

## Objective

Extend the SQLite storage layer to support efficient vector storage and retrieval with similarity search capabilities.

## Context

The `contextmcp-storage` crate already has an `embeddings` table defined in the schema. However, vector retrieval and similarity search methods are not implemented. This subtask adds the storage operations needed for semantic search.

## Implementation

1. **Vector Storage Methods** (`crates/contextmcp-storage/src/sqlite.rs`)
   - Implement `get_embeddings_for_file(file_id)` - retrieve all embeddings for a file
   - Implement `get_all_embeddings()` - retrieve all stored embeddings for search
   - Implement `delete_embeddings_for_file(file_id)` - cleanup on file changes
   - Add `get_embedding_count()` for statistics

2. **Vector Search Method** (`crates/contextmcp-storage/src/sqlite.rs`)
   - Implement `search_similar_vectors(query_embedding, limit)` - brute-force similarity search
   - Return results with: chunk_text, file_path, similarity_score, symbol info
   - Use cosine similarity for scoring

3. **Batch Operations**
   - Implement `insert_embeddings_batch()` for efficient bulk inserts
   - Use SQLite transactions for atomicity

4. **Index Optimization**
   - Add index on `embeddings.symbol_id` if not exists
   - Consider adding file_id column for direct file queries

5. **Tests** (`crates/contextmcp-storage/src/sqlite.rs`)
   - Test embedding insert/retrieve roundtrip
   - Test vector search returns correct order
   - Test batch insert performance
   - Test cleanup on file deletion

## Acceptance Criteria

- [ ] `insert_embedding()` stores vectors correctly (already exists, verify)
- [ ] `get_embeddings_for_file()` retrieves embeddings for a specific file
- [ ] `search_similar_vectors()` returns results ordered by similarity
- [ ] Batch operations work within transactions
- [ ] Tests pass with `cargo test -p contextmcp-storage`
- [ ] No breaking changes to existing Database API

## Files to Create/Modify

- `crates/contextmcp-storage/src/sqlite.rs` - Add vector storage/search methods
- `crates/contextmcp-storage/src/lib.rs` - Export new types if needed

## Integration Points

- **Provides**: Vector storage and retrieval API for semantic search
- **Consumes**: None directly (uses existing SQLite infrastructure)
- **Conflicts**: Avoid modifying `tantivy_index.rs` (handled by hybrid-search)

## Technical Notes

- Existing schema: `embeddings(id, symbol_id, chunk_text, embedding BLOB, chunk_type)`
- BLOB format: f32 little-endian bytes
- Use existing `insert_embedding()` as reference for byte conversion
- May need to add `file_id` to embeddings table for efficient per-file queries
