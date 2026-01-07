---
id: hybrid-search-integration
name: Integrate Hybrid Search with Server
priority: 3
dependencies: [semantic-search-impl]
estimated_hours: 5
tags: [integration, search, server]
---

## Objective

Complete the hybrid search implementation combining keyword and semantic search, and integrate it into the MCP server with proper indexing pipeline.

## Context

The `contextmcp-search` crate has a partial `HybridSearch` implementation (keyword only). The server needs to generate embeddings during indexing and expose search via MCP tools.

## Implementation

1. **Hybrid Search Completion** (`crates/contextmcp-search/src/hybrid.rs`)
   - Integrate `SemanticSearch` for `SearchMode::Semantic`
   - Implement `SearchMode::Hybrid` using Reciprocal Rank Fusion (RRF)
   - Combine scores: `rrf_score = 1/(k + rank_keyword) + 1/(k + rank_semantic)`

2. **Indexing Pipeline** (`crates/contextmcp-index/src/codebase.rs`)
   - Add `EmbeddingEngine` to `CodebaseIndexer`
   - Generate embeddings during `index_file()`:
     - Use `Chunker` to split code into chunks
     - Generate embedding for each chunk
     - Store in database via `insert_embedding()`
   - Add config option to enable/disable embedding generation

3. **Server Integration** (`crates/contextmcp-server/src/server.rs`)
   - Add `EmbeddingEngine` to `ContextMcpServer`
   - Initialize during server startup
   - Update `index()` method to generate embeddings
   - Add semantic search MCP tool

4. **MCP Tool Implementation** (`crates/contextmcp-server/src/tools/`)
   - Create `semantic_search` tool:
     - Input: query string, optional limit
     - Output: list of matches with paths, snippets, scores
   - Update `hybrid_search` tool to support all modes

5. **Configuration** (`crates/contextmcp-core/src/config.rs`)
   - Verify `enable_embeddings` config option works
   - Add `search_mode` config option (keyword/semantic/hybrid)
   - Add `hybrid_weight` for balancing results

6. **Tests**
   - Test hybrid search combines results correctly
   - Test RRF scoring produces expected order
   - Test server integration end-to-end
   - Test MCP tool responses

## Acceptance Criteria

- [ ] `HybridSearch` supports all three SearchMode options
- [ ] RRF combining produces sensible merged results
- [ ] Indexing generates embeddings when enabled
- [ ] `semantic_search` MCP tool works correctly
- [ ] Configuration options control behavior
- [ ] Tests pass: `cargo test -p contextmcp-search -p contextmcp-server`
- [ ] Full integration test: index + search works end-to-end

## Files to Create/Modify

- `crates/contextmcp-search/src/hybrid.rs` - Complete hybrid implementation
- `crates/contextmcp-index/src/codebase.rs` - Add embedding generation
- `crates/contextmcp-server/src/server.rs` - Integrate embedding engine
- `crates/contextmcp-server/src/tools/mod.rs` - Add semantic search tool
- `crates/contextmcp-core/src/config.rs` - Add search config options

## Integration Points

- **Provides**: Complete semantic search feature via MCP
- **Consumes**: All previous subtasks
- **Conflicts**: None - final integration layer

## Technical Notes

- RRF constant k typically = 60
- Consider async indexing for large codebases
- May need progress reporting during embedding generation
- Test with small codebase first for quick iteration
