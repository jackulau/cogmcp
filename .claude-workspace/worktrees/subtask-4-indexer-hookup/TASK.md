---
id: subtask-4-indexer-hookup
name: Hook Parser into Indexing Pipeline
priority: 2
dependencies: [subtask-1-symbol-types, subtask-2-parser-integration, subtask-3-storage-schema]
estimated_hours: 3
tags: [index, integration]
---

## Objective

Integrate the enhanced tree-sitter symbol extraction into the indexing pipeline, ensuring parsed symbols flow through to storage correctly.

## Context

This is the integration task that connects the enhanced parser (subtask-2) with the extended storage (subtask-3) using the new types (subtask-1). The `CodebaseIndexer` needs to be updated to:
- Call the enhanced parser methods
- Transform extracted symbols to storage format
- Persist symbols with full metadata
- Handle symbol relationships (parent/child)

## Implementation

1. Update `CodebaseIndexer` in `crates/contextmcp-index/src/codebase.rs`:
   - Modify `index_file()` to extract enhanced symbol metadata
   - Pass visibility, modifiers, and type params to storage
   - Handle nested symbols by establishing parent relationships

2. Create symbol relationship handling:
   - Track parent symbol IDs during extraction
   - Insert parent symbols first, then children with references
   - Handle impl blocks associating methods with structs/traits

3. Update re-indexing flow:
   - Clear existing symbols for a file before re-indexing
   - Ensure symbol relationships are updated correctly
   - Handle renamed/moved symbols gracefully

4. Add indexing statistics:
   - Track symbols indexed by kind
   - Track symbols with visibility extracted
   - Report extraction coverage in `index_status`

5. Update server tools:
   - `get_file_outline` should return enhanced symbol info
   - `find_symbol` should support filtering by visibility
   - `index_status` should show enhanced metrics

6. Integration tests:
   - Index a sample Rust file, verify all symbols extracted
   - Index a sample TypeScript file, verify exports detected
   - Test re-indexing preserves relationships

## Acceptance Criteria

- [ ] `index_file()` extracts and stores enhanced metadata
- [ ] Nested symbols correctly reference their parents
- [ ] Re-indexing updates all symbol metadata
- [ ] `get_file_outline` returns visibility and modifiers
- [ ] `index_status` shows symbol extraction metrics
- [ ] Full `cargo build` succeeds
- [ ] `cargo test` passes all tests
- [ ] Integration test with real code files passes

## Files to Create/Modify

- `crates/contextmcp-index/src/codebase.rs` - Update indexing pipeline
- `crates/contextmcp-server/src/server.rs` - Update tool handlers
- `crates/contextmcp-index/src/lib.rs` - Export integration helpers

## Integration Points

- **Provides**: Complete tree-sitter to index pipeline
- **Consumes**:
  - Extended types from subtask-1
  - Enhanced parser from subtask-2
  - Extended storage from subtask-3
- **Conflicts**: This task integrates all others - work on this last
