---
id: subtask-3-storage-schema
name: Extend Storage Schema for Symbols
priority: 1
dependencies: []
estimated_hours: 3
tags: [storage, sqlite, schema]
---

## Objective

Extend the SQLite database schema and storage layer to persist enhanced symbol metadata from tree-sitter extraction.

## Context

The current `symbols` table stores basic symbol information. To support the enhanced tree-sitter extraction, the schema needs additional columns for visibility, modifiers, parent relationships, and type parameters. This task updates the schema and the corresponding Rust code.

## Implementation

1. Update SQLite schema in `crates/contextmcp-storage/src/sqlite.rs`:
   - Add new columns to `symbols` table:
     - `visibility TEXT` (null for unknown)
     - `is_async INTEGER DEFAULT 0`
     - `is_static INTEGER DEFAULT 0`
     - `is_abstract INTEGER DEFAULT 0`
     - `is_exported INTEGER DEFAULT 0`
     - `parent_symbol_id INTEGER REFERENCES symbols(id)`
     - `type_parameters TEXT` (JSON array)
     - `parameters TEXT` (JSON array of {name, type} objects)
     - `return_type TEXT`
   - Create index on `parent_symbol_id` for hierarchy queries

2. Create migration function:
   - `migrate_symbols_table_v2()` that adds columns if not present
   - Handle both fresh installs and upgrades gracefully

3. Update `Database` methods:
   - Modify `insert_symbol()` to accept extended metadata
   - Add `get_symbol_children(symbol_id)` to query nested symbols
   - Add `get_symbols_by_visibility(visibility)` for filtering
   - Update `get_file_symbols()` to return enhanced data

4. Add serialization helpers:
   - `serialize_type_params(Vec<String>) -> String` (JSON)
   - `deserialize_type_params(String) -> Vec<String>`
   - Similar for parameters

5. Update tests to verify new schema and methods

## Acceptance Criteria

- [ ] Schema migration adds new columns without data loss
- [ ] `insert_symbol()` persists all extended metadata
- [ ] `get_symbol_children()` returns nested symbols correctly
- [ ] JSON serialization for type_parameters and parameters works
- [ ] Existing queries continue to work (backward compatible)
- [ ] `cargo test -p contextmcp-storage` passes
- [ ] Fresh database creation includes new schema

## Files to Create/Modify

- `crates/contextmcp-storage/src/sqlite.rs` - Schema updates, new methods
- `crates/contextmcp-storage/src/lib.rs` - Export new types if needed

## Integration Points

- **Provides**: Persistent storage for enhanced symbol metadata
- **Consumes**: Extended types (will integrate after subtask-1)
- **Conflicts**: Avoid editing `crates/contextmcp-index/` (handled by subtask-2)
