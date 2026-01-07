---
id: subtask-1-symbol-types
name: Extend Symbol Types for Tree-Sitter
priority: 1
dependencies: []
estimated_hours: 2
tags: [core, types]
---

## Objective

Extend the core type system to fully support tree-sitter extracted symbol metadata for indexing.

## Context

The current `SymbolKind` and `ExtractedSymbol` types in `contextmcp-core` need to be enhanced to capture all the rich metadata that tree-sitter can extract. This includes better support for:
- Visibility/access modifiers (public, private, protected)
- Symbol relationships (parent class/module, implementations)
- Generic type parameters
- Additional symbol kinds (constants, type aliases, macros)

This is a foundational task that other subtasks depend on for type definitions.

## Implementation

1. Extend `SymbolKind` enum in `crates/contextmcp-core/src/types.rs`:
   - Add missing kinds: `Macro`, `TypeAlias`, `Constant`, `Static`, `Field`, `Parameter`
   - Ensure all tree-sitter extractable symbol types are covered

2. Create new types for symbol metadata:
   - `SymbolVisibility` enum (Public, Private, Protected, Internal, Crate)
   - `SymbolModifiers` struct (is_async, is_static, is_abstract, etc.)

3. Extend `ExtractedSymbol` in `crates/contextmcp-index/src/parser.rs`:
   - Add `visibility: Option<SymbolVisibility>`
   - Add `modifiers: SymbolModifiers`
   - Add `parent_symbol: Option<String>` for nested symbols
   - Add `type_parameters: Vec<String>` for generics

4. Update `SymbolInfo` in `crates/contextmcp-core/src/types.rs`:
   - Add fields to match extended `ExtractedSymbol`
   - Ensure conversion between types is seamless

5. Add unit tests for new types in `crates/contextmcp-core/src/types.rs`

## Acceptance Criteria

- [ ] `SymbolKind` enum covers all tree-sitter extractable symbol types
- [ ] `SymbolVisibility` enum defined with all common visibility levels
- [ ] `SymbolModifiers` struct captures boolean modifiers
- [ ] `ExtractedSymbol` extended with new metadata fields
- [ ] All new types derive necessary traits (Debug, Clone, Serialize, etc.)
- [ ] Unit tests pass for new type conversions
- [ ] `cargo build` succeeds with no warnings

## Files to Create/Modify

- `crates/contextmcp-core/src/types.rs` - Extend SymbolKind, add SymbolVisibility, SymbolModifiers
- `crates/contextmcp-index/src/parser.rs` - Update ExtractedSymbol struct

## Integration Points

- **Provides**: Extended type definitions for symbol metadata
- **Consumes**: None (core dependency)
- **Conflicts**: Avoid editing `crates/contextmcp-storage/` (handled by subtask-3)
