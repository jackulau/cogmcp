---
id: subtask-2-parser-integration
name: Enhance Tree-Sitter Parser Extraction
priority: 1
dependencies: []
estimated_hours: 4
tags: [index, parser, tree-sitter]
---

## Objective

Enhance the tree-sitter parser to extract richer symbol metadata and improve extraction accuracy for all supported languages.

## Context

The current `CodeParser` in `contextmcp-index` extracts basic symbol information but doesn't capture all available metadata from tree-sitter ASTs. This task improves extraction to include:
- Visibility modifiers
- Method/function parameters
- Return types
- Generic type parameters
- Nested symbol relationships
- More accurate signature generation

## Implementation

1. Enhance Rust parsing in `crates/contextmcp-index/src/parser.rs`:
   - Extract visibility from `visibility_modifier` nodes
   - Parse generic parameters from `type_parameters` nodes
   - Extract impl blocks and associate methods with their types
   - Capture const/static values where available

2. Enhance TypeScript/JavaScript parsing:
   - Extract export modifiers
   - Parse class member visibility (public/private/protected)
   - Extract interface implementations
   - Handle arrow functions and method expressions

3. Enhance Python parsing:
   - Detect `@staticmethod`, `@classmethod`, `@property` decorators
   - Extract type hints from annotations
   - Detect dunder methods and categorize appropriately

4. Improve signature generation:
   - Include parameter types in signatures
   - Include return types where available
   - Format generics properly

5. Add helper functions:
   - `extract_visibility(node)` - Get visibility from AST node
   - `extract_type_params(node)` - Get generic parameters
   - `extract_parameters(node)` - Get function parameters with types

6. Update tests for enhanced extraction

## Acceptance Criteria

- [ ] Rust parser extracts visibility modifiers correctly
- [ ] TypeScript parser extracts export/access modifiers
- [ ] Python parser detects decorators
- [ ] Signatures include parameter and return types
- [ ] Generic type parameters extracted for all languages
- [ ] Nested symbols correctly associate with parent
- [ ] All existing parser tests still pass
- [ ] New unit tests cover enhanced extraction
- [ ] `cargo test -p contextmcp-index` passes

## Files to Create/Modify

- `crates/contextmcp-index/src/parser.rs` - Enhance all extraction methods
- `crates/contextmcp-index/src/lib.rs` - Export new helper functions if needed

## Integration Points

- **Provides**: Enhanced symbol extraction with rich metadata
- **Consumes**: Tree-sitter language grammars (already available)
- **Conflicts**: Coordinate with subtask-1 for type definitions; avoid `crates/contextmcp-storage/`
