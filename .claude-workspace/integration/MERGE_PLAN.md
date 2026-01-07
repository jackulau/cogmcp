---
target_branch: main
merge_order: [onnx-embedding-engine, vector-storage-layer, semantic-search-impl, hybrid-search-integration]
created: 2026-01-05
---

# Merge Plan

## Overview

This document specifies the merge order for the vector storage and semantic search feature branches. Follow this order to respect dependencies and avoid conflicts.

## Pre-Merge Checklist

Before starting merges, verify:
- [ ] All subtask branches have passing tests
- [ ] Code review completed for each branch
- [ ] No uncommitted changes in any worktree

## Merge Order (Dependency-Based)

Execute merges in this order to respect dependencies:

### Wave 1 (No Dependencies - Can Merge in Any Order)

1. **onnx-embedding-engine** (`parallel/onnx-embedding-engine`)
   - Provides: EmbeddingEngine API
   - Files: `crates/contextmcp-embeddings/*`
   - No conflicts expected

2. **vector-storage-layer** (`parallel/vector-storage-layer`)
   - Provides: Vector storage/search in SQLite
   - Files: `crates/contextmcp-storage/sqlite.rs`
   - No conflicts expected

### Wave 2 (Depends on Wave 1)

3. **semantic-search-impl** (`parallel/semantic-search-impl`)
   - Depends on: onnx-embedding-engine, vector-storage-layer
   - Files: `crates/contextmcp-search/semantic.rs`
   - May need dependency updates after Wave 1 merges

### Wave 3 (Depends on Wave 2)

4. **hybrid-search-integration** (`parallel/hybrid-search-integration`)
   - Depends on: semantic-search-impl
   - Files: Multiple crates (search, index, server)
   - Final integration - test thoroughly

## Merge Commands

```bash
# Ensure clean main branch
git checkout main
git pull origin main

# ═══════════════════════════════════════════════
# Wave 1: No dependencies (can merge in parallel)
# ═══════════════════════════════════════════════

# Merge onnx-embedding-engine
git merge --no-ff parallel/onnx-embedding-engine -m "feat: complete ONNX embedding engine

- Implement model download and management
- Complete ONNX Runtime inference
- Add tokenization and batch embedding
- Add unit tests for embedding generation"

# Verify after merge
cargo test -p contextmcp-embeddings

# Merge vector-storage-layer
git merge --no-ff parallel/vector-storage-layer -m "feat: add vector storage layer

- Add vector retrieval methods to SQLite
- Implement similarity search
- Add batch insert operations
- Add unit tests for vector storage"

# Verify after merge
cargo test -p contextmcp-storage

# ═══════════════════════════════════════════════
# Wave 2: Depends on Wave 1
# ═══════════════════════════════════════════════

# Merge semantic-search-impl
git merge --no-ff parallel/semantic-search-impl -m "feat: implement semantic search

- Complete SemanticSearch with embedding engine integration
- Add search options (min_similarity, filters)
- Implement query embedding caching
- Add unit tests for semantic search"

# Verify after merge
cargo test -p contextmcp-search

# ═══════════════════════════════════════════════
# Wave 3: Final integration
# ═══════════════════════════════════════════════

# Merge hybrid-search-integration
git merge --no-ff parallel/hybrid-search-integration -m "feat: integrate hybrid search and MCP tools

- Complete hybrid search with RRF scoring
- Add embedding generation to indexing pipeline
- Expose semantic search as MCP tool
- Add integration tests"

# Full verification
cargo test --workspace
cargo build --release
```

## Conflict Resolution

If conflicts occur during merge:

1. **Identify the conflicting files**
   ```bash
   git status
   ```

2. **For Cargo.toml conflicts** (most common):
   - Keep both dependency additions
   - Ensure version compatibility
   - Run `cargo check` after resolution

3. **For code conflicts**:
   - Prefer the newer (dependent) branch's changes
   - Ensure imports are updated correctly
   - Run tests after resolution

4. **After resolving**:
   ```bash
   git add <resolved-files>
   git commit -m "resolve: merge conflicts from <branch>"
   cargo test -p <affected-crate>
   ```

## Post-Merge Verification

After all merges complete:

```bash
# Full test suite
cargo test --workspace

# Build release
cargo build --release

# Integration test (manual)
./target/release/contextmcp --help

# Test semantic search (if test script exists)
# ./scripts/test-semantic-search.sh
```

## Rollback Procedure

If issues are found after merge:

```bash
# Find the commit before the problematic merge
git log --oneline -10

# Reset to that commit (preserves work in worktrees)
git reset --hard <commit-sha>

# Force push if already pushed (coordinate with team)
# git push --force-with-lease origin main
```

## Cleanup After Successful Merge

```bash
# Remove merged branches
git branch -d parallel/onnx-embedding-engine
git branch -d parallel/vector-storage-layer
git branch -d parallel/semantic-search-impl
git branch -d parallel/hybrid-search-integration

# Remove worktrees
git worktree remove .claude-workspace/worktrees/onnx-embedding-engine/worktree
git worktree remove .claude-workspace/worktrees/vector-storage-layer/worktree
git worktree remove .claude-workspace/worktrees/semantic-search-impl/worktree
git worktree remove .claude-workspace/worktrees/hybrid-search-integration/worktree

# Optionally remove remote branches
# git push origin --delete parallel/onnx-embedding-engine
# ...
```

## Merge Log

| Subtask | Merged At | Merged By | Commit SHA | Notes |
|---------|-----------|-----------|------------|-------|
| onnx-embedding-engine | - | - | - | pending |
| vector-storage-layer | - | - | - | pending |
| semantic-search-impl | - | - | - | pending |
| hybrid-search-integration | - | - | - | pending |
