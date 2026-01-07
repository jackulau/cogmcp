---
id: mcp-stdio-transport
name: Stdio Transport Layer
priority: 1
dependencies: []
estimated_hours: 4
tags: [transport, io, async]
---

## Objective

Implement async stdin/stdout transport for reading and writing newline-delimited JSON messages.

## Context

MCP servers communicate via stdio using newline-delimited JSON. This subtask creates the transport layer that handles low-level I/O without knowledge of the protocol semantics. It uses Tokio's async I/O for non-blocking operation.

## Implementation

1. Create `/crates/contextmcp-server/src/transport/mod.rs` with:
   - `StdioTransport` struct wrapping stdin/stdout
   - Async read method returning raw JSON strings
   - Async write method accepting serializable messages
   - Proper error handling for I/O failures

2. Create `/crates/contextmcp-server/src/transport/stdio.rs`:
   - Use `tokio::io::BufReader` for efficient line reading
   - Use `tokio::io::BufWriter` for buffered output
   - Handle partial reads and write flushing
   - Log messages at trace level for debugging

3. Add integration test with mock stdin/stdout

## Acceptance Criteria

- [ ] `StdioTransport::new()` creates transport from stdio handles
- [ ] `transport.read_message()` returns next JSON line from stdin
- [ ] `transport.write_message(msg)` serializes and writes to stdout with newline
- [ ] Handles EOF gracefully (returns None)
- [ ] Handles malformed input (returns error, continues)
- [ ] `cargo test -p contextmcp-server` passes
- [ ] Can be tested with mock readers/writers

## Files to Create/Modify

- `crates/contextmcp-server/src/transport/mod.rs` - Module exports
- `crates/contextmcp-server/src/transport/stdio.rs` - Stdio implementation
- `crates/contextmcp-server/src/lib.rs` - Add transport module export
- `crates/contextmcp-server/Cargo.toml` - Ensure tokio features are present

## Integration Points

- **Provides**: Transport layer for server loop to read/write messages
- **Consumes**: tokio async I/O, serde_json for serialization
- **Conflicts**: None - this is a new module with no file overlap
