---
id: mcp-server-loop
name: Main Server Loop Integration
priority: 3
dependencies: [mcp-message-types, mcp-stdio-transport, mcp-request-handler]
estimated_hours: 3
tags: [integration, main, server]
---

## Objective

Integrate all components into the main server loop that processes MCP requests over stdio.

## Context

This is the integration subtask that ties together the transport, protocol types, and request handler into a complete working MCP server. It replaces the TODO placeholder in main.rs with a proper message loop.

## Implementation

1. Modify `/crates/contextmcp-server/src/main.rs`:
   - Create StdioTransport
   - Create RequestHandler with server reference
   - Implement main loop: read request -> handle -> write response
   - Handle graceful shutdown on EOF
   - Add proper logging at info/debug levels

2. Create `/crates/contextmcp-server/src/runner.rs`:
   - `ServerRunner` struct combining all components
   - `run()` method with the main event loop
   - Error recovery for individual request failures
   - Shutdown handling

3. Add end-to-end test with scripted input/output

## Acceptance Criteria

- [ ] Server starts and waits for input on stdin
- [ ] Initialize handshake completes successfully
- [ ] tools/list returns all 7 tools
- [ ] tools/call executes tools and returns results
- [ ] Server handles malformed requests gracefully
- [ ] Server exits cleanly on EOF/shutdown
- [ ] All tests pass: `cargo test -p contextmcp-server`
- [ ] Manual test: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | cargo run`

## Files to Create/Modify

- `crates/contextmcp-server/src/main.rs` - Replace TODO with server loop
- `crates/contextmcp-server/src/runner.rs` - Server runner abstraction
- `crates/contextmcp-server/src/lib.rs` - Add runner module export

## Integration Points

- **Provides**: Complete working MCP server binary
- **Consumes**: All other subtasks (transport, protocol, handler)
- **Conflicts**: Modifies main.rs (sole owner of this file)
