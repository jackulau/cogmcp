---
id: mcp-message-types
name: MCP Message Types and Protocol Models
priority: 1
dependencies: []
estimated_hours: 3
tags: [core, protocol, types]
---

## Objective

Define JSON-RPC 2.0 message types and MCP protocol structures for request/response handling.

## Context

The MCP protocol uses JSON-RPC 2.0 as its transport format. This subtask creates the foundational types that all other MCP components will use. The `rmcp` crate provides some types, but we need additional structures for proper message routing and error handling.

## Implementation

1. Create `/crates/contextmcp-server/src/protocol.rs` with:
   - JSON-RPC 2.0 request/response/notification structures
   - MCP-specific message types (initialize, tools/list, tools/call, etc.)
   - Error response types with proper MCP error codes
   - Serialization/deserialization implementations using serde

2. Create `/crates/contextmcp-server/src/protocol/mod.rs` to organize:
   - `messages.rs` - Core JSON-RPC structures
   - `methods.rs` - MCP method constants and routing helpers
   - `errors.rs` - MCP error codes and error response builders

3. Add unit tests for serialization round-trips

## Acceptance Criteria

- [ ] JSON-RPC 2.0 Request struct with id, method, params fields
- [ ] JSON-RPC 2.0 Response struct with id, result/error fields
- [ ] JSON-RPC 2.0 Notification struct (no id)
- [ ] MCP Error codes defined (ParseError, InvalidRequest, MethodNotFound, etc.)
- [ ] All types implement Serialize/Deserialize
- [ ] Unit tests pass for serialization round-trips
- [ ] `cargo test -p contextmcp-server` passes

## Files to Create/Modify

- `crates/contextmcp-server/src/protocol/mod.rs` - Module organization
- `crates/contextmcp-server/src/protocol/messages.rs` - JSON-RPC structures
- `crates/contextmcp-server/src/protocol/methods.rs` - MCP method constants
- `crates/contextmcp-server/src/protocol/errors.rs` - Error types and codes
- `crates/contextmcp-server/src/lib.rs` - Add protocol module export

## Integration Points

- **Provides**: Message types for stdio transport and request handler
- **Consumes**: serde, serde_json from workspace dependencies
- **Conflicts**: None - this is a new module with no file overlap
