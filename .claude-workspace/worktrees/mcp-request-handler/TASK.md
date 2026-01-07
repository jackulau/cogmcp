---
id: mcp-request-handler
name: MCP Request Handler and Router
priority: 2
dependencies: [mcp-message-types]
estimated_hours: 4
tags: [protocol, routing, handler]
---

## Objective

Implement request routing and handler dispatch for MCP protocol methods.

## Context

The MCP protocol defines several methods: initialize, tools/list, tools/call, etc. This subtask creates the router that maps incoming requests to the appropriate handlers and formats responses. It depends on the message types from subtask 1.

## Implementation

1. Create `/crates/contextmcp-server/src/handler/mod.rs` with:
   - `RequestHandler` struct holding reference to `ContextMcpServer`
   - Method dispatch based on request method name
   - Response building with proper JSON-RPC format

2. Create `/crates/contextmcp-server/src/handler/methods.rs`:
   - `handle_initialize()` - Return server info and capabilities
   - `handle_initialized()` - Acknowledge initialization complete
   - `handle_tools_list()` - Return available tools from server.rs
   - `handle_tools_call()` - Route to server.call_tool() and format result
   - `handle_ping()` - Simple ping response

3. Wire existing `ContextMcpServer::list_tools()` and `call_tool()` to handlers

4. Add unit tests for each handler method

## Acceptance Criteria

- [ ] `RequestHandler::new(server)` creates handler with server reference
- [ ] `handler.handle(request)` dispatches to correct method handler
- [ ] Initialize returns server info and capabilities
- [ ] tools/list returns tool definitions in MCP format
- [ ] tools/call invokes tool and wraps result in MCP CallToolResult
- [ ] Unknown methods return MethodNotFound error
- [ ] All tests pass: `cargo test -p contextmcp-server`

## Files to Create/Modify

- `crates/contextmcp-server/src/handler/mod.rs` - Handler struct and dispatch
- `crates/contextmcp-server/src/handler/methods.rs` - Individual method handlers
- `crates/contextmcp-server/src/lib.rs` - Add handler module export
- `crates/contextmcp-server/src/server.rs` - May need to adjust method signatures

## Integration Points

- **Provides**: Request handling for server loop
- **Consumes**: Protocol types from mcp-message-types, server methods from server.rs
- **Conflicts**: Minor changes to server.rs (coordinate with team)
