//! Individual MCP method handlers
//!
//! This module contains the implementation of each MCP method handler.
//! These functions are called by the RequestHandler to process specific requests.

use crate::protocol::{RpcError, MCP_VERSION};
use crate::server::CogMcpServer;
use serde_json::{json, Value};

/// Handle the initialize request
///
/// Returns server info and capabilities, and marks the server as initialized.
pub fn handle_initialize(
    server: &CogMcpServer,
    params: Option<Value>,
    initialized: &mut bool,
) -> Result<Value, RpcError> {
    // Check if already initialized
    if *initialized {
        return Err(RpcError::server_error(-32007, "Server already initialized"));
    }

    // Validate client protocol version if provided
    if let Some(ref p) = params {
        if let Some(client_version) = p.get("protocolVersion").and_then(|v| v.as_str()) {
            // We accept any version for now, but could validate here
            tracing::info!("Client protocol version: {}", client_version);
        }

        // Log client info if provided
        if let Some(client_info) = p.get("clientInfo") {
            if let Some(name) = client_info.get("name").and_then(|v| v.as_str()) {
                tracing::info!("Client: {}", name);
            }
        }
    }

    // Mark as initialized
    *initialized = true;

    // Build server info response
    let server_info = CogMcpServer::server_info();
    let capabilities = CogMcpServer::capabilities();

    Ok(json!({
        "protocolVersion": MCP_VERSION,
        "capabilities": capabilities,
        "serverInfo": {
            "name": server_info.name,
            "version": server_info.version
        },
        "instructions": format!(
            "CogMCP provides intelligent code context for AI assistants. \
             Root: {}",
            server.root.display()
        )
    }))
}

/// Handle the initialized notification
///
/// This is a notification sent by the client after it has processed
/// the initialize response. It indicates the client is ready.
pub fn handle_initialized() {
    tracing::info!("Client initialization complete");
}

/// Handle the ping request
///
/// Simple ping/pong for health checks.
pub fn handle_ping() -> Result<Value, RpcError> {
    Ok(json!({}))
}

/// Handle the tools/list request
///
/// Returns the list of available tools in MCP format.
pub fn handle_tools_list(server: &CogMcpServer, params: Option<Value>) -> Result<Value, RpcError> {
    // Handle pagination cursor if provided (not implemented yet)
    let _cursor = params
        .as_ref()
        .and_then(|p| p.get("cursor"))
        .and_then(|c| c.as_str());

    let tools = server.list_tools();

    // Convert tools to JSON format
    let tools_json: Vec<Value> = tools
        .into_iter()
        .map(|tool| {
            json!({
                "name": tool.name.as_ref(),
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        })
        .collect();

    Ok(json!({
        "tools": tools_json
    }))
}

/// Handle the tools/call request
///
/// Invokes a tool and returns the result in MCP CallToolResult format.
pub fn handle_tools_call(server: &CogMcpServer, params: Option<Value>) -> Result<Value, RpcError> {
    let params = params.ok_or_else(|| {
        RpcError::invalid_params_with_message("Missing params for tools/call")
    })?;

    // Extract tool name
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcError::invalid_params_with_message("Missing 'name' parameter"))?;

    // Extract arguments (default to empty object)
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    // Call the tool
    let result = server.call_tool(name, arguments);

    // Format as CallToolResult
    match result {
        Ok(content) => Ok(json!({
            "content": [{
                "type": "text",
                "text": content
            }],
            "isError": false
        })),
        Err(error) => Ok(json!({
            "content": [{
                "type": "text",
                "text": error
            }],
            "isError": true
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn create_test_server() -> Arc<CogMcpServer> {
        Arc::new(
            CogMcpServer::in_memory(PathBuf::from("/tmp/test"))
                .expect("Failed to create test server"),
        )
    }

    #[test]
    fn test_handle_initialize_success() {
        let server = create_test_server();
        let mut initialized = false;

        let params = Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }));

        let result = handle_initialize(&server, params, &mut initialized);

        assert!(result.is_ok());
        assert!(initialized);

        let value = result.unwrap();
        assert_eq!(value["protocolVersion"], MCP_VERSION);
        assert!(value.get("capabilities").is_some());
        assert!(value.get("serverInfo").is_some());
        assert_eq!(value["serverInfo"]["name"], "cogmcp");
    }

    #[test]
    fn test_handle_initialize_already_initialized() {
        let server = create_test_server();
        let mut initialized = true;

        let params = Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }));

        let result = handle_initialize(&server, params, &mut initialized);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, -32007);
    }

    #[test]
    fn test_handle_ping() {
        let result = handle_ping();

        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.is_object());
    }

    #[test]
    fn test_handle_tools_list() {
        let server = create_test_server();
        let result = handle_tools_list(&server, None);

        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value.get("tools").is_some());

        let tools = value["tools"].as_array().unwrap();
        assert!(!tools.is_empty());

        // Check that tools have required fields
        for tool in tools {
            assert!(tool.get("name").is_some());
            assert!(tool.get("inputSchema").is_some());
        }
    }

    #[test]
    fn test_handle_tools_list_with_cursor() {
        let server = create_test_server();
        let params = Some(json!({
            "cursor": "some-cursor"
        }));

        let result = handle_tools_list(&server, params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_handle_tools_call_ping() {
        let server = create_test_server();
        let params = Some(json!({
            "name": "ping",
            "arguments": {}
        }));

        let result = handle_tools_call(&server, params);

        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value.get("content").is_some());
        assert_eq!(value["isError"], false);

        let content = value["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
    }

    #[test]
    fn test_handle_tools_call_missing_name() {
        let server = create_test_server();
        let params = Some(json!({
            "arguments": {}
        }));

        let result = handle_tools_call(&server, params);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.code, -32602); // Invalid params
    }

    #[test]
    fn test_handle_tools_call_no_params() {
        let server = create_test_server();
        let result = handle_tools_call(&server, None);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.code, -32602); // Invalid params
    }

    #[test]
    fn test_handle_tools_call_unknown_tool() {
        let server = create_test_server();
        let params = Some(json!({
            "name": "nonexistent_tool",
            "arguments": {}
        }));

        let result = handle_tools_call(&server, params);

        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["isError"], true);
    }

    #[test]
    fn test_handle_tools_call_default_arguments() {
        let server = create_test_server();
        let params = Some(json!({
            "name": "ping"
        }));

        let result = handle_tools_call(&server, params);

        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["isError"], false);
    }
}
