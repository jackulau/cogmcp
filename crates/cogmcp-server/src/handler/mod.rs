//! MCP Request Handler and Router
//!
//! This module provides request routing and handler dispatch for MCP protocol methods.
//! It maps incoming JSON-RPC requests to the appropriate handlers and formats responses.

pub mod methods;

use crate::protocol::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpMethod, RpcError,
};
use crate::server::CogMcpServer;
use serde_json::Value;
use std::sync::Arc;

pub use methods::*;

/// Request handler for MCP protocol
///
/// Holds a reference to the server and dispatches incoming requests
/// to the appropriate method handlers.
pub struct RequestHandler {
    server: Arc<CogMcpServer>,
    initialized: bool,
}

impl RequestHandler {
    /// Create a new request handler with a reference to the server
    pub fn new(server: Arc<CogMcpServer>) -> Self {
        Self {
            server,
            initialized: false,
        }
    }

    /// Handle an incoming JSON-RPC request and return a response
    pub fn handle(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();
        let method_name = &request.method;

        // Parse the method name
        let method = match McpMethod::from_str(method_name) {
            Some(m) => m,
            None => {
                return JsonRpcResponse::error(
                    Some(id),
                    RpcError::method_not_found_with_name(method_name),
                );
            }
        };

        // Check initialization state (initialize must be called first)
        if !self.initialized && method != McpMethod::Initialize {
            return JsonRpcResponse::error(
                Some(id),
                RpcError::server_error(-32006, "Server not initialized"),
            );
        }

        // Dispatch to the appropriate handler
        let result = match method {
            McpMethod::Initialize => self.handle_initialize(request.params),
            McpMethod::Ping => self.handle_ping(),
            McpMethod::ToolsList => self.handle_tools_list(request.params),
            McpMethod::ToolsCall => self.handle_tools_call(request.params),
            // Methods not yet implemented
            McpMethod::ResourcesList
            | McpMethod::ResourcesRead
            | McpMethod::ResourcesTemplatesList
            | McpMethod::ResourcesSubscribe
            | McpMethod::ResourcesUnsubscribe
            | McpMethod::PromptsList
            | McpMethod::PromptsGet
            | McpMethod::LoggingSetLevel
            | McpMethod::SamplingCreateMessage
            | McpMethod::RootsList
            | McpMethod::CompletionComplete => {
                Err(RpcError::method_not_found_with_name(method_name))
            }
            // Notifications shouldn't reach here as requests
            McpMethod::Initialized
            | McpMethod::Cancelled
            | McpMethod::Progress
            | McpMethod::ResourcesUpdated
            | McpMethod::ResourcesListChanged
            | McpMethod::LoggingMessage
            | McpMethod::RootsListChanged => {
                Err(RpcError::invalid_request())
            }
        };

        match result {
            Ok(value) => JsonRpcResponse::success(id, value),
            Err(error) => JsonRpcResponse::error(Some(id), error),
        }
    }

    /// Handle a JSON-RPC notification (no response expected)
    pub fn handle_notification(&mut self, notification: JsonRpcNotification) {
        let method_name = &notification.method;

        let method = match McpMethod::from_str(method_name) {
            Some(m) => m,
            None => return, // Ignore unknown notifications
        };

        match method {
            McpMethod::Initialized => self.handle_initialized(notification.params),
            McpMethod::Cancelled => self.handle_cancelled(notification.params),
            McpMethod::Progress => self.handle_progress(notification.params),
            _ => {} // Ignore non-notification methods sent as notifications
        }
    }

    /// Handle initialize request
    fn handle_initialize(&mut self, params: Option<Value>) -> Result<Value, RpcError> {
        handle_initialize(&self.server, params, &mut self.initialized)
    }

    /// Handle initialized notification
    fn handle_initialized(&mut self, _params: Option<Value>) {
        // Notification that client has completed initialization
        // Nothing to do here for now
    }

    /// Handle cancelled notification
    fn handle_cancelled(&mut self, _params: Option<Value>) {
        // Handle request cancellation
        // Not implemented yet
    }

    /// Handle progress notification
    fn handle_progress(&mut self, _params: Option<Value>) {
        // Handle progress updates
        // Not implemented yet
    }

    /// Handle ping request
    fn handle_ping(&self) -> Result<Value, RpcError> {
        handle_ping()
    }

    /// Handle tools/list request
    fn handle_tools_list(&self, params: Option<Value>) -> Result<Value, RpcError> {
        handle_tools_list(&self.server, params)
    }

    /// Handle tools/call request
    fn handle_tools_call(&self, params: Option<Value>) -> Result<Value, RpcError> {
        handle_tools_call(&self.server, params)
    }

    /// Check if the handler has been initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn create_test_handler() -> RequestHandler {
        let server = Arc::new(
            CogMcpServer::in_memory(PathBuf::from("/tmp/test"))
                .expect("Failed to create test server"),
        );
        RequestHandler::new(server)
    }

    #[test]
    fn test_handler_creation() {
        let handler = create_test_handler();
        assert!(!handler.is_initialized());
    }

    #[test]
    fn test_handle_before_initialize() {
        let mut handler = create_test_handler();
        let request = JsonRpcRequest::without_params(1, "tools/list");

        let response = handler.handle(request);

        assert!(response.is_error());
        assert_eq!(response.error.unwrap().code, -32006);
    }

    #[test]
    fn test_handle_unknown_method() {
        let mut handler = create_test_handler();

        // First initialize
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Then test unknown method
        let request = JsonRpcRequest::without_params(2, "unknown/method");
        let response = handler.handle(request);

        assert!(response.is_error());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_handle_initialize() {
        let mut handler = create_test_handler();
        let request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );

        let response = handler.handle(request);

        assert!(response.is_success());
        assert!(handler.is_initialized());

        let result = response.result.unwrap();
        assert!(result.get("protocolVersion").is_some());
        assert!(result.get("capabilities").is_some());
        assert!(result.get("serverInfo").is_some());
    }

    #[test]
    fn test_handle_ping() {
        let mut handler = create_test_handler();

        // Initialize first
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Test ping
        let request = JsonRpcRequest::without_params(2, "ping");
        let response = handler.handle(request);

        assert!(response.is_success());
    }

    #[test]
    fn test_handle_tools_list() {
        let mut handler = create_test_handler();

        // Initialize first
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Test tools/list
        let request = JsonRpcRequest::without_params(2, "tools/list");
        let response = handler.handle(request);

        assert!(response.is_success());

        let result = response.result.unwrap();
        assert!(result.get("tools").is_some());
        assert!(result["tools"].is_array());
    }

    #[test]
    fn test_handle_tools_call() {
        let mut handler = create_test_handler();

        // Initialize first
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Test tools/call with ping tool
        let request = JsonRpcRequest::with_params(
            2,
            "tools/call",
            json!({
                "name": "ping",
                "arguments": {}
            }),
        );
        let response = handler.handle(request);

        assert!(response.is_success());

        let result = response.result.unwrap();
        assert!(result.get("content").is_some());
        assert!(result["content"].is_array());
    }

    #[test]
    fn test_handle_tools_call_unknown_tool() {
        let mut handler = create_test_handler();

        // Initialize first
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Test tools/call with unknown tool
        let request = JsonRpcRequest::with_params(
            2,
            "tools/call",
            json!({
                "name": "unknown_tool",
                "arguments": {}
            }),
        );
        let response = handler.handle(request);

        // The tool returns an error result, not an RPC error
        assert!(response.is_success());
        let result = response.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn test_handle_notification_initialized() {
        let mut handler = create_test_handler();

        // Initialize first
        let init_request = JsonRpcRequest::with_params(
            1,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }),
        );
        handler.handle(init_request);

        // Send initialized notification (should not error)
        let notification =
            JsonRpcNotification::without_params("notifications/initialized");
        handler.handle_notification(notification);

        // Handler should still be in valid state
        assert!(handler.is_initialized());
    }
}
