//! MCP Request Handler and Router
//!
//! This module provides request routing and handler dispatch for MCP protocol methods.
//! It maps incoming JSON-RPC requests to the appropriate handlers and formats responses.

pub mod methods;

use crate::protocol::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpMethod, RpcError,
};
use crate::response_builder::{StreamingResponseBuilder, StreamingThreshold};
use crate::server::CogMcpServer;
use crate::streaming::{FormattedResult, StreamingConfig, StreamingFormatter};
use serde_json::Value;
use std::fmt::Display;
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

// ============================================================================
// Streaming Helper Functions
// ============================================================================

/// Create a streaming response for a collection of results
///
/// This helper formats results into chunks suitable for streaming MCP responses.
/// It uses the default streaming configuration.
pub fn format_streaming_response<T: Display>(
    results: &[T],
    threshold: Option<StreamingThreshold>,
) -> Value {
    let threshold = threshold.unwrap_or_default();
    let total_size: usize = results.iter().map(|r| r.to_string().len()).sum();

    // Check if we should use streaming
    if !threshold.should_stream(results.len(), total_size) {
        // Use simple single-chunk response
        let content: String = results
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n\n");

        return serde_json::json!({
            "content": [{
                "type": "text",
                "text": content
            }],
            "isError": false
        });
    }

    // Use streaming formatter for larger result sets
    let mut formatter = StreamingFormatter::new();
    let chunks = formatter.format_all(results);

    let mut builder = StreamingResponseBuilder::new()
        .with_total_items(results.len());

    builder.add_chunks(chunks);
    builder.build_combined()
}

/// Create a streaming response builder with custom configuration
///
/// Use this when you need more control over the streaming behavior.
pub fn create_streaming_builder(
    config: StreamingConfig,
    total_items: Option<usize>,
) -> (StreamingFormatter, StreamingResponseBuilder) {
    let formatter = StreamingFormatter::with_config(config);
    let mut builder = StreamingResponseBuilder::new();

    if let Some(total) = total_items {
        builder = builder.with_total_items(total);
    }

    (formatter, builder)
}

/// Format search results into a streaming-ready response
///
/// This is a convenience function specifically for search results.
pub fn format_search_results(
    results: Vec<FormattedResult>,
    query: &str,
) -> Value {
    if results.is_empty() {
        return serde_json::json!({
            "content": [{
                "type": "text",
                "text": format!("No results found for: {}", query)
            }],
            "isError": false
        });
    }

    let header = format!("## Search results for: {}\n\n", query);
    let threshold = StreamingThreshold::default();

    // Format each result
    let formatted: Vec<String> = results
        .iter()
        .map(|r| r.to_string())
        .collect();

    let total_size: usize = formatted.iter().map(|s| s.len()).sum();

    if !threshold.should_stream(results.len(), total_size) {
        // Simple response
        let content = format!("{}{}", header, formatted.join("\n\n"));
        return serde_json::json!({
            "content": [{
                "type": "text",
                "text": content
            }],
            "isError": false
        });
    }

    // Streaming response
    let mut builder = StreamingResponseBuilder::new()
        .with_total_items(results.len());

    builder.add_chunk(header);
    builder.add_chunks(formatted);

    builder.build_combined()
}

/// Check if streaming should be used for a given result set
pub fn should_use_streaming<T: Display>(
    results: &[T],
    threshold: Option<StreamingThreshold>,
) -> bool {
    let threshold = threshold.unwrap_or_default();
    let total_size: usize = results.iter().map(|r| r.to_string().len()).sum();
    threshold.should_stream(results.len(), total_size)
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

    // ========================================================================
    // Streaming Helper Tests
    // ========================================================================

    #[test]
    fn test_format_streaming_response_small() {
        let results = vec!["Result 1", "Result 2", "Result 3"];
        let response = format_streaming_response(&results, None);

        assert_eq!(response["isError"], false);
        assert!(response["content"].is_array());
        assert_eq!(response["content"].as_array().unwrap().len(), 1);

        let text = response["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Result 1"));
        assert!(text.contains("Result 2"));
        assert!(text.contains("Result 3"));
    }

    #[test]
    fn test_format_streaming_response_large() {
        // Create enough results to trigger streaming
        let results: Vec<String> = (0..20)
            .map(|i| format!("Result {} with some additional content to increase size", i))
            .collect();

        let threshold = StreamingThreshold::new(10, 100);
        let response = format_streaming_response(&results, Some(threshold));

        assert_eq!(response["isError"], false);
        assert!(response["content"].is_array());

        // Content should be present
        let text = response["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Result 0"));
        assert!(text.contains("Result 19"));
    }

    #[test]
    fn test_should_use_streaming_by_count() {
        let results: Vec<&str> = (0..15).map(|_| "item").collect();
        let threshold = StreamingThreshold::new(10, 10000);

        assert!(should_use_streaming(&results, Some(threshold)));
    }

    #[test]
    fn test_should_use_streaming_by_size() {
        let results = vec!["a".repeat(5000), "b".repeat(5000)];
        let threshold = StreamingThreshold::new(100, 8000);

        assert!(should_use_streaming(&results, Some(threshold)));
    }

    #[test]
    fn test_should_not_use_streaming() {
        let results = vec!["small", "results"];
        let threshold = StreamingThreshold::new(10, 10000);

        assert!(!should_use_streaming(&results, Some(threshold)));
    }

    #[test]
    fn test_format_search_results_empty() {
        let results: Vec<FormattedResult> = vec![];
        let response = format_search_results(results, "test query");

        let text = response["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No results found"));
        assert!(text.contains("test query"));
    }

    #[test]
    fn test_format_search_results_with_results() {
        let results = vec![
            FormattedResult::new("src/main.rs", "fn main() {}")
                .with_line(1)
                .with_score(0.95),
            FormattedResult::new("src/lib.rs", "pub mod test;")
                .with_line(5)
                .with_score(0.85),
        ];

        let response = format_search_results(results, "function definition");

        assert_eq!(response["isError"], false);
        let text = response["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Search results for: function definition"));
        assert!(text.contains("src/main.rs:1"));
        assert!(text.contains("fn main()"));
    }

    #[test]
    fn test_create_streaming_builder() {
        let config = StreamingConfig::default().with_chunk_size(2048);
        let (formatter, builder) = create_streaming_builder(config, Some(100));

        assert_eq!(formatter.total_items(), 0); // Not set until format_all or set_total
        assert_eq!(builder.items_processed(), 0);
    }
}
