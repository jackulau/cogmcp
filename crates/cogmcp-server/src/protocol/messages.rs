//! JSON-RPC 2.0 message structures
//!
//! This module defines the core JSON-RPC 2.0 request, response, and notification types
//! used in the MCP protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::RpcError;

/// JSON-RPC 2.0 version string
pub const JSONRPC_VERSION: &str = "2.0";

/// Request ID type - can be a string, number, or null
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String ID
    String(String),
    /// Integer ID
    Number(i64),
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<i32> for RequestId {
    fn from(n: i32) -> Self {
        RequestId::Number(n as i64)
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestId::String(s) => write!(f, "{}", s),
            RequestId::Number(n) => write!(f, "{}", n),
        }
    }
}

/// JSON-RPC 2.0 Request
///
/// A request object sent from client to server (or vice versa) that expects a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,

    /// Unique identifier for this request
    pub id: RequestId,

    /// Method name to invoke
    pub method: String,

    /// Optional parameters for the method
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: id.into(),
            method: method.into(),
            params,
        }
    }

    /// Create a request with no parameters
    pub fn without_params(id: impl Into<RequestId>, method: impl Into<String>) -> Self {
        Self::new(id, method, None)
    }

    /// Create a request with JSON object parameters
    pub fn with_params(
        id: impl Into<RequestId>,
        method: impl Into<String>,
        params: Value,
    ) -> Self {
        Self::new(id, method, Some(params))
    }
}

/// JSON-RPC 2.0 Notification
///
/// A notification is a request that does not expect a response.
/// It has no `id` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,

    /// Method name to invoke
    pub method: String,

    /// Optional parameters for the method
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    /// Create a new JSON-RPC notification
    pub fn new(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
        }
    }

    /// Create a notification with no parameters
    pub fn without_params(method: impl Into<String>) -> Self {
        Self::new(method, None)
    }

    /// Create a notification with JSON object parameters
    pub fn with_params(method: impl Into<String>, params: Value) -> Self {
        Self::new(method, Some(params))
    }
}

/// JSON-RPC 2.0 Response
///
/// A response object sent in reply to a request.
/// Contains either a result or an error, but never both.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,

    /// Request ID this response corresponds to
    pub id: Option<RequestId>,

    /// Successful result value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,

    /// Error object if the request failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl JsonRpcResponse {
    /// Create a successful response
    pub fn success(id: impl Into<RequestId>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(id.into()),
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: Option<RequestId>, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Check if this response is successful
    pub fn is_success(&self) -> bool {
        self.result.is_some() && self.error.is_none()
    }

    /// Check if this response is an error
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// A generic JSON-RPC message that could be a request, notification, or response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    /// A request expecting a response
    Request(JsonRpcRequest),
    /// A notification (no response expected)
    Notification(JsonRpcNotification),
    /// A response to a previous request
    Response(JsonRpcResponse),
}

impl JsonRpcMessage {
    /// Try to parse a JSON string into a JSON-RPC message
    pub fn parse(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize this message to a JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize this message to a pretty-printed JSON string
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_request_serialization() {
        let request = JsonRpcRequest::with_params(
            1,
            "tools/list",
            json!({"cursor": null}),
        );

        let json = serde_json::to_string(&request).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.jsonrpc, "2.0");
        assert_eq!(parsed.id, RequestId::Number(1));
        assert_eq!(parsed.method, "tools/list");
    }

    #[test]
    fn test_request_with_string_id() {
        let request = JsonRpcRequest::without_params("req-123", "initialize");

        let json = serde_json::to_string(&request).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, RequestId::String("req-123".to_string()));
    }

    #[test]
    fn test_notification_serialization() {
        let notification = JsonRpcNotification::with_params(
            "notifications/initialized",
            json!({}),
        );

        let json = serde_json::to_string(&notification).unwrap();
        assert!(!json.contains("\"id\""));

        let parsed: JsonRpcNotification = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.method, "notifications/initialized");
    }

    #[test]
    fn test_success_response_serialization() {
        let response = JsonRpcResponse::success(
            1,
            json!({"tools": []}),
        );

        let json = serde_json::to_string(&response).unwrap();
        let parsed: JsonRpcResponse = serde_json::from_str(&json).unwrap();

        assert!(parsed.is_success());
        assert!(!parsed.is_error());
        assert_eq!(parsed.id, Some(RequestId::Number(1)));
    }

    #[test]
    fn test_error_response_serialization() {
        let error = RpcError::new(
            -32600,
            "Invalid Request".to_string(),
            None,
        );
        let response = JsonRpcResponse::error(Some(RequestId::Number(1)), error);

        let json = serde_json::to_string(&response).unwrap();
        let parsed: JsonRpcResponse = serde_json::from_str(&json).unwrap();

        assert!(parsed.is_error());
        assert!(!parsed.is_success());
        assert_eq!(parsed.error.unwrap().code, -32600);
    }

    #[test]
    fn test_request_id_display() {
        let string_id = RequestId::String("test-id".to_string());
        let number_id = RequestId::Number(42);

        assert_eq!(format!("{}", string_id), "test-id");
        assert_eq!(format!("{}", number_id), "42");
    }

    #[test]
    fn test_message_round_trip() {
        let original = JsonRpcRequest::with_params(
            "abc-123",
            "tools/call",
            json!({
                "name": "ping",
                "arguments": {}
            }),
        );

        let json = serde_json::to_string(&original).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(original.id, parsed.id);
        assert_eq!(original.method, parsed.method);
        assert_eq!(original.params, parsed.params);
    }
}
