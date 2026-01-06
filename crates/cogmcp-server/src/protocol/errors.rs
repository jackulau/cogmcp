//! MCP error codes and error response builders
//!
//! This module defines JSON-RPC 2.0 error codes, MCP-specific error codes,
//! and utilities for building error responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// Standard JSON-RPC 2.0 error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Invalid JSON was received
    ParseError,
    /// The JSON sent is not a valid Request object
    InvalidRequest,
    /// The method does not exist or is not available
    MethodNotFound,
    /// Invalid method parameter(s)
    InvalidParams,
    /// Internal JSON-RPC error
    InternalError,
    /// Server-defined error (code in range -32000 to -32099)
    ServerError(i32),
}

impl ErrorCode {
    /// Get the numeric error code
    pub fn code(&self) -> i32 {
        match self {
            Self::ParseError => -32700,
            Self::InvalidRequest => -32600,
            Self::MethodNotFound => -32601,
            Self::InvalidParams => -32602,
            Self::InternalError => -32603,
            Self::ServerError(code) => *code,
        }
    }

    /// Get the default message for this error code
    pub fn default_message(&self) -> &'static str {
        match self {
            Self::ParseError => "Parse error",
            Self::InvalidRequest => "Invalid Request",
            Self::MethodNotFound => "Method not found",
            Self::InvalidParams => "Invalid params",
            Self::InternalError => "Internal error",
            Self::ServerError(_) => "Server error",
        }
    }

    /// Create an ErrorCode from a numeric code
    pub fn from_code(code: i32) -> Self {
        match code {
            -32700 => Self::ParseError,
            -32600 => Self::InvalidRequest,
            -32601 => Self::MethodNotFound,
            -32602 => Self::InvalidParams,
            -32603 => Self::InternalError,
            c if (-32099..=-32000).contains(&c) => Self::ServerError(c),
            c => Self::ServerError(c), // Allow non-standard codes
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.default_message())
    }
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code
    pub code: i32,

    /// Human-readable error message
    pub message: String,

    /// Optional additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// Create a new RPC error
    pub fn new(code: i32, message: String, data: Option<Value>) -> Self {
        Self {
            code,
            message,
            data,
        }
    }

    /// Create an error from an ErrorCode with default message
    pub fn from_code(code: ErrorCode) -> Self {
        Self {
            code: code.code(),
            message: code.default_message().to_string(),
            data: None,
        }
    }

    /// Create an error from an ErrorCode with a custom message
    pub fn from_code_with_message(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code.code(),
            message: message.into(),
            data: None,
        }
    }

    /// Create an error with additional data
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Create a parse error
    pub fn parse_error() -> Self {
        Self::from_code(ErrorCode::ParseError)
    }

    /// Create an invalid request error
    pub fn invalid_request() -> Self {
        Self::from_code(ErrorCode::InvalidRequest)
    }

    /// Create a method not found error
    pub fn method_not_found() -> Self {
        Self::from_code(ErrorCode::MethodNotFound)
    }

    /// Create a method not found error with the method name
    pub fn method_not_found_with_name(method: &str) -> Self {
        Self::from_code_with_message(
            ErrorCode::MethodNotFound,
            format!("Method not found: {}", method),
        )
    }

    /// Create an invalid params error
    pub fn invalid_params() -> Self {
        Self::from_code(ErrorCode::InvalidParams)
    }

    /// Create an invalid params error with details
    pub fn invalid_params_with_message(message: impl Into<String>) -> Self {
        Self::from_code_with_message(ErrorCode::InvalidParams, message)
    }

    /// Create an internal error
    pub fn internal_error() -> Self {
        Self::from_code(ErrorCode::InternalError)
    }

    /// Create an internal error with details
    pub fn internal_error_with_message(message: impl Into<String>) -> Self {
        Self::from_code_with_message(ErrorCode::InternalError, message)
    }

    /// Create a server error with a custom code and message
    pub fn server_error(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

/// High-level MCP error type for application use
#[derive(Debug, Clone)]
pub enum McpError {
    /// JSON-RPC protocol error
    Rpc(RpcError),

    /// Tool not found
    ToolNotFound(String),

    /// Resource not found
    ResourceNotFound(String),

    /// Prompt not found
    PromptNotFound(String),

    /// Invalid tool arguments
    InvalidToolArguments(String),

    /// Tool execution failed
    ToolExecutionError(String),

    /// Resource read error
    ResourceReadError(String),

    /// Not initialized
    NotInitialized,

    /// Already initialized
    AlreadyInitialized,

    /// Request cancelled
    Cancelled,

    /// Rate limited
    RateLimited,

    /// Custom error with code and message
    Custom { code: i32, message: String },
}

impl McpError {
    /// Convert this error to an RpcError for sending in a response
    pub fn to_rpc_error(&self) -> RpcError {
        match self {
            Self::Rpc(e) => e.clone(),
            Self::ToolNotFound(name) => RpcError::server_error(
                -32001,
                format!("Tool not found: {}", name),
            ),
            Self::ResourceNotFound(uri) => RpcError::server_error(
                -32002,
                format!("Resource not found: {}", uri),
            ),
            Self::PromptNotFound(name) => RpcError::server_error(
                -32003,
                format!("Prompt not found: {}", name),
            ),
            Self::InvalidToolArguments(msg) => {
                RpcError::invalid_params_with_message(format!("Invalid tool arguments: {}", msg))
            }
            Self::ToolExecutionError(msg) => RpcError::server_error(
                -32004,
                format!("Tool execution error: {}", msg),
            ),
            Self::ResourceReadError(msg) => RpcError::server_error(
                -32005,
                format!("Resource read error: {}", msg),
            ),
            Self::NotInitialized => RpcError::server_error(-32006, "Server not initialized"),
            Self::AlreadyInitialized => RpcError::server_error(-32007, "Server already initialized"),
            Self::Cancelled => RpcError::server_error(-32008, "Request cancelled"),
            Self::RateLimited => RpcError::server_error(-32009, "Rate limited"),
            Self::Custom { code, message } => RpcError::server_error(*code, message.clone()),
        }
    }

    /// Get the error code
    pub fn code(&self) -> i32 {
        self.to_rpc_error().code
    }
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rpc(e) => write!(f, "{}", e),
            Self::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            Self::ResourceNotFound(uri) => write!(f, "Resource not found: {}", uri),
            Self::PromptNotFound(name) => write!(f, "Prompt not found: {}", name),
            Self::InvalidToolArguments(msg) => write!(f, "Invalid tool arguments: {}", msg),
            Self::ToolExecutionError(msg) => write!(f, "Tool execution error: {}", msg),
            Self::ResourceReadError(msg) => write!(f, "Resource read error: {}", msg),
            Self::NotInitialized => write!(f, "Server not initialized"),
            Self::AlreadyInitialized => write!(f, "Server already initialized"),
            Self::Cancelled => write!(f, "Request cancelled"),
            Self::RateLimited => write!(f, "Rate limited"),
            Self::Custom { code, message } => write!(f, "[{}] {}", code, message),
        }
    }
}

impl std::error::Error for McpError {}

impl From<RpcError> for McpError {
    fn from(e: RpcError) -> Self {
        Self::Rpc(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_error_code_values() {
        assert_eq!(ErrorCode::ParseError.code(), -32700);
        assert_eq!(ErrorCode::InvalidRequest.code(), -32600);
        assert_eq!(ErrorCode::MethodNotFound.code(), -32601);
        assert_eq!(ErrorCode::InvalidParams.code(), -32602);
        assert_eq!(ErrorCode::InternalError.code(), -32603);
        assert_eq!(ErrorCode::ServerError(-32000).code(), -32000);
    }

    #[test]
    fn test_error_code_from_code() {
        assert_eq!(ErrorCode::from_code(-32700), ErrorCode::ParseError);
        assert_eq!(ErrorCode::from_code(-32600), ErrorCode::InvalidRequest);
        assert_eq!(ErrorCode::from_code(-32050), ErrorCode::ServerError(-32050));
    }

    #[test]
    fn test_rpc_error_serialization() {
        let error = RpcError::method_not_found_with_name("unknown/method");
        let json = serde_json::to_string(&error).unwrap();
        let parsed: RpcError = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.code, -32601);
        assert!(parsed.message.contains("unknown/method"));
    }

    #[test]
    fn test_rpc_error_with_data() {
        let error = RpcError::invalid_params_with_message("Missing required field")
            .with_data(json!({"field": "name"}));

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"data\""));

        let parsed: RpcError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.data.unwrap()["field"], "name");
    }

    #[test]
    fn test_mcp_error_to_rpc() {
        let mcp_error = McpError::ToolNotFound("ping".to_string());
        let rpc_error = mcp_error.to_rpc_error();

        assert_eq!(rpc_error.code, -32001);
        assert!(rpc_error.message.contains("ping"));
    }

    #[test]
    fn test_mcp_error_display() {
        let error = McpError::NotInitialized;
        assert_eq!(format!("{}", error), "Server not initialized");

        let error = McpError::ToolNotFound("search".to_string());
        assert!(format!("{}", error).contains("search"));
    }

    #[test]
    fn test_rpc_error_convenience_methods() {
        assert_eq!(RpcError::parse_error().code, -32700);
        assert_eq!(RpcError::invalid_request().code, -32600);
        assert_eq!(RpcError::method_not_found().code, -32601);
        assert_eq!(RpcError::invalid_params().code, -32602);
        assert_eq!(RpcError::internal_error().code, -32603);
    }
}
