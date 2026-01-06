//! MCP Protocol types and structures
//!
//! This module defines JSON-RPC 2.0 message types and MCP-specific protocol structures.

pub mod errors;
pub mod messages;
pub mod methods;

pub use errors::{ErrorCode, McpError, RpcError};
pub use messages::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId};
pub use methods::{McpMethod, MCP_VERSION};
