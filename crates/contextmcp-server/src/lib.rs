//! ContextMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod handler;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod transport;

pub use handler::RequestHandler;
pub use protocol::{
    ErrorCode, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, McpMethod,
    RequestId, RpcError, MCP_VERSION,
};
pub use server::ContextMcpServer;
pub use transport::StdioTransport;
