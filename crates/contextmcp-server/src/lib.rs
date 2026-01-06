//! ContextMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod server;
pub mod tools;
pub mod transport;

pub use server::ContextMcpServer;
pub use transport::StdioTransport;
