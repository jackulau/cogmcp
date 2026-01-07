//! CogMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod server;
pub mod tools;
pub mod transport;

pub use server::CogMcpServer;
pub use transport::StdioTransport;
