//! CogMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod handler;
pub mod protocol;
pub mod server;
pub mod tools;

pub use handler::RequestHandler;
pub use server::CogMcpServer;
