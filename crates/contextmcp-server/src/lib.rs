//! CogMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod runner;
pub mod server;
pub mod tools;

pub use runner::{RunnerConfig, ServerRunner};
pub use server::CogMcpServer;
