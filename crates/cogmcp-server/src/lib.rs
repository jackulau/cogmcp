//! CogMCP Server - MCP server implementation
//!
//! This crate implements the MCP server with all context tools.

pub mod handler;
pub mod protocol;
pub mod response_builder;
pub mod runner;
pub mod server;
pub mod status;
pub mod streaming;
pub mod tools;
pub mod transport;

pub use handler::{
    create_streaming_builder, format_search_results, format_streaming_response,
    should_use_streaming, RequestHandler,
};
pub use response_builder::{
    ProgressNotification, ProgressToken, StreamingResponseBuilder, StreamingThreshold, TextContent,
};
pub use runner::{RunnerConfig, ServerRunner};
pub use server::CogMcpServer;
pub use streaming::{FormattedResult, StreamingConfig, StreamingFormatter};
pub use transport::StdioTransport;
