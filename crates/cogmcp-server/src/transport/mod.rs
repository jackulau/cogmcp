//! Transport layer for MCP communication
//!
//! This module provides async transport implementations for reading and writing
//! newline-delimited JSON messages over stdio.

mod stdio;

pub use stdio::StdioTransport;
