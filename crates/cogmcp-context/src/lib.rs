//! CogMCP Context - Context management and compression
//!
//! This crate provides intelligent context selection, compression,
//! and prioritization for efficient token usage.

pub mod prioritizer;
pub mod chunker;
pub mod formatter;

pub use prioritizer::ContextPrioritizer;
pub use chunker::ContextChunker;
pub use formatter::ContextFormatter;
