//! CogMCP Index - Code parsing and indexing
//!
//! This crate provides file indexing, code parsing with tree-sitter,
//! symbol extraction, and git integration.

pub mod codebase;
pub mod parser;
pub mod parser_pool;
pub mod git;
pub mod dependencies;

pub use codebase::{CodebaseIndexer, IndexResult};
pub use parser::{CodeParser, ExtractedSymbol};
pub use parser_pool::{FileEntry, ParseResult, ParserPool, ParserPoolConfig};
