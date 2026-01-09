//! CogMCP Index - Code parsing and indexing
//!
//! This crate provides file indexing, code parsing with tree-sitter,
//! symbol extraction, and git integration.

pub mod codebase;
pub mod parser;
pub mod git;
pub mod dependencies;
pub mod parallel_walker;

pub use codebase::{CodebaseIndexer, IndexProgress, IndexResult};
pub use parser::{CodeParser, ExtractedSymbol};
