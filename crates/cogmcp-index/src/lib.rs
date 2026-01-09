//! CogMCP Index - Code parsing and indexing
//!
//! This crate provides file indexing, code parsing with tree-sitter,
//! symbol extraction, and git integration.

pub mod codebase;
pub mod parser;
pub mod git;
pub mod git_scoring;
pub mod dependencies;

pub use codebase::{CodebaseIndexer, IndexResult};
pub use parser::{CodeParser, ExtractedSymbol};
pub use git_scoring::{GitActivityScore, GitActivityScorer, GitScoringConfig};
