//! CogMCP Storage - SQLite and Tantivy storage layer
//!
//! This crate provides persistent storage for the file index,
//! symbols, embeddings, and full-text search.

pub mod sqlite;
pub mod tantivy_index;
pub mod cache;

pub use sqlite::{Database, EmbeddingInput, EmbeddingRow, SimilarityResult, FileRow, SymbolRow, IndexStats};
pub use tantivy_index::FullTextIndex;
