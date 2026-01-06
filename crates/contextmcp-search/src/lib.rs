//! ContextMCP Search - Text and semantic search
//!
//! This crate provides search functionality including full-text search,
//! semantic search with embeddings, and hybrid search combining both.

pub mod text;
pub mod semantic;
pub mod hybrid;

pub use hybrid::HybridSearch;
