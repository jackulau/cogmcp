//! CogMCP Search - Text and semantic search
//!
//! This crate provides search functionality including full-text search,
//! semantic search with embeddings, and hybrid search combining both.

pub mod text;
pub mod semantic;
pub mod hybrid;
pub mod hnsw;

pub use hybrid::{HybridSearch, HybridSearchConfig, HybridSearchResult, MatchType, SearchMode};
pub use semantic::{ChunkType, SemanticSearch, SemanticSearchOptions, SemanticSearchResult};
pub use hnsw::{HnswConfig, HnswIndex, HnswSearchResult};
