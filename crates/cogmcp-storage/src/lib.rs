//! CogMCP Storage - SQLite and Tantivy storage layer
//!
//! This crate provides persistent storage for the file index,
//! symbols, embeddings, and full-text search.

pub mod cache;
pub mod lru_cache;
pub mod sqlite;
pub mod tantivy_index;

pub use lru_cache::{LruCache, LruCacheWithTtl};
pub use sqlite::{
    deserialize_parameters, deserialize_type_params, serialize_parameters, serialize_type_params,
    Database, EmbeddingInput, EmbeddingRow, ExtendedSymbolMetadata, FileRow, IndexStats,
    ParameterInfo, SimilarityResult, SymbolRow,
};
pub use tantivy_index::FullTextIndex;
