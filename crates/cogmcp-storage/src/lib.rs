//! CogMCP Storage - SQLite and Tantivy storage layer
//!
//! This crate provides persistent storage for the file index,
//! symbols, embeddings, and full-text search.

pub mod cache;
pub mod pool;
pub mod sqlite;
pub mod tantivy_index;

pub use pool::{ConnectionPool, PoolConfig, PoolState};
pub use sqlite::{
    deserialize_parameters, deserialize_type_params, serialize_parameters, serialize_type_params,
    Database, EmbeddingInput, EmbeddingRow, ExtendedIndexStats, ExtendedSymbolMetadata, FileInsert,
    FileRow, IndexStats, ParameterInfo, SimilarityResult, SymbolInsert, SymbolRow,
};
pub use tantivy_index::FullTextIndex;
