//! ContextMCP Embeddings - Text embedding for semantic search
//!
//! This crate provides embedding generation for semantic search functionality.
//! It supports ONNX runtime for model inference when models are available,
//! or can operate in mock mode for testing.

pub mod model;
pub mod inference;

pub use model::ModelConfig;
pub use inference::EmbeddingEngine;
