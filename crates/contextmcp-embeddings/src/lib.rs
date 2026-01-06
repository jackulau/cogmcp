//! ContextMCP Embeddings - Text embedding generation using ONNX Runtime
//!
//! This crate provides text embedding generation using the all-MiniLM-L6-v2 model
//! via ONNX Runtime. It produces 384-dimensional semantic embeddings suitable
//! for similarity search and semantic retrieval.
//!
//! # Example
//!
//! ```no_run
//! use contextmcp_embeddings::{EmbeddingEngine, ModelManager};
//!
//! // Ensure model is downloaded
//! let manager = ModelManager::new().unwrap();
//! let config = manager.ensure_model_available().unwrap();
//!
//! // Create embedding engine (mutable for inference)
//! let mut engine = EmbeddingEngine::new(config).unwrap();
//!
//! // Generate embeddings
//! let embedding = engine.embed("Hello, world!").unwrap();
//! assert_eq!(embedding.len(), 384);
//!
//! // Compare similarity
//! let other = engine.embed("Hi there!").unwrap();
//! let similarity = EmbeddingEngine::cosine_similarity(&embedding, &other);
//! println!("Similarity: {}", similarity);
//! ```

pub mod inference;
pub mod model;
pub mod tokenizer;

pub use inference::EmbeddingEngine;
pub use model::{ModelConfig, ModelManager};
pub use tokenizer::{BatchTokenizedInput, TokenizedInput, Tokenizer};
