//! CogMCP Core - Configuration, error types, and shared utilities
//!
//! This crate provides the foundational types and utilities used across
//! all CogMCP crates.

pub mod actionable_error;
pub mod config;
pub mod error;
pub mod streaming;
pub mod types;

pub use actionable_error::ActionableError;
pub use config::{Config, ReloadResult, SearchConfig, SharedConfig};
pub use error::{Error, Result};
pub use streaming::{ResultStream, StreamChunk, StreamingConfig, StreamingResult};
