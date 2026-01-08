//! CogMCP Core - Configuration, error types, and shared utilities
//!
//! This crate provides the foundational types and utilities used across
//! all CogMCP crates.

pub mod config;
pub mod error;
pub mod types;

pub use config::{Config, SearchConfig};
pub use error::{ActionableError, Error, ErrorCode, Result};
