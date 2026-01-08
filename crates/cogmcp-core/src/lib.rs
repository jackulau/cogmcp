//! CogMCP Core - Configuration, error types, and shared utilities
//!
//! This crate provides the foundational types and utilities used across
//! all CogMCP crates.

pub mod actionable_error;
pub mod config;
pub mod error;
pub mod types;

pub use actionable_error::{ActionableError, ErrorCode};
pub use config::{Config, SearchConfig};
pub use error::{Error, Result};
