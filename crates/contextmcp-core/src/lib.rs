//! ContextMCP Core - Configuration, error types, and shared utilities
//!
//! This crate provides the foundational types and utilities used across
//! all ContextMCP crates.

pub mod config;
pub mod error;
pub mod types;

pub use config::Config;
pub use error::{Error, Result};
