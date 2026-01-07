//! CogMCP Watcher - File system watching
//!
//! This crate provides smart file watching with hot/warm/cold tiering
//! for efficient incremental updates.

pub mod realtime;
pub mod prioritizer;

pub use realtime::FileWatcher;
pub use prioritizer::FilePrioritizer;
