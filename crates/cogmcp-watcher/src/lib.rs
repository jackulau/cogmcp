//! CogMCP Watcher - File system watching
//!
//! This crate provides smart file watching with hot/warm/cold tiering
//! for efficient incremental updates.

pub mod debouncer;
pub mod prioritizer;
pub mod realtime;

pub use debouncer::{spawn_debounce_checker, FileDebouncer};
pub use prioritizer::FilePrioritizer;
pub use realtime::FileWatcher;
