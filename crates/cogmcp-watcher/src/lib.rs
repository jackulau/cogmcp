//! CogMCP Watcher - File system watching
//!
//! This crate provides smart file watching with hot/warm/cold tiering
//! for efficient incremental updates.

pub mod handler;
pub mod prioritizer;
pub mod realtime;

pub use handler::{FnCallback, IndexAction, IndexCallback, WatcherEventHandler};
pub use prioritizer::FilePrioritizer;
pub use realtime::{ChangeKind, FileChangeEvent, FileWatcher};
