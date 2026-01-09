//! CogMCP Watcher - File system watching
//!
//! This crate provides smart file watching with hot/warm/cold tiering
//! for efficient incremental updates, plus config file watching with
//! automatic reload support.

pub mod config_watcher;
pub mod prioritizer;
pub mod realtime;

pub use config_watcher::{
    AutoReloadingConfig, ConfigChangeEvent, ConfigChangeKind, ConfigWatcher, ConfigWatcherOptions,
    ReloadEvent, SharedConfigWatcher,
};
pub use prioritizer::FilePrioritizer;
pub use realtime::FileWatcher;
