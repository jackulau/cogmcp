//! CogMCP Context - Context management and compression
//!
//! This crate provides intelligent context selection, compression,
//! and prioritization for efficient token usage.

pub mod api;
pub mod chunker;
pub mod formatter;
pub mod prioritizer;

pub use api::{
    calculate_file_priorities, FileScoreMetadata, PriorityQuery, PriorityResult, ScoreBreakdown,
};
pub use chunker::ContextChunker;
pub use formatter::ContextFormatter;
pub use prioritizer::{ContextPrioritizer, PriorityWeights};
