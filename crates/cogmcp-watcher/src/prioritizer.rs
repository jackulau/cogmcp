//! File priority scoring and tiering

use cogmcp_core::types::PriorityTier;
use dashmap::DashMap;
use std::time::{Duration, Instant};

/// File access record
struct FileAccess {
    last_accessed: Instant,
    access_count: u32,
    is_entry_point: bool,
    reference_count: u32,
}

/// Manages file priorities for watching
pub struct FilePrioritizer {
    files: DashMap<String, FileAccess>,
    hot_threshold: Duration,
}

impl FilePrioritizer {
    /// Create a new prioritizer
    pub fn new(hot_threshold_seconds: u64) -> Self {
        Self {
            files: DashMap::new(),
            hot_threshold: Duration::from_secs(hot_threshold_seconds),
        }
    }

    /// Record a file access
    pub fn record_access(&self, path: &str) {
        self.files
            .entry(path.to_string())
            .and_modify(|access| {
                access.last_accessed = Instant::now();
                access.access_count += 1;
            })
            .or_insert(FileAccess {
                last_accessed: Instant::now(),
                access_count: 1,
                is_entry_point: false,
                reference_count: 0,
            });
    }

    /// Mark a file as an entry point (always hot)
    pub fn mark_entry_point(&self, path: &str) {
        self.files
            .entry(path.to_string())
            .and_modify(|access| {
                access.is_entry_point = true;
            })
            .or_insert(FileAccess {
                last_accessed: Instant::now(),
                access_count: 0,
                is_entry_point: true,
                reference_count: 0,
            });
    }

    /// Update reference count for a file
    pub fn set_reference_count(&self, path: &str, count: u32) {
        self.files
            .entry(path.to_string())
            .and_modify(|access| {
                access.reference_count = count;
            })
            .or_insert(FileAccess {
                last_accessed: Instant::now(),
                access_count: 0,
                is_entry_point: false,
                reference_count: count,
            });
    }

    /// Get the priority tier for a file
    pub fn get_tier(&self, path: &str) -> PriorityTier {
        if let Some(access) = self.files.get(path) {
            // Entry points are always hot
            if access.is_entry_point {
                return PriorityTier::Hot;
            }

            // Recently accessed files are hot
            if access.last_accessed.elapsed() < self.hot_threshold {
                return PriorityTier::Hot;
            }

            // Frequently referenced files are warm
            if access.reference_count > 5 || access.access_count > 10 {
                return PriorityTier::Warm;
            }
        }

        PriorityTier::Cold
    }

    /// Get all hot files
    pub fn get_hot_files(&self) -> Vec<String> {
        self.files
            .iter()
            .filter(|entry| self.get_tier(entry.key()) == PriorityTier::Hot)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Calculate priority score for a file (0.0 - 1.0)
    pub fn calculate_score(&self, path: &str) -> f32 {
        if let Some(access) = self.files.get(path) {
            let mut score = 0.0f32;

            // Entry point bonus
            if access.is_entry_point {
                score += 0.3;
            }

            // Recency score (decays over time)
            let age_secs = access.last_accessed.elapsed().as_secs_f32();
            let recency = (-age_secs / 3600.0).exp(); // 1 hour half-life
            score += 0.25 * recency;

            // Access frequency score
            let freq = (access.access_count as f32).ln_1p() / 10.0;
            score += 0.25 * freq.min(1.0);

            // Reference count score
            let refs = (access.reference_count as f32).ln_1p() / 5.0;
            score += 0.20 * refs.min(1.0);

            return score.min(1.0);
        }

        0.0
    }
}

impl Default for FilePrioritizer {
    fn default() -> Self {
        Self::new(300) // 5 minutes
    }
}
