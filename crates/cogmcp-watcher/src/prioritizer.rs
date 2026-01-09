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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_new_prioritizer() {
        let prioritizer = FilePrioritizer::new(60);
        assert_eq!(prioritizer.hot_threshold, Duration::from_secs(60));
        assert!(prioritizer.files.is_empty());
    }

    #[test]
    fn test_default_prioritizer() {
        let prioritizer = FilePrioritizer::default();
        assert_eq!(prioritizer.hot_threshold, Duration::from_secs(300));
    }

    #[test]
    fn test_record_access_new_file() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("test.rs");

        assert!(prioritizer.files.contains_key("test.rs"));
        let access = prioritizer.files.get("test.rs").unwrap();
        assert_eq!(access.access_count, 1);
        assert!(!access.is_entry_point);
        assert_eq!(access.reference_count, 0);
    }

    #[test]
    fn test_record_access_increments_count() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("test.rs");
        prioritizer.record_access("test.rs");
        prioritizer.record_access("test.rs");

        let access = prioritizer.files.get("test.rs").unwrap();
        assert_eq!(access.access_count, 3);
    }

    #[test]
    fn test_mark_entry_point_new_file() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.mark_entry_point("main.rs");

        let access = prioritizer.files.get("main.rs").unwrap();
        assert!(access.is_entry_point);
        assert_eq!(access.access_count, 0);
    }

    #[test]
    fn test_mark_entry_point_existing_file() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("main.rs");
        prioritizer.record_access("main.rs");
        prioritizer.mark_entry_point("main.rs");

        let access = prioritizer.files.get("main.rs").unwrap();
        assert!(access.is_entry_point);
        assert_eq!(access.access_count, 2);
    }

    #[test]
    fn test_set_reference_count_new_file() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.set_reference_count("lib.rs", 10);

        let access = prioritizer.files.get("lib.rs").unwrap();
        assert_eq!(access.reference_count, 10);
        assert!(!access.is_entry_point);
    }

    #[test]
    fn test_set_reference_count_existing_file() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("lib.rs");
        prioritizer.set_reference_count("lib.rs", 15);

        let access = prioritizer.files.get("lib.rs").unwrap();
        assert_eq!(access.reference_count, 15);
        assert_eq!(access.access_count, 1);
    }

    #[test]
    fn test_get_tier_unknown_file_is_cold() {
        let prioritizer = FilePrioritizer::new(60);
        assert_eq!(prioritizer.get_tier("unknown.rs"), PriorityTier::Cold);
    }

    #[test]
    fn test_get_tier_entry_point_is_hot() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.mark_entry_point("main.rs");
        assert_eq!(prioritizer.get_tier("main.rs"), PriorityTier::Hot);
    }

    #[test]
    fn test_get_tier_recently_accessed_is_hot() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("recent.rs");
        assert_eq!(prioritizer.get_tier("recent.rs"), PriorityTier::Hot);
    }

    #[test]
    fn test_get_tier_high_reference_count_is_warm() {
        let prioritizer = FilePrioritizer::new(0); // 0 second threshold - immediately cold by time
        prioritizer.set_reference_count("referenced.rs", 10);

        // Sleep a tiny bit to ensure time passes
        sleep(Duration::from_millis(10));

        assert_eq!(prioritizer.get_tier("referenced.rs"), PriorityTier::Warm);
    }

    #[test]
    fn test_get_tier_high_access_count_is_warm() {
        let prioritizer = FilePrioritizer::new(0); // 0 second threshold - immediately cold by time

        // Record many accesses
        for _ in 0..15 {
            prioritizer.record_access("frequently_accessed.rs");
        }

        // Sleep a tiny bit to ensure time passes
        sleep(Duration::from_millis(10));

        assert_eq!(
            prioritizer.get_tier("frequently_accessed.rs"),
            PriorityTier::Warm
        );
    }

    #[test]
    fn test_get_tier_old_low_usage_is_cold() {
        let prioritizer = FilePrioritizer::new(0); // 0 second threshold
        prioritizer.record_access("old.rs");
        prioritizer.set_reference_count("old.rs", 2);

        // Sleep to ensure time passes
        sleep(Duration::from_millis(10));

        assert_eq!(prioritizer.get_tier("old.rs"), PriorityTier::Cold);
    }

    #[test]
    fn test_calculate_score_unknown_file() {
        let prioritizer = FilePrioritizer::new(60);
        assert_eq!(prioritizer.calculate_score("unknown.rs"), 0.0);
    }

    #[test]
    fn test_calculate_score_entry_point_bonus() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.mark_entry_point("main.rs");

        let score = prioritizer.calculate_score("main.rs");
        // Entry point bonus is 0.3, plus some recency bonus
        assert!(score >= 0.3);
    }

    #[test]
    fn test_calculate_score_recently_accessed() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("recent.rs");

        let score = prioritizer.calculate_score("recent.rs");
        // Should have recency score (close to 0.25) and some access frequency
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_calculate_score_high_frequency() {
        let prioritizer = FilePrioritizer::new(60);

        // Access many times
        for _ in 0..100 {
            prioritizer.record_access("frequent.rs");
        }

        let score = prioritizer.calculate_score("frequent.rs");
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_calculate_score_high_references() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.set_reference_count("referenced.rs", 50);

        let score = prioritizer.calculate_score("referenced.rs");
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_calculate_score_max_bounded() {
        let prioritizer = FilePrioritizer::new(60);

        // Entry point + high access + high references
        prioritizer.mark_entry_point("maxed.rs");
        for _ in 0..1000 {
            prioritizer.record_access("maxed.rs");
        }
        prioritizer.set_reference_count("maxed.rs", 100);

        let score = prioritizer.calculate_score("maxed.rs");
        // Score is bounded at 1.0 but components may not sum exactly to 1.0
        // Entry point: 0.3, Recency: ~0.25, Freq: 0.25, Refs: 0.20 = ~1.0
        assert!(score > 0.8);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_get_hot_files_empty() {
        let prioritizer = FilePrioritizer::new(60);
        assert!(prioritizer.get_hot_files().is_empty());
    }

    #[test]
    fn test_get_hot_files_returns_entry_points() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.mark_entry_point("main.rs");
        prioritizer.mark_entry_point("lib.rs");
        prioritizer.record_access("cold.rs");

        // Wait for cold.rs to become cold
        sleep(Duration::from_millis(10));

        // But with 60 second threshold, recent access still hot
        let hot_files = prioritizer.get_hot_files();
        assert!(hot_files.contains(&"main.rs".to_string()));
        assert!(hot_files.contains(&"lib.rs".to_string()));
    }

    #[test]
    fn test_get_hot_files_returns_recently_accessed() {
        let prioritizer = FilePrioritizer::new(60);
        prioritizer.record_access("recent.rs");

        let hot_files = prioritizer.get_hot_files();
        assert!(hot_files.contains(&"recent.rs".to_string()));
    }

    #[test]
    fn test_entry_point_always_hot_regardless_of_time() {
        let prioritizer = FilePrioritizer::new(0); // 0 second threshold
        prioritizer.mark_entry_point("main.rs");

        // Even with 0 threshold, entry points are always hot
        sleep(Duration::from_millis(10));
        assert_eq!(prioritizer.get_tier("main.rs"), PriorityTier::Hot);
    }

    #[test]
    fn test_tier_boundary_reference_count() {
        let prioritizer = FilePrioritizer::new(0);

        // Exactly at boundary (5 refs is NOT warm, >5 is)
        prioritizer.set_reference_count("boundary5.rs", 5);
        sleep(Duration::from_millis(10));
        assert_eq!(prioritizer.get_tier("boundary5.rs"), PriorityTier::Cold);

        // Just above boundary
        prioritizer.set_reference_count("boundary6.rs", 6);
        sleep(Duration::from_millis(10));
        assert_eq!(prioritizer.get_tier("boundary6.rs"), PriorityTier::Warm);
    }

    #[test]
    fn test_tier_boundary_access_count() {
        let prioritizer = FilePrioritizer::new(0);

        // Exactly at boundary (10 accesses is NOT warm, >10 is)
        for _ in 0..10 {
            prioritizer.record_access("boundary10.rs");
        }
        sleep(Duration::from_millis(10));
        assert_eq!(prioritizer.get_tier("boundary10.rs"), PriorityTier::Cold);

        // Just above boundary
        for _ in 0..11 {
            prioritizer.record_access("boundary11.rs");
        }
        sleep(Duration::from_millis(10));
        assert_eq!(prioritizer.get_tier("boundary11.rs"), PriorityTier::Warm);
    }
}
