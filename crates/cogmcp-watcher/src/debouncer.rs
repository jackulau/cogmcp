//! Debouncing for Warm-tier file changes
//!
//! This module implements a debouncing system that prevents excessive reindexing
//! during rapid file modifications. Warm files are indexed with a configurable delay
//! to batch rapid changes together.

use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Manages debouncing for file change events
///
/// Files are tracked in a pending map with timestamps. A file is only ready
/// for processing once the debounce duration has elapsed since its last change.
pub struct FileDebouncer {
    /// Map of pending files to their last change timestamp
    pending: DashMap<PathBuf, Instant>,
    /// Duration to wait before processing a file
    debounce_duration: Duration,
}

impl FileDebouncer {
    /// Create a new FileDebouncer with the specified debounce duration in milliseconds
    pub fn new(debounce_ms: u64) -> Self {
        Self {
            pending: DashMap::new(),
            debounce_duration: Duration::from_millis(debounce_ms),
        }
    }

    /// Check if a file should be processed (debounce period has elapsed)
    ///
    /// Returns true if the file is pending and enough time has passed since
    /// the last change event.
    pub fn should_process(&self, path: &Path) -> bool {
        if let Some(entry) = self.pending.get(path) {
            entry.elapsed() >= self.debounce_duration
        } else {
            false
        }
    }

    /// Mark a file as pending (received a change event)
    ///
    /// If the file is already pending, updates the timestamp to the current time.
    /// This resets the debounce timer for rapid consecutive changes.
    pub fn mark_pending(&self, path: &Path) {
        self.pending.insert(path.to_path_buf(), Instant::now());
    }

    /// Mark a file as processed and remove from pending
    ///
    /// Call this after successfully processing a file to remove it from tracking.
    pub fn mark_processed(&self, path: &Path) {
        self.pending.remove(path);
    }

    /// Get all files that are ready for processing
    ///
    /// Returns files whose debounce period has elapsed. Does not remove them
    /// from the pending map - call `mark_processed` after successful processing.
    pub fn get_ready_files(&self) -> Vec<PathBuf> {
        self.pending
            .iter()
            .filter(|entry| entry.value().elapsed() >= self.debounce_duration)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get the number of pending files
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clear all pending files
    pub fn clear(&self) {
        self.pending.clear();
    }
}

impl Default for FileDebouncer {
    fn default() -> Self {
        Self::new(1000) // Default 1 second debounce
    }
}

/// Starts a background task that periodically checks for ready files
///
/// This function spawns an async task that polls the debouncer at regular intervals
/// and sends ready files through the provided channel.
///
/// # Arguments
///
/// * `debouncer` - Arc-wrapped FileDebouncer to check for ready files
/// * `tx` - Channel sender for ready file paths
/// * `poll_interval_ms` - How often to check for ready files (in milliseconds)
///
/// # Returns
///
/// A handle to the spawned task that can be used for cancellation
pub fn spawn_debounce_checker(
    debouncer: std::sync::Arc<FileDebouncer>,
    tx: mpsc::Sender<PathBuf>,
    poll_interval_ms: u64,
) -> tokio::task::JoinHandle<()> {
    let poll_interval = Duration::from_millis(poll_interval_ms);

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(poll_interval);

        loop {
            interval.tick().await;

            let ready_files = debouncer.get_ready_files();
            for path in ready_files {
                // Mark as processed before sending to avoid duplicate sends
                debouncer.mark_processed(&path);

                if tx.send(path).await.is_err() {
                    // Receiver dropped, exit the task
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_new_debouncer() {
        let debouncer = FileDebouncer::new(500);
        assert_eq!(debouncer.debounce_duration, Duration::from_millis(500));
        assert_eq!(debouncer.pending_count(), 0);
    }

    #[test]
    fn test_default_debouncer() {
        let debouncer = FileDebouncer::default();
        assert_eq!(debouncer.debounce_duration, Duration::from_millis(1000));
    }

    #[test]
    fn test_mark_pending() {
        let debouncer = FileDebouncer::new(100);
        let path = PathBuf::from("/test/file.rs");

        assert!(!debouncer.should_process(&path));
        assert_eq!(debouncer.pending_count(), 0);

        debouncer.mark_pending(&path);
        assert_eq!(debouncer.pending_count(), 1);
    }

    #[test]
    fn test_should_process_before_debounce() {
        let debouncer = FileDebouncer::new(1000);
        let path = PathBuf::from("/test/file.rs");

        debouncer.mark_pending(&path);

        // Should not be ready immediately
        assert!(!debouncer.should_process(&path));
    }

    #[test]
    fn test_should_process_after_debounce() {
        let debouncer = FileDebouncer::new(50); // Short debounce for testing
        let path = PathBuf::from("/test/file.rs");

        debouncer.mark_pending(&path);
        assert!(!debouncer.should_process(&path));

        // Wait for debounce period
        sleep(Duration::from_millis(60));

        assert!(debouncer.should_process(&path));
    }

    #[test]
    fn test_mark_processed() {
        let debouncer = FileDebouncer::new(50);
        let path = PathBuf::from("/test/file.rs");

        debouncer.mark_pending(&path);
        assert_eq!(debouncer.pending_count(), 1);

        debouncer.mark_processed(&path);
        assert_eq!(debouncer.pending_count(), 0);
        assert!(!debouncer.should_process(&path));
    }

    #[test]
    fn test_get_ready_files() {
        let debouncer = FileDebouncer::new(30); // Very short debounce
        let path1 = PathBuf::from("/test/file1.rs");
        let path2 = PathBuf::from("/test/file2.rs");

        debouncer.mark_pending(&path1);
        debouncer.mark_pending(&path2);

        // Nothing ready yet
        assert!(debouncer.get_ready_files().is_empty());

        // Wait for debounce
        sleep(Duration::from_millis(50));

        let ready = debouncer.get_ready_files();
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&path1));
        assert!(ready.contains(&path2));
    }

    #[test]
    fn test_debounce_reset_on_new_change() {
        let debouncer = FileDebouncer::new(50);
        let path = PathBuf::from("/test/file.rs");

        debouncer.mark_pending(&path);

        // Wait half the debounce time
        sleep(Duration::from_millis(30));

        // New change resets the timer
        debouncer.mark_pending(&path);

        // After original debounce time, still not ready because timer was reset
        sleep(Duration::from_millis(30));
        assert!(!debouncer.should_process(&path));

        // After full debounce from the reset, it should be ready
        sleep(Duration::from_millis(30));
        assert!(debouncer.should_process(&path));
    }

    #[test]
    fn test_clear() {
        let debouncer = FileDebouncer::new(100);

        debouncer.mark_pending(&PathBuf::from("/test/file1.rs"));
        debouncer.mark_pending(&PathBuf::from("/test/file2.rs"));
        debouncer.mark_pending(&PathBuf::from("/test/file3.rs"));

        assert_eq!(debouncer.pending_count(), 3);

        debouncer.clear();
        assert_eq!(debouncer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_spawn_debounce_checker() {
        let debouncer = std::sync::Arc::new(FileDebouncer::new(30));
        let (tx, mut rx) = mpsc::channel(10);

        let path = PathBuf::from("/test/async_file.rs");
        debouncer.mark_pending(&path);

        // Spawn the checker with 10ms poll interval
        let handle = spawn_debounce_checker(debouncer.clone(), tx, 10);

        // Should receive the file after debounce period
        let received = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;

        assert!(received.is_ok());
        assert_eq!(received.unwrap(), Some(path.clone()));

        // File should be removed from pending after being sent
        assert_eq!(debouncer.pending_count(), 0);

        handle.abort();
    }

    #[tokio::test]
    async fn test_checker_handles_multiple_files() {
        let debouncer = std::sync::Arc::new(FileDebouncer::new(20));
        let (tx, mut rx) = mpsc::channel(10);

        let path1 = PathBuf::from("/test/file1.rs");
        let path2 = PathBuf::from("/test/file2.rs");

        debouncer.mark_pending(&path1);
        debouncer.mark_pending(&path2);

        let handle = spawn_debounce_checker(debouncer.clone(), tx, 10);

        // Collect received files
        let mut received = Vec::new();
        for _ in 0..2 {
            if let Ok(Some(path)) =
                tokio::time::timeout(Duration::from_millis(200), rx.recv()).await
            {
                received.push(path);
            }
        }

        assert_eq!(received.len(), 2);
        assert!(received.contains(&path1));
        assert!(received.contains(&path2));

        handle.abort();
    }
}
