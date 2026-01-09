//! Watcher event handler
//!
//! Processes file change events and triggers incremental indexing
//! with tier-based behavior (hot/warm/cold).

use crate::prioritizer::FilePrioritizer;
use crate::realtime::{ChangeKind, FileChangeEvent};
use cogmcp_core::types::PriorityTier;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Action to take for a file change event
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexAction {
    /// Index the file immediately (hot files)
    IndexImmediate,
    /// Queue for debounced indexing (warm files)
    QueueForIndex,
    /// Ignore, will be indexed on next full reindex (cold files)
    Ignore,
    /// Remove the file from the index (deleted files)
    RemoveFromIndex,
}

/// Callback trait for handling index actions
pub trait IndexCallback: Send + Sync {
    /// Called when a file should be indexed immediately
    fn index_file(&self, path: &Path);

    /// Called when a file should be queued for debounced indexing
    fn queue_file(&self, path: &Path);

    /// Called when a file should be removed from the index
    fn remove_file(&self, path: &Path);
}

/// Handler for file watcher events
///
/// Consumes events from FileWatcher and decides what action to take
/// based on the file's priority tier from FilePrioritizer.
pub struct WatcherEventHandler<C: IndexCallback> {
    prioritizer: Arc<FilePrioritizer>,
    callback: Arc<C>,
}

impl<C: IndexCallback + 'static> WatcherEventHandler<C> {
    /// Create a new event handler
    pub fn new(prioritizer: Arc<FilePrioritizer>, callback: Arc<C>) -> Self {
        Self {
            prioritizer,
            callback,
        }
    }

    /// Determine what action to take for a file change event
    pub fn determine_action(&self, event: &FileChangeEvent) -> IndexAction {
        match event.kind {
            ChangeKind::Deleted => IndexAction::RemoveFromIndex,
            ChangeKind::Created | ChangeKind::Modified | ChangeKind::Renamed => {
                let tier = self.prioritizer.get_tier(&event.path);
                match tier {
                    PriorityTier::Hot => IndexAction::IndexImmediate,
                    PriorityTier::Warm => IndexAction::QueueForIndex,
                    PriorityTier::Cold => IndexAction::Ignore,
                }
            }
        }
    }

    /// Process a single file change event
    pub fn handle_event(&self, event: &FileChangeEvent) {
        let action = self.determine_action(event);
        let path = Path::new(&event.path);

        debug!(
            path = %event.path,
            kind = ?event.kind,
            action = ?action,
            "Processing file change event"
        );

        match action {
            IndexAction::IndexImmediate => {
                info!(path = %event.path, "Triggering immediate index for hot file");
                self.callback.index_file(path);
            }
            IndexAction::QueueForIndex => {
                debug!(path = %event.path, "Queueing warm file for debounced index");
                self.callback.queue_file(path);
            }
            IndexAction::Ignore => {
                debug!(path = %event.path, "Ignoring cold file change");
            }
            IndexAction::RemoveFromIndex => {
                info!(path = %event.path, "Removing deleted file from index");
                self.callback.remove_file(path);
            }
        }
    }

    /// Start the event processing loop
    ///
    /// This consumes events from the broadcast receiver and processes them
    /// according to their priority tier. Returns when the channel is closed.
    pub async fn run(&self, mut event_rx: broadcast::Receiver<FileChangeEvent>) {
        info!("Starting watcher event handler loop");

        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    self.handle_event(&event);
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    warn!(
                        count,
                        "Event handler lagged behind, missed events - consider full reindex"
                    );
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Event channel closed, stopping handler");
                    break;
                }
            }
        }
    }

    /// Spawn the event handler as an async task
    pub fn spawn(
        self: Arc<Self>,
        event_rx: broadcast::Receiver<FileChangeEvent>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run(event_rx).await;
        })
    }
}

/// Simple function-based callback implementation
pub struct FnCallback<F, Q, R>
where
    F: Fn(&Path) + Send + Sync,
    Q: Fn(&Path) + Send + Sync,
    R: Fn(&Path) + Send + Sync,
{
    on_index: F,
    on_queue: Q,
    on_remove: R,
}

impl<F, Q, R> FnCallback<F, Q, R>
where
    F: Fn(&Path) + Send + Sync,
    Q: Fn(&Path) + Send + Sync,
    R: Fn(&Path) + Send + Sync,
{
    /// Create a new function-based callback
    pub fn new(on_index: F, on_queue: Q, on_remove: R) -> Self {
        Self {
            on_index,
            on_queue,
            on_remove,
        }
    }
}

impl<F, Q, R> IndexCallback for FnCallback<F, Q, R>
where
    F: Fn(&Path) + Send + Sync,
    Q: Fn(&Path) + Send + Sync,
    R: Fn(&Path) + Send + Sync,
{
    fn index_file(&self, path: &Path) {
        (self.on_index)(path);
    }

    fn queue_file(&self, path: &Path) {
        (self.on_queue)(path);
    }

    fn remove_file(&self, path: &Path) {
        (self.on_remove)(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test callback that counts invocations
    struct TestCallback {
        index_count: AtomicUsize,
        queue_count: AtomicUsize,
        remove_count: AtomicUsize,
    }

    impl TestCallback {
        fn new() -> Self {
            Self {
                index_count: AtomicUsize::new(0),
                queue_count: AtomicUsize::new(0),
                remove_count: AtomicUsize::new(0),
            }
        }
    }

    impl IndexCallback for TestCallback {
        fn index_file(&self, _path: &Path) {
            self.index_count.fetch_add(1, Ordering::SeqCst);
        }

        fn queue_file(&self, _path: &Path) {
            self.queue_count.fetch_add(1, Ordering::SeqCst);
        }

        fn remove_file(&self, _path: &Path) {
            self.remove_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_deleted_file_action() {
        let prioritizer = Arc::new(FilePrioritizer::default());
        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(prioritizer, callback);

        let event = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Deleted,
        };

        assert_eq!(handler.determine_action(&event), IndexAction::RemoveFromIndex);
    }

    #[test]
    fn test_hot_file_action() {
        let prioritizer = Arc::new(FilePrioritizer::default());
        // Mark file as entry point to make it hot
        prioritizer.mark_entry_point("test.rs");

        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(prioritizer, callback);

        let event = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Modified,
        };

        assert_eq!(handler.determine_action(&event), IndexAction::IndexImmediate);
    }

    #[test]
    fn test_warm_file_action() {
        let prioritizer = Arc::new(FilePrioritizer::new(0)); // 0 second threshold - nothing is "recently accessed"
        // Set high reference count to make it warm
        prioritizer.set_reference_count("test.rs", 10);

        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(prioritizer, callback);

        let event = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Modified,
        };

        assert_eq!(handler.determine_action(&event), IndexAction::QueueForIndex);
    }

    #[test]
    fn test_cold_file_action() {
        let prioritizer = Arc::new(FilePrioritizer::new(0)); // 0 second threshold

        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(prioritizer, callback);

        let event = FileChangeEvent {
            path: "unknown.rs".to_string(),
            kind: ChangeKind::Created,
        };

        assert_eq!(handler.determine_action(&event), IndexAction::Ignore);
    }

    #[test]
    fn test_handle_event_calls_callback() {
        let prioritizer = Arc::new(FilePrioritizer::default());
        prioritizer.mark_entry_point("hot.rs");
        prioritizer.set_reference_count("warm.rs", 10);

        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(Arc::clone(&prioritizer), Arc::clone(&callback));

        // Test hot file
        handler.handle_event(&FileChangeEvent {
            path: "hot.rs".to_string(),
            kind: ChangeKind::Modified,
        });
        assert_eq!(callback.index_count.load(Ordering::SeqCst), 1);

        // Test warm file - need to wait a bit for warm to not be considered recently accessed
        let prioritizer2 = Arc::new(FilePrioritizer::new(0));
        prioritizer2.set_reference_count("warm.rs", 10);
        let handler2 = WatcherEventHandler::new(prioritizer2, Arc::clone(&callback));
        handler2.handle_event(&FileChangeEvent {
            path: "warm.rs".to_string(),
            kind: ChangeKind::Modified,
        });
        assert_eq!(callback.queue_count.load(Ordering::SeqCst), 1);

        // Test deleted file
        handler.handle_event(&FileChangeEvent {
            path: "deleted.rs".to_string(),
            kind: ChangeKind::Deleted,
        });
        assert_eq!(callback.remove_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_all_change_kinds() {
        let prioritizer = Arc::new(FilePrioritizer::default());
        prioritizer.mark_entry_point("test.rs");

        let callback = Arc::new(TestCallback::new());
        let handler = WatcherEventHandler::new(prioritizer, callback);

        // Created should trigger action based on tier
        let created = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Created,
        };
        assert_eq!(handler.determine_action(&created), IndexAction::IndexImmediate);

        // Modified should trigger action based on tier
        let modified = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Modified,
        };
        assert_eq!(handler.determine_action(&modified), IndexAction::IndexImmediate);

        // Renamed should trigger action based on tier
        let renamed = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Renamed,
        };
        assert_eq!(handler.determine_action(&renamed), IndexAction::IndexImmediate);

        // Deleted always removes from index
        let deleted = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Deleted,
        };
        assert_eq!(handler.determine_action(&deleted), IndexAction::RemoveFromIndex);
    }

    #[tokio::test]
    async fn test_event_loop() {
        let prioritizer = Arc::new(FilePrioritizer::default());
        prioritizer.mark_entry_point("hot.rs");

        let callback = Arc::new(TestCallback::new());
        let handler = Arc::new(WatcherEventHandler::new(
            Arc::clone(&prioritizer),
            Arc::clone(&callback),
        ));

        // Create a channel
        let (tx, rx) = broadcast::channel(10);

        // Spawn the handler
        let handle = handler.spawn(rx);

        // Send some events
        tx.send(FileChangeEvent {
            path: "hot.rs".to_string(),
            kind: ChangeKind::Modified,
        })
        .unwrap();

        tx.send(FileChangeEvent {
            path: "deleted.rs".to_string(),
            kind: ChangeKind::Deleted,
        })
        .unwrap();

        // Give time for events to be processed
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Drop sender to close channel
        drop(tx);

        // Wait for handler to finish
        handle.await.unwrap();

        assert_eq!(callback.index_count.load(Ordering::SeqCst), 1);
        assert_eq!(callback.remove_count.load(Ordering::SeqCst), 1);
    }
}
