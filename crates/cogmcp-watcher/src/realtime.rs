//! Real-time file watching

use cogmcp_core::Result;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;
use tokio::sync::broadcast;

/// File change event
#[derive(Debug, Clone)]
pub struct FileChangeEvent {
    pub path: String,
    pub kind: ChangeKind,
}

/// Kind of file change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Created,
    Modified,
    Deleted,
    Renamed,
}

/// File system watcher for real-time updates
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    event_tx: broadcast::Sender<FileChangeEvent>,
}

impl FileWatcher {
    /// Create a new file watcher for the given root directory
    pub fn new(root: &Path) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(1000);
        let tx = event_tx.clone();

        let (notify_tx, notify_rx) = mpsc::channel();

        let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                let _ = notify_tx.send(event);
            }
        })
        .map_err(|e| cogmcp_core::Error::FileSystem(format!("Failed to create watcher: {}", e)))?;

        // Start background task to process events
        let root_path = root.to_path_buf();
        std::thread::spawn(move || {
            while let Ok(event) = notify_rx.recv() {
                for path in event.paths {
                    let relative = path
                        .strip_prefix(&root_path)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    let kind = match event.kind {
                        notify::EventKind::Create(_) => ChangeKind::Created,
                        notify::EventKind::Modify(_) => ChangeKind::Modified,
                        notify::EventKind::Remove(_) => ChangeKind::Deleted,
                        _ => continue,
                    };

                    let _ = tx.send(FileChangeEvent { path: relative, kind });
                }
            }
        });

        // Start watching (but don't block on result for now)
        let mut watcher = watcher;
        let _ = watcher.watch(root, RecursiveMode::Recursive);

        Ok(Self {
            _watcher: watcher,
            event_tx,
        })
    }

    /// Subscribe to file change events
    pub fn subscribe(&self) -> broadcast::Receiver<FileChangeEvent> {
        self.event_tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_change_kind_equality() {
        assert_eq!(ChangeKind::Created, ChangeKind::Created);
        assert_eq!(ChangeKind::Modified, ChangeKind::Modified);
        assert_eq!(ChangeKind::Deleted, ChangeKind::Deleted);
        assert_eq!(ChangeKind::Renamed, ChangeKind::Renamed);

        assert_ne!(ChangeKind::Created, ChangeKind::Modified);
        assert_ne!(ChangeKind::Created, ChangeKind::Deleted);
        assert_ne!(ChangeKind::Created, ChangeKind::Renamed);
    }

    #[test]
    fn test_change_kind_clone() {
        let kind = ChangeKind::Modified;
        let cloned = kind;
        assert_eq!(kind, cloned);
    }

    #[test]
    fn test_change_kind_debug() {
        let kind = ChangeKind::Created;
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("Created"));
    }

    #[test]
    fn test_file_change_event_clone() {
        let event = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Modified,
        };
        let cloned = event.clone();
        assert_eq!(event.path, cloned.path);
        assert_eq!(event.kind, cloned.kind);
    }

    #[test]
    fn test_file_change_event_debug() {
        let event = FileChangeEvent {
            path: "test.rs".to_string(),
            kind: ChangeKind::Created,
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("test.rs"));
        assert!(debug_str.contains("Created"));
    }

    #[test]
    fn test_file_watcher_new_success() {
        let temp_dir = TempDir::new().unwrap();
        let watcher = FileWatcher::new(temp_dir.path());
        assert!(watcher.is_ok());
    }

    #[test]
    fn test_file_watcher_subscribe_returns_receiver() {
        let temp_dir = TempDir::new().unwrap();
        let watcher = FileWatcher::new(temp_dir.path()).unwrap();
        let _receiver = watcher.subscribe();
        // Just verify we can subscribe without panic
    }

    #[test]
    fn test_file_watcher_multiple_subscribers() {
        let temp_dir = TempDir::new().unwrap();
        let watcher = FileWatcher::new(temp_dir.path()).unwrap();

        let _receiver1 = watcher.subscribe();
        let _receiver2 = watcher.subscribe();
        let _receiver3 = watcher.subscribe();
        // Verify multiple subscriptions work
    }

    #[test]
    fn test_file_watcher_new_nonexistent_path() {
        // Notify typically allows watching paths that don't exist yet
        // but the behavior can vary - just ensure it doesn't panic
        let result = FileWatcher::new(Path::new("/nonexistent/path/that/should/not/exist"));
        // The result may be Ok or Err depending on platform, just don't panic
        let _ = result;
    }
}
