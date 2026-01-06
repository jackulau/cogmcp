//! Real-time file watching

use contextmcp_core::Result;
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
        .map_err(|e| contextmcp_core::Error::FileSystem(format!("Failed to create watcher: {}", e)))?;

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
