//! Connection pool for SQLite databases
//!
//! This module provides a connection pool implementation for SQLite that:
//! - Maintains a pool of pre-initialized connections
//! - Supports configurable pool size and timeout
//! - Applies WAL mode and other pragmas for optimal concurrent access
//! - Handles connection acquisition with timeout behavior

use cogmcp_core::{Error, Result};
use parking_lot::{Condvar, Mutex};
use rusqlite::Connection;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the connection pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,
    /// Minimum number of connections to maintain
    pub min_connections: usize,
    /// Timeout for acquiring a connection (None = wait forever)
    pub acquire_timeout: Option<Duration>,
    /// Path to the database file (None for in-memory)
    pub db_path: Option<PathBuf>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 4,
            min_connections: 1,
            acquire_timeout: Some(Duration::from_secs(30)),
            db_path: None,
        }
    }
}

impl PoolConfig {
    /// Create a new pool config with the given max connections
    pub fn new(max_connections: usize) -> Self {
        Self {
            max_connections,
            min_connections: 1,
            acquire_timeout: Some(Duration::from_secs(30)),
            db_path: None,
        }
    }

    /// Set the database path
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    /// Set the acquire timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = Some(timeout);
        self
    }

    /// Set no acquire timeout (wait forever)
    pub fn without_timeout(mut self) -> Self {
        self.acquire_timeout = None;
        self
    }

    /// Set minimum connections
    pub fn with_min_connections(mut self, min: usize) -> Self {
        self.min_connections = min;
        self
    }
}

/// A pooled connection that returns to the pool when dropped
pub struct PooledConnection {
    conn: Option<Connection>,
    pool: Arc<ConnectionPoolInner>,
}

impl PooledConnection {
    /// Get a reference to the underlying connection
    pub fn connection(&self) -> &Connection {
        self.conn.as_ref().expect("Connection should exist")
    }

    /// Get a mutable reference to the underlying connection
    pub fn connection_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("Connection should exist")
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.return_connection(conn);
        }
    }
}

impl std::ops::Deref for PooledConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        self.connection()
    }
}

impl std::ops::DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection_mut()
    }
}

/// Internal pool state
struct ConnectionPoolInner {
    config: PoolConfig,
    state: Mutex<PoolState>,
    available: Condvar,
}

struct PoolState {
    /// Available connections in the pool
    connections: VecDeque<Connection>,
    /// Number of connections currently checked out
    checked_out: usize,
    /// Total connections created
    total_created: usize,
}

impl ConnectionPoolInner {
    fn return_connection(&self, conn: Connection) {
        let mut state = self.state.lock();
        state.connections.push_back(conn);
        state.checked_out -= 1;
        self.available.notify_one();
    }
}

/// A connection pool for SQLite databases
pub struct ConnectionPool {
    inner: Arc<ConnectionPoolInner>,
}

impl ConnectionPool {
    /// Create a new connection pool with the given configuration
    pub fn new(config: PoolConfig) -> Result<Self> {
        let mut connections = VecDeque::with_capacity(config.max_connections);

        // Create minimum number of connections
        for _ in 0..config.min_connections {
            let conn = Self::create_connection(&config)?;
            connections.push_back(conn);
        }

        let inner = Arc::new(ConnectionPoolInner {
            config,
            state: Mutex::new(PoolState {
                connections,
                checked_out: 0,
                total_created: 0,
            }),
            available: Condvar::new(),
        });

        // Update total_created
        {
            let mut state = inner.state.lock();
            state.total_created = state.connections.len();
        }

        Ok(Self { inner })
    }

    /// Create a new in-memory connection pool
    pub fn in_memory(max_connections: usize) -> Result<Self> {
        let config = PoolConfig::new(max_connections);
        Self::new(config)
    }

    /// Create a connection pool for a file-backed database
    pub fn open(path: &Path, max_connections: usize) -> Result<Self> {
        let config = PoolConfig::new(max_connections).with_path(path);
        Self::new(config)
    }

    /// Get a connection from the pool
    pub fn get(&self) -> Result<PooledConnection> {
        let deadline = self
            .inner
            .config
            .acquire_timeout
            .map(|t| Instant::now() + t);

        let mut state = self.inner.state.lock();

        loop {
            // Try to get an existing connection
            if let Some(conn) = state.connections.pop_front() {
                state.checked_out += 1;
                return Ok(PooledConnection {
                    conn: Some(conn),
                    pool: Arc::clone(&self.inner),
                });
            }

            // Try to create a new connection if below max
            let total_connections = state.connections.len() + state.checked_out;
            if total_connections < self.inner.config.max_connections {
                let conn = Self::create_connection(&self.inner.config)?;
                state.total_created += 1;
                state.checked_out += 1;
                return Ok(PooledConnection {
                    conn: Some(conn),
                    pool: Arc::clone(&self.inner),
                });
            }

            // Wait for a connection to become available
            if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(Error::Storage(
                        "Timeout waiting for connection from pool".to_string(),
                    ));
                }
                let result = self.inner.available.wait_for(&mut state, remaining);
                if result.timed_out() {
                    return Err(Error::Storage(
                        "Timeout waiting for connection from pool".to_string(),
                    ));
                }
            } else {
                self.inner.available.wait(&mut state);
            }
        }
    }

    /// Try to get a connection without waiting
    pub fn try_get(&self) -> Result<Option<PooledConnection>> {
        let mut state = self.inner.state.lock();

        // Try to get an existing connection
        if let Some(conn) = state.connections.pop_front() {
            state.checked_out += 1;
            return Ok(Some(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(&self.inner),
            }));
        }

        // Try to create a new connection if below max
        let total_connections = state.connections.len() + state.checked_out;
        if total_connections < self.inner.config.max_connections {
            let conn = Self::create_connection(&self.inner.config)?;
            state.total_created += 1;
            state.checked_out += 1;
            return Ok(Some(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(&self.inner),
            }));
        }

        Ok(None)
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let state = self.inner.state.lock();
        PoolStats {
            available: state.connections.len(),
            checked_out: state.checked_out,
            total_created: state.total_created,
            max_connections: self.inner.config.max_connections,
        }
    }

    /// Create a new connection with proper pragmas
    fn create_connection(config: &PoolConfig) -> Result<Connection> {
        let conn = if let Some(ref path) = config.db_path {
            Connection::open(path)
                .map_err(|e| Error::Storage(format!("Failed to open database: {}", e)))?
        } else {
            Connection::open_in_memory()
                .map_err(|e| Error::Storage(format!("Failed to create in-memory database: {}", e)))?
        };

        // Apply performance pragmas
        Self::apply_pragmas(&conn)?;

        Ok(conn)
    }

    /// Apply SQLite pragmas for optimal concurrent access
    fn apply_pragmas(conn: &Connection) -> Result<()> {
        // Enable WAL mode for better concurrent access
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = 268435456;
            PRAGMA busy_timeout = 5000;
            "#,
        )
        .map_err(|e| Error::Storage(format!("Failed to apply pragmas: {}", e)))?;

        Ok(())
    }
}

impl Clone for ConnectionPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Statistics about the connection pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of available connections
    pub available: usize,
    /// Number of connections currently checked out
    pub checked_out: usize,
    /// Total number of connections created
    pub total_created: usize,
    /// Maximum allowed connections
    pub max_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    // ========================================================================
    // Pool Creation Tests
    // ========================================================================

    #[test]
    fn test_pool_creation_default_config() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config).unwrap();

        let stats = pool.stats();
        assert!(stats.available >= 1, "Should have at least min connections");
        assert_eq!(stats.checked_out, 0, "No connections should be checked out");
    }

    #[test]
    fn test_pool_creation_custom_config() {
        let config = PoolConfig::new(8)
            .with_min_connections(2)
            .with_timeout(Duration::from_secs(5));

        let pool = ConnectionPool::new(config).unwrap();

        let stats = pool.stats();
        assert!(stats.available >= 2, "Should have at least 2 min connections");
        assert_eq!(stats.max_connections, 8, "Max should be 8");
    }

    #[test]
    fn test_pool_in_memory() {
        let pool = ConnectionPool::in_memory(4).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.max_connections, 4);
    }

    #[test]
    fn test_pool_file_backed() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_pool.db");

        // Clean up if exists
        let _ = std::fs::remove_file(&db_path);

        let pool = ConnectionPool::open(&db_path, 4).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.max_connections, 4);

        // Verify file was created
        assert!(db_path.exists(), "Database file should be created");

        // Clean up
        drop(pool);
        let _ = std::fs::remove_file(&db_path);
    }

    // ========================================================================
    // Connection Acquisition and Release Tests
    // ========================================================================

    #[test]
    fn test_connection_acquisition() {
        let pool = ConnectionPool::in_memory(4).unwrap();

        let conn = pool.get().unwrap();

        let stats = pool.stats();
        assert_eq!(stats.checked_out, 1, "One connection should be checked out");

        // Verify connection is usable
        let result: i64 = conn.query_row("SELECT 1", [], |row| row.get(0)).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_connection_release() {
        let pool = ConnectionPool::in_memory(4).unwrap();

        {
            let _conn = pool.get().unwrap();
            assert_eq!(pool.stats().checked_out, 1);
        }

        // Connection should be returned to pool
        assert_eq!(pool.stats().checked_out, 0);
        assert!(pool.stats().available >= 1);
    }

    #[test]
    fn test_multiple_connections() {
        let pool = ConnectionPool::in_memory(4).unwrap();

        let conn1 = pool.get().unwrap();
        let conn2 = pool.get().unwrap();
        let conn3 = pool.get().unwrap();

        let stats = pool.stats();
        assert_eq!(stats.checked_out, 3);

        // All connections should be usable
        let r1: i64 = conn1.query_row("SELECT 1", [], |row| row.get(0)).unwrap();
        let r2: i64 = conn2.query_row("SELECT 2", [], |row| row.get(0)).unwrap();
        let r3: i64 = conn3.query_row("SELECT 3", [], |row| row.get(0)).unwrap();
        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_eq!(r3, 3);
    }

    #[test]
    fn test_try_get_success() {
        let pool = ConnectionPool::in_memory(4).unwrap();

        let conn = pool.try_get().unwrap();
        assert!(conn.is_some(), "Should get a connection");
    }

    #[test]
    fn test_try_get_pool_exhausted() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        let _conn1 = pool.get().unwrap();
        let _conn2 = pool.get().unwrap();

        // Pool is now exhausted
        let result = pool.try_get().unwrap();
        assert!(result.is_none(), "Should return None when pool exhausted");
    }

    // ========================================================================
    // Pool Exhaustion and Timeout Tests
    // ========================================================================

    #[test]
    fn test_pool_exhaustion_timeout() {
        let config = PoolConfig::new(1).with_timeout(Duration::from_millis(100));

        let pool = ConnectionPool::new(config).unwrap();

        // Hold the only connection
        let _conn = pool.get().unwrap();

        // Try to get another connection - should timeout
        let start = Instant::now();
        let result = pool.get();
        let elapsed = start.elapsed();

        assert!(result.is_err(), "Should timeout");
        assert!(
            elapsed >= Duration::from_millis(100),
            "Should wait at least 100ms"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "Should not wait too long"
        );
    }

    #[test]
    fn test_pool_waits_and_gets_returned_connection() {
        let config = PoolConfig::new(1).with_timeout(Duration::from_secs(5));

        let pool = ConnectionPool::new(config).unwrap();
        let pool_clone = pool.clone();

        // First, grab the connection in the main thread to ensure it's checked out
        let conn = pool.get().unwrap();

        // Use a barrier to coordinate between threads
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);

        // Spawn a thread that will wait for our signal, then try to get a connection
        let handle = thread::spawn(move || {
            // Signal that we're ready to wait
            barrier_clone.wait();
            // Now try to get a connection - this should block until main thread releases
            let start = Instant::now();
            let result = pool_clone.get();
            let elapsed = start.elapsed();
            (result, elapsed)
        });

        // Wait for the other thread to start waiting
        barrier.wait();
        // Give the thread time to actually start waiting on the pool
        thread::sleep(Duration::from_millis(50));

        // Now release our connection
        drop(conn);

        // Wait for the other thread to finish
        let (result, elapsed) = handle.join().unwrap();

        assert!(result.is_ok(), "Should get connection after it's returned");
        // The thread should have waited at least a short time (it started waiting, then we released)
        assert!(
            elapsed >= Duration::from_millis(10),
            "Should have waited for connection, elapsed: {:?}",
            elapsed
        );
    }

    // ========================================================================
    // Connection Initialization (Pragmas) Tests
    // ========================================================================

    #[test]
    fn test_wal_mode_applied() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_wal.db");

        // Clean up if exists
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));

        let pool = ConnectionPool::open(&db_path, 2).unwrap();

        let conn = pool.get().unwrap();

        // Verify WAL mode is enabled
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();

        assert_eq!(journal_mode.to_lowercase(), "wal", "WAL mode should be enabled");

        // Clean up
        drop(conn);
        drop(pool);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    #[test]
    fn test_busy_timeout_applied() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        let conn = pool.get().unwrap();

        // Verify busy_timeout is set
        let timeout: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();

        assert_eq!(timeout, 5000, "Busy timeout should be 5000ms");
    }

    // ========================================================================
    // Concurrent Access Tests
    // ========================================================================

    #[test]
    fn test_concurrent_acquisitions() {
        let pool = ConnectionPool::in_memory(4).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        for _ in 0..10 {
            let pool = pool.clone();
            let counter = Arc::clone(&counter);

            let handle = thread::spawn(move || {
                for _ in 0..5 {
                    let conn = pool.get().unwrap();
                    let _: i64 = conn.query_row("SELECT 1", [], |row| row.get(0)).unwrap();
                    counter.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 50, "All operations should complete");
    }

    #[test]
    fn test_connection_reuse() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        // Get and release connections multiple times
        for i in 0..10 {
            let conn = pool.get().unwrap();
            let result: i64 = conn
                .query_row(&format!("SELECT {}", i), [], |row| row.get(0))
                .unwrap();
            assert_eq!(result, i as i64);
        }

        let stats = pool.stats();
        assert!(
            stats.total_created <= 2,
            "Should not create more than max connections"
        );
    }

    #[test]
    fn test_pool_clone_shares_state() {
        let pool1 = ConnectionPool::in_memory(4).unwrap();
        let pool2 = pool1.clone();

        let _conn1 = pool1.get().unwrap();

        // Both pools should see the same state
        assert_eq!(pool1.stats().checked_out, 1);
        assert_eq!(pool2.stats().checked_out, 1);

        let _conn2 = pool2.get().unwrap();

        assert_eq!(pool1.stats().checked_out, 2);
        assert_eq!(pool2.stats().checked_out, 2);
    }

    // ========================================================================
    // PooledConnection Tests
    // ========================================================================

    #[test]
    fn test_pooled_connection_deref() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        let conn = pool.get().unwrap();

        // Use deref to access Connection methods directly
        let result: i64 = conn.query_row("SELECT 42", [], |row| row.get(0)).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_pooled_connection_deref_mut() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        let conn = pool.get().unwrap();

        // Create a table using connection
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", [])
            .unwrap();

        conn.execute("INSERT INTO test (id) VALUES (1)", [])
            .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM test", [], |row| row.get(0))
            .unwrap();

        assert_eq!(count, 1);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_zero_min_connections() {
        let config = PoolConfig::new(4).with_min_connections(0);

        let pool = ConnectionPool::new(config).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.available, 0, "Should start with 0 connections");

        // Should create connection on demand
        let conn = pool.get().unwrap();
        let result: i64 = conn.query_row("SELECT 1", [], |row| row.get(0)).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_min_equals_max() {
        let config = PoolConfig::new(3).with_min_connections(3);

        let pool = ConnectionPool::new(config).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.available, 3, "Should pre-create all connections");
        assert_eq!(stats.total_created, 3);
    }

    #[test]
    fn test_rapid_acquire_release() {
        let pool = ConnectionPool::in_memory(2).unwrap();

        for _ in 0..100 {
            let conn = pool.get().unwrap();
            let _: i64 = conn.query_row("SELECT 1", [], |row| row.get(0)).unwrap();
            // Connection dropped immediately
        }

        let stats = pool.stats();
        assert!(
            stats.total_created <= 2,
            "Should reuse connections efficiently"
        );
    }
}
