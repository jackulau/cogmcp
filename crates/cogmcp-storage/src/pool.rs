//! SQLite connection pool using r2d2
//!
//! Provides a connection pool for SQLite to support concurrent database access
//! with proper connection initialization (WAL mode, pragmas).

use cogmcp_core::{Error, Result};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Global counter for creating unique in-memory database names
static IN_MEMORY_DB_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for the connection pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    pub max_size: u32,
    /// Minimum number of idle connections to maintain
    pub min_idle: Option<u32>,
    /// Timeout for acquiring a connection from the pool
    pub connection_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 10,
            min_idle: Some(2),
            connection_timeout: Duration::from_secs(30),
        }
    }
}

impl PoolConfig {
    /// Create a configuration suitable for testing with minimal resources
    /// Uses a single connection to avoid locking issues with in-memory shared databases
    pub fn for_testing() -> Self {
        Self {
            max_size: 1,
            min_idle: Some(1),
            connection_timeout: Duration::from_secs(5),
        }
    }
}

/// Convert from application-level pool configuration to storage-level pool configuration
impl From<&cogmcp_core::config::PoolConfig> for PoolConfig {
    fn from(app_config: &cogmcp_core::config::PoolConfig) -> Self {
        Self {
            max_size: app_config.max_connections,
            min_idle: app_config.min_idle,
            connection_timeout: Duration::from_secs(app_config.connection_timeout_secs),
        }
    }
}

/// Custom connection initializer that sets up SQLite pragmas
#[derive(Debug)]
struct ConnectionInitializer;

impl r2d2::CustomizeConnection<Connection, rusqlite::Error> for ConnectionInitializer {
    fn on_acquire(&self, conn: &mut Connection) -> std::result::Result<(), rusqlite::Error> {
        // Enable WAL mode for better concurrent access
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA cache_size=-64000;
             PRAGMA foreign_keys=ON;",
        )?;
        Ok(())
    }
}

/// SQLite connection pool wrapper
pub struct ConnectionPool {
    pool: Pool<SqliteConnectionManager>,
}

impl ConnectionPool {
    /// Create a new connection pool for the database at the given path
    pub fn new(path: &Path, config: PoolConfig) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        Self::build_pool(manager, config)
    }

    /// Create an in-memory connection pool (for testing)
    ///
    /// Note: In-memory databases with connection pools require special handling.
    /// Each connection in the pool would normally create a separate in-memory database.
    /// We use a unique named shared cache URI to make all connections in this pool
    /// share the same database, while keeping different pool instances isolated.
    pub fn in_memory(config: PoolConfig) -> Result<Self> {
        // Generate a unique name for this in-memory database instance
        let db_id = IN_MEMORY_DB_COUNTER.fetch_add(1, Ordering::SeqCst);
        let uri = format!("file:memdb_{}?mode=memory&cache=shared", db_id);
        let manager = SqliteConnectionManager::file(&uri);
        Self::build_pool(manager, config)
    }

    /// Build the pool from a manager with the given configuration
    fn build_pool(manager: SqliteConnectionManager, config: PoolConfig) -> Result<Self> {
        let mut builder = Pool::builder()
            .max_size(config.max_size)
            .connection_timeout(config.connection_timeout)
            .connection_customizer(Box::new(ConnectionInitializer));

        if let Some(min_idle) = config.min_idle {
            builder = builder.min_idle(Some(min_idle));
        }

        let pool = builder.build(manager).map_err(|e| {
            Error::Storage(format!("Failed to create connection pool: {}", e))
        })?;

        Ok(Self { pool })
    }

    /// Get a connection from the pool
    pub fn get(&self) -> Result<PooledConnection<SqliteConnectionManager>> {
        self.pool.get().map_err(|e| {
            Error::Storage(format!("Failed to get connection from pool: {}", e))
        })
    }

    /// Get the current pool state (for diagnostics)
    pub fn state(&self) -> PoolState {
        let state = self.pool.state();
        PoolState {
            connections: state.connections,
            idle_connections: state.idle_connections,
        }
    }
}

/// Pool state information
#[derive(Debug, Clone)]
pub struct PoolState {
    /// Total number of connections managed by the pool
    pub connections: u32,
    /// Number of idle connections in the pool
    pub idle_connections: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = ConnectionPool::in_memory(PoolConfig::for_testing()).unwrap();
        let state = pool.state();
        assert!(state.connections > 0);
    }

    #[test]
    fn test_connection_acquisition() {
        let pool = ConnectionPool::in_memory(PoolConfig::for_testing()).unwrap();
        let conn = pool.get().unwrap();

        // Verify WAL mode is set
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        // In-memory databases may use 'memory' instead of 'wal'
        assert!(journal_mode == "wal" || journal_mode == "memory");
    }

    #[test]
    fn test_multiple_connections() {
        let config = PoolConfig {
            max_size: 5,
            min_idle: Some(2),
            connection_timeout: Duration::from_secs(5),
        };
        let pool = ConnectionPool::in_memory(config).unwrap();

        // Acquire multiple connections
        let conn1 = pool.get().unwrap();
        let conn2 = pool.get().unwrap();
        let conn3 = pool.get().unwrap();

        // Create a table using one connection
        conn1
            .execute(
                "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, value TEXT)",
                [],
            )
            .unwrap();

        // Insert from another connection
        conn2
            .execute("INSERT INTO test_table (value) VALUES ('test')", [])
            .unwrap();

        // Read from a third connection
        let count: i64 = conn3
            .query_row("SELECT COUNT(*) FROM test_table", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
