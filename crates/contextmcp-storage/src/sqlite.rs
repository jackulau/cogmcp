//! SQLite database for structured data storage

use cogmcp_core::{Error, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;

/// SQLite database wrapper with connection pooling
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Open or create a database at the given path
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| Error::Storage(format!("Failed to open database: {}", e)))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| Error::Storage(format!("Failed to create in-memory database: {}", e)))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(SCHEMA).map_err(|e| {
            Error::Storage(format!("Failed to initialize schema: {}", e))
        })?;
        Ok(())
    }

    /// Execute a query with the connection
    pub fn with_connection<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self.conn.lock();
        f(&conn)
    }

    /// Insert or update a file in the index
    pub fn upsert_file(
        &self,
        path: &str,
        hash: &str,
        modified_at: i64,
        size: u64,
        language: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO files (path, hash, modified_at, size, language, indexed_at)
            VALUES (?1, ?2, ?3, ?4, ?5, strftime('%s', 'now'))
            ON CONFLICT(path) DO UPDATE SET
                hash = excluded.hash,
                modified_at = excluded.modified_at,
                size = excluded.size,
                language = excluded.language,
                indexed_at = excluded.indexed_at
            "#,
            params![path, hash, modified_at, size as i64, language],
        )
        .map_err(|e| Error::Storage(format!("Failed to upsert file: {}", e)))?;

        // Get the file ID (last_insert_rowid returns 0 for updates)
        let id: i64 = conn
            .query_row("SELECT id FROM files WHERE path = ?1", params![path], |row| {
                row.get(0)
            })
            .map_err(|e| Error::Storage(format!("Failed to get file id: {}", e)))?;
        Ok(id)
    }

    /// Get a file by path
    pub fn get_file_by_path(&self, path: &str) -> Result<Option<FileRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, path, hash, modified_at, size, language, priority_score,
                       last_accessed, indexed_at
                FROM files WHERE path = ?1
                "#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let result = stmt
            .query_row(params![path], |row| {
                Ok(FileRow {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    hash: row.get(2)?,
                    modified_at: row.get(3)?,
                    size: row.get(4)?,
                    language: row.get(5)?,
                    priority_score: row.get(6)?,
                    last_accessed: row.get(7)?,
                    indexed_at: row.get(8)?,
                })
            })
            .ok();

        Ok(result)
    }

    /// Insert a symbol
    pub fn insert_symbol(
        &self,
        file_id: i64,
        name: &str,
        kind: &str,
        start_line: u32,
        end_line: u32,
        signature: Option<&str>,
        doc_comment: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO symbols (file_id, name, kind, start_line, end_line, signature, doc_comment)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![file_id, name, kind, start_line, end_line, signature, doc_comment],
        )
        .map_err(|e| Error::Storage(format!("Failed to insert symbol: {}", e)))?;

        Ok(conn.last_insert_rowid())
    }

    /// Delete all symbols for a file
    pub fn delete_symbols_for_file(&self, file_id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM symbols WHERE file_id = ?1", params![file_id])
            .map_err(|e| Error::Storage(format!("Failed to delete symbols: {}", e)))?;
        Ok(())
    }

    /// Find symbols by name
    pub fn find_symbols_by_name(&self, name: &str, fuzzy: bool) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();
        let pattern = if fuzzy {
            format!("%{}%", name)
        } else {
            name.to_string()
        };

        let query = if fuzzy {
            "SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path
             FROM symbols s JOIN files f ON s.file_id = f.id
             WHERE s.name LIKE ?1"
        } else {
            "SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path
             FROM symbols s JOIN files f ON s.file_id = f.id
             WHERE s.name = ?1"
        };

        let mut stmt = conn
            .prepare(query)
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok(SymbolRow {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    start_line: row.get(4)?,
                    end_line: row.get(5)?,
                    signature: row.get(6)?,
                    doc_comment: row.get(7)?,
                    file_path: row.get(8)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query symbols: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get all files in the index
    pub fn get_all_files(&self) -> Result<Vec<FileRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, path, hash, modified_at, size, language, priority_score,
                       last_accessed, indexed_at
                FROM files
                "#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(FileRow {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    hash: row.get(2)?,
                    modified_at: row.get(3)?,
                    size: row.get(4)?,
                    language: row.get(5)?,
                    priority_score: row.get(6)?,
                    last_accessed: row.get(7)?,
                    indexed_at: row.get(8)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query files: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Store an embedding for a symbol/chunk
    pub fn insert_embedding(
        &self,
        symbol_id: Option<i64>,
        chunk_text: &str,
        embedding: &[f32],
        chunk_type: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock();
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        conn.execute(
            r#"
            INSERT INTO embeddings (symbol_id, chunk_text, embedding, chunk_type)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![symbol_id, chunk_text, embedding_bytes, chunk_type],
        )
        .map_err(|e| Error::Storage(format!("Failed to insert embedding: {}", e)))?;

        Ok(conn.last_insert_rowid())
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Result<IndexStats> {
        let conn = self.conn.lock();

        let file_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .map_err(|e| Error::Storage(format!("Failed to count files: {}", e)))?;

        let symbol_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))
            .map_err(|e| Error::Storage(format!("Failed to count symbols: {}", e)))?;

        let embedding_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .map_err(|e| Error::Storage(format!("Failed to count embeddings: {}", e)))?;

        Ok(IndexStats {
            file_count: file_count as u64,
            symbol_count: symbol_count as u64,
            embedding_count: embedding_count as u64,
        })
    }
}

/// Row from the files table
#[derive(Debug, Clone)]
pub struct FileRow {
    pub id: i64,
    pub path: String,
    pub hash: String,
    pub modified_at: i64,
    pub size: i64,
    pub language: Option<String>,
    pub priority_score: f64,
    pub last_accessed: Option<i64>,
    pub indexed_at: Option<i64>,
}

/// Row from the symbols table with joined file path
#[derive(Debug, Clone)]
pub struct SymbolRow {
    pub id: i64,
    pub file_id: i64,
    pub name: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub file_path: String,
}

/// Index statistics
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub file_count: u64,
    pub symbol_count: u64,
    pub embedding_count: u64,
}

/// Database schema
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    hash TEXT NOT NULL,
    modified_at INTEGER NOT NULL,
    size INTEGER NOT NULL,
    language TEXT,
    priority_score REAL DEFAULT 0.5,
    last_accessed INTEGER,
    indexed_at INTEGER
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    signature TEXT,
    doc_comment TEXT
);

CREATE TABLE IF NOT EXISTS symbol_references (
    id INTEGER PRIMARY KEY,
    from_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    to_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    ref_type TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    chunk_text TEXT,
    embedding BLOB,
    chunk_type TEXT
);

CREATE TABLE IF NOT EXISTS dependencies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    dep_type TEXT,
    source TEXT,
    doc_url TEXT
);

CREATE TABLE IF NOT EXISTS git_commits (
    id INTEGER PRIMARY KEY,
    hash TEXT UNIQUE NOT NULL,
    author TEXT,
    message TEXT,
    timestamp INTEGER
);

CREATE TABLE IF NOT EXISTS git_file_changes (
    id INTEGER PRIMARY KEY,
    commit_id INTEGER REFERENCES git_commits(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    change_type TEXT,
    lines_added INTEGER,
    lines_removed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_embeddings_symbol ON embeddings(symbol_id);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_database() {
        let db = Database::in_memory().unwrap();
        let stats = db.get_stats().unwrap();
        assert_eq!(stats.file_count, 0);
    }

    #[test]
    fn test_upsert_file() {
        let db = Database::in_memory().unwrap();
        let id = db
            .upsert_file("src/main.rs", "abc123", 1234567890, 1024, "rust")
            .unwrap();
        assert!(id > 0);

        let file = db.get_file_by_path("src/main.rs").unwrap().unwrap();
        assert_eq!(file.path, "src/main.rs");
        assert_eq!(file.hash, "abc123");
    }
}
