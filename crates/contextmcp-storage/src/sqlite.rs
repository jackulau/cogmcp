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

    /// Insert a symbol with basic metadata (backward compatible)
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
        self.insert_symbol_extended(
            file_id,
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            None,         // visibility
            false, false, false, false, false, false, // modifiers
            None,         // parent_symbol_id
            None,         // type_parameters
            None,         // parameters
            None,         // return_type
        )
    }

    /// Insert a symbol with extended metadata
    #[allow(clippy::too_many_arguments)]
    pub fn insert_symbol_extended(
        &self,
        file_id: i64,
        name: &str,
        kind: &str,
        start_line: u32,
        end_line: u32,
        signature: Option<&str>,
        doc_comment: Option<&str>,
        visibility: Option<&str>,
        is_async: bool,
        is_static: bool,
        is_abstract: bool,
        is_exported: bool,
        is_const: bool,
        is_unsafe: bool,
        parent_symbol_id: Option<i64>,
        type_parameters: Option<&str>,
        parameters: Option<&str>,
        return_type: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO symbols (
                file_id, name, kind, start_line, end_line, signature, doc_comment,
                visibility, is_async, is_static, is_abstract, is_exported, is_const, is_unsafe,
                parent_symbol_id, type_parameters, parameters, return_type
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
            "#,
            params![
                file_id, name, kind, start_line, end_line, signature, doc_comment,
                visibility, is_async as i32, is_static as i32, is_abstract as i32,
                is_exported as i32, is_const as i32, is_unsafe as i32,
                parent_symbol_id, type_parameters, parameters, return_type
            ],
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

    /// Update a symbol's parent reference
    pub fn update_symbol_parent(&self, symbol_id: i64, parent_id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE symbols SET parent_symbol_id = ?1 WHERE id = ?2",
            params![parent_id, symbol_id],
        )
        .map_err(|e| Error::Storage(format!("Failed to update symbol parent: {}", e)))?;
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
            r#"SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path,
                      s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported, s.is_const, s.is_unsafe,
                      s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
             FROM symbols s JOIN files f ON s.file_id = f.id
             WHERE s.name LIKE ?1"#
        } else {
            r#"SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path,
                      s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported, s.is_const, s.is_unsafe,
                      s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
             FROM symbols s JOIN files f ON s.file_id = f.id
             WHERE s.name = ?1"#
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
                    visibility: row.get(9)?,
                    is_async: row.get::<_, i32>(10)? != 0,
                    is_static: row.get::<_, i32>(11)? != 0,
                    is_abstract: row.get::<_, i32>(12)? != 0,
                    is_exported: row.get::<_, i32>(13)? != 0,
                    is_const: row.get::<_, i32>(14)? != 0,
                    is_unsafe: row.get::<_, i32>(15)? != 0,
                    parent_symbol_id: row.get(16)?,
                    type_parameters: row.get(17)?,
                    parameters: row.get(18)?,
                    return_type: row.get(19)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query symbols: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Find symbols by visibility
    pub fn find_symbols_by_visibility(&self, visibility: &str) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare(
                r#"SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path,
                          s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported, s.is_const, s.is_unsafe,
                          s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.visibility = ?1"#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![visibility], |row| {
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
                    visibility: row.get(9)?,
                    is_async: row.get::<_, i32>(10)? != 0,
                    is_static: row.get::<_, i32>(11)? != 0,
                    is_abstract: row.get::<_, i32>(12)? != 0,
                    is_exported: row.get::<_, i32>(13)? != 0,
                    is_const: row.get::<_, i32>(14)? != 0,
                    is_unsafe: row.get::<_, i32>(15)? != 0,
                    parent_symbol_id: row.get(16)?,
                    type_parameters: row.get(17)?,
                    parameters: row.get(18)?,
                    return_type: row.get(19)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query symbols: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get child symbols (symbols with given parent_symbol_id)
    pub fn get_symbol_children(&self, parent_id: i64) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare(
                r#"SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path,
                          s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported, s.is_const, s.is_unsafe,
                          s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.parent_symbol_id = ?1"#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![parent_id], |row| {
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
                    visibility: row.get(9)?,
                    is_async: row.get::<_, i32>(10)? != 0,
                    is_static: row.get::<_, i32>(11)? != 0,
                    is_abstract: row.get::<_, i32>(12)? != 0,
                    is_exported: row.get::<_, i32>(13)? != 0,
                    is_const: row.get::<_, i32>(14)? != 0,
                    is_unsafe: row.get::<_, i32>(15)? != 0,
                    parent_symbol_id: row.get(16)?,
                    type_parameters: row.get(17)?,
                    parameters: row.get(18)?,
                    return_type: row.get(19)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query symbols: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get symbols for a file with enhanced data
    pub fn get_file_symbols(&self, file_id: i64) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare(
                r#"SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line, s.signature, s.doc_comment, f.path,
                          s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported, s.is_const, s.is_unsafe,
                          s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.file_id = ?1
                   ORDER BY s.start_line"#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![file_id], |row| {
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
                    visibility: row.get(9)?,
                    is_async: row.get::<_, i32>(10)? != 0,
                    is_static: row.get::<_, i32>(11)? != 0,
                    is_abstract: row.get::<_, i32>(12)? != 0,
                    is_exported: row.get::<_, i32>(13)? != 0,
                    is_const: row.get::<_, i32>(14)? != 0,
                    is_unsafe: row.get::<_, i32>(15)? != 0,
                    parent_symbol_id: row.get(16)?,
                    type_parameters: row.get(17)?,
                    parameters: row.get(18)?,
                    return_type: row.get(19)?,
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

    /// Get extended index statistics including symbol breakdown
    pub fn get_extended_stats(&self) -> Result<ExtendedIndexStats> {
        let conn = self.conn.lock();
        let mut stats = ExtendedIndexStats::default();

        // Basic counts
        stats.file_count = conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get::<_, i64>(0))
            .map_err(|e| Error::Storage(format!("Failed to count files: {}", e)))? as u64;

        stats.symbol_count = conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get::<_, i64>(0))
            .map_err(|e| Error::Storage(format!("Failed to count symbols: {}", e)))? as u64;

        stats.embedding_count = conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get::<_, i64>(0))
            .map_err(|e| Error::Storage(format!("Failed to count embeddings: {}", e)))? as u64;

        // Symbols by kind
        let mut stmt = conn
            .prepare("SELECT kind, COUNT(*) FROM symbols GROUP BY kind")
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map_err(|e| Error::Storage(format!("Failed to query symbol kinds: {}", e)))?;
        for row in rows {
            let (kind, count) = row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?;
            stats.symbols_by_kind.insert(kind, count as u64);
        }

        // Symbols by visibility
        let mut stmt = conn
            .prepare("SELECT COALESCE(visibility, 'unknown'), COUNT(*) FROM symbols GROUP BY visibility")
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map_err(|e| Error::Storage(format!("Failed to query symbol visibility: {}", e)))?;
        for row in rows {
            let (visibility, count) = row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?;
            stats.symbols_by_visibility.insert(visibility, count as u64);
        }

        // Count symbols with visibility set
        stats.symbols_with_visibility = conn
            .query_row(
                "SELECT COUNT(*) FROM symbols WHERE visibility IS NOT NULL AND visibility != 'unknown'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| Error::Storage(format!("Failed to count visibility: {}", e)))? as u64;

        // Count symbols with parent
        stats.symbols_with_parent = conn
            .query_row(
                "SELECT COUNT(*) FROM symbols WHERE parent_symbol_id IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| Error::Storage(format!("Failed to count parent relationships: {}", e)))? as u64;

        Ok(stats)
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
    pub visibility: Option<String>,
    pub is_async: bool,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_exported: bool,
    pub is_const: bool,
    pub is_unsafe: bool,
    pub parent_symbol_id: Option<i64>,
    pub type_parameters: Option<String>,
    pub parameters: Option<String>,
    pub return_type: Option<String>,
}

/// Index statistics
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub file_count: u64,
    pub symbol_count: u64,
    pub embedding_count: u64,
}

/// Extended index statistics including symbol breakdown
#[derive(Debug, Clone, Default)]
pub struct ExtendedIndexStats {
    pub file_count: u64,
    pub symbol_count: u64,
    pub embedding_count: u64,
    pub symbols_by_kind: std::collections::HashMap<String, u64>,
    pub symbols_by_visibility: std::collections::HashMap<String, u64>,
    pub symbols_with_visibility: u64,
    pub symbols_with_parent: u64,
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
    doc_comment TEXT,
    visibility TEXT,
    is_async INTEGER DEFAULT 0,
    is_static INTEGER DEFAULT 0,
    is_abstract INTEGER DEFAULT 0,
    is_exported INTEGER DEFAULT 0,
    is_const INTEGER DEFAULT 0,
    is_unsafe INTEGER DEFAULT 0,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    type_parameters TEXT,
    parameters TEXT,
    return_type TEXT
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
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id);
CREATE INDEX IF NOT EXISTS idx_symbols_visibility ON symbols(visibility);
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

    #[test]
    fn test_insert_symbol_extended() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash123", 1234567890, 1024, "rust")
            .unwrap();

        let symbol_id = db
            .insert_symbol_extended(
                file_id,
                "my_function",
                "function",
                10,
                20,
                Some("pub async fn my_function()"),
                Some("A test function"),
                Some("public"),
                true,  // is_async
                false, // is_static
                false, // is_abstract
                false, // is_exported
                false, // is_const
                false, // is_unsafe
                None,  // parent_symbol_id
                Some("[\"T\", \"U\"]"),
                Some("[{\"name\": \"x\", \"type_annotation\": \"i32\"}]"),
                Some("Result<()>"),
            )
            .unwrap();

        assert!(symbol_id > 0);

        // Find the symbol and verify fields
        let symbols = db.find_symbols_by_name("my_function", false).unwrap();
        assert_eq!(symbols.len(), 1);

        let sym = &symbols[0];
        assert_eq!(sym.name, "my_function");
        assert_eq!(sym.kind, "function");
        assert_eq!(sym.visibility, Some("public".to_string()));
        assert!(sym.is_async);
        assert!(!sym.is_static);
        assert_eq!(sym.return_type, Some("Result<()>".to_string()));
    }

    #[test]
    fn test_symbol_parent_relationship() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash123", 1234567890, 1024, "rust")
            .unwrap();

        // Insert parent symbol (struct)
        let parent_id = db
            .insert_symbol_extended(
                file_id,
                "MyStruct",
                "struct",
                1,
                10,
                Some("pub struct MyStruct"),
                None,
                Some("public"),
                false, false, false, false, false, false,
                None, None, None, None,
            )
            .unwrap();

        // Insert child symbol (method)
        let child_id = db
            .insert_symbol_extended(
                file_id,
                "my_method",
                "function",
                5,
                8,
                Some("pub fn my_method(&self)"),
                None,
                Some("public"),
                false, false, false, false, false, false,
                None, None, None, None,
            )
            .unwrap();

        // Update parent relationship
        db.update_symbol_parent(child_id, parent_id).unwrap();

        // Verify child was updated
        let children = db.get_symbol_children(parent_id).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "my_method");
        assert_eq!(children[0].parent_symbol_id, Some(parent_id));
    }

    #[test]
    fn test_find_symbols_by_visibility() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash123", 1234567890, 1024, "rust")
            .unwrap();

        // Insert public function
        db.insert_symbol_extended(
            file_id,
            "public_fn",
            "function",
            1, 5,
            None, None,
            Some("public"),
            false, false, false, false, false, false,
            None, None, None, None,
        ).unwrap();

        // Insert private function
        db.insert_symbol_extended(
            file_id,
            "private_fn",
            "function",
            6, 10,
            None, None,
            Some("private"),
            false, false, false, false, false, false,
            None, None, None, None,
        ).unwrap();

        // Find public symbols
        let public_symbols = db.find_symbols_by_visibility("public").unwrap();
        assert_eq!(public_symbols.len(), 1);
        assert_eq!(public_symbols[0].name, "public_fn");

        // Find private symbols
        let private_symbols = db.find_symbols_by_visibility("private").unwrap();
        assert_eq!(private_symbols.len(), 1);
        assert_eq!(private_symbols[0].name, "private_fn");
    }

    #[test]
    fn test_extended_stats() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash123", 1234567890, 1024, "rust")
            .unwrap();

        // Insert symbols with different kinds and visibilities
        db.insert_symbol_extended(
            file_id, "func1", "function", 1, 5, None, None,
            Some("public"), false, false, false, false, false, false,
            None, None, None, None,
        ).unwrap();

        let parent_id = db.insert_symbol_extended(
            file_id, "struct1", "struct", 6, 15, None, None,
            Some("public"), false, false, false, false, false, false,
            None, None, None, None,
        ).unwrap();

        let child_id = db.insert_symbol_extended(
            file_id, "method1", "function", 10, 14, None, None,
            Some("private"), false, false, false, false, false, false,
            None, None, None, None,
        ).unwrap();

        db.update_symbol_parent(child_id, parent_id).unwrap();

        // Get extended stats
        let stats = db.get_extended_stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.symbol_count, 3);

        // Check symbols by kind
        assert_eq!(*stats.symbols_by_kind.get("function").unwrap_or(&0), 2);
        assert_eq!(*stats.symbols_by_kind.get("struct").unwrap_or(&0), 1);

        // Check symbols by visibility
        assert_eq!(*stats.symbols_by_visibility.get("public").unwrap_or(&0), 2);
        assert_eq!(*stats.symbols_by_visibility.get("private").unwrap_or(&0), 1);

        // Check visibility extraction (all 3 have visibility set)
        assert_eq!(stats.symbols_with_visibility, 3);

        // Check parent relationships (1 has parent)
        assert_eq!(stats.symbols_with_parent, 1);
    }
}
