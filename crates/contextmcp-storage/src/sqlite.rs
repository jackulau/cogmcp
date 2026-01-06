//! SQLite database for structured data storage

use contextmcp_core::{Error, Result};
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
        file_id: Option<i64>,
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
            INSERT INTO embeddings (symbol_id, file_id, chunk_text, embedding, chunk_type)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![symbol_id, file_id, chunk_text, embedding_bytes, chunk_type],
        )
        .map_err(|e| Error::Storage(format!("Failed to insert embedding: {}", e)))?;

        Ok(conn.last_insert_rowid())
    }

    /// Insert multiple embeddings in a single transaction for efficiency
    pub fn insert_embeddings_batch(
        &self,
        embeddings: &[EmbeddingInput],
    ) -> Result<Vec<i64>> {
        let conn = self.conn.lock();

        let mut ids = Vec::with_capacity(embeddings.len());

        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| Error::Storage(format!("Failed to begin transaction: {}", e)))?;

        let result = (|| {
            let mut stmt = conn
                .prepare(
                    r#"
                    INSERT INTO embeddings (symbol_id, file_id, chunk_text, embedding, chunk_type)
                    VALUES (?1, ?2, ?3, ?4, ?5)
                    "#,
                )
                .map_err(|e| Error::Storage(format!("Failed to prepare statement: {}", e)))?;

            for emb in embeddings {
                let embedding_bytes: Vec<u8> = emb.embedding
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();

                stmt.execute(params![
                    emb.symbol_id,
                    emb.file_id,
                    emb.chunk_text,
                    embedding_bytes,
                    emb.chunk_type
                ])
                .map_err(|e| Error::Storage(format!("Failed to insert embedding: {}", e)))?;

                ids.push(conn.last_insert_rowid());
            }

            Ok(())
        })();

        match result {
            Ok(()) => {
                conn.execute("COMMIT", [])
                    .map_err(|e| Error::Storage(format!("Failed to commit transaction: {}", e)))?;
                Ok(ids)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    /// Get all embeddings for a specific file
    pub fn get_embeddings_for_file(&self, file_id: i64) -> Result<Vec<EmbeddingRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT e.id, e.symbol_id, e.file_id, e.chunk_text, e.embedding, e.chunk_type,
                       s.name as symbol_name, s.kind as symbol_kind, f.path as file_path
                FROM embeddings e
                LEFT JOIN symbols s ON e.symbol_id = s.id
                LEFT JOIN files f ON e.file_id = f.id
                WHERE e.file_id = ?1
                "#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![file_id], |row| {
                let embedding_bytes: Vec<u8> = row.get(4)?;
                let embedding = bytes_to_f32_vec(&embedding_bytes);
                Ok(EmbeddingRow {
                    id: row.get(0)?,
                    symbol_id: row.get(1)?,
                    file_id: row.get(2)?,
                    chunk_text: row.get(3)?,
                    embedding,
                    chunk_type: row.get(5)?,
                    symbol_name: row.get(6)?,
                    symbol_kind: row.get(7)?,
                    file_path: row.get(8)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query embeddings: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get all embeddings in the database
    pub fn get_all_embeddings(&self) -> Result<Vec<EmbeddingRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT e.id, e.symbol_id, e.file_id, e.chunk_text, e.embedding, e.chunk_type,
                       s.name as symbol_name, s.kind as symbol_kind, f.path as file_path
                FROM embeddings e
                LEFT JOIN symbols s ON e.symbol_id = s.id
                LEFT JOIN files f ON e.file_id = f.id
                "#,
            )
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map([], |row| {
                let embedding_bytes: Vec<u8> = row.get(4)?;
                let embedding = bytes_to_f32_vec(&embedding_bytes);
                Ok(EmbeddingRow {
                    id: row.get(0)?,
                    symbol_id: row.get(1)?,
                    file_id: row.get(2)?,
                    chunk_text: row.get(3)?,
                    embedding,
                    chunk_type: row.get(5)?,
                    symbol_name: row.get(6)?,
                    symbol_kind: row.get(7)?,
                    file_path: row.get(8)?,
                })
            })
            .map_err(|e| Error::Storage(format!("Failed to query embeddings: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Delete all embeddings for a specific file
    pub fn delete_embeddings_for_file(&self, file_id: i64) -> Result<u64> {
        let conn = self.conn.lock();
        let deleted = conn
            .execute("DELETE FROM embeddings WHERE file_id = ?1", params![file_id])
            .map_err(|e| Error::Storage(format!("Failed to delete embeddings: {}", e)))?;
        Ok(deleted as u64)
    }

    /// Get the count of embeddings in the database
    pub fn get_embedding_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .map_err(|e| Error::Storage(format!("Failed to count embeddings: {}", e)))?;
        Ok(count as u64)
    }

    /// Search for similar vectors using cosine similarity
    /// Returns results ordered by similarity score (highest first)
    pub fn search_similar_vectors(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<SimilarityResult>> {
        let all_embeddings = self.get_all_embeddings()?;

        let mut results: Vec<SimilarityResult> = all_embeddings
            .into_iter()
            .filter_map(|emb| {
                let score = cosine_similarity(query_embedding, &emb.embedding);
                if score.is_finite() {
                    Some(SimilarityResult {
                        embedding_id: emb.id,
                        chunk_text: emb.chunk_text,
                        file_path: emb.file_path,
                        symbol_name: emb.symbol_name,
                        symbol_kind: emb.symbol_kind,
                        similarity_score: score,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity score descending
        results.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take the top `limit` results
        results.truncate(limit);

        Ok(results)
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

/// Input for batch embedding insert
#[derive(Debug, Clone)]
pub struct EmbeddingInput {
    pub symbol_id: Option<i64>,
    pub file_id: Option<i64>,
    pub chunk_text: String,
    pub embedding: Vec<f32>,
    pub chunk_type: String,
}

/// Row from the embeddings table with joined data
#[derive(Debug, Clone)]
pub struct EmbeddingRow {
    pub id: i64,
    pub symbol_id: Option<i64>,
    pub file_id: Option<i64>,
    pub chunk_text: String,
    pub embedding: Vec<f32>,
    pub chunk_type: String,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub file_path: Option<String>,
}

/// Result from similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub embedding_id: i64,
    pub chunk_text: String,
    pub file_path: Option<String>,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub similarity_score: f32,
}

/// Convert bytes to f32 vector (little-endian format)
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().expect("chunk size is 4");
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
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
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_embeddings_file ON embeddings(file_id);
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
    fn test_insert_embedding_roundtrip() {
        let db = Database::in_memory().unwrap();

        // Create a file first
        let file_id = db
            .upsert_file("src/lib.rs", "def456", 1234567890, 2048, "rust")
            .unwrap();

        // Insert an embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let emb_id = db
            .insert_embedding(None, Some(file_id), "test chunk", &embedding, "code")
            .unwrap();
        assert!(emb_id > 0);

        // Retrieve embeddings for the file
        let embeddings = db.get_embeddings_for_file(file_id).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].chunk_text, "test chunk");
        assert_eq!(embeddings[0].chunk_type, "code");
        assert_eq!(embeddings[0].embedding, embedding);
    }

    #[test]
    fn test_get_all_embeddings() {
        let db = Database::in_memory().unwrap();

        let file_id1 = db
            .upsert_file("src/a.rs", "aaa", 1234567890, 1024, "rust")
            .unwrap();
        let file_id2 = db
            .upsert_file("src/b.rs", "bbb", 1234567890, 1024, "rust")
            .unwrap();

        let emb1 = vec![0.1, 0.2, 0.3];
        let emb2 = vec![0.4, 0.5, 0.6];

        db.insert_embedding(None, Some(file_id1), "chunk a", &emb1, "code")
            .unwrap();
        db.insert_embedding(None, Some(file_id2), "chunk b", &emb2, "code")
            .unwrap();

        let all = db.get_all_embeddings().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_delete_embeddings_for_file() {
        let db = Database::in_memory().unwrap();

        let file_id = db
            .upsert_file("src/delete.rs", "del", 1234567890, 1024, "rust")
            .unwrap();

        let emb = vec![0.1, 0.2, 0.3];
        db.insert_embedding(None, Some(file_id), "chunk 1", &emb, "code")
            .unwrap();
        db.insert_embedding(None, Some(file_id), "chunk 2", &emb, "code")
            .unwrap();

        assert_eq!(db.get_embedding_count().unwrap(), 2);

        let deleted = db.delete_embeddings_for_file(file_id).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(db.get_embedding_count().unwrap(), 0);
    }

    #[test]
    fn test_get_embedding_count() {
        let db = Database::in_memory().unwrap();

        assert_eq!(db.get_embedding_count().unwrap(), 0);

        let file_id = db
            .upsert_file("src/count.rs", "cnt", 1234567890, 1024, "rust")
            .unwrap();

        let emb = vec![0.1, 0.2, 0.3];
        db.insert_embedding(None, Some(file_id), "chunk", &emb, "code")
            .unwrap();

        assert_eq!(db.get_embedding_count().unwrap(), 1);
    }

    #[test]
    fn test_search_similar_vectors() {
        let db = Database::in_memory().unwrap();

        let file_id = db
            .upsert_file("src/search.rs", "src", 1234567890, 1024, "rust")
            .unwrap();

        // Insert embeddings with different vectors
        // Vector 1: [1, 0, 0] - should be most similar to query
        db.insert_embedding(None, Some(file_id), "chunk x", &[1.0, 0.0, 0.0], "code")
            .unwrap();
        // Vector 2: [0, 1, 0] - orthogonal to query
        db.insert_embedding(None, Some(file_id), "chunk y", &[0.0, 1.0, 0.0], "code")
            .unwrap();
        // Vector 3: [0.7, 0.7, 0] - partially similar
        db.insert_embedding(None, Some(file_id), "chunk z", &[0.7, 0.7, 0.0], "code")
            .unwrap();

        // Search with query vector similar to [1, 0, 0]
        let query = vec![0.9, 0.1, 0.0];
        let results = db.search_similar_vectors(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be the most similar (chunk x)
        assert_eq!(results[0].chunk_text, "chunk x");
        // Similarity scores should be in descending order
        assert!(results[0].similarity_score >= results[1].similarity_score);
        assert!(results[1].similarity_score >= results[2].similarity_score);
    }

    #[test]
    fn test_search_similar_vectors_respects_limit() {
        let db = Database::in_memory().unwrap();

        let file_id = db
            .upsert_file("src/limit.rs", "lim", 1234567890, 1024, "rust")
            .unwrap();

        let emb = vec![0.1, 0.2, 0.3];
        for i in 0..10 {
            db.insert_embedding(None, Some(file_id), &format!("chunk {}", i), &emb, "code")
                .unwrap();
        }

        let results = db.search_similar_vectors(&emb, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_insert_embeddings_batch() {
        let db = Database::in_memory().unwrap();

        let file_id = db
            .upsert_file("src/batch.rs", "btch", 1234567890, 1024, "rust")
            .unwrap();

        let embeddings = vec![
            EmbeddingInput {
                symbol_id: None,
                file_id: Some(file_id),
                chunk_text: "batch chunk 1".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                chunk_type: "code".to_string(),
            },
            EmbeddingInput {
                symbol_id: None,
                file_id: Some(file_id),
                chunk_text: "batch chunk 2".to_string(),
                embedding: vec![0.4, 0.5, 0.6],
                chunk_type: "code".to_string(),
            },
            EmbeddingInput {
                symbol_id: None,
                file_id: Some(file_id),
                chunk_text: "batch chunk 3".to_string(),
                embedding: vec![0.7, 0.8, 0.9],
                chunk_type: "code".to_string(),
            },
        ];

        let ids = db.insert_embeddings_batch(&embeddings).unwrap();
        assert_eq!(ids.len(), 3);

        let all = db.get_embeddings_for_file(file_id).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_vec() {
        let original = vec![0.1f32, 0.2f32, 0.3f32];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let restored = bytes_to_f32_vec(&bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_embedding_with_symbol() {
        let db = Database::in_memory().unwrap();

        let file_id = db
            .upsert_file("src/sym.rs", "sym", 1234567890, 1024, "rust")
            .unwrap();

        let symbol_id = db
            .insert_symbol(file_id, "my_function", "function", 10, 20, Some("fn my_function()"), None)
            .unwrap();

        let emb = vec![0.1, 0.2, 0.3];
        db.insert_embedding(Some(symbol_id), Some(file_id), "function body", &emb, "function")
            .unwrap();

        let embeddings = db.get_embeddings_for_file(file_id).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].symbol_name, Some("my_function".to_string()));
        assert_eq!(embeddings[0].symbol_kind, Some("function".to_string()));
    }
}
