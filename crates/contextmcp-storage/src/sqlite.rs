//! SQLite database for structured data storage

use contextmcp_core::{Error, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// A parameter with name and optional type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: Option<String>,
}

/// Extended symbol metadata for insertion
#[derive(Debug, Clone, Default)]
pub struct ExtendedSymbolMetadata {
    pub visibility: Option<String>,
    pub is_async: bool,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_exported: bool,
    pub parent_symbol_id: Option<i64>,
    pub type_parameters: Vec<String>,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: Option<String>,
}

/// Serialize type parameters to JSON
pub fn serialize_type_params(params: &[String]) -> Option<String> {
    if params.is_empty() {
        None
    } else {
        serde_json::to_string(params).ok()
    }
}

/// Deserialize type parameters from JSON
pub fn deserialize_type_params(json: Option<&str>) -> Vec<String> {
    json.and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default()
}

/// Serialize parameters to JSON
pub fn serialize_parameters(params: &[ParameterInfo]) -> Option<String> {
    if params.is_empty() {
        None
    } else {
        serde_json::to_string(params).ok()
    }
}

/// Deserialize parameters from JSON
pub fn deserialize_parameters(json: Option<&str>) -> Vec<ParameterInfo> {
    json.and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default()
}

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
        drop(conn);

        // Run migrations for existing databases
        self.migrate_symbols_table_v2()?;
        Ok(())
    }

    /// Migration to add extended symbol metadata columns (v2)
    /// This handles both fresh installs (no-op) and upgrades gracefully.
    fn migrate_symbols_table_v2(&self) -> Result<()> {
        let conn = self.conn.lock();

        // Check if migration is needed by looking for one of the new columns
        let needs_migration: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('symbols') WHERE name = 'visibility'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count == 0)
            .unwrap_or(true);

        if !needs_migration {
            return Ok(());
        }

        // Add new columns one by one (SQLite doesn't support ADD COLUMN in batch)
        let migrations = [
            "ALTER TABLE symbols ADD COLUMN visibility TEXT",
            "ALTER TABLE symbols ADD COLUMN is_async INTEGER DEFAULT 0",
            "ALTER TABLE symbols ADD COLUMN is_static INTEGER DEFAULT 0",
            "ALTER TABLE symbols ADD COLUMN is_abstract INTEGER DEFAULT 0",
            "ALTER TABLE symbols ADD COLUMN is_exported INTEGER DEFAULT 0",
            "ALTER TABLE symbols ADD COLUMN parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL",
            "ALTER TABLE symbols ADD COLUMN type_parameters TEXT",
            "ALTER TABLE symbols ADD COLUMN parameters TEXT",
            "ALTER TABLE symbols ADD COLUMN return_type TEXT",
        ];

        for migration in migrations {
            // Ignore errors for columns that already exist
            let _ = conn.execute(migration, []);
        }

        // Create index on parent_symbol_id if it doesn't exist
        let _ = conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id)",
            [],
        );

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

    /// Insert a symbol with extended metadata
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
            &ExtendedSymbolMetadata::default(),
        )
    }

    /// Insert a symbol with full extended metadata
    pub fn insert_symbol_extended(
        &self,
        file_id: i64,
        name: &str,
        kind: &str,
        start_line: u32,
        end_line: u32,
        signature: Option<&str>,
        doc_comment: Option<&str>,
        extended: &ExtendedSymbolMetadata,
    ) -> Result<i64> {
        let conn = self.conn.lock();

        let type_params_json = serialize_type_params(&extended.type_parameters);
        let params_json = serialize_parameters(&extended.parameters);

        conn.execute(
            r#"
            INSERT INTO symbols (
                file_id, name, kind, start_line, end_line, signature, doc_comment,
                visibility, is_async, is_static, is_abstract, is_exported,
                parent_symbol_id, type_parameters, parameters, return_type
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
            "#,
            params![
                file_id,
                name,
                kind,
                start_line,
                end_line,
                signature,
                doc_comment,
                extended.visibility,
                extended.is_async as i32,
                extended.is_static as i32,
                extended.is_abstract as i32,
                extended.is_exported as i32,
                extended.parent_symbol_id,
                type_params_json,
                params_json,
                extended.return_type
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

    /// Find symbols by name
    pub fn find_symbols_by_name(&self, name: &str, fuzzy: bool) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();
        let pattern = if fuzzy {
            format!("%{}%", name)
        } else {
            name.to_string()
        };

        let base_query = r#"
            SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line,
                   s.signature, s.doc_comment, f.path,
                   s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported,
                   s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
            FROM symbols s JOIN files f ON s.file_id = f.id
        "#;

        let query = if fuzzy {
            format!("{} WHERE s.name LIKE ?1", base_query)
        } else {
            format!("{} WHERE s.name = ?1", base_query)
        };

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![pattern], |row| row_to_symbol_row(row))
            .map_err(|e| Error::Storage(format!("Failed to query symbols: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get children of a symbol (nested symbols)
    pub fn get_symbol_children(&self, symbol_id: i64) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();
        let query = r#"
            SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line,
                   s.signature, s.doc_comment, f.path,
                   s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported,
                   s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.parent_symbol_id = ?1
            ORDER BY s.start_line
        "#;

        let mut stmt = conn
            .prepare(query)
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![symbol_id], |row| row_to_symbol_row(row))
            .map_err(|e| Error::Storage(format!("Failed to query symbol children: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get symbols by visibility
    pub fn get_symbols_by_visibility(&self, visibility: &str) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();
        let query = r#"
            SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line,
                   s.signature, s.doc_comment, f.path,
                   s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported,
                   s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.visibility = ?1
            ORDER BY f.path, s.start_line
        "#;

        let mut stmt = conn
            .prepare(query)
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![visibility], |row| row_to_symbol_row(row))
            .map_err(|e| Error::Storage(format!("Failed to query symbols by visibility: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Storage(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }

    /// Get all symbols for a file with extended metadata
    pub fn get_file_symbols(&self, file_id: i64) -> Result<Vec<SymbolRow>> {
        let conn = self.conn.lock();
        let query = r#"
            SELECT s.id, s.file_id, s.name, s.kind, s.start_line, s.end_line,
                   s.signature, s.doc_comment, f.path,
                   s.visibility, s.is_async, s.is_static, s.is_abstract, s.is_exported,
                   s.parent_symbol_id, s.type_parameters, s.parameters, s.return_type
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.file_id = ?1
            ORDER BY s.start_line
        "#;

        let mut stmt = conn
            .prepare(query)
            .map_err(|e| Error::Storage(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params![file_id], |row| row_to_symbol_row(row))
            .map_err(|e| Error::Storage(format!("Failed to query file symbols: {}", e)))?;

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

/// Helper function to convert a database row to SymbolRow
/// Expected column order:
/// 0: id, 1: file_id, 2: name, 3: kind, 4: start_line, 5: end_line,
/// 6: signature, 7: doc_comment, 8: file_path,
/// 9: visibility, 10: is_async, 11: is_static, 12: is_abstract, 13: is_exported,
/// 14: parent_symbol_id, 15: type_parameters, 16: parameters, 17: return_type
fn row_to_symbol_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SymbolRow> {
    let type_params_json: Option<String> = row.get(15)?;
    let params_json: Option<String> = row.get(16)?;

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
        is_async: row.get::<_, Option<i32>>(10)?.unwrap_or(0) != 0,
        is_static: row.get::<_, Option<i32>>(11)?.unwrap_or(0) != 0,
        is_abstract: row.get::<_, Option<i32>>(12)?.unwrap_or(0) != 0,
        is_exported: row.get::<_, Option<i32>>(13)?.unwrap_or(0) != 0,
        parent_symbol_id: row.get(14)?,
        type_parameters: deserialize_type_params(type_params_json.as_deref()),
        parameters: deserialize_parameters(params_json.as_deref()),
        return_type: row.get(17)?,
    })
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
    // Extended metadata (v2)
    pub visibility: Option<String>,
    pub is_async: bool,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_exported: bool,
    pub parent_symbol_id: Option<i64>,
    pub type_parameters: Vec<String>,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: Option<String>,
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
    doc_comment TEXT,
    -- Extended symbol metadata (v2)
    visibility TEXT,
    is_async INTEGER DEFAULT 0,
    is_static INTEGER DEFAULT 0,
    is_abstract INTEGER DEFAULT 0,
    is_exported INTEGER DEFAULT 0,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    type_parameters TEXT,  -- JSON array
    parameters TEXT,       -- JSON array of {name, type} objects
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
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id);
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
    fn test_insert_symbol_basic() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Test basic insert (backward compatible)
        let symbol_id = db
            .insert_symbol(file_id, "my_function", "function", 10, 20, Some("fn my_function()"), None)
            .unwrap();
        assert!(symbol_id > 0);

        // Verify it can be found
        let symbols = db.find_symbols_by_name("my_function", false).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "my_function");
        assert_eq!(symbols[0].kind, "function");
        // Extended fields should have defaults
        assert!(!symbols[0].is_async);
        assert!(!symbols[0].is_static);
    }

    #[test]
    fn test_insert_symbol_extended() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Create extended metadata
        let extended = ExtendedSymbolMetadata {
            visibility: Some("pub".to_string()),
            is_async: true,
            is_static: false,
            is_abstract: false,
            is_exported: true,
            parent_symbol_id: None,
            type_parameters: vec!["T".to_string(), "U".to_string()],
            parameters: vec![
                ParameterInfo { name: "input".to_string(), param_type: Some("T".to_string()) },
                ParameterInfo { name: "count".to_string(), param_type: Some("usize".to_string()) },
            ],
            return_type: Some("Result<U>".to_string()),
        };

        let symbol_id = db
            .insert_symbol_extended(
                file_id,
                "async_transform",
                "function",
                50,
                100,
                Some("pub async fn async_transform<T, U>(input: T, count: usize) -> Result<U>"),
                Some("/// Transforms input asynchronously"),
                &extended,
            )
            .unwrap();
        assert!(symbol_id > 0);

        // Find the symbol and verify extended metadata
        let symbols = db.find_symbols_by_name("async_transform", false).unwrap();
        assert_eq!(symbols.len(), 1);
        let sym = &symbols[0];
        assert_eq!(sym.visibility, Some("pub".to_string()));
        assert!(sym.is_async);
        assert!(!sym.is_static);
        assert!(sym.is_exported);
        assert_eq!(sym.type_parameters, vec!["T", "U"]);
        assert_eq!(sym.parameters.len(), 2);
        assert_eq!(sym.parameters[0].name, "input");
        assert_eq!(sym.parameters[0].param_type, Some("T".to_string()));
        assert_eq!(sym.return_type, Some("Result<U>".to_string()));
    }

    #[test]
    fn test_symbol_children() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Create a parent class/struct
        let parent_id = db
            .insert_symbol_extended(
                file_id,
                "MyClass",
                "class",
                10,
                100,
                Some("class MyClass"),
                None,
                &ExtendedSymbolMetadata {
                    visibility: Some("pub".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();

        // Create child methods
        let _method1_id = db
            .insert_symbol_extended(
                file_id,
                "method_a",
                "method",
                20,
                30,
                Some("fn method_a(&self)"),
                None,
                &ExtendedSymbolMetadata {
                    visibility: Some("pub".to_string()),
                    parent_symbol_id: Some(parent_id),
                    ..Default::default()
                },
            )
            .unwrap();

        let _method2_id = db
            .insert_symbol_extended(
                file_id,
                "method_b",
                "method",
                40,
                50,
                Some("fn method_b(&self)"),
                None,
                &ExtendedSymbolMetadata {
                    visibility: Some("private".to_string()),
                    parent_symbol_id: Some(parent_id),
                    is_static: true,
                    ..Default::default()
                },
            )
            .unwrap();

        // Get children
        let children = db.get_symbol_children(parent_id).unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].name, "method_a");
        assert_eq!(children[1].name, "method_b");
        assert!(children[1].is_static);
    }

    #[test]
    fn test_get_symbols_by_visibility() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Create symbols with different visibility
        db.insert_symbol_extended(
            file_id, "pub_func", "function", 10, 20, None, None,
            &ExtendedSymbolMetadata { visibility: Some("pub".to_string()), ..Default::default() },
        ).unwrap();

        db.insert_symbol_extended(
            file_id, "private_func", "function", 30, 40, None, None,
            &ExtendedSymbolMetadata { visibility: Some("private".to_string()), ..Default::default() },
        ).unwrap();

        db.insert_symbol_extended(
            file_id, "another_pub", "function", 50, 60, None, None,
            &ExtendedSymbolMetadata { visibility: Some("pub".to_string()), ..Default::default() },
        ).unwrap();

        // Query by visibility
        let pub_symbols = db.get_symbols_by_visibility("pub").unwrap();
        assert_eq!(pub_symbols.len(), 2);

        let private_symbols = db.get_symbols_by_visibility("private").unwrap();
        assert_eq!(private_symbols.len(), 1);
        assert_eq!(private_symbols[0].name, "private_func");
    }

    #[test]
    fn test_get_file_symbols() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/lib.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Insert multiple symbols
        db.insert_symbol(file_id, "func_c", "function", 50, 60, None, None).unwrap();
        db.insert_symbol(file_id, "func_a", "function", 10, 20, None, None).unwrap();
        db.insert_symbol(file_id, "func_b", "function", 30, 40, None, None).unwrap();

        // Get symbols - should be ordered by start_line
        let symbols = db.get_file_symbols(file_id).unwrap();
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].name, "func_a");
        assert_eq!(symbols[1].name, "func_b");
        assert_eq!(symbols[2].name, "func_c");
    }

    #[test]
    fn test_json_serialization() {
        // Test type parameters serialization
        let params = vec!["T".to_string(), "U".to_string()];
        let json = serialize_type_params(&params);
        assert!(json.is_some());
        let deserialized = deserialize_type_params(json.as_deref());
        assert_eq!(deserialized, params);

        // Empty params should return None
        let empty: Vec<String> = vec![];
        assert!(serialize_type_params(&empty).is_none());
        assert!(deserialize_type_params(None).is_empty());

        // Test parameter info serialization
        let params = vec![
            ParameterInfo { name: "x".to_string(), param_type: Some("i32".to_string()) },
            ParameterInfo { name: "y".to_string(), param_type: None },
        ];
        let json = serialize_parameters(&params);
        assert!(json.is_some());
        let deserialized = deserialize_parameters(json.as_deref());
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized[0].name, "x");
        assert_eq!(deserialized[0].param_type, Some("i32".to_string()));
        assert_eq!(deserialized[1].name, "y");
        assert_eq!(deserialized[1].param_type, None);
    }

    #[test]
    fn test_existing_queries_backward_compatible() {
        let db = Database::in_memory().unwrap();
        let file_id = db
            .upsert_file("src/main.rs", "hash1", 1234567890, 512, "rust")
            .unwrap();

        // Use old-style insert (no extended metadata)
        db.insert_symbol(file_id, "old_func", "function", 10, 20, Some("fn old_func()"), None).unwrap();

        // find_symbols_by_name should still work
        let symbols = db.find_symbols_by_name("old_func", false).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "old_func");

        // Extended fields should have sensible defaults
        assert!(symbols[0].visibility.is_none());
        assert!(!symbols[0].is_async);
        assert!(symbols[0].type_parameters.is_empty());
        assert!(symbols[0].parameters.is_empty());

        // fuzzy search should still work
        let fuzzy = db.find_symbols_by_name("old", true).unwrap();
        assert_eq!(fuzzy.len(), 1);

        // Stats should still work
        let stats = db.get_stats().unwrap();
        assert_eq!(stats.symbol_count, 1);
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
