//! Tantivy full-text search index

use cogmcp_core::{Error, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};

/// Full-text search index using Tantivy
pub struct FullTextIndex {
    index: Index,
    reader: IndexReader,
    writer: parking_lot::Mutex<IndexWriter>,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    path_field: Field,
    content_field: Field,
    line_number_field: Field,
}

impl FullTextIndex {
    /// Create or open a Tantivy index at the given path
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let schema = Self::build_schema();
        let path_field = schema.get_field("path").unwrap();
        let content_field = schema.get_field("content").unwrap();
        let line_number_field = schema.get_field("line_number").unwrap();

        let index = Index::open_or_create(
            tantivy::directory::MmapDirectory::open(path)
                .map_err(|e| Error::Storage(format!("Failed to open index directory: {}", e)))?,
            schema.clone(),
        )
        .map_err(|e| Error::Storage(format!("Failed to open index: {}", e)))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::Storage(format!("Failed to create reader: {}", e)))?;

        let writer = index
            .writer(50_000_000) // 50MB buffer
            .map_err(|e| Error::Storage(format!("Failed to create writer: {}", e)))?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema,
            path_field,
            content_field,
            line_number_field,
        })
    }

    /// Create an in-memory index (for testing)
    pub fn in_memory() -> Result<Self> {
        let schema = Self::build_schema();
        let path_field = schema.get_field("path").unwrap();
        let content_field = schema.get_field("content").unwrap();
        let line_number_field = schema.get_field("line_number").unwrap();

        let index = Index::create_in_ram(schema.clone());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::Storage(format!("Failed to create reader: {}", e)))?;

        let writer = index
            .writer(50_000_000)
            .map_err(|e| Error::Storage(format!("Failed to create writer: {}", e)))?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema,
            path_field,
            content_field,
            line_number_field,
        })
    }

    fn build_schema() -> Schema {
        let mut schema_builder = Schema::builder();

        // File path (stored, indexed for filtering)
        schema_builder.add_text_field("path", STRING | STORED);

        // Content (indexed for full-text search, stored for snippets)
        schema_builder.add_text_field("content", TEXT | STORED);

        // Line number (stored for result display)
        schema_builder.add_u64_field("line_number", INDEXED | STORED);

        schema_builder.build()
    }

    /// Index a file's content line by line
    pub fn index_file(&self, path: &str, content: &str) -> Result<()> {
        let writer = self.writer.lock();

        // Delete existing documents for this file
        let path_term = tantivy::Term::from_field_text(self.path_field, path);
        writer.delete_term(path_term);

        // Index each line
        for (line_num, line) in content.lines().enumerate() {
            if !line.trim().is_empty() {
                writer
                    .add_document(doc!(
                        self.path_field => path,
                        self.content_field => line,
                        self.line_number_field => (line_num + 1) as u64
                    ))
                    .map_err(|e| Error::Storage(format!("Failed to add document: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Commit pending changes
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        writer
            .commit()
            .map_err(|e| Error::Storage(format!("Failed to commit: {}", e)))?;
        // Reload the reader to see the committed changes
        self.reader
            .reload()
            .map_err(|e| Error::Storage(format!("Failed to reload reader: {}", e)))?;
        Ok(())
    }

    /// Search for content matching a query
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| Error::Search(format!("Failed to parse query: {}", e)))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| Error::Search(format!("Search failed: {}", e)))?;

        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| Error::Search(format!("Failed to retrieve document: {}", e)))?;

            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let content = doc
                .get_first(self.content_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let line_number = doc
                .get_first(self.line_number_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            results.push(SearchHit {
                path,
                content,
                line_number,
                score: _score,
            });
        }

        Ok(results)
    }

    /// Delete all documents for a file
    pub fn delete_file(&self, path: &str) -> Result<()> {
        let writer = self.writer.lock();
        let path_term = tantivy::Term::from_field_text(self.path_field, path);
        writer.delete_term(path_term);
        Ok(())
    }
}

/// A search hit from the full-text index
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub path: String,
    pub content: String,
    pub line_number: u32,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_and_search() {
        let index = FullTextIndex::in_memory().unwrap();

        index
            .index_file(
                "src/main.rs",
                "fn main() {\n    println!(\"Hello, world!\");\n}",
            )
            .unwrap();
        index.commit().unwrap();

        let results = index.search("println", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].path, "src/main.rs");
    }
}
