//! Main MCP server implementation

use cogmcp_core::{Config, Result};
use cogmcp_embeddings::{EmbeddingEngine, ModelConfig};
use cogmcp_index::{CodeParser, CodebaseIndexer};
use cogmcp_search::{HybridSearch, SearchMode, SemanticSearch};
use cogmcp_storage::{Database, FullTextIndex};
use cogmcp_watcher::{FileDebouncer, FilePrioritizer, FileWatcher};
use parking_lot::Mutex;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolResult, Content, Implementation, ListToolsResult, ServerCapabilities, ServerInfo, Tool,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// MCP server for context management
#[derive(Clone)]
pub struct CogMcpServer {
    pub root: PathBuf,
    pub config: Config,
    pub db: Arc<Database>,
    pub text_index: Arc<FullTextIndex>,
    pub parser: Arc<CodeParser>,
    pub embedding_engine: Option<Arc<Mutex<EmbeddingEngine>>>,
    pub semantic_search: Option<Arc<SemanticSearch>>,
    /// File watcher for real-time file change monitoring
    pub watcher: Option<Arc<FileWatcher>>,
    /// File prioritizer for determining Hot/Warm/Cold tiers
    pub prioritizer: Option<Arc<FilePrioritizer>>,
    /// File debouncer for Warm-tier files
    pub debouncer: Option<Arc<FileDebouncer>>,
}

impl CogMcpServer {
    /// Create a new server instance
    pub fn new(root: PathBuf) -> Result<Self> {
        let config = Config::load()?;
        let db = Arc::new(Database::open(&Config::database_path()?)?);
        let text_index = Arc::new(FullTextIndex::open(&Config::tantivy_path()?)?);
        let parser = Arc::new(CodeParser::new());

        // Initialize embedding engine if enabled
        let (embedding_engine, semantic_search) = if config.indexing.enable_embeddings {
            match Self::init_embedding_engine(&config, db.clone()) {
                Ok((engine, search)) => (Some(engine), Some(search)),
                Err(e) => {
                    warn!("Failed to initialize embedding engine: {}. Semantic search disabled.", e);
                    (None, None)
                }
            }
        } else {
            debug!("Embeddings disabled in configuration");
            (None, None)
        };

        // Initialize watcher components if enabled
        let (watcher, prioritizer, debouncer) = if config.watching.enabled {
            match Self::init_watcher(&root, &config) {
                Ok((w, p, d)) => {
                    info!("File watcher initialized for {}", root.display());
                    (Some(w), Some(p), Some(d))
                }
                Err(e) => {
                    warn!("Failed to initialize file watcher: {}. File watching disabled.", e);
                    (None, None, None)
                }
            }
        } else {
            debug!("File watching disabled in configuration");
            (None, None, None)
        };

        Ok(Self {
            root,
            config,
            db,
            text_index,
            parser,
            embedding_engine,
            semantic_search,
            watcher,
            prioritizer,
            debouncer,
        })
    }

    /// Create a server with in-memory storage (for testing)
    pub fn in_memory(root: PathBuf) -> Result<Self> {
        let config = Config::default();
        let db = Arc::new(Database::in_memory()?);
        let text_index = Arc::new(FullTextIndex::in_memory()?);
        let parser = Arc::new(CodeParser::new());

        Ok(Self {
            root,
            config,
            db,
            text_index,
            parser,
            embedding_engine: None,
            semantic_search: None,
            watcher: None,
            prioritizer: None,
            debouncer: None,
        })
    }

    /// Initialize the embedding engine and semantic search
    fn init_embedding_engine(
        config: &Config,
        db: Arc<Database>,
    ) -> Result<(Arc<Mutex<EmbeddingEngine>>, Arc<SemanticSearch>)> {
        let model_path = config
            .indexing
            .embedding_model
            .clone()
            .or_else(|| Config::model_path().ok().map(|p| p.to_string_lossy().to_string()))
            .unwrap_or_default();

        let model_config = ModelConfig {
            model_path,
            ..Default::default()
        };

        let engine = Arc::new(Mutex::new(EmbeddingEngine::new(model_config)?));
        let semantic = Arc::new(SemanticSearch::new(engine.clone(), db));

        {
            let engine_guard = engine.lock();
            info!(
                "Embedding engine initialized (model loaded: {}, dim: {})",
                engine_guard.is_loaded(),
                engine_guard.embedding_dim()
            );
        }

        Ok((engine, semantic))
    }

    /// Initialize the file watcher components
    fn init_watcher(
        root: &Path,
        config: &Config,
    ) -> Result<(Arc<FileWatcher>, Arc<FilePrioritizer>, Arc<FileDebouncer>)> {
        let watcher = Arc::new(FileWatcher::new(root)?);
        let prioritizer = Arc::new(FilePrioritizer::new(config.watching.hot_threshold_seconds));
        let debouncer = Arc::new(FileDebouncer::new(config.watching.debounce_ms));

        Ok((watcher, prioritizer, debouncer))
    }

    /// Check if file watching is enabled and available
    pub fn has_watcher(&self) -> bool {
        self.watcher.is_some()
    }

    /// Get the file prioritizer if available
    pub fn prioritizer(&self) -> Option<&Arc<FilePrioritizer>> {
        self.prioritizer.as_ref()
    }

    /// Get the file debouncer if available
    pub fn debouncer(&self) -> Option<&Arc<FileDebouncer>> {
        self.debouncer.as_ref()
    }

    /// Get the file watcher if available
    pub fn watcher(&self) -> Option<&Arc<FileWatcher>> {
        self.watcher.as_ref()
    }

    /// Check if semantic search is available
    pub fn has_semantic_search(&self) -> bool {
        self.semantic_search
            .as_ref()
            .map_or(false, |s| s.is_available())
    }

    /// Index the codebase
    pub fn index(&self) -> Result<()> {
        let indexer = if let Some(ref engine) = self.embedding_engine {
            CodebaseIndexer::with_embedding_engine(
                self.root.clone(),
                self.config.clone(),
                self.parser.clone(),
                engine.clone(),
            )?
        } else {
            CodebaseIndexer::new(self.root.clone(), self.config.clone(), self.parser.clone())?
        };

        let result = indexer.index_all(&self.db, &self.text_index)?;
        info!(
            "Indexed {} files ({} skipped, {} errors, {} symbols)",
            result.indexed, result.skipped, result.errors, result.symbols_extracted
        );

        // Invalidate semantic search cache after indexing
        if let Some(ref semantic) = self.semantic_search {
            semantic.invalidate_cache();
        }

        Ok(())
    }

    /// Index a single file incrementally
    ///
    /// This is called by the watcher when a Hot file is modified,
    /// or when a debounced Warm file is ready to be processed.
    pub fn index_file_incremental(&self, path: &Path) -> Result<()> {
        let full_path = self.root.join(path);
        debug!(path = %path.display(), "Starting incremental index for file");

        let indexer = if let Some(ref engine) = self.embedding_engine {
            CodebaseIndexer::with_embedding_engine(
                self.root.clone(),
                self.config.clone(),
                self.parser.clone(),
                engine.clone(),
            )?
        } else {
            CodebaseIndexer::new(self.root.clone(), self.config.clone(), self.parser.clone())?
        };

        // Index the single file
        indexer.index_file(&full_path, &self.db, &self.text_index)?;
        self.text_index.commit()?;

        // Invalidate semantic search cache
        if let Some(ref semantic) = self.semantic_search {
            semantic.invalidate_cache();
        }

        info!(path = %path.display(), "Incremental index complete for file");
        Ok(())
    }

    /// Remove a file from the index
    ///
    /// This is called by the watcher when a file is deleted.
    pub fn remove_file_from_index(&self, path: &Path) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        debug!(path = %path_str, "Removing file from index");

        // Remove from database (this also removes associated symbols and embeddings)
        let removed = self.db.delete_file_by_path(&path_str)?;
        if removed {
            info!(path = %path_str, "Removed file from database");
        } else {
            debug!(path = %path_str, "File not found in database (may not have been indexed)");
        }

        // Remove from full-text index
        self.text_index.delete_file(&path_str)?;
        self.text_index.commit()?;

        // Invalidate semantic search cache
        if let Some(ref semantic) = self.semantic_search {
            semantic.invalidate_cache();
        }

        info!(path = %path_str, "File removed from index");
        Ok(())
    }

    /// Get server info for MCP
    pub fn server_info() -> Implementation {
        Implementation {
            name: "cogmcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            title: None,
            icons: None,
            website_url: None,
        }
    }

    /// Get server capabilities
    pub fn capabilities() -> ServerCapabilities {
        ServerCapabilities::builder()
            .enable_tools()
            .build()
    }

    /// List available tools
    pub fn list_tools(&self) -> Vec<Tool> {
        fn make_tool(name: &'static str, description: &'static str, schema: serde_json::Value) -> Tool {
            Tool {
                name: name.into(),
                title: None,
                description: Some(description.into()),
                input_schema: serde_json::from_value(schema).unwrap(),
                output_schema: None,
                annotations: None,
                icons: None,
                meta: None,
            }
        }

        vec![
            make_tool(
                "ping",
                "Check if the CogMCP server is running",
                json!({ "type": "object", "properties": {} }),
            ),
            make_tool(
                "context_grep",
                "Search for content matching a pattern (like grep)",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Search pattern" },
                        "limit": { "type": "integer", "description": "Maximum results" }
                    },
                    "required": ["pattern"]
                }),
            ),
            make_tool(
                "context_search",
                "Search for code using natural language",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Natural language query" },
                        "limit": { "type": "integer", "description": "Maximum results" },
                        "mode": { "type": "string", "description": "Search mode: keyword, semantic, or hybrid" }
                    },
                    "required": ["query"]
                }),
            ),
            make_tool(
                "find_symbol",
                "Find symbol definitions (functions, classes, etc.)",
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Symbol name" },
                        "kind": { "type": "string", "description": "Symbol kind filter (function, class, method, etc.)" },
                        "visibility": { "type": "string", "description": "Visibility filter (public, private, protected)" },
                        "fuzzy": { "type": "boolean", "description": "Enable fuzzy matching" }
                    },
                    "required": ["name"]
                }),
            ),
            make_tool(
                "get_file_outline",
                "Get the outline (symbols) of a file",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string", "description": "Path to the file" }
                    },
                    "required": ["file_path"]
                }),
            ),
            make_tool(
                "index_status",
                "Get the current indexing status and statistics",
                json!({ "type": "object", "properties": {} }),
            ),
            make_tool(
                "reindex",
                "Trigger re-indexing of the codebase",
                json!({
                    "type": "object",
                    "properties": {
                        "force": { "type": "boolean", "description": "Force full reindex" }
                    }
                }),
            ),
            make_tool(
                "semantic_search",
                "Search for code using semantic similarity (requires embeddings)",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Natural language search query" },
                        "limit": { "type": "integer", "description": "Maximum results (default: 10)" }
                    },
                    "required": ["query"]
                }),
            ),
        ]
    }

    /// Call a tool by name
    pub fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> std::result::Result<String, String> {
        match name {
            "ping" => Ok(self.ping()),
            "context_grep" => {
                let pattern = arguments["pattern"]
                    .as_str()
                    .ok_or("Missing pattern")?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(50) as usize;
                Ok(self.context_grep(&pattern, limit))
            }
            "context_search" => {
                let query = arguments["query"]
                    .as_str()
                    .ok_or("Missing query")?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(20) as usize;
                let mode = arguments["mode"].as_str().unwrap_or("hybrid");
                Ok(self.context_search(&query, limit, mode))
            }
            "find_symbol" => {
                let name = arguments["name"]
                    .as_str()
                    .ok_or("Missing name")?
                    .to_string();
                let kind = arguments["kind"].as_str().map(|s| s.to_string());
                let visibility = arguments["visibility"].as_str().map(|s| s.to_string());
                let fuzzy = arguments["fuzzy"].as_bool().unwrap_or(false);
                Ok(self.find_symbol_with_visibility(
                    &name,
                    kind.as_deref(),
                    visibility.as_deref(),
                    fuzzy,
                ))
            }
            "get_file_outline" => {
                let file_path = arguments["file_path"]
                    .as_str()
                    .ok_or("Missing file_path")?
                    .to_string();
                Ok(self.get_file_outline(&file_path))
            }
            "index_status" => Ok(self.index_status()),
            "reindex" => Ok(self.reindex()),
            "semantic_search" => {
                let query = arguments["query"]
                    .as_str()
                    .ok_or("Missing query")?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(10) as usize;
                Ok(self.semantic_search(&query, limit))
            }
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // Tool implementations

    fn ping(&self) -> String {
        format!(
            "CogMCP server is running.\nVersion: {}\nRoot: {}",
            env!("CARGO_PKG_VERSION"),
            self.root.display()
        )
    }

    fn context_grep(&self, pattern: &str, limit: usize) -> String {
        match self.text_index.search(pattern, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return "No matches found.".to_string();
                }
                let mut output = String::new();
                for hit in results {
                    output.push_str(&format!(
                        "{}:{}: {}\n",
                        hit.path,
                        hit.line_number,
                        hit.content.trim()
                    ));
                }
                output
            }
            Err(e) => format!("Search failed: {}", e),
        }
    }

    fn context_search(&self, query: &str, limit: usize, mode: &str) -> String {
        // Use configured default mode if not specified
        let mode_str = if mode.is_empty() {
            &self.config.search.default_mode
        } else {
            mode
        };
        let search_mode = SearchMode::from_str(mode_str);

        // Create hybrid search with semantic capability if available
        let search = if let Some(ref semantic) = self.semantic_search {
            let config = cogmcp_search::HybridSearchConfig {
                keyword_weight: self.config.search.keyword_weight,
                semantic_weight: self.config.search.semantic_weight,
                min_similarity: self.config.search.min_similarity,
                rrf_k: self.config.search.rrf_k,
            };
            HybridSearch::with_semantic(&self.text_index, semantic.clone()).with_config(config)
        } else {
            HybridSearch::new(&self.text_index)
        };

        match search.search(query, search_mode, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return "No matches found.".to_string();
                }

                let mode_info = if search.has_semantic() {
                    format!("(mode: {:?}, semantic enabled)", search_mode)
                } else {
                    format!("(mode: {:?}, semantic disabled)", search_mode)
                };

                let mut output = String::new();
                output.push_str(&format!("## Search results for: {} {}\n\n", query, mode_info));
                for hit in results {
                    output.push_str(&format!("### `{}`", hit.path));
                    if let Some(line) = hit.line_number {
                        output.push_str(&format!(":{}", line));
                    }
                    output.push_str(&format!(" (score: {:.2}, type: {:?})\n", hit.score, hit.match_type));
                    output.push_str("```\n");
                    output.push_str(&hit.content);
                    output.push_str("\n```\n\n");
                }
                output
            }
            Err(e) => format!("Search failed: {}", e),
        }
    }

    fn find_symbol_with_visibility(
        &self,
        name: &str,
        kind: Option<&str>,
        visibility: Option<&str>,
        fuzzy: bool,
    ) -> String {
        match self.db.find_symbols_by_name(name, fuzzy) {
            Ok(symbols) => {
                if symbols.is_empty() {
                    return format!("No symbols found matching '{}'", name);
                }

                // Filter by kind if specified
                let symbols: Vec<_> = if let Some(k) = kind {
                    symbols
                        .into_iter()
                        .filter(|s| s.kind.to_lowercase() == k.to_lowercase())
                        .collect()
                } else {
                    symbols
                };

                // Filter by visibility if specified
                let symbols: Vec<_> = if let Some(v) = visibility {
                    symbols
                        .into_iter()
                        .filter(|s| {
                            s.visibility
                                .as_ref()
                                .map(|vis| vis.to_lowercase() == v.to_lowercase())
                                .unwrap_or(false)
                        })
                        .collect()
                } else {
                    symbols
                };

                if symbols.is_empty() {
                    return format!("No symbols found matching '{}' with given filters", name);
                }

                let mut output = String::new();
                output.push_str(&format!("## Symbols matching '{}'\n\n", name));
                for sym in symbols {
                    // Build modifiers string
                    let mut modifiers = Vec::new();
                    if sym.is_async {
                        modifiers.push("async");
                    }
                    if sym.is_static {
                        modifiers.push("static");
                    }
                    if sym.is_abstract {
                        modifiers.push("abstract");
                    }
                    if sym.is_exported {
                        modifiers.push("exported");
                    }
                    if sym.is_const {
                        modifiers.push("const");
                    }
                    if sym.is_unsafe {
                        modifiers.push("unsafe");
                    }

                    let visibility_str = sym.visibility.as_deref().unwrap_or("unknown");
                    let modifiers_str = if modifiers.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", modifiers.join(", "))
                    };

                    output.push_str(&format!(
                        "- **{}** ({}, {}{}) in `{}:{}-{}`\n",
                        sym.name, sym.kind, visibility_str, modifiers_str,
                        sym.file_path, sym.start_line, sym.end_line
                    ));
                    if let Some(sig) = &sym.signature {
                        output.push_str(&format!("  `{}`\n", sig.trim()));
                    }
                    if let Some(ret_type) = &sym.return_type {
                        output.push_str(&format!("  Returns: `{}`\n", ret_type));
                    }
                }
                output
            }
            Err(e) => format!("Symbol search failed: {}", e),
        }
    }

    fn get_file_outline(&self, file_path: &str) -> String {
        use cogmcp_core::types::Language;

        let full_path = self.root.join(file_path);
        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(e) => return format!("Failed to read file: {}", e),
        };

        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);

        match self.parser.parse(&content, language) {
            Ok(symbols) => {
                if symbols.is_empty() {
                    return format!("No symbols found in {}", file_path);
                }
                let mut output = String::new();
                output.push_str(&format!("## Outline: {}\n\n", file_path));
                for sym in symbols {
                    // Build modifiers string
                    let mut modifiers = Vec::new();
                    if sym.modifiers.is_async {
                        modifiers.push("async");
                    }
                    if sym.modifiers.is_static {
                        modifiers.push("static");
                    }
                    if sym.modifiers.is_abstract {
                        modifiers.push("abstract");
                    }
                    if sym.modifiers.is_exported {
                        modifiers.push("exported");
                    }
                    if sym.modifiers.is_const {
                        modifiers.push("const");
                    }
                    if sym.modifiers.is_unsafe {
                        modifiers.push("unsafe");
                    }

                    let visibility_str = sym.visibility.as_str();
                    let modifiers_str = if modifiers.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", modifiers.join(", "))
                    };

                    let parent_str = if let Some(parent) = &sym.parent_name {
                        format!(" (in {})", parent)
                    } else {
                        String::new()
                    };

                    output.push_str(&format!(
                        "- **{}** ({}, {}{}) lines {}-{}{}\n",
                        sym.name,
                        format!("{:?}", sym.kind).to_lowercase(),
                        visibility_str,
                        modifiers_str,
                        sym.start_line,
                        sym.end_line,
                        parent_str
                    ));
                    if let Some(sig) = &sym.signature {
                        output.push_str(&format!("  `{}`\n", sig.trim()));
                    }
                    if let Some(ret_type) = &sym.return_type {
                        output.push_str(&format!("  Returns: `{}`\n", ret_type));
                    }
                }
                output
            }
            Err(e) => format!("Parse failed: {}", e),
        }
    }

    fn index_status(&self) -> String {
        match self.db.get_extended_stats() {
            Ok(stats) => {
                let mut output = format!(
                    "## Index Status\n\n\
                    - **Files indexed:** {}\n\
                    - **Symbols extracted:** {}\n\
                    - **Embeddings stored:** {}\n\
                    - **Root directory:** {}\n",
                    stats.file_count,
                    stats.symbol_count,
                    stats.embedding_count,
                    self.root.display()
                );

                // Add symbol breakdown by kind
                if !stats.symbols_by_kind.is_empty() {
                    output.push_str("\n### Symbols by Kind\n\n");
                    let mut kinds: Vec<_> = stats.symbols_by_kind.iter().collect();
                    kinds.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
                    for (kind, count) in kinds {
                        output.push_str(&format!("- {}: {}\n", kind, count));
                    }
                }

                // Add visibility stats
                if !stats.symbols_by_visibility.is_empty() {
                    output.push_str("\n### Symbols by Visibility\n\n");
                    let mut visibilities: Vec<_> = stats.symbols_by_visibility.iter().collect();
                    visibilities.sort_by(|a, b| b.1.cmp(a.1));
                    for (vis, count) in visibilities {
                        output.push_str(&format!("- {}: {}\n", vis, count));
                    }
                }

                // Add extraction coverage
                if stats.symbol_count > 0 {
                    let visibility_coverage =
                        (stats.symbols_with_visibility as f64 / stats.symbol_count as f64) * 100.0;
                    let parent_coverage =
                        (stats.symbols_with_parent as f64 / stats.symbol_count as f64) * 100.0;
                    output.push_str("\n### Extraction Coverage\n\n");
                    output.push_str(&format!(
                        "- Visibility extracted: {:.1}% ({} symbols)\n",
                        visibility_coverage, stats.symbols_with_visibility
                    ));
                    output.push_str(&format!(
                        "- Parent relationships: {:.1}% ({} symbols)\n",
                        parent_coverage, stats.symbols_with_parent
                    ));
                }

                output
            }
            Err(e) => format!("Failed to get stats: {}", e),
        }
    }

    fn reindex(&self) -> String {
        let indexer_result = if let Some(ref engine) = self.embedding_engine {
            CodebaseIndexer::with_embedding_engine(
                self.root.clone(),
                self.config.clone(),
                self.parser.clone(),
                engine.clone(),
            )
        } else {
            CodebaseIndexer::new(self.root.clone(), self.config.clone(), self.parser.clone())
        };

        match indexer_result {
            Ok(indexer) => match indexer.index_all(&self.db, &self.text_index) {
                Ok(result) => {
                    let mut output = format!(
                        "## Reindex Complete\n\n\
                        - **Files indexed:** {}\n\
                        - **Files skipped:** {}\n\
                        - **Errors:** {}\n\
                        - **Symbols extracted:** {}\n\
                        - **Symbols with visibility:** {}\n",
                        result.indexed,
                        result.skipped,
                        result.errors,
                        result.symbols_extracted,
                        result.symbols_with_visibility
                    );

                    // Add breakdown by kind
                    if !result.symbols_by_kind.is_empty() {
                        output.push_str("\n### Symbols by Kind\n\n");
                        let mut kinds: Vec<_> = result.symbols_by_kind.iter().collect();
                        kinds.sort_by(|a, b| b.1.cmp(a.1));
                        for (kind, count) in kinds {
                            output.push_str(&format!("- {}: {}\n", kind, count));
                        }
                    }

                    output
                }
                Err(e) => format!("Indexing failed: {}", e),
            },
            Err(e) => format!("Failed to create indexer: {}", e),
        }
    }

    /// Semantic search for code using natural language
    fn semantic_search(&self, query: &str, limit: usize) -> String {
        let Some(ref semantic) = self.semantic_search else {
            return "Semantic search is not available. Enable embeddings in configuration.".to_string();
        };

        if !semantic.is_available() {
            return "Semantic search is not available. Model not loaded.".to_string();
        }

        match semantic.search(query, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return "No matches found.".to_string();
                }

                let mut output = String::new();
                output.push_str(&format!("## Semantic search results for: {}\n\n", query));
                for result in results {
                    output.push_str(&format!(
                        "### `{}`",
                        result.path
                    ));
                    if let Some(line) = result.start_line {
                        output.push_str(&format!(":{}", line));
                    }
                    output.push_str(&format!(
                        " (similarity: {:.2}, type: {:?})\n",
                        result.similarity, result.chunk_type
                    ));
                    output.push_str("```\n");
                    output.push_str(&result.chunk_text);
                    output.push_str("\n```\n\n");
                }
                output
            }
            Err(e) => format!("Semantic search failed: {}", e),
        }
    }
}

/// MCP ServerHandler implementation
impl ServerHandler for CogMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: Self::capabilities(),
            server_info: Self::server_info(),
            instructions: Some(
                "CogMCP provides intelligent code context for AI assistants. \
                 Use context_search for natural language queries, find_symbol for \
                 symbol lookups, and context_grep for pattern matching."
                    .to_string(),
            ),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParam>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> std::result::Result<ListToolsResult, rmcp::model::ErrorData> {
        Ok(ListToolsResult {
            tools: self.list_tools(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParam,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> std::result::Result<CallToolResult, rmcp::model::ErrorData> {
        let name = request.name.as_ref();
        let arguments = request
            .arguments
            .map(|v| serde_json::Value::Object(v))
            .unwrap_or(json!({}));

        match self.call_tool(name, arguments) {
            Ok(result) => Ok(CallToolResult {
                content: vec![Content::text(result)],
                is_error: None,
                meta: None,
                structured_content: None,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![Content::text(format!("Error: {}", e))],
                is_error: Some(true),
                meta: None,
                structured_content: None,
            }),
        }
    }
}
