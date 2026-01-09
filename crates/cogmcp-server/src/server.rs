//! Main MCP server implementation

use cogmcp_core::{Config, Result, SharedConfig};
use cogmcp_embeddings::{EmbeddingEngine, ModelConfig};
use cogmcp_index::{CodeParser, CodebaseIndexer};
use cogmcp_search::{HybridSearch, SearchMode, SemanticSearch};
use cogmcp_storage::{Database, FullTextIndex};
use parking_lot::Mutex;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolResult, Content, Implementation, ListToolsResult, ServerCapabilities, ServerInfo, Tool,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Format an actionable error message with suggestions
fn format_error(message: &str, suggestions: &[&str]) -> String {
    if suggestions.is_empty() {
        return message.to_string();
    }

    let mut output = format!("{}\n\nSuggested actions:", message);
    for (i, suggestion) in suggestions.iter().enumerate() {
        output.push_str(&format!("\n  {}. {}", i + 1, suggestion));
    }
    output
}

/// MCP server for context management
#[derive(Clone)]
pub struct CogMcpServer {
    pub root: PathBuf,
    pub shared_config: SharedConfig,
    pub db: Arc<Database>,
    pub text_index: Arc<FullTextIndex>,
    pub parser: Arc<CodeParser>,
    pub embedding_engine: Option<Arc<Mutex<EmbeddingEngine>>>,
    pub semantic_search: Option<Arc<SemanticSearch>>,
    /// Server runtime status tracking
    pub status: Arc<ServerStatus>,
}

impl CogMcpServer {
    /// Create a new server instance
    pub fn new(root: PathBuf) -> Result<Self> {
        let shared_config = SharedConfig::load()?;
        let config = shared_config.get();
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

        Ok(Self {
            root,
            shared_config,
            db,
            text_index,
            parser,
            embedding_engine,
            semantic_search,
            status: Arc::new(ServerStatus::new()),
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> Config {
        self.shared_config.get()
    }

    /// Create a server with in-memory storage (for testing)
    pub fn in_memory(root: PathBuf) -> Result<Self> {
        let shared_config = SharedConfig::default();
        let db = Arc::new(Database::in_memory()?);
        let text_index = Arc::new(FullTextIndex::in_memory()?);
        let parser = Arc::new(CodeParser::new());

        Ok(Self {
            root,
            shared_config,
            db,
            text_index,
            parser,
            embedding_engine: None,
            semantic_search: None,
            status: Arc::new(ServerStatus::new()),
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
        let semantic = Arc::new(SemanticSearch::new(engine.clone(), db, &config.cache));

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

    /// Check if semantic search is available
    pub fn has_semantic_search(&self) -> bool {
        self.semantic_search
            .as_ref()
            .map_or(false, |s| s.is_available())
    }

    /// Check if index has any content and return an error message if empty
    fn check_index_status(&self) -> Option<String> {
        match self.db.get_stats() {
            Ok(stats) if stats.file_count == 0 => Some(format_error(
                "The code index is empty.",
                &[
                    "Run the `reindex` tool to index the codebase",
                    "Check that the root directory contains source files",
                    "Verify .gitignore isn't excluding all files",
                ],
            )),
            _ => None,
        }
    }

    /// Index the codebase
    pub fn index(&self) -> Result<()> {
        let config = self.config();
        let indexer = if let Some(ref engine) = self.embedding_engine {
            CodebaseIndexer::with_embedding_engine(
                self.root.clone(),
                config.clone(),
                self.parser.clone(),
                engine.clone(),
            )?
        } else {
            CodebaseIndexer::new(self.root.clone(), config, self.parser.clone())?
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
            make_tool(
                "reload_config",
                "Reload configuration from disk. Returns the new config summary.",
                json!({
                    "type": "object",
                    "properties": {
                        "validate_only": {
                            "type": "boolean",
                            "description": "If true, only validate the config without applying"
                        }
                    }
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
        let result = match name {
            "ping" => Ok(self.ping()),
            "context_grep" => {
                let pattern = arguments["pattern"]
                    .as_str()
                    .ok_or_else(|| {
                        format_error(
                            "Missing required parameter: 'pattern'",
                            &[
                                "Provide a search pattern string",
                                "Example: {\"pattern\": \"TODO\", \"limit\": 50}",
                            ],
                        )
                    })?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(50) as usize;
                self.context_grep(&pattern, limit)
            }
            "context_search" => {
                let query = arguments["query"]
                    .as_str()
                    .ok_or_else(|| {
                        format_error(
                            "Missing required parameter: 'query'",
                            &[
                                "Provide a search query string",
                                "Example: {\"query\": \"authentication logic\", \"mode\": \"hybrid\"}",
                            ],
                        )
                    })?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(20) as usize;
                let mode = arguments["mode"].as_str().unwrap_or("hybrid");
                self.context_search(&query, limit, mode)
            }
            "find_symbol" => {
                let name = arguments["name"]
                    .as_str()
                    .ok_or_else(|| {
                        format_error(
                            "Missing required parameter: 'name'",
                            &[
                                "Provide a symbol name to search for",
                                "Example: {\"name\": \"MyClass\", \"fuzzy\": true}",
                            ],
                        )
                    })?
                    .to_string();
                let kind = arguments["kind"].as_str().map(|s| s.to_string());
                let visibility = arguments["visibility"].as_str().map(|s| s.to_string());
                let fuzzy = arguments["fuzzy"].as_bool().unwrap_or(false);
                self.find_symbol_with_visibility(
                    &name,
                    kind.as_deref(),
                    visibility.as_deref(),
                    fuzzy,
                )
            }
            "get_file_outline" => {
                let file_path = arguments["file_path"]
                    .as_str()
                    .ok_or_else(|| {
                        format_error(
                            "Missing required parameter: 'file_path'",
                            &[
                                "Provide a path to the file (relative to project root)",
                                "Example: {\"file_path\": \"src/main.rs\"}",
                            ],
                        )
                    })?
                    .to_string();
                self.get_file_outline(&file_path)
            }
            "index_status" => self.index_status(),
            "reindex" => self.reindex(),
            "semantic_search" => {
                let query = arguments["query"]
                    .as_str()
                    .ok_or_else(|| {
                        format_error(
                            "Missing required parameter: 'query'",
                            &[
                                "Provide a natural language search query",
                                "Example: {\"query\": \"function that handles authentication\", \"limit\": 10}",
                            ],
                        )
                    })?
                    .to_string();
                let limit = arguments["limit"].as_u64().unwrap_or(10) as usize;
                self.semantic_search(&query, limit)
            }
            "reload_config" => {
                let validate_only = arguments["validate_only"].as_bool().unwrap_or(false);
                Ok(self.reload_config(validate_only))
            }
            _ => Err(format!("Unknown tool: {}", name)),
        };

        // Track the call in status
        match &result {
            Ok(_) => self.status.record_tool_call(name),
            Err(_) => self.status.record_error(name),
        }

        result
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
        // Check index first
        if let Some(err) = self.check_index_status() {
            return err;
        }

        match self.text_index.search(pattern, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return format_error(
                        &format!("No matches found for pattern '{}'.", pattern),
                        &[
                            "Try a less specific pattern",
                            "Check spelling and case sensitivity",
                            "Use `index_status` to see indexed file count",
                            "Run `reindex` if files were recently added",
                        ],
                    );
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
                Ok(output)
            }
            Err(e) => format_error(
                "Search operation failed.",
                &[
                    &format!("Error details: {}", e),
                    "Check if the pattern is a valid regex",
                    "Escape special characters: . * + ? [ ] ( ) { } | \\ ^",
                    "Try a simpler search pattern",
                ],
            ),
        }
    }

    fn context_search(&self, query: &str, limit: usize, mode: &str) -> String {
        let current_config = self.config();
        // Use configured default mode if not specified
        let mode_str = if mode.is_empty() {
            &current_config.search.default_mode
        } else {
            mode
        };
        let search_mode = SearchMode::from_str(mode_str);

        // Create hybrid search with semantic capability if available
        let search = if let Some(ref semantic) = self.semantic_search {
            let hybrid_config = cogmcp_search::HybridSearchConfig {
                keyword_weight: current_config.search.keyword_weight,
                semantic_weight: current_config.search.semantic_weight,
                min_similarity: current_config.search.min_similarity,
                rrf_k: current_config.search.rrf_k,
            };
            HybridSearch::with_semantic(&self.text_index, semantic.clone()).with_config(hybrid_config)
        } else {
            HybridSearch::new(&self.text_index)
        };

        match search.search(query, search_mode, limit) {
            Ok(results) => {
                if results.is_empty() {
                    let mode_hint = if !search.has_semantic() && mode == "semantic" {
                        " Note: Semantic search is disabled. Try mode='keyword' or enable embeddings."
                    } else {
                        ""
                    };
                    return format_error(
                        &format!("No results found for '{}'.{}", query, mode_hint),
                        &[
                            "Try different keywords or phrasing",
                            "Use broader search terms",
                            "Check `index_status` to verify files are indexed",
                            "Try mode='keyword' for literal text matching",
                        ],
                    );
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
                Ok(output)
            }
            Err(e) => format_error(
                "Search operation failed.",
                &[
                    &format!("Error details: {}", e),
                    "Run `reindex` if the index may be corrupted",
                    "Check disk space and permissions",
                ],
            ),
        }
    }

    fn find_symbol_with_visibility(
        &self,
        name: &str,
        kind: Option<&str>,
        visibility: Option<&str>,
        fuzzy: bool,
    ) -> String {
        // Check index first
        if let Some(err) = self.check_index_status() {
            return err;
        }

        match self.db.find_symbols_by_name(name, fuzzy) {
            Ok(symbols) => {
                if symbols.is_empty() {
                    return format_error(
                        &format!("No symbols found matching '{}'.", name),
                        &[
                            "Try enabling fuzzy search: {\"name\": \"...\", \"fuzzy\": true}",
                            "Check the exact symbol name and spelling",
                            "Run `reindex` if the code was recently changed",
                            "Use `context_grep` to search file contents instead",
                        ],
                    );
                }

                let original_count = symbols.len();

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
                    return format_error(
                        &format!("No symbols matching '{}' with the specified filters.", name),
                        &[
                            &format!(
                                "Found {} symbols matching '{}' without filters",
                                original_count, name
                            ),
                            "Try removing the 'kind' or 'visibility' filter",
                            "Available kinds: function, class, method, struct, enum, etc.",
                            "Available visibilities: public, private, protected",
                        ],
                    );
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
                        sym.name, sym.kind, visibility_str, modifiers_str, sym.file_path,
                        sym.start_line, sym.end_line
                    ));
                    if let Some(sig) = &sym.signature {
                        output.push_str(&format!("  `{}`\n", sig.trim()));
                    }
                    if let Some(ret_type) = &sym.return_type {
                        output.push_str(&format!("  Returns: `{}`\n", ret_type));
                    }
                }
                Ok(output)
            }
            Err(e) => format_error(
                "Symbol search failed.",
                &[
                    &format!("Error details: {}", e),
                    "Run `reindex` to rebuild the symbol index",
                    "Try `context_grep` as an alternative",
                ],
            ),
        }
    }

    fn get_file_outline(&self, file_path: &str) -> std::result::Result<String, ActionableError> {
        use cogmcp_core::types::Language;

        let full_path = self.root.join(file_path);
        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(e) => {
                let suggestions: Vec<&str> =
                    if e.kind() == std::io::ErrorKind::NotFound {
                        vec![
                            "Check that the path is relative to the project root",
                            "Verify the file exists in the repository",
                            "Use `context_grep` to search for the file name",
                        ]
                    } else {
                        vec![
                            "Check file permissions",
                            "Ensure the file is not locked by another process",
                        ]
                    };
                return format_error(
                    &format!("Cannot read file: {} ({})", file_path, e),
                    &suggestions,
                );
            }
        };

        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);

        match self.parser.parse(&content, language) {
            Ok(symbols) => {
                if symbols.is_empty() {
                    return format_error(
                        &format!("No symbols found in '{}'.", file_path),
                        &[
                            &format!(
                                "File extension '{}' may not have symbol extraction support",
                                ext
                            ),
                            "Supported: .rs, .py, .js, .ts, .go, .java, .c, .cpp, .rb",
                            "Try `context_grep` to search file contents instead",
                        ],
                    );
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
                Ok(output)
            }
            Err(e) => format_error(
                &format!("Failed to parse file: {}", file_path),
                &[
                    &format!("Parse error: {}", e),
                    "Check that the file contains valid syntax",
                    "The file may have unsupported language features",
                ],
            ),
        }
    }

    fn index_status(&self) -> std::result::Result<String, ActionableError> {
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

                // Add search cache statistics
                if let Some(ref semantic) = self.semantic_search {
                    let cache_stats = semantic.cache_stats();
                    output.push_str("\n### Search Cache\n\n");
                    output.push_str(&format!("- Cache hits: {}\n", cache_stats.hits));
                    output.push_str(&format!("- Cache misses: {}\n", cache_stats.misses));
                    output.push_str(&format!("- Hit rate: {:.1}%\n", cache_stats.hit_rate() * 100.0));
                    output.push_str(&format!("- Result cache size: {}\n", cache_stats.result_cache_size));
                    output.push_str(&format!("- Embedding cache size: {}\n", cache_stats.embedding_cache_size));
                    output.push_str(&format!("- Index version: {}\n", cache_stats.index_version));
                }

                output
            }
            Err(e) => format_error(
                "Failed to retrieve index statistics.",
                &[
                    &format!("Error details: {}", e),
                    "Run `reindex` to rebuild the index",
                    "Check disk space and file permissions",
                ],
            ),
        }
    }

    fn reindex(&self) -> String {
        let config = self.config();
        let indexer_result = if let Some(ref engine) = self.embedding_engine {
            CodebaseIndexer::with_embedding_engine(
                self.root.clone(),
                config.clone(),
                self.parser.clone(),
                engine.clone(),
            )
        } else {
            CodebaseIndexer::new(self.root.clone(), config, self.parser.clone())
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

                    Ok(output)
                }
                Err(e) => format_error(
                    "Indexing operation failed.",
                    &[
                        &format!("Error details: {}", e),
                        "Check disk space availability",
                        "Verify write permissions to the data directory",
                        "Try removing the data directory and re-running",
                    ],
                ),
            },
            Err(e) => format_error(
                "Failed to initialize indexer.",
                &[
                    &format!("Error details: {}", e),
                    "Check configuration file for errors",
                    "Verify the root directory is accessible",
                ],
            ),
        }
    }

    /// Semantic search for code using natural language
    fn semantic_search(&self, query: &str, limit: usize) -> std::result::Result<String, ActionableError> {
        let Some(ref semantic) = self.semantic_search else {
            return format_error(
                "Semantic search is not available.",
                &[
                    "Enable embeddings in config: indexing.enable_embeddings = true",
                    "Use `context_search` with mode='keyword' for text search",
                    "Check server logs for embedding initialization errors",
                ],
            );
        };

        if !semantic.is_available() {
            return format_error(
                "Semantic search model is not loaded.",
                &[
                    "Check network connectivity for model download",
                    "Verify the model cache directory is writable",
                    "Run `reindex` after fixing the issue",
                    "Use `context_search` as an alternative",
                ],
            );
        }

        match semantic.search(query, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return format_error(
                        &format!("No semantic matches found for '{}'.", query),
                        &[
                            "Try rephrasing with more code-specific terms",
                            "Use `context_search` with mode='hybrid' for combined results",
                            "Run `reindex` if code was recently added",
                            "Check `index_status` to verify embeddings are generated",
                        ],
                    );
                }

                let mut output = String::new();
                output.push_str(&format!("## Semantic search results for: {}\n\n", query));
                for result in results {
                    output.push_str(&format!("### `{}`", result.path));
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
                Ok(output)
            }
            Err(e) => format_error(
                "Semantic search failed.",
                &[
                    &format!("Error details: {}", e),
                    "Run `reindex` to regenerate embeddings",
                    "Try `context_search` with mode='keyword'",
                ],
            ),
        }
    }

    /// Reload configuration from disk
    fn reload_config(&self, validate_only: bool) -> String {
        let action = if validate_only { "Validating" } else { "Reloading" };
        info!("{} configuration", action);

        let result = if validate_only {
            self.shared_config.validate()
        } else {
            self.shared_config.reload()
        };

        match result {
            Ok(reload_result) => {
                let source = match reload_result.source_path {
                    Some(path) => format!("`{}`", path.display()),
                    None => "defaults (no config file found)".to_string(),
                };

                let action_past = if validate_only { "validated" } else { "reloaded" };
                let mut output = format!("## Configuration {}\n\n", action_past);
                output.push_str(&format!("**Source:** {}\n\n", source));
                output.push_str("### Current Settings\n\n");
                output.push_str("```\n");
                output.push_str(&reload_result.config_summary);
                output.push_str("\n```\n");

                if validate_only {
                    output.push_str("\n*Configuration is valid. Call without `validate_only` to apply.*");
                }

                output
            }
            Err(e) => {
                let action_noun = if validate_only { "validation" } else { "reload" };
                format!(
                    "## Configuration {} failed\n\n**Error:** {}\n\n\
                     The previous configuration remains active.",
                    action_noun, e
                )
            }
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
            Err(actionable_error) => {
                // Build metadata with error details for programmatic handling
                let mut meta_obj = serde_json::Map::new();
                meta_obj.insert("errorCode".to_string(), json!(actionable_error.code.code()));
                meta_obj.insert("suggestions".to_string(), json!(actionable_error.suggestions));
                if let Some(cause) = &actionable_error.cause {
                    meta_obj.insert("cause".to_string(), json!(cause));
                }
                let meta = Some(rmcp::model::Meta(meta_obj));

                Ok(CallToolResult {
                    content: vec![Content::text(actionable_error.to_user_message())],
                    is_error: Some(true),
                    meta,
                    structured_content: None,
                })
            }
        }
    }
}
