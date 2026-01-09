//! Main MCP server implementation

use cogmcp_context::{ContextPrioritizer, prioritizer::PriorityWeights};
use cogmcp_core::{Config, Result};
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

        Ok(Self {
            root,
            config,
            db,
            text_index,
            parser,
            embedding_engine,
            semantic_search,
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
                "get_relevant_context",
                "Get the most relevant code context based on a multi-factor priority score",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Optional natural language query to filter by relevance" },
                        "paths": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional list of paths to filter by"
                        },
                        "limit": { "type": "integer", "description": "Maximum number of results (default: 10)" },
                        "min_score": { "type": "number", "description": "Minimum priority score threshold (0.0-1.0, default: 0.0)" },
                        "include_content": { "type": "boolean", "description": "Include file content in results (default: true)" },
                        "weights": {
                            "type": "object",
                            "description": "Custom weights for priority calculation",
                            "properties": {
                                "recency": { "type": "number", "description": "Weight for recency (default: 0.25)" },
                                "relevance": { "type": "number", "description": "Weight for query relevance (default: 0.30)" },
                                "centrality": { "type": "number", "description": "Weight for import centrality (default: 0.20)" },
                                "git_activity": { "type": "number", "description": "Weight for git activity (default: 0.15)" },
                                "user_focus": { "type": "number", "description": "Weight for user focus/recency (default: 0.10)" }
                            }
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
            "get_relevant_context" => {
                let query = arguments["query"].as_str().map(|s| s.to_string());
                let paths: Option<Vec<String>> = arguments["paths"]
                    .as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect());
                let limit = arguments["limit"].as_u64().unwrap_or(10) as usize;
                let min_score = arguments["min_score"].as_f64().unwrap_or(0.0) as f32;
                let include_content = arguments["include_content"].as_bool().unwrap_or(true);
                let weights = arguments["weights"].as_object().map(|obj| PriorityWeights {
                    recency: obj.get("recency").and_then(|v| v.as_f64()).unwrap_or(0.25) as f32,
                    relevance: obj.get("relevance").and_then(|v| v.as_f64()).unwrap_or(0.30) as f32,
                    centrality: obj.get("centrality").and_then(|v| v.as_f64()).unwrap_or(0.20) as f32,
                    git_activity: obj.get("git_activity").and_then(|v| v.as_f64()).unwrap_or(0.15) as f32,
                    user_focus: obj.get("user_focus").and_then(|v| v.as_f64()).unwrap_or(0.10) as f32,
                });
                Ok(self.get_relevant_context(query.as_deref(), paths.as_deref(), limit, min_score, include_content, weights))
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

    /// Get the most relevant code context based on a multi-factor priority score
    fn get_relevant_context(
        &self,
        query: Option<&str>,
        paths: Option<&[String]>,
        limit: usize,
        min_score: f32,
        include_content: bool,
        custom_weights: Option<PriorityWeights>,
    ) -> String {
        let weights = custom_weights.unwrap_or_default();
        let prioritizer = ContextPrioritizer::new(weights.clone());

        // Get all indexed files
        let files = match self.db.list_files() {
            Ok(files) => files,
            Err(e) => return format!("Failed to get files: {}", e),
        };

        if files.is_empty() {
            return "No files indexed. Run reindex first.".to_string();
        }

        // Filter by paths if specified
        let files: Vec<_> = if let Some(filter_paths) = paths {
            files
                .into_iter()
                .filter(|f| {
                    filter_paths
                        .iter()
                        .any(|p| f.path.starts_with(p) || f.path.contains(p))
                })
                .collect()
        } else {
            files
        };

        if files.is_empty() {
            return "No files match the path filter.".to_string();
        }

        // Calculate priority scores for each file
        let mut scored_files: Vec<(String, f32, ScoreBreakdown)> = files
            .iter()
            .map(|file| {
                // Calculate recency score based on file modification time
                let indexed_at = file.indexed_at.unwrap_or(file.modified_at);
                let recency_score = Self::calculate_recency_score(indexed_at);

                // Calculate relevance score using text search if query is provided
                let relevance_score = if let Some(q) = query {
                    self.calculate_relevance_score(&file.path, q)
                } else {
                    0.5 // Default middle score if no query
                };

                // Calculate centrality score (based on how many symbols reference this file)
                let centrality_score = self.calculate_centrality_score(&file.path);

                // Git activity score (placeholder - would use git integration)
                let git_activity_score = 0.5; // Default score

                // User focus score (placeholder - would track user interactions)
                let user_focus_score = 0.5; // Default score

                let total_score = prioritizer.calculate_priority(
                    recency_score,
                    relevance_score,
                    centrality_score,
                    git_activity_score,
                    user_focus_score,
                );

                let breakdown = ScoreBreakdown {
                    recency: recency_score,
                    relevance: relevance_score,
                    centrality: centrality_score,
                    git_activity: git_activity_score,
                    user_focus: user_focus_score,
                };

                (file.path.clone(), total_score, breakdown)
            })
            .filter(|(_, score, _)| *score >= min_score)
            .collect();

        // Sort by score (highest first)
        scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        scored_files.truncate(limit);

        if scored_files.is_empty() {
            return "No files meet the minimum score threshold.".to_string();
        }

        // Format output
        let mut output = String::new();
        output.push_str("## Relevant Context\n\n");
        output.push_str(&format!(
            "Found {} relevant files (weights: recency={:.2}, relevance={:.2}, centrality={:.2}, git_activity={:.2}, user_focus={:.2})\n\n",
            scored_files.len(),
            weights.recency,
            weights.relevance,
            weights.centrality,
            weights.git_activity,
            weights.user_focus
        ));

        for (path, score, breakdown) in &scored_files {
            output.push_str(&format!("### `{}` (score: {:.3})\n", path, score));
            output.push_str(&format!(
                "  Breakdown: recency={:.2}, relevance={:.2}, centrality={:.2}, git={:.2}, focus={:.2}\n",
                breakdown.recency,
                breakdown.relevance,
                breakdown.centrality,
                breakdown.git_activity,
                breakdown.user_focus
            ));

            if include_content {
                let full_path = self.root.join(path);
                if let Ok(content) = std::fs::read_to_string(&full_path) {
                    let preview = if content.len() > 500 {
                        format!("{}...", &content[..500])
                    } else {
                        content
                    };
                    output.push_str("```\n");
                    output.push_str(&preview);
                    output.push_str("\n```\n");
                }
            }
            output.push('\n');
        }

        output
    }

    /// Calculate recency score based on when the file was indexed
    fn calculate_recency_score(indexed_at: i64) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let age_seconds = now - indexed_at;
        let age_hours = age_seconds as f32 / 3600.0;

        // Exponential decay: score decreases as file gets older
        // Half-life of ~24 hours
        (-age_hours / 24.0).exp().clamp(0.0, 1.0)
    }

    /// Calculate relevance score for a file based on a query
    fn calculate_relevance_score(&self, path: &str, query: &str) -> f32 {
        // Use text search to find matches in the file
        if let Ok(results) = self.text_index.search(query, 100) {
            let matching_count = results.iter().filter(|r| r.path == path).count();
            if matching_count > 0 {
                // Normalize: more matches = higher score, capped at 1.0
                (matching_count as f32 / 10.0).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Calculate centrality score based on symbol references
    fn calculate_centrality_score(&self, path: &str) -> f32 {
        // Count symbols in this file
        if let Ok(symbols) = self.db.get_symbols_by_file(path) {
            let symbol_count = symbols.len();
            // Normalize: more symbols = more central
            (symbol_count as f32 / 20.0).min(1.0)
        } else {
            0.0
        }
    }
}

/// Score breakdown for debugging and transparency
#[derive(Debug, Clone)]
struct ScoreBreakdown {
    recency: f32,
    relevance: f32,
    centrality: f32,
    git_activity: f32,
    user_focus: f32,
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
