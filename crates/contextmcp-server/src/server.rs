//! Main MCP server implementation

use contextmcp_core::{Config, Result};
use contextmcp_index::{CodeParser, CodebaseIndexer};
use contextmcp_search::HybridSearch;
use contextmcp_storage::{Database, FullTextIndex};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolResult, Content, Implementation, ListToolsResult, ServerCapabilities, ServerInfo, Tool,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

/// MCP server for context management
#[derive(Clone)]
pub struct ContextMcpServer {
    pub root: PathBuf,
    pub config: Config,
    pub db: Arc<Database>,
    pub text_index: Arc<FullTextIndex>,
    pub parser: Arc<CodeParser>,
}

impl ContextMcpServer {
    /// Create a new server instance
    pub fn new(root: PathBuf) -> Result<Self> {
        let config = Config::load()?;
        let db = Arc::new(Database::open(&Config::database_path()?)?);
        let text_index = Arc::new(FullTextIndex::open(&Config::tantivy_path()?)?);
        let parser = Arc::new(CodeParser::new());

        Ok(Self {
            root,
            config,
            db,
            text_index,
            parser,
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
        })
    }

    /// Index the codebase
    pub fn index(&self) -> Result<()> {
        let indexer = CodebaseIndexer::new(self.root.clone(), self.config.clone(), self.parser.clone())?;
        indexer.index_all(&self.db, &self.text_index)?;
        Ok(())
    }

    /// Get server info for MCP
    pub fn server_info() -> Implementation {
        Implementation {
            name: "contextmcp".to_string(),
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
                "Check if the ContextMCP server is running",
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
                        "kind": { "type": "string", "description": "Symbol kind filter" },
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
                let fuzzy = arguments["fuzzy"].as_bool().unwrap_or(false);
                Ok(self.find_symbol(&name, kind.as_deref(), fuzzy))
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
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // Tool implementations

    fn ping(&self) -> String {
        format!(
            "ContextMCP server is running.\nVersion: {}\nRoot: {}",
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
        use contextmcp_search::hybrid::SearchMode;

        let search_mode = match mode {
            "keyword" => SearchMode::Keyword,
            "semantic" => SearchMode::Semantic,
            _ => SearchMode::Hybrid,
        };

        let search = HybridSearch::new(&self.text_index);
        match search.search(query, search_mode, limit) {
            Ok(results) => {
                if results.is_empty() {
                    return "No matches found.".to_string();
                }
                let mut output = String::new();
                output.push_str(&format!("## Search results for: {}\n\n", query));
                for hit in results {
                    output.push_str(&format!("### `{}`", hit.path));
                    if let Some(line) = hit.line_number {
                        output.push_str(&format!(":{}", line));
                    }
                    output.push_str(&format!(" (score: {:.2})\n", hit.score));
                    output.push_str("```\n");
                    output.push_str(&hit.content);
                    output.push_str("\n```\n\n");
                }
                output
            }
            Err(e) => format!("Search failed: {}", e),
        }
    }

    fn find_symbol(&self, name: &str, kind: Option<&str>, fuzzy: bool) -> String {
        match self.db.find_symbols_by_name(name, fuzzy) {
            Ok(symbols) => {
                if symbols.is_empty() {
                    return format!("No symbols found matching '{}'", name);
                }
                let symbols: Vec<_> = if let Some(k) = kind {
                    symbols
                        .into_iter()
                        .filter(|s| s.kind.to_lowercase() == k.to_lowercase())
                        .collect()
                } else {
                    symbols
                };

                let mut output = String::new();
                output.push_str(&format!("## Symbols matching '{}'\n\n", name));
                for sym in symbols {
                    output.push_str(&format!(
                        "- **{}** ({}) in `{}:{}-{}`\n",
                        sym.name, sym.kind, sym.file_path, sym.start_line, sym.end_line
                    ));
                    if let Some(sig) = &sym.signature {
                        output.push_str(&format!("  `{}`\n", sig.trim()));
                    }
                }
                output
            }
            Err(e) => format!("Symbol search failed: {}", e),
        }
    }

    fn get_file_outline(&self, file_path: &str) -> String {
        use contextmcp_core::types::Language;

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
                    output.push_str(&format!(
                        "- **{}** ({}) lines {}-{}\n",
                        sym.name,
                        format!("{:?}", sym.kind).to_lowercase(),
                        sym.start_line,
                        sym.end_line
                    ));
                    if let Some(sig) = &sym.signature {
                        output.push_str(&format!("  `{}`\n", sig.trim()));
                    }
                }
                output
            }
            Err(e) => format!("Parse failed: {}", e),
        }
    }

    fn index_status(&self) -> String {
        match self.db.get_stats() {
            Ok(stats) => {
                format!(
                    "## Index Status\n\n\
                    - **Files indexed:** {}\n\
                    - **Symbols extracted:** {}\n\
                    - **Embeddings stored:** {}\n\
                    - **Root directory:** {}\n",
                    stats.file_count,
                    stats.symbol_count,
                    stats.embedding_count,
                    self.root.display()
                )
            }
            Err(e) => format!("Failed to get stats: {}", e),
        }
    }

    fn reindex(&self) -> String {
        match CodebaseIndexer::new(self.root.clone(), self.config.clone(), self.parser.clone()) {
            Ok(indexer) => match indexer.index_all(&self.db, &self.text_index) {
                Ok(result) => {
                    format!(
                        "## Reindex Complete\n\n\
                        - **Files indexed:** {}\n\
                        - **Files skipped:** {}\n\
                        - **Errors:** {}\n",
                        result.indexed, result.skipped, result.errors
                    )
                }
                Err(e) => format!("Indexing failed: {}", e),
            },
            Err(e) => format!("Failed to create indexer: {}", e),
        }
    }
}

/// MCP ServerHandler implementation
impl ServerHandler for ContextMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: Self::capabilities(),
            server_info: Self::server_info(),
            instructions: Some(
                "ContextMCP provides intelligent code context for AI assistants. \
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
