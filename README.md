# CogMCP

A local-first, high-performance MCP server that provides intelligent code context to AI coding assistants like Claude Code and Cursor.

## Why CogMCP?

| Feature | CogMCP | Cloud-based alternatives |
|---------|------------|-------------------------|
| **Privacy** | 100% local - code never leaves your machine | Code sent to cloud |
| **Latency** | <10ms local | 100-500ms network |
| **Rate Limits** | Unlimited | Often limited |
| **Cost** | Free & open source | Often paid |
| **Offline** | Yes | No |
| **Git Integration** | Deep (blame, history, diff) | Limited |

## Installation

### Quick Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/cogmcp/cogmcp/main/install.sh | bash
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/cogmcp/cogmcp.git
cd cogmcp

# Build release binary
cargo build --release

# Install to your PATH
cp target/release/cogmcp ~/.local/bin/
```

### Homebrew (macOS)

```bash
brew install cogmcp/tap/cogmcp
```

## Setup

After installation, run the setup command to generate your MCP configuration:

```bash
cd /path/to/your/project
cogmcp setup
```

This will output the configuration to add to your IDE.

### Claude Code CLI

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cogmcp": {
      "command": "/path/to/cogmcp",
      "env": {
        "COGMCP_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

### Cursor IDE

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "cogmcp": {
      "command": "/path/to/cogmcp",
      "env": {
        "COGMCP_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `ping` | Check if the server is running |
| `context_grep` | Pattern search (like grep) |
| `context_search` | Natural language search |
| `find_symbol` | Find symbol definitions |
| `get_file_outline` | Get file structure |
| `index_status` | View indexing stats |
| `reindex` | Re-index the codebase |
| `semantic_search` | Semantic code search (if embeddings enabled) |

## Usage

### Command Line

```bash
# Start server for current directory
cogmcp

# Start server for specific project
cogmcp --root /path/to/project

# Generate setup instructions
cogmcp setup

# Print version
cogmcp --version

# Print help
cogmcp --help
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COGMCP_ROOT` | Project root directory | Current directory |
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | info |

## Features

### Smart Indexing

- **Multi-language support**: Rust, TypeScript, JavaScript, Python
- **Symbol extraction**: Functions, classes, methods, structs, enums, traits
- **Visibility tracking**: public, private, protected, crate-level
- **Parent relationships**: Tracks nested symbols (methods in classes, etc.)

### Fast Search

- **Full-text search** via Tantivy
- **Symbol search** with fuzzy matching
- **Visibility filtering** (find all public functions)
- **Kind filtering** (find all classes, all functions, etc.)

### Git Integration

- Respects `.gitignore`
- Tracks file modification times
- Smart re-indexing

## Configuration

CogMCP stores configuration in `~/.config/cogmcp/config.toml`:

```toml
[indexing]
max_file_size = 500  # KB
ignore_patterns = ["node_modules/**", "target/**", ".git/**"]
include_types = ["rs", "ts", "tsx", "js", "jsx", "py"]
enable_embeddings = false

[search]
default_mode = "keyword"
keyword_weight = 0.5
semantic_weight = 0.5
min_similarity = 0.3
```

## Architecture

```
cogmcp/
├── crates/
│   ├── cogmcp-core/       # Config, types, errors
│   ├── cogmcp-storage/    # SQLite + Tantivy
│   ├── cogmcp-index/      # File & symbol indexing
│   ├── cogmcp-search/     # Search implementations
│   ├── cogmcp-embeddings/ # Optional embeddings
│   ├── cogmcp-watcher/    # File watching
│   ├── cogmcp-context/    # Context management
│   └── cogmcp-server/     # MCP server
```

## Development

```bash
# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run

# Build release
cargo build --release
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the contributing guidelines first.
