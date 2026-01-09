//! CogMCP - Local-first context MCP server
//!
//! A high-performance, privacy-preserving context management MCP server
//! that runs entirely on your local machine.

use cogmcp_server::runner::{RunnerConfig, ServerRunner};
use std::env;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_help() {
    eprintln!(
        r#"CogMCP v{VERSION} - Local-first context MCP server

USAGE:
    cogmcp [OPTIONS] [COMMAND]

COMMANDS:
    serve           Start the MCP server (default)
    setup           Generate MCP configuration for Claude Code
    version         Print version information

OPTIONS:
    -h, --help      Print this help message
    -v, --version   Print version
    --root <PATH>   Set the project root directory

ENVIRONMENT:
    COGMCP_ROOT   Project root directory (default: current directory)
    RUST_LOG          Log level (default: info)

EXAMPLES:
    # Start server for current directory
    cogmcp

    # Start server for specific project
    cogmcp --root /path/to/project

    # Generate Claude Code configuration
    cogmcp setup

    # Generate config for specific project
    cogmcp setup --root /path/to/project
"#
    );
}

fn print_version() {
    println!("cogmcp {}", VERSION);
}

fn generate_setup(root: &PathBuf) {
    let binary_path = env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "cogmcp".to_string());

    let root_path = root.canonicalize().unwrap_or_else(|_| root.clone());

    println!(
        r#"
CogMCP Setup
================

Add the following to your Claude Code MCP settings:

For Claude Code CLI (~/.claude/settings.json):
----------------------------------------------
{{
  "mcpServers": {{
    "cogmcp": {{
      "command": "{binary_path}",
      "env": {{
        "COGMCP_ROOT": "{root}"
      }}
    }}
  }}
}}

For Cursor IDE (.cursor/mcp.json in project root):
--------------------------------------------------
{{
  "mcpServers": {{
    "cogmcp": {{
      "command": "{binary_path}",
      "env": {{
        "COGMCP_ROOT": "{root}"
      }}
    }}
  }}
}}

Quick Install (copy to clipboard):
----------------------------------
macOS:   echo '{{"mcpServers":{{"cogmcp":{{"command":"{binary_path}","env":{{"COGMCP_ROOT":"{root}"}}}}}}}}' | pbcopy
Linux:   echo '{{"mcpServers":{{"cogmcp":{{"command":"{binary_path}","env":{{"COGMCP_ROOT":"{root}"}}}}}}}}' | xclip -selection clipboard

Available Tools:
----------------
- ping              Check if server is running
- context_grep      Pattern search (like grep)
- context_search    Natural language search
- find_symbol       Find symbol definitions
- get_file_outline  Get file structure
- index_status      View indexing stats
- reindex           Re-index the codebase
- semantic_search   Semantic code search (if embeddings enabled)
- reload_config     Reload configuration from disk

"#,
        binary_path = binary_path,
        root = root_path.display()
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut root: Option<PathBuf> = None;
    let mut command = "serve";
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "-v" | "--version" | "version" => {
                print_version();
                return Ok(());
            }
            "--root" => {
                i += 1;
                if i < args.len() {
                    root = Some(PathBuf::from(&args[i]));
                } else {
                    eprintln!("Error: --root requires a path argument");
                    std::process::exit(1);
                }
            }
            "setup" => command = "setup",
            "serve" => command = "serve",
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                eprintln!("Run 'cogmcp --help' for usage");
                std::process::exit(1);
            }
            _ => {}
        }
        i += 1;
    }

    // Determine root directory
    let root = root.or_else(|| env::var("COGMCP_ROOT").ok().map(PathBuf::from))
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    match command {
        "setup" => {
            generate_setup(&root);
            Ok(())
        }
        "serve" | _ => {
            // Initialize logging (to stderr so it doesn't interfere with MCP stdio)
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
                )
                .with_writer(std::io::stderr)
                .init();

            tracing::info!("Starting CogMCP server v{}", VERSION);
            tracing::info!("Root directory: {}", root.display());

            // Create and run the server
            let config = RunnerConfig::new(root).index_on_startup(true);

            let runner = ServerRunner::new(config)
                .map_err(|e| anyhow::anyhow!("Failed to create server: {}", e))?;

            // Log available tools before starting
            tracing::info!("Available tools:");
            for tool in runner.server().list_tools() {
                tracing::info!("  - {}: {}", tool.name, tool.description.unwrap_or_default());
            }

            // Run the server (blocks until shutdown)
            runner.run().await
        }
    }
}
