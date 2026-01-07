//! Shared types used across CogMCP crates

use serde::{Deserialize, Serialize};

/// Unique identifier for indexed files
pub type FileId = i64;

/// Unique identifier for symbols
pub type SymbolId = i64;

/// Visibility/access modifier for symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SymbolVisibility {
    /// Public - accessible from anywhere
    Public,
    /// Private - accessible only within the same scope
    Private,
    /// Protected - accessible within class and subclasses
    Protected,
    /// Internal - accessible within the same module/package
    Internal,
    /// Crate-level visibility (Rust pub(crate))
    Crate,
    /// Unknown or not applicable
    #[default]
    Unknown,
}

impl SymbolVisibility {
    /// Parse visibility from a string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "public" | "pub" => Self::Public,
            "private" | "priv" => Self::Private,
            "protected" => Self::Protected,
            "internal" => Self::Internal,
            "crate" | "pub(crate)" => Self::Crate,
            _ => Self::Unknown,
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Private => "private",
            Self::Protected => "protected",
            Self::Internal => "internal",
            Self::Crate => "crate",
            Self::Unknown => "unknown",
        }
    }
}

/// Modifiers that can be applied to symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct SymbolModifiers {
    /// Is the symbol async (Rust async fn, JS async function)
    pub is_async: bool,
    /// Is the symbol static (class static method, Rust static)
    pub is_static: bool,
    /// Is the symbol abstract (abstract class/method)
    pub is_abstract: bool,
    /// Is the symbol exported (JS/TS export)
    pub is_exported: bool,
    /// Is the symbol const (Rust const, JS const)
    pub is_const: bool,
    /// Is the symbol unsafe (Rust unsafe)
    pub is_unsafe: bool,
}

/// Supported programming languages for parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    C,
    Cpp,
    Json,
    Toml,
    Yaml,
    Markdown,
    Unknown,
}

impl Language {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Language::Rust,
            "ts" | "tsx" => Language::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
            "py" | "pyi" => Language::Python,
            "go" => Language::Go,
            "java" => Language::Java,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Language::Cpp,
            "json" => Language::Json,
            "toml" => Language::Toml,
            "yaml" | "yml" => Language::Yaml,
            "md" | "markdown" => Language::Markdown,
            _ => Language::Unknown,
        }
    }

    /// Get file extensions for this language
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::TypeScript => &["ts", "tsx"],
            Language::JavaScript => &["js", "jsx", "mjs", "cjs"],
            Language::Python => &["py", "pyi"],
            Language::Go => &["go"],
            Language::Java => &["java"],
            Language::C => &["c", "h"],
            Language::Cpp => &["cpp", "cc", "cxx", "hpp", "hxx"],
            Language::Json => &["json"],
            Language::Toml => &["toml"],
            Language::Yaml => &["yaml", "yml"],
            Language::Markdown => &["md", "markdown"],
            Language::Unknown => &[],
        }
    }
}

/// Kind of symbol (function, class, variable, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Module,
    Variable,
    Constant,
    Field,
    Property,
    Type,
    Import,
    Unknown,
}

/// Information about a file in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: FileId,
    pub path: String,
    pub hash: String,
    pub modified_at: i64,
    pub size: u64,
    pub language: Language,
    pub priority_score: f32,
    pub last_accessed: Option<i64>,
    pub indexed_at: Option<i64>,
}

/// Information about a symbol in the code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub id: SymbolId,
    pub file_id: FileId,
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: u32,
    pub end_line: u32,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    /// Visibility of the symbol
    pub visibility: SymbolVisibility,
    /// Symbol modifiers (async, static, etc.)
    pub modifiers: SymbolModifiers,
    /// Parent symbol ID for nested symbols
    pub parent_id: Option<SymbolId>,
    /// Generic type parameters
    pub type_parameters: Vec<String>,
    /// Function/method parameters
    pub parameters: Vec<ParameterInfo>,
    /// Return type for functions/methods
    pub return_type: Option<String>,
}

/// Information about a function/method parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub type_annotation: Option<String>,
}

/// A location in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub start_column: Option<u32>,
    pub end_column: Option<u32>,
}

/// Search result with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub location: Location,
    pub snippet: String,
    pub score: f32,
    pub match_type: MatchType,
}

/// Type of search match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchType {
    Exact,
    Fuzzy,
    Semantic,
    Regex,
}

/// Priority tier for file watching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorityTier {
    Hot,  // Real-time updates
    Warm, // Debounced updates
    Cold, // On-demand only
}
