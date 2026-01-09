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
    pub fn parse(s: &str) -> Self {
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

impl std::str::FromStr for SymbolVisibility {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::parse(s))
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
    Ruby,
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
            "rb" | "rake" | "gemspec" => Language::Ruby,
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
            Language::Ruby => &["rb", "rake", "gemspec"],
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
    Static,
    Field,
    Property,
    Type,
    TypeAlias,
    Macro,
    Parameter,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_kind_serialization() {
        // Test that all new SymbolKind variants serialize correctly
        let kinds = vec![
            (SymbolKind::Function, "function"),
            (SymbolKind::TypeAlias, "type_alias"),
            (SymbolKind::Macro, "macro"),
            (SymbolKind::Static, "static"),
            (SymbolKind::Parameter, "parameter"),
        ];

        for (kind, expected) in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", expected));
        }
    }

    #[test]
    fn test_symbol_visibility_default() {
        let visibility = SymbolVisibility::default();
        assert_eq!(visibility, SymbolVisibility::Unknown);
    }

    #[test]
    fn test_symbol_visibility_serialization() {
        let visibilities = vec![
            (SymbolVisibility::Public, "public"),
            (SymbolVisibility::Private, "private"),
            (SymbolVisibility::Protected, "protected"),
            (SymbolVisibility::Internal, "internal"),
            (SymbolVisibility::Crate, "crate"),
        ];

        for (vis, expected) in visibilities {
            let json = serde_json::to_string(&vis).unwrap();
            assert_eq!(json, format!("\"{}\"", expected));
        }
    }

    #[test]
    fn test_symbol_modifiers_default() {
        let modifiers = SymbolModifiers::default();
        assert!(!modifiers.is_async);
        assert!(!modifiers.is_static);
        assert!(!modifiers.is_abstract);
        assert!(!modifiers.is_const);
        assert!(!modifiers.is_exported);
        assert!(!modifiers.is_unsafe);
    }

    #[test]
    fn test_symbol_modifiers_serialization() {
        let modifiers = SymbolModifiers {
            is_async: true,
            is_static: false,
            is_abstract: true,
            is_exported: false,
            is_const: false,
            is_unsafe: true,
        };

        let json = serde_json::to_string(&modifiers).unwrap();
        let deserialized: SymbolModifiers = serde_json::from_str(&json).unwrap();

        assert_eq!(modifiers, deserialized);
    }

    #[test]
    fn test_symbol_info_with_new_fields() {
        let symbol = SymbolInfo {
            id: 1,
            file_id: 2,
            name: "test_function".to_string(),
            kind: SymbolKind::Function,
            start_line: 10,
            end_line: 20,
            signature: Some("pub async fn test_function<T>(arg: T)".to_string()),
            doc_comment: Some("/// Test doc".to_string()),
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers {
                is_async: true,
                ..Default::default()
            },
            parent_id: Some(100),
            type_parameters: vec!["T".to_string()],
            parameters: vec![],
            return_type: None,
        };

        // Test serialization roundtrip
        let json = serde_json::to_string(&symbol).unwrap();
        let deserialized: SymbolInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(symbol.name, deserialized.name);
        assert_eq!(symbol.visibility, deserialized.visibility);
        assert_eq!(symbol.modifiers.is_async, deserialized.modifiers.is_async);
        assert_eq!(symbol.parent_id, deserialized.parent_id);
        assert_eq!(symbol.type_parameters, deserialized.type_parameters);
    }

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("rb"), Language::Ruby);
        assert_eq!(Language::from_extension("rake"), Language::Ruby);
        assert_eq!(Language::from_extension("gemspec"), Language::Ruby);
        assert_eq!(Language::from_extension("unknown"), Language::Unknown);
    }
}
