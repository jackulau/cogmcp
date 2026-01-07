//! Shared types used across CogMCP crates

use serde::{Deserialize, Serialize};

/// Unique identifier for indexed files
pub type FileId = i64;

/// Unique identifier for symbols
pub type SymbolId = i64;

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

/// Visibility/access modifier for symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SymbolVisibility {
    /// Public visibility (accessible from anywhere)
    Public,
    /// Private visibility (only accessible within defining scope)
    #[default]
    Private,
    /// Protected visibility (accessible within class hierarchy)
    Protected,
    /// Internal visibility (accessible within module/package)
    Internal,
    /// Crate visibility (Rust's pub(crate))
    Crate,
}

/// Modifiers that can apply to symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct SymbolModifiers {
    /// Whether the symbol is async
    pub is_async: bool,
    /// Whether the symbol is static (class-level, not instance)
    pub is_static: bool,
    /// Whether the symbol is abstract
    pub is_abstract: bool,
    /// Whether the symbol is const (compile-time constant)
    pub is_const: bool,
    /// Whether the symbol is final/sealed (cannot be overridden)
    pub is_final: bool,
    /// Whether the symbol is unsafe (Rust)
    pub is_unsafe: bool,
    /// Whether the symbol is extern (foreign function)
    pub is_extern: bool,
    /// Whether the symbol is virtual (can be overridden)
    pub is_virtual: bool,
    /// Whether the symbol is mutable
    pub is_mut: bool,
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
    /// Visibility/access modifier
    pub visibility: Option<SymbolVisibility>,
    /// Symbol modifiers (async, static, etc.)
    pub modifiers: SymbolModifiers,
    /// Name of the parent symbol (for nested symbols like methods in a class)
    pub parent_symbol: Option<String>,
    /// Generic type parameters (e.g., ["T", "U"] for fn foo<T, U>)
    pub type_parameters: Vec<String>,
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
        assert_eq!(visibility, SymbolVisibility::Private);
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
        assert!(!modifiers.is_final);
        assert!(!modifiers.is_unsafe);
        assert!(!modifiers.is_extern);
        assert!(!modifiers.is_virtual);
        assert!(!modifiers.is_mut);
    }

    #[test]
    fn test_symbol_modifiers_serialization() {
        let modifiers = SymbolModifiers {
            is_async: true,
            is_static: false,
            is_abstract: true,
            is_const: false,
            is_final: false,
            is_unsafe: true,
            is_extern: false,
            is_virtual: false,
            is_mut: true,
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
            visibility: Some(SymbolVisibility::Public),
            modifiers: SymbolModifiers {
                is_async: true,
                ..Default::default()
            },
            parent_symbol: Some("TestModule".to_string()),
            type_parameters: vec!["T".to_string()],
        };

        // Test serialization roundtrip
        let json = serde_json::to_string(&symbol).unwrap();
        let deserialized: SymbolInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(symbol.name, deserialized.name);
        assert_eq!(symbol.visibility, deserialized.visibility);
        assert_eq!(symbol.modifiers.is_async, deserialized.modifiers.is_async);
        assert_eq!(symbol.parent_symbol, deserialized.parent_symbol);
        assert_eq!(symbol.type_parameters, deserialized.type_parameters);
    }

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("unknown"), Language::Unknown);
    }
}
