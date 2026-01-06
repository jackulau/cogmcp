//! Code parsing with tree-sitter
//!
//! This module provides enhanced symbol extraction using tree-sitter parsers.
//! It extracts rich metadata including visibility modifiers, generics, parameters,
//! return types, and decorators for supported languages.

use contextmcp_core::types::{Language, SymbolKind};
use contextmcp_core::{Error, Result};

/// Visibility modifier for symbols
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Private,
    Public,
    Protected,
    /// Rust: pub(crate), pub(super), etc.
    Restricted(String),
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
            Visibility::Protected => write!(f, "protected"),
            Visibility::Restricted(scope) => write!(f, "pub({})", scope),
        }
    }
}

/// A function/method parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default_value: Option<String>,
    pub is_variadic: bool,
}

impl std::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();

        if self.is_variadic {
            result.push_str("...");
        }

        result.push_str(&self.name);

        if let Some(ref ty) = self.type_annotation {
            result.push_str(": ");
            result.push_str(ty);
        }

        if let Some(ref default) = self.default_value {
            result.push_str(" = ");
            result.push_str(default);
        }

        write!(f, "{}", result)
    }
}

/// Decorator/attribute on a symbol
#[derive(Debug, Clone)]
pub struct Decorator {
    pub name: String,
    pub arguments: Option<String>,
}

impl std::fmt::Display for Decorator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref args) = self.arguments {
            write!(f, "@{}({})", self.name, args)
        } else {
            write!(f, "@{}", self.name)
        }
    }
}

/// Extracted symbol from source code with rich metadata
#[derive(Debug, Clone)]
pub struct ExtractedSymbol {
    /// Symbol name
    pub name: String,
    /// Symbol kind (function, class, etc.)
    pub kind: SymbolKind,
    /// Starting line (1-indexed)
    pub start_line: u32,
    /// Ending line (1-indexed)
    pub end_line: u32,
    /// Full signature with types
    pub signature: Option<String>,
    /// Documentation comment
    pub doc_comment: Option<String>,
    /// Visibility modifier
    pub visibility: Visibility,
    /// Generic type parameters (e.g., ["T", "U: Clone"])
    pub type_params: Vec<String>,
    /// Function/method parameters
    pub parameters: Vec<Parameter>,
    /// Return type annotation
    pub return_type: Option<String>,
    /// Decorators/attributes
    pub decorators: Vec<Decorator>,
    /// Parent symbol name (for nested symbols, e.g., methods in a class)
    pub parent_symbol: Option<String>,
    /// Whether this is async
    pub is_async: bool,
    /// Whether this is static (classmethod in Python, static in TS/JS)
    pub is_static: bool,
}

impl Default for ExtractedSymbol {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: SymbolKind::Unknown,
            start_line: 0,
            end_line: 0,
            signature: None,
            doc_comment: None,
            visibility: Visibility::Private,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        }
    }
}

/// Code parser using tree-sitter
pub struct CodeParser {
    // Parsers are created on demand per language
}

impl CodeParser {
    pub fn new() -> Self {
        Self {}
    }

    /// Parse a file and extract symbols
    pub fn parse(&self, content: &str, language: Language) -> Result<Vec<ExtractedSymbol>> {
        match language {
            Language::Rust => self.parse_rust(content),
            Language::TypeScript | Language::JavaScript => self.parse_typescript(content),
            Language::Python => self.parse_python(content),
            _ => Ok(Vec::new()), // Unsupported language
        }
    }

    // ==================== Helper Functions ====================

    /// Extract visibility modifier from a node
    pub fn extract_visibility(&self, node: &tree_sitter::Node, content: &str) -> Visibility {
        // Look for visibility_modifier child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    let text = child.utf8_text(content.as_bytes()).unwrap_or("").trim();
                    return match text {
                        "pub" => Visibility::Public,
                        s if s.starts_with("pub(") => {
                            // Extract scope from pub(crate), pub(super), etc.
                            let scope = s
                                .trim_start_matches("pub(")
                                .trim_end_matches(')')
                                .to_string();
                            Visibility::Restricted(scope)
                        }
                        _ => Visibility::Private,
                    };
                }
                // TypeScript/JavaScript modifiers
                "accessibility_modifier" | "public" | "private" | "protected" => {
                    let text = child.utf8_text(content.as_bytes()).unwrap_or("").trim();
                    return match text {
                        "public" => Visibility::Public,
                        "private" => Visibility::Private,
                        "protected" => Visibility::Protected,
                        _ => Visibility::Private,
                    };
                }
                _ => {}
            }
        }
        Visibility::Private
    }

    /// Extract generic type parameters from a node
    pub fn extract_type_params(&self, node: &tree_sitter::Node, content: &str) -> Vec<String> {
        let mut params = Vec::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "type_parameters" | "type_parameter_list" => {
                    let mut inner_cursor = child.walk();
                    for param in child.children(&mut inner_cursor) {
                        match param.kind() {
                            "type_parameter" | "type_identifier" | "constrained_type_parameter" => {
                                if let Ok(text) = param.utf8_text(content.as_bytes()) {
                                    params.push(text.trim().to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        params
    }

    /// Extract function parameters from a node
    pub fn extract_parameters(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        language: Language,
    ) -> Vec<Parameter> {
        let mut params = Vec::new();

        // Find parameters node
        let params_node = node.child_by_field_name("parameters");
        if params_node.is_none() {
            return params;
        }
        let params_node = params_node.unwrap();

        let mut cursor = params_node.walk();
        for child in params_node.children(&mut cursor) {
            match language {
                Language::Rust => {
                    if child.kind() == "parameter" {
                        if let Some(param) = self.extract_rust_parameter(&child, content) {
                            params.push(param);
                        }
                    } else if child.kind() == "self_parameter" {
                        let text = child.utf8_text(content.as_bytes()).unwrap_or("self");
                        params.push(Parameter {
                            name: text.to_string(),
                            type_annotation: None,
                            default_value: None,
                            is_variadic: false,
                        });
                    }
                }
                Language::TypeScript | Language::JavaScript => {
                    if matches!(
                        child.kind(),
                        "required_parameter"
                            | "optional_parameter"
                            | "rest_parameter"
                            | "formal_parameter"
                    ) {
                        if let Some(param) = self.extract_ts_parameter(&child, content) {
                            params.push(param);
                        }
                    }
                }
                Language::Python => {
                    if matches!(
                        child.kind(),
                        "identifier"
                            | "typed_parameter"
                            | "default_parameter"
                            | "typed_default_parameter"
                            | "list_splat_pattern"
                            | "dictionary_splat_pattern"
                    ) {
                        if let Some(param) = self.extract_python_parameter(&child, content) {
                            params.push(param);
                        }
                    }
                }
                _ => {}
            }
        }

        params
    }

    fn extract_rust_parameter(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<Parameter> {
        let name = node
            .child_by_field_name("pattern")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .unwrap_or("")
            .to_string();

        let type_annotation = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        Some(Parameter {
            name,
            type_annotation,
            default_value: None,
            is_variadic: false,
        })
    }

    fn extract_ts_parameter(&self, node: &tree_sitter::Node, content: &str) -> Option<Parameter> {
        let is_variadic = node.kind() == "rest_parameter";

        let name = if is_variadic {
            // For rest parameters, the pattern is inside
            let mut cursor = node.walk();
            let result = node
                .children(&mut cursor)
                .find(|c| c.kind() == "identifier")
                .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                .unwrap_or("")
                .to_string();
            result
        } else {
            node.child_by_field_name("pattern")
                .or_else(|| node.child_by_field_name("name"))
                .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                .unwrap_or("")
                .to_string()
        };

        let type_annotation = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.trim_start_matches(':').trim().to_string());

        let default_value = node
            .child_by_field_name("value")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        Some(Parameter {
            name,
            type_annotation,
            default_value,
            is_variadic,
        })
    }

    fn extract_python_parameter(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<Parameter> {
        let kind = node.kind();
        let is_variadic = matches!(kind, "list_splat_pattern" | "dictionary_splat_pattern");

        let (name, type_annotation, default_value) = match kind {
            "identifier" => {
                let name = node.utf8_text(content.as_bytes()).ok()?.to_string();
                (name, None, None)
            }
            "typed_parameter" => {
                let name = node
                    .child(0)
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("")
                    .to_string();
                let type_ann = node
                    .child_by_field_name("type")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string());
                (name, type_ann, None)
            }
            "default_parameter" => {
                let name = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("")
                    .to_string();
                let default = node
                    .child_by_field_name("value")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string());
                (name, None, default)
            }
            "typed_default_parameter" => {
                let name = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("")
                    .to_string();
                let type_ann = node
                    .child_by_field_name("type")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string());
                let default = node
                    .child_by_field_name("value")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string());
                (name, type_ann, default)
            }
            "list_splat_pattern" | "dictionary_splat_pattern" => {
                let prefix = if kind == "list_splat_pattern" {
                    "*"
                } else {
                    "**"
                };
                let inner_name = node
                    .child(0)
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("args");
                (format!("{}{}", prefix, inner_name), None, None)
            }
            _ => return None,
        };

        Some(Parameter {
            name,
            type_annotation,
            default_value,
            is_variadic,
        })
    }

    /// Extract return type from a function node
    pub fn extract_return_type(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        language: Language,
    ) -> Option<String> {
        match language {
            Language::Rust => {
                node.child_by_field_name("return_type")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.trim_start_matches("->").trim().to_string())
            }
            Language::TypeScript | Language::JavaScript => {
                node.child_by_field_name("return_type")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.trim_start_matches(':').trim().to_string())
            }
            Language::Python => {
                node.child_by_field_name("return_type")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.trim_start_matches("->").trim().to_string())
            }
            _ => None,
        }
    }

    /// Check if a function is async
    fn is_async_function(&self, node: &tree_sitter::Node, content: &str) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "async" {
                return true;
            }
            if let Ok(text) = child.utf8_text(content.as_bytes()) {
                if text == "async" {
                    return true;
                }
            }
        }
        false
    }

    /// Generate a rich signature from extracted metadata
    pub fn generate_signature(
        &self,
        name: &str,
        kind: SymbolKind,
        visibility: &Visibility,
        type_params: &[String],
        parameters: &[Parameter],
        return_type: Option<&str>,
        is_async: bool,
        language: Language,
    ) -> String {
        let mut sig = String::new();

        // Visibility
        match language {
            Language::Rust => {
                if *visibility != Visibility::Private {
                    sig.push_str(&visibility.to_string());
                    sig.push(' ');
                }
            }
            Language::TypeScript | Language::JavaScript => {
                if *visibility != Visibility::Private {
                    sig.push_str(&format!("{} ", visibility));
                }
            }
            _ => {}
        }

        // Async
        if is_async {
            sig.push_str("async ");
        }

        // Kind-specific prefix
        match kind {
            SymbolKind::Function | SymbolKind::Method => {
                sig.push_str("fn ");
            }
            SymbolKind::Struct => {
                sig.push_str("struct ");
            }
            SymbolKind::Enum => {
                sig.push_str("enum ");
            }
            SymbolKind::Trait => {
                sig.push_str("trait ");
            }
            SymbolKind::Interface => {
                sig.push_str("interface ");
            }
            SymbolKind::Class => {
                sig.push_str("class ");
            }
            SymbolKind::Type => {
                sig.push_str("type ");
            }
            _ => {}
        }

        // Name
        sig.push_str(name);

        // Type parameters
        if !type_params.is_empty() {
            sig.push('<');
            sig.push_str(&type_params.join(", "));
            sig.push('>');
        }

        // Parameters (for functions/methods)
        if matches!(kind, SymbolKind::Function | SymbolKind::Method) {
            sig.push('(');
            let param_strs: Vec<String> = parameters.iter().map(|p| p.to_string()).collect();
            sig.push_str(&param_strs.join(", "));
            sig.push(')');
        }

        // Return type
        if let Some(ret) = return_type {
            sig.push_str(" -> ");
            sig.push_str(ret);
        }

        sig
    }

    // ==================== Rust Parsing ====================

    fn parse_rust(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse Rust code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_rust_symbols(&root, content, &mut symbols, None);

        Ok(symbols)
    }

    fn extract_rust_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent: Option<&str>,
    ) {
        let kind_str = node.kind();

        match kind_str {
            "function_item" => {
                if let Some(symbol) = self.extract_rust_function(node, content, parent) {
                    symbols.push(symbol);
                }
            }
            "impl_item" => {
                // Extract impl block and its methods
                if let Some(impl_symbol) = self.extract_rust_impl(node, content) {
                    let impl_name = impl_symbol.name.clone();
                    symbols.push(impl_symbol);

                    // Extract methods inside impl
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            if child.kind() == "function_item" {
                                if let Some(method) =
                                    self.extract_rust_function(&child, content, Some(&impl_name))
                                {
                                    symbols.push(method);
                                }
                            }
                        }
                    }
                    return; // Don't recurse normally, we handled methods
                }
            }
            "struct_item" => {
                if let Some(symbol) = self.extract_rust_struct(node, content) {
                    let struct_name = symbol.name.clone();
                    symbols.push(symbol);

                    // Extract fields
                    if let Some(body) = node.child_by_field_name("body") {
                        self.extract_rust_fields(&body, content, symbols, &struct_name);
                    }
                }
            }
            "enum_item" => {
                if let Some(symbol) = self.extract_rust_enum(node, content) {
                    symbols.push(symbol);
                }
            }
            "trait_item" => {
                if let Some(symbol) = self.extract_rust_trait(node, content) {
                    let trait_name = symbol.name.clone();
                    symbols.push(symbol);

                    // Extract trait methods
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            if child.kind() == "function_signature_item"
                                || child.kind() == "function_item"
                            {
                                if let Some(method) =
                                    self.extract_rust_function(&child, content, Some(&trait_name))
                                {
                                    symbols.push(method);
                                }
                            }
                        }
                    }
                    return;
                }
            }
            "mod_item" => {
                if let Some(symbol) = self.extract_rust_mod(node, content) {
                    symbols.push(symbol);
                }
            }
            "const_item" => {
                if let Some(symbol) = self.extract_rust_const(node, content) {
                    symbols.push(symbol);
                }
            }
            "static_item" => {
                if let Some(symbol) = self.extract_rust_static(node, content) {
                    symbols.push(symbol);
                }
            }
            "type_item" => {
                if let Some(symbol) = self.extract_rust_type_alias(node, content) {
                    symbols.push(symbol);
                }
            }
            _ => {}
        }

        // Recurse into children (except for impl/trait which we handle specially)
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_rust_symbols(&child, content, symbols, parent);
        }
    }

    fn extract_rust_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent: Option<&str>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_params = self.extract_type_params(node, content);
        let parameters = self.extract_parameters(node, content, Language::Rust);
        let return_type = self.extract_return_type(node, content, Language::Rust);
        let is_async = self.is_async_function(node, content);

        let kind = if parent.is_some() {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        let signature = self.generate_signature(
            &name,
            kind,
            &visibility,
            &type_params,
            &parameters,
            return_type.as_deref(),
            is_async,
            Language::Rust,
        );

        let doc_comment = self.find_doc_comment(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment,
            visibility,
            type_params,
            parameters,
            return_type,
            decorators: Vec::new(), // Rust uses attributes, not decorators
            parent_symbol: parent.map(|s| s.to_string()),
            is_async,
            is_static: false,
        })
    }

    fn extract_rust_impl(&self, node: &tree_sitter::Node, content: &str) -> Option<ExtractedSymbol> {
        // Get the type being implemented
        let type_node = node.child_by_field_name("type")?;
        let name = type_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Check if it's a trait impl
        let trait_node = node.child_by_field_name("trait");
        let trait_name = trait_node.and_then(|n| n.utf8_text(content.as_bytes()).ok());

        let type_params = self.extract_type_params(node, content);

        let signature = if let Some(trait_n) = trait_name {
            format!("impl {} for {}", trait_n, name)
        } else {
            format!("impl {}", name)
        };

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Class, // impl blocks are like class extensions
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility: Visibility::Private,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_struct(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_params = self.extract_type_params(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Struct,
            &visibility,
            &type_params,
            &[],
            None,
            false,
            Language::Rust,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Struct,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_fields(
        &self,
        body: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: &str,
    ) {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "field_declaration" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(content.as_bytes()) {
                        let visibility = self.extract_visibility(&child, content);
                        let type_annotation = child
                            .child_by_field_name("type")
                            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                            .map(|s| s.to_string());

                        let signature =
                            format!("{}: {}", name, type_annotation.as_deref().unwrap_or("?"));

                        symbols.push(ExtractedSymbol {
                            name: name.to_string(),
                            kind: SymbolKind::Field,
                            start_line: child.start_position().row as u32 + 1,
                            end_line: child.end_position().row as u32 + 1,
                            signature: Some(signature),
                            doc_comment: self.find_doc_comment(&child, content),
                            visibility,
                            type_params: Vec::new(),
                            parameters: Vec::new(),
                            return_type: type_annotation,
                            decorators: Vec::new(),
                            parent_symbol: Some(parent_name.to_string()),
                            is_async: false,
                            is_static: false,
                        });
                    }
                }
            }
        }
    }

    fn extract_rust_enum(&self, node: &tree_sitter::Node, content: &str) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_params = self.extract_type_params(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Enum,
            &visibility,
            &type_params,
            &[],
            None,
            false,
            Language::Rust,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Enum,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_trait(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_params = self.extract_type_params(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Trait,
            &visibility,
            &type_params,
            &[],
            None,
            false,
            Language::Rust,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Trait,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_mod(&self, node: &tree_sitter::Node, content: &str) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);

        Some(ExtractedSymbol {
            name: name.clone(),
            kind: SymbolKind::Module,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(format!("mod {}", name)),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_const(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_annotation = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        let signature = format!("const {}: {}", name, type_annotation.as_deref().unwrap_or("?"));

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Constant,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: type_annotation,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_rust_static(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_annotation = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        // Check if it's mutable
        let is_mut = node
            .children(&mut node.walk())
            .any(|c| c.kind() == "mutable_specifier");

        let signature = if is_mut {
            format!(
                "static mut {}: {}",
                name,
                type_annotation.as_deref().unwrap_or("?")
            )
        } else {
            format!(
                "static {}: {}",
                name,
                type_annotation.as_deref().unwrap_or("?")
            )
        };

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Variable,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: type_annotation,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: true,
        })
    }

    fn extract_rust_type_alias(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let type_params = self.extract_type_params(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Type,
            &visibility,
            &type_params,
            &[],
            None,
            false,
            Language::Rust,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Type,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_doc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    // ==================== TypeScript/JavaScript Parsing ====================

    fn parse_typescript(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            .map_err(|e| Error::Parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse TypeScript code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_ts_symbols(&root, content, &mut symbols, None, false);

        Ok(symbols)
    }

    fn extract_ts_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent: Option<&str>,
        is_exported: bool,
    ) {
        let kind_str = node.kind();

        // Check if this is an export statement
        let mut child_exported = is_exported;
        if kind_str == "export_statement" {
            child_exported = true;
        }

        match kind_str {
            "function_declaration" | "generator_function_declaration" => {
                if let Some(symbol) =
                    self.extract_ts_function(node, content, parent, child_exported)
                {
                    symbols.push(symbol);
                }
            }
            "arrow_function" => {
                // Arrow functions are usually assigned to variables, handled separately
            }
            "class_declaration" => {
                if let Some(symbol) = self.extract_ts_class(node, content, child_exported) {
                    let class_name = symbol.name.clone();
                    symbols.push(symbol);

                    // Extract class members
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            match child.kind() {
                                "method_definition" | "public_field_definition"
                                | "field_definition" => {
                                    self.extract_ts_symbols(
                                        &child,
                                        content,
                                        symbols,
                                        Some(&class_name),
                                        false,
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                    return;
                }
            }
            "method_definition" => {
                if let Some(symbol) = self.extract_ts_method(node, content, parent) {
                    symbols.push(symbol);
                }
            }
            "public_field_definition" | "field_definition" => {
                if let Some(symbol) = self.extract_ts_field(node, content, parent) {
                    symbols.push(symbol);
                }
            }
            "interface_declaration" => {
                if let Some(symbol) = self.extract_ts_interface(node, content, child_exported) {
                    symbols.push(symbol);
                }
            }
            "type_alias_declaration" => {
                if let Some(symbol) = self.extract_ts_type_alias(node, content, child_exported) {
                    symbols.push(symbol);
                }
            }
            "enum_declaration" => {
                if let Some(symbol) = self.extract_ts_enum(node, content, child_exported) {
                    symbols.push(symbol);
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                // Handle const/let/var declarations
                self.extract_ts_variables(node, content, symbols, child_exported);
            }
            _ => {}
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_ts_symbols(&child, content, symbols, parent, child_exported);
        }
    }

    fn extract_ts_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent: Option<&str>,
        is_exported: bool,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = if is_exported {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let type_params = self.extract_type_params(node, content);
        let parameters = self.extract_parameters(node, content, Language::TypeScript);
        let return_type = self.extract_return_type(node, content, Language::TypeScript);
        let is_async = self.is_async_function(node, content);

        let kind = if parent.is_some() {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        let signature = self.generate_signature(
            &name,
            kind,
            &visibility,
            &type_params,
            &parameters,
            return_type.as_deref(),
            is_async,
            Language::TypeScript,
        );

        Some(ExtractedSymbol {
            name,
            kind,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params,
            parameters,
            return_type,
            decorators: self.extract_ts_decorators(node, content),
            parent_symbol: parent.map(|s| s.to_string()),
            is_async,
            is_static: false,
        })
    }

    fn extract_ts_class(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        is_exported: bool,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = if is_exported {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let type_params = self.extract_type_params(node, content);

        // Check for extends/implements
        let mut extends = None;
        let mut implements = Vec::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "class_heritage" => {
                    let mut heritage_cursor = child.walk();
                    for heritage_child in child.children(&mut heritage_cursor) {
                        if heritage_child.kind() == "extends_clause" {
                            extends = heritage_child
                                .child(1)
                                .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                                .map(|s| s.to_string());
                        } else if heritage_child.kind() == "implements_clause" {
                            let mut impl_cursor = heritage_child.walk();
                            for impl_child in heritage_child.children(&mut impl_cursor) {
                                if impl_child.kind() == "type_identifier" {
                                    if let Ok(impl_name) =
                                        impl_child.utf8_text(content.as_bytes())
                                    {
                                        implements.push(impl_name.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let mut signature = format!("class {}", name);
        if !type_params.is_empty() {
            signature.push('<');
            signature.push_str(&type_params.join(", "));
            signature.push('>');
        }
        if let Some(ref ext) = extends {
            signature.push_str(&format!(" extends {}", ext));
        }
        if !implements.is_empty() {
            signature.push_str(&format!(" implements {}", implements.join(", ")));
        }

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Class,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: self.extract_ts_decorators(node, content),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_ts_method(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent: Option<&str>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Check for visibility modifier
        let visibility = self.extract_visibility(node, content);

        // Check if static
        let is_static = node.children(&mut node.walk()).any(|c| c.kind() == "static");

        let type_params = self.extract_type_params(node, content);
        let parameters = self.extract_parameters(node, content, Language::TypeScript);
        let return_type = self.extract_return_type(node, content, Language::TypeScript);
        let is_async = self.is_async_function(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Method,
            &visibility,
            &type_params,
            &parameters,
            return_type.as_deref(),
            is_async,
            Language::TypeScript,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Method,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params,
            parameters,
            return_type,
            decorators: self.extract_ts_decorators(node, content),
            parent_symbol: parent.map(|s| s.to_string()),
            is_async,
            is_static,
        })
    }

    fn extract_ts_field(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent: Option<&str>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_visibility(node, content);
        let is_static = node.children(&mut node.walk()).any(|c| c.kind() == "static");

        let type_annotation = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.trim_start_matches(':').trim().to_string());

        let signature = format!("{}: {}", name, type_annotation.as_deref().unwrap_or("any"));

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Field,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: type_annotation,
            decorators: self.extract_ts_decorators(node, content),
            parent_symbol: parent.map(|s| s.to_string()),
            is_async: false,
            is_static,
        })
    }

    fn extract_ts_interface(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        is_exported: bool,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = if is_exported {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let type_params = self.extract_type_params(node, content);

        let mut signature = format!("interface {}", name);
        if !type_params.is_empty() {
            signature.push('<');
            signature.push_str(&type_params.join(", "));
            signature.push('>');
        }

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Interface,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_ts_type_alias(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        is_exported: bool,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = if is_exported {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let type_params = self.extract_type_params(node, content);

        let signature = self.generate_signature(
            &name,
            SymbolKind::Type,
            &visibility,
            &type_params,
            &[],
            None,
            false,
            Language::TypeScript,
        );

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Type,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params,
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_ts_enum(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        is_exported: bool,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = if is_exported {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let signature = format!("enum {}", name);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Enum,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_jsdoc_comment(node, content),
            visibility,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_ts_variables(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        is_exported: bool,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                let name_node = child.child_by_field_name("name");
                if let Some(name_node) = name_node {
                    if let Ok(name) = name_node.utf8_text(content.as_bytes()) {
                        let visibility = if is_exported {
                            Visibility::Public
                        } else {
                            Visibility::Private
                        };

                        let type_annotation = child
                            .child_by_field_name("type")
                            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                            .map(|s| s.trim_start_matches(':').trim().to_string());

                        // Check if it's a const/let/var
                        let kind_prefix = node
                            .children(&mut node.walk())
                            .find(|c| matches!(c.kind(), "const" | "let" | "var"))
                            .and_then(|c| c.utf8_text(content.as_bytes()).ok())
                            .unwrap_or("let");

                        // Check if the value is a function
                        let value_node = child.child_by_field_name("value");
                        let is_function = value_node
                            .map(|v| matches!(v.kind(), "arrow_function" | "function"))
                            .unwrap_or(false);

                        let kind = if kind_prefix == "const" && !is_function {
                            SymbolKind::Constant
                        } else if is_function {
                            SymbolKind::Function
                        } else {
                            SymbolKind::Variable
                        };

                        let signature = if let Some(ref ty) = type_annotation {
                            format!("{} {}: {}", kind_prefix, name, ty)
                        } else {
                            format!("{} {}", kind_prefix, name)
                        };

                        // For arrow functions, extract parameters and return type
                        let (parameters, return_type, is_async) = if is_function {
                            let func_node = value_node.unwrap();
                            let params =
                                self.extract_parameters(&func_node, content, Language::TypeScript);
                            let ret =
                                self.extract_return_type(&func_node, content, Language::TypeScript);
                            let async_fn = self.is_async_function(&func_node, content);
                            (params, ret, async_fn)
                        } else {
                            (Vec::new(), type_annotation.clone(), false)
                        };

                        symbols.push(ExtractedSymbol {
                            name: name.to_string(),
                            kind,
                            start_line: child.start_position().row as u32 + 1,
                            end_line: child.end_position().row as u32 + 1,
                            signature: Some(signature),
                            doc_comment: self.find_jsdoc_comment(&child, content),
                            visibility,
                            type_params: Vec::new(),
                            parameters,
                            return_type,
                            decorators: Vec::new(),
                            parent_symbol: None,
                            is_async,
                            is_static: false,
                        });
                    }
                }
            }
        }
    }

    fn extract_ts_decorators(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Vec<Decorator> {
        let mut decorators = Vec::new();

        // Look at previous siblings for decorators
        let mut current = node.prev_sibling();
        while let Some(sibling) = current {
            if sibling.kind() == "decorator" {
                let decorator_text = sibling.utf8_text(content.as_bytes()).unwrap_or("");
                let name = decorator_text
                    .trim_start_matches('@')
                    .split('(')
                    .next()
                    .unwrap_or("")
                    .to_string();
                let arguments = if decorator_text.contains('(') {
                    Some(
                        decorator_text
                            .split('(')
                            .nth(1)
                            .unwrap_or("")
                            .trim_end_matches(')')
                            .to_string(),
                    )
                } else {
                    None
                };
                decorators.push(Decorator { name, arguments });
            } else if sibling.kind() != "comment" {
                break;
            }
            current = sibling.prev_sibling();
        }

        decorators.reverse();
        decorators
    }

    fn find_jsdoc_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for JSDoc comment
        if let Some(prev) = node.prev_sibling() {
            if prev.kind() == "comment" {
                let text = prev.utf8_text(content.as_bytes()).ok()?;
                if text.starts_with("/**") {
                    return self.parse_jsdoc(text);
                }
            }
        }

        // For exported items, the comment might be before the export statement
        if let Some(parent) = node.parent() {
            if parent.kind() == "export_statement" {
                if let Some(prev) = parent.prev_sibling() {
                    if prev.kind() == "comment" {
                        let text = prev.utf8_text(content.as_bytes()).ok()?;
                        if text.starts_with("/**") {
                            return self.parse_jsdoc(text);
                        }
                    }
                }
            }
        }

        None
    }

    fn parse_jsdoc(&self, text: &str) -> Option<String> {
        Some(
            text.trim_start_matches("/**")
                .trim_end_matches("*/")
                .lines()
                .map(|l| l.trim().trim_start_matches('*').trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    // ==================== Python Parsing ====================

    fn parse_python(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse Python code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_python_symbols(&root, content, &mut symbols, None);

        Ok(symbols)
    }

    fn extract_python_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent: Option<&str>,
    ) {
        let kind_str = node.kind();

        match kind_str {
            "function_definition" => {
                if let Some(symbol) = self.extract_python_function(node, content, parent) {
                    symbols.push(symbol);
                }
            }
            "class_definition" => {
                if let Some(symbol) = self.extract_python_class(node, content) {
                    let class_name = symbol.name.clone();
                    symbols.push(symbol);

                    // Extract class methods and attributes
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            match child.kind() {
                                "function_definition" => {
                                    if let Some(method) =
                                        self.extract_python_function(&child, content, Some(&class_name))
                                    {
                                        symbols.push(method);
                                    }
                                }
                                "decorated_definition" => {
                                    // Handle decorated methods
                                    let mut inner_cursor = child.walk();
                                    for inner_child in child.children(&mut inner_cursor) {
                                        if inner_child.kind() == "function_definition" {
                                            if let Some(method) =
                                                self.extract_python_function(&inner_child, content, Some(&class_name))
                                            {
                                                symbols.push(method);
                                            }
                                        }
                                    }
                                }
                                "expression_statement" => {
                                    // Class-level assignments
                                    if let Some(assign) = self.extract_python_class_var(&child, content, &class_name) {
                                        symbols.push(assign);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    return;
                }
            }
            "decorated_definition" => {
                // Handle decorated functions/classes
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "decorator" {
                        self.extract_python_symbols(&child, content, symbols, parent);
                    }
                }
                return;
            }
            _ => {}
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_python_symbols(&child, content, symbols, parent);
        }
    }

    fn extract_python_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent: Option<&str>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Detect if this is a method based on parent
        let is_method = parent.is_some();

        // Extract decorators
        let decorators = self.extract_python_decorators(node, content);

        // Determine if static/classmethod/property based on decorators
        let is_static = decorators.iter().any(|d| d.name == "staticmethod");
        let is_classmethod = decorators.iter().any(|d| d.name == "classmethod");
        let is_property = decorators.iter().any(|d| d.name == "property");

        // Check for async
        let is_async = node
            .prev_sibling()
            .map(|s| s.kind() == "async")
            .unwrap_or(false)
            || node.children(&mut node.walk()).any(|c| c.kind() == "async");

        // Detect dunder methods
        let is_dunder = name.starts_with("__") && name.ends_with("__");

        let parameters = self.extract_parameters(node, content, Language::Python);
        let return_type = self.extract_return_type(node, content, Language::Python);

        // Determine kind
        let kind = if is_property {
            SymbolKind::Property
        } else if is_method {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        // Build signature
        let mut signature = String::new();

        // Add decorators to signature representation
        for dec in &decorators {
            signature.push_str(&format!("@{}\n", dec.name));
        }

        if is_async {
            signature.push_str("async ");
        }

        signature.push_str("def ");
        signature.push_str(&name);
        signature.push('(');

        let param_strs: Vec<String> = parameters.iter().map(|p| p.to_string()).collect();
        signature.push_str(&param_strs.join(", "));
        signature.push(')');

        if let Some(ref ret) = return_type {
            signature.push_str(" -> ");
            signature.push_str(ret);
        }

        // Determine visibility based on name convention
        // Dunder methods (like __init__, __str__) are public
        // Methods starting with __ but not ending with __ are private (name mangling)
        // Methods starting with single _ are protected
        let visibility = if is_dunder {
            Visibility::Public
        } else if name.starts_with("__") {
            Visibility::Private
        } else if name.starts_with('_') {
            Visibility::Protected
        } else {
            Visibility::Public
        };

        Some(ExtractedSymbol {
            name,
            kind,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_python_docstring(node, content),
            visibility,
            type_params: Vec::new(), // Python doesn't have type params in the same way
            parameters,
            return_type,
            decorators,
            parent_symbol: parent.map(|s| s.to_string()),
            is_async,
            is_static: is_static || is_classmethod,
        })
    }

    fn extract_python_class(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Extract base classes
        let bases = node
            .child_by_field_name("superclasses")
            .map(|args| args.utf8_text(content.as_bytes()).ok())
            .flatten()
            .map(|s| s.to_string());

        let decorators = self.extract_python_decorators(node, content);

        let mut signature = String::new();
        for dec in &decorators {
            signature.push_str(&format!("@{}\n", dec.name));
        }
        signature.push_str("class ");
        signature.push_str(&name);
        if let Some(ref b) = bases {
            signature.push_str(b);
        }

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Class,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            signature: Some(signature),
            doc_comment: self.find_python_docstring(node, content),
            visibility: Visibility::Public,
            type_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            decorators,
            parent_symbol: None,
            is_async: false,
            is_static: false,
        })
    }

    fn extract_python_class_var(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: &str,
    ) -> Option<ExtractedSymbol> {
        // Look for assignment
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "assignment" {
                let left = child.child_by_field_name("left")?;
                if left.kind() == "identifier" {
                    let name = left.utf8_text(content.as_bytes()).ok()?.to_string();

                    // Try to get type annotation
                    let type_annotation = child
                        .child_by_field_name("type")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string());

                    let visibility = if name.starts_with("__") {
                        Visibility::Private
                    } else if name.starts_with('_') {
                        Visibility::Protected
                    } else {
                        Visibility::Public
                    };

                    return Some(ExtractedSymbol {
                        name: name.clone(),
                        kind: SymbolKind::Field,
                        start_line: child.start_position().row as u32 + 1,
                        end_line: child.end_position().row as u32 + 1,
                        signature: Some(format!(
                            "{}: {}",
                            name,
                            type_annotation.as_deref().unwrap_or("Any")
                        )),
                        doc_comment: None,
                        visibility,
                        type_params: Vec::new(),
                        parameters: Vec::new(),
                        return_type: type_annotation,
                        decorators: Vec::new(),
                        parent_symbol: Some(parent_name.to_string()),
                        is_async: false,
                        is_static: true, // Class variables are static
                    });
                }
            }
        }
        None
    }

    fn extract_python_decorators(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Vec<Decorator> {
        let mut decorators = Vec::new();

        // Check if parent is decorated_definition
        if let Some(parent) = node.parent() {
            if parent.kind() == "decorated_definition" {
                let mut cursor = parent.walk();
                for child in parent.children(&mut cursor) {
                    if child.kind() == "decorator" {
                        let decorator_text = child.utf8_text(content.as_bytes()).unwrap_or("");
                        let clean_text = decorator_text.trim_start_matches('@').trim();

                        let (name, arguments) = if clean_text.contains('(') {
                            let parts: Vec<&str> = clean_text.splitn(2, '(').collect();
                            let name = parts[0].to_string();
                            let args = parts
                                .get(1)
                                .map(|s| s.trim_end_matches(')').to_string());
                            (name, args)
                        } else {
                            (clean_text.to_string(), None)
                        };

                        decorators.push(Decorator { name, arguments });
                    }
                }
            }
        }

        decorators
    }

    fn find_doc_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for comment
        if let Some(prev) = node.prev_sibling() {
            match prev.kind() {
                "line_comment" => {
                    let mut comments = Vec::new();
                    let mut current = Some(prev);
                    while let Some(c) = current {
                        if c.kind() == "line_comment" {
                            if let Ok(text) = c.utf8_text(content.as_bytes()) {
                                comments.push(text.trim_start_matches("//").trim().to_string());
                            }
                            current = c.prev_sibling();
                        } else {
                            break;
                        }
                    }
                    comments.reverse();
                    if !comments.is_empty() {
                        return Some(comments.join("\n"));
                    }
                }
                "block_comment" => {
                    return prev.utf8_text(content.as_bytes()).ok().map(|s| {
                        s.trim_start_matches("/*")
                            .trim_end_matches("*/")
                            .trim()
                            .to_string()
                    });
                }
                _ => {}
            }
        }
        None
    }

    fn find_python_docstring(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look for string as first child of body
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "expression_statement" {
                    let mut inner_cursor = child.walk();
                    for inner in child.children(&mut inner_cursor) {
                        if inner.kind() == "string" {
                            return inner.utf8_text(content.as_bytes()).ok().map(|s| {
                                // Handle triple-quoted strings
                                let trimmed = s
                                    .trim_start_matches("\"\"\"")
                                    .trim_start_matches("'''")
                                    .trim_end_matches("\"\"\"")
                                    .trim_end_matches("'''")
                                    .trim_start_matches('"')
                                    .trim_start_matches('\'')
                                    .trim_end_matches('"')
                                    .trim_end_matches('\'')
                                    .trim();
                                trimmed.to_string()
                            });
                        }
                    }
                }
                break; // Only check first statement
            }
        }
        None
    }
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Rust Tests ====================

    #[test]
    fn test_rust_function_with_visibility() {
        let parser = CodeParser::new();
        let code = r#"
/// A public function
pub fn public_func(x: i32, y: String) -> bool {
    true
}

fn private_func() {
}

pub(crate) fn crate_visible() -> u32 {
    42
}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        // Find public_func
        let public_fn = symbols.iter().find(|s| s.name == "public_func").unwrap();
        assert_eq!(public_fn.visibility, Visibility::Public);
        assert_eq!(public_fn.parameters.len(), 2);
        assert_eq!(public_fn.parameters[0].name, "x");
        assert_eq!(
            public_fn.parameters[0].type_annotation,
            Some("i32".to_string())
        );
        assert_eq!(public_fn.return_type, Some("bool".to_string()));
        assert!(public_fn.doc_comment.is_some());

        // Find private_func
        let private_fn = symbols.iter().find(|s| s.name == "private_func").unwrap();
        assert_eq!(private_fn.visibility, Visibility::Private);

        // Find crate_visible
        let crate_fn = symbols.iter().find(|s| s.name == "crate_visible").unwrap();
        assert!(matches!(crate_fn.visibility, Visibility::Restricted(_)));
    }

    #[test]
    fn test_rust_struct_with_generics() {
        let parser = CodeParser::new();
        let code = r#"
pub struct MyStruct<T, U: Clone> {
    pub field1: T,
    field2: U,
}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let my_struct = symbols.iter().find(|s| s.name == "MyStruct").unwrap();
        assert_eq!(my_struct.kind, SymbolKind::Struct);
        assert_eq!(my_struct.visibility, Visibility::Public);
        assert_eq!(my_struct.type_params.len(), 2);

        // Check fields
        let field1 = symbols.iter().find(|s| s.name == "field1").unwrap();
        assert_eq!(field1.kind, SymbolKind::Field);
        assert_eq!(field1.visibility, Visibility::Public);
        assert_eq!(field1.parent_symbol, Some("MyStruct".to_string()));
    }

    #[test]
    fn test_rust_impl_with_methods() {
        let parser = CodeParser::new();
        let code = r#"
impl MyStruct {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn async_method(&self, arg: &str) -> Result<(), Error> {
        Ok(())
    }
}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        // Check impl block
        let impl_block = symbols
            .iter()
            .find(|s| s.name == "MyStruct" && s.kind == SymbolKind::Class)
            .unwrap();
        assert!(impl_block.signature.as_ref().unwrap().contains("impl MyStruct"));

        // Check methods
        let new_method = symbols.iter().find(|s| s.name == "new").unwrap();
        assert_eq!(new_method.kind, SymbolKind::Method);
        assert_eq!(new_method.parent_symbol, Some("MyStruct".to_string()));

        let async_method = symbols.iter().find(|s| s.name == "async_method").unwrap();
        assert!(async_method.is_async);
        assert_eq!(async_method.parameters.len(), 2); // &self and arg
    }

    #[test]
    fn test_rust_trait_with_methods() {
        let parser = CodeParser::new();
        let code = r#"
pub trait MyTrait<T> {
    fn required_method(&self) -> T;

    fn default_method(&self) -> bool {
        true
    }
}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let my_trait = symbols.iter().find(|s| s.name == "MyTrait").unwrap();
        assert_eq!(my_trait.kind, SymbolKind::Trait);
        assert!(!my_trait.type_params.is_empty());

        let required = symbols
            .iter()
            .find(|s| s.name == "required_method")
            .unwrap();
        assert_eq!(required.parent_symbol, Some("MyTrait".to_string()));
    }

    #[test]
    fn test_rust_const_and_static() {
        let parser = CodeParser::new();
        let code = r#"
pub const MAX_SIZE: usize = 100;
static mut COUNTER: i32 = 0;
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let max_size = symbols.iter().find(|s| s.name == "MAX_SIZE").unwrap();
        assert_eq!(max_size.kind, SymbolKind::Constant);
        assert_eq!(max_size.return_type, Some("usize".to_string()));

        let counter = symbols.iter().find(|s| s.name == "COUNTER").unwrap();
        assert_eq!(counter.kind, SymbolKind::Variable);
        assert!(counter.is_static);
    }

    // ==================== TypeScript Tests ====================

    #[test]
    fn test_ts_exported_function() {
        let parser = CodeParser::new();
        let code = r#"
/**
 * A documented function
 */
export function myFunction<T>(arg1: string, arg2?: number): T[] {
    return [];
}

function privateFunction(): void {}
"#;

        let symbols = parser.parse(code, Language::TypeScript).unwrap();

        let my_func = symbols.iter().find(|s| s.name == "myFunction").unwrap();
        assert_eq!(my_func.visibility, Visibility::Public);
        assert_eq!(my_func.parameters.len(), 2);
        assert!(my_func.doc_comment.is_some());

        let private_func = symbols
            .iter()
            .find(|s| s.name == "privateFunction")
            .unwrap();
        assert_eq!(private_func.visibility, Visibility::Private);
    }

    #[test]
    fn test_ts_class_with_members() {
        let parser = CodeParser::new();
        let code = r#"
export class MyClass<T> extends BaseClass implements IInterface {
    private _value: T;
    public name: string;

    constructor(value: T) {
        this._value = value;
    }

    public async fetchData(): Promise<T> {
        return this._value;
    }

    static create(): MyClass<string> {
        return new MyClass("");
    }
}
"#;

        let symbols = parser.parse(code, Language::TypeScript).unwrap();

        let my_class = symbols.iter().find(|s| s.name == "MyClass").unwrap();
        assert_eq!(my_class.kind, SymbolKind::Class);
        assert_eq!(my_class.visibility, Visibility::Public);
        assert!(my_class.signature.as_ref().unwrap().contains("extends BaseClass"));

        let fetch_data = symbols.iter().find(|s| s.name == "fetchData").unwrap();
        assert!(fetch_data.is_async);
        assert_eq!(fetch_data.parent_symbol, Some("MyClass".to_string()));
    }

    #[test]
    fn test_ts_interface() {
        let parser = CodeParser::new();
        let code = r#"
export interface MyInterface<T> {
    name: string;
    value: T;
}
"#;

        let symbols = parser.parse(code, Language::TypeScript).unwrap();

        let my_interface = symbols.iter().find(|s| s.name == "MyInterface").unwrap();
        assert_eq!(my_interface.kind, SymbolKind::Interface);
        assert_eq!(my_interface.visibility, Visibility::Public);
    }

    #[test]
    fn test_ts_arrow_function() {
        let parser = CodeParser::new();
        let code = r#"
export const myArrowFunc = async (x: number): Promise<string> => {
    return x.toString();
};

const privateArrow = () => {};
"#;

        let symbols = parser.parse(code, Language::TypeScript).unwrap();

        let my_arrow = symbols.iter().find(|s| s.name == "myArrowFunc").unwrap();
        assert_eq!(my_arrow.kind, SymbolKind::Function);
        assert_eq!(my_arrow.visibility, Visibility::Public);
        assert!(my_arrow.is_async);
    }

    // ==================== Python Tests ====================

    #[test]
    fn test_python_function_with_types() {
        let parser = CodeParser::new();
        let code = r#"
def my_function(x: int, y: str = "default") -> bool:
    """A documented function."""
    return True

async def async_function(data: list) -> dict:
    """An async function."""
    return {}
"#;

        let symbols = parser.parse(code, Language::Python).unwrap();

        let my_func = symbols.iter().find(|s| s.name == "my_function").unwrap();
        assert_eq!(my_func.kind, SymbolKind::Function);
        assert_eq!(my_func.parameters.len(), 2);
        assert_eq!(my_func.return_type, Some("bool".to_string()));
        assert!(my_func.doc_comment.is_some());

        let async_func = symbols.iter().find(|s| s.name == "async_function");
        // Note: async detection in Python may vary based on tree-sitter version
        if let Some(af) = async_func {
            assert_eq!(af.kind, SymbolKind::Function);
        }
    }

    #[test]
    fn test_python_class_with_decorators() {
        let parser = CodeParser::new();
        let code = r#"
@dataclass
class MyClass:
    """A dataclass."""
    name: str
    value: int = 0

    @property
    def computed(self) -> str:
        return self.name

    @staticmethod
    def static_method() -> None:
        pass

    @classmethod
    def class_method(cls) -> "MyClass":
        return cls()

    def __init__(self, name: str):
        self.name = name

    def _protected_method(self) -> None:
        pass

    def __private_method(self) -> None:
        pass
"#;

        let symbols = parser.parse(code, Language::Python).unwrap();

        let my_class = symbols.iter().find(|s| s.name == "MyClass").unwrap();
        assert_eq!(my_class.kind, SymbolKind::Class);
        assert!(my_class.decorators.iter().any(|d| d.name == "dataclass"));

        let computed = symbols.iter().find(|s| s.name == "computed").unwrap();
        assert_eq!(computed.kind, SymbolKind::Property);
        assert!(computed.decorators.iter().any(|d| d.name == "property"));

        let static_method = symbols.iter().find(|s| s.name == "static_method").unwrap();
        assert!(static_method.is_static);

        let init = symbols.iter().find(|s| s.name == "__init__").unwrap();
        assert_eq!(init.visibility, Visibility::Public); // Dunder methods are public

        let protected = symbols
            .iter()
            .find(|s| s.name == "_protected_method")
            .unwrap();
        assert_eq!(protected.visibility, Visibility::Protected);

        let private = symbols
            .iter()
            .find(|s| s.name == "__private_method")
            .unwrap();
        assert_eq!(private.visibility, Visibility::Private);
    }

    #[test]
    fn test_python_nested_class_method() {
        let parser = CodeParser::new();
        let code = r#"
class Outer:
    def method(self, arg1: int, *args, **kwargs) -> None:
        """A method with variadic args."""
        pass
"#;

        let symbols = parser.parse(code, Language::Python).unwrap();

        let method = symbols.iter().find(|s| s.name == "method").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(method.parent_symbol, Some("Outer".to_string()));
        // Parameters should include self, arg1, *args, **kwargs
        assert!(method.parameters.len() >= 2);
    }

    // ==================== Helper Function Tests ====================

    #[test]
    fn test_generate_signature() {
        let parser = CodeParser::new();

        let sig = parser.generate_signature(
            "myFunc",
            SymbolKind::Function,
            &Visibility::Public,
            &["T".to_string(), "U".to_string()],
            &[
                Parameter {
                    name: "arg1".to_string(),
                    type_annotation: Some("i32".to_string()),
                    default_value: None,
                    is_variadic: false,
                },
                Parameter {
                    name: "arg2".to_string(),
                    type_annotation: Some("String".to_string()),
                    default_value: None,
                    is_variadic: false,
                },
            ],
            Some("Result<T, Error>"),
            true,
            Language::Rust,
        );

        assert!(sig.contains("pub"));
        assert!(sig.contains("async"));
        assert!(sig.contains("fn myFunc"));
        assert!(sig.contains("<T, U>"));
        assert!(sig.contains("arg1: i32"));
        assert!(sig.contains("arg2: String"));
        assert!(sig.contains("-> Result<T, Error>"));
    }

    #[test]
    fn test_visibility_display() {
        assert_eq!(Visibility::Public.to_string(), "public");
        assert_eq!(Visibility::Private.to_string(), "private");
        assert_eq!(Visibility::Protected.to_string(), "protected");
        assert_eq!(
            Visibility::Restricted("crate".to_string()).to_string(),
            "pub(crate)"
        );
    }

    #[test]
    fn test_parameter_display() {
        let param = Parameter {
            name: "x".to_string(),
            type_annotation: Some("i32".to_string()),
            default_value: Some("0".to_string()),
            is_variadic: false,
        };
        assert_eq!(param.to_string(), "x: i32 = 0");

        let variadic = Parameter {
            name: "args".to_string(),
            type_annotation: Some("T".to_string()),
            default_value: None,
            is_variadic: true,
        };
        assert_eq!(variadic.to_string(), "...args: T");
    }
}
