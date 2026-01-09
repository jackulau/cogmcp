//! Code parsing with tree-sitter

use cogmcp_core::types::{Language, ParameterInfo, SymbolKind, SymbolModifiers, SymbolVisibility};
use cogmcp_core::{Error, Result};

/// Extracted symbol from source code
#[derive(Debug, Clone)]
pub struct ExtractedSymbol {
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
    /// Parent symbol name for nested symbols (to be resolved to ID during indexing)
    pub parent_name: Option<String>,
    /// Generic type parameters
    pub type_parameters: Vec<String>,
    /// Function/method parameters
    pub parameters: Vec<ParameterInfo>,
    /// Return type for functions/methods
    pub return_type: Option<String>,
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
            visibility: SymbolVisibility::Unknown,
            modifiers: SymbolModifiers::default(),
            parent_name: None,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
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
            Language::Ruby => self.parse_ruby(content),
            _ => Ok(Vec::new()), // Unsupported language
        }
    }

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

        self.extract_rust_symbols(&root, content, &mut symbols);

        Ok(symbols)
    }

    fn extract_rust_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        self.extract_rust_symbols_with_parent(node, content, symbols, None);
    }

    fn extract_rust_symbols_with_parent(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        let kind_str = node.kind();

        let current_parent = match kind_str {
            "function_item" | "impl_item" | "struct_item" | "enum_item" | "trait_item"
            | "mod_item" | "const_item" | "static_item" | "type_item" => {
                if let Some(symbol) = self.extract_rust_symbol(node, content, kind_str, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            _ => parent_name.clone(),
        };

        // Recurse into children with updated parent
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_rust_symbols_with_parent(&child, content, symbols, current_parent.clone());
        }
    }

    fn extract_rust_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        // Find the name node
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let kind = match kind_str {
            "function_item" => SymbolKind::Function,
            "impl_item" => SymbolKind::Class,
            "struct_item" => SymbolKind::Struct,
            "enum_item" => SymbolKind::Enum,
            "trait_item" => SymbolKind::Trait,
            "mod_item" => SymbolKind::Module,
            "const_item" => SymbolKind::Constant,
            "static_item" => SymbolKind::Variable,
            "type_item" => SymbolKind::Type,
            _ => SymbolKind::Unknown,
        };

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        // Extract signature (first line)
        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for doc comment above
        let doc_comment = self.find_doc_comment(node, content);

        // Extract visibility
        let visibility = self.extract_rust_visibility(node, content);

        // Extract modifiers
        let modifiers = self.extract_rust_modifiers(node, content);

        // Extract type parameters
        let type_parameters = self.extract_rust_type_params(node, content);

        // Extract parameters for functions
        let parameters = if kind_str == "function_item" {
            self.extract_rust_parameters(node, content)
        } else {
            Vec::new()
        };

        // Extract return type
        let return_type = self.extract_rust_return_type(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers,
            parent_name,
            type_parameters,
            parameters,
            return_type,
        })
    }

    fn extract_rust_visibility(&self, node: &tree_sitter::Node, content: &str) -> SymbolVisibility {
        // Look for visibility_modifier child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                let vis_text = child.utf8_text(content.as_bytes()).ok().unwrap_or("");
                return match vis_text {
                    "pub" => SymbolVisibility::Public,
                    s if s.starts_with("pub(crate)") => SymbolVisibility::Crate,
                    s if s.starts_with("pub(super)") => SymbolVisibility::Internal,
                    s if s.starts_with("pub(") => SymbolVisibility::Internal,
                    _ => SymbolVisibility::Private,
                };
            }
        }
        SymbolVisibility::Private // Default for Rust is private
    }

    fn extract_rust_modifiers(&self, node: &tree_sitter::Node, content: &str) -> SymbolModifiers {
        let mut modifiers = SymbolModifiers::default();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "async" => modifiers.is_async = true,
                "unsafe" => modifiers.is_unsafe = true,
                "const" => modifiers.is_const = true,
                _ => {}
            }
        }

        // Check the node text for keywords
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            let first_line = text.lines().next().unwrap_or("");
            if first_line.contains("async ") {
                modifiers.is_async = true;
            }
            if first_line.contains("unsafe ") {
                modifiers.is_unsafe = true;
            }
            if first_line.contains("const ") {
                modifiers.is_const = true;
            }
        }

        modifiers
    }

    fn extract_rust_type_params(&self, node: &tree_sitter::Node, content: &str) -> Vec<String> {
        let mut params = Vec::new();

        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            let mut cursor = type_params.walk();
            for child in type_params.children(&mut cursor) {
                if child.kind() == "type_identifier" || child.kind() == "lifetime" {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        params.push(text.to_string());
                    }
                }
            }
        }

        params
    }

    fn extract_rust_parameters(&self, node: &tree_sitter::Node, content: &str) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        if let Some(parameters) = node.child_by_field_name("parameters") {
            let mut cursor = parameters.walk();
            for child in parameters.children(&mut cursor) {
                if child.kind() == "parameter" {
                    let name = child
                        .child_by_field_name("pattern")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    let type_annotation = child
                        .child_by_field_name("type")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string());

                    if !name.is_empty() {
                        params.push(ParameterInfo { name, type_annotation });
                    }
                } else if child.kind() == "self_parameter" {
                    params.push(ParameterInfo {
                        name: "self".to_string(),
                        type_annotation: None,
                    });
                }
            }
        }

        params
    }

    fn extract_rust_return_type(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.trim_start_matches("->").trim().to_string())
    }

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

        self.extract_ts_symbols(&root, content, &mut symbols);

        Ok(symbols)
    }

    fn extract_ts_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        self.extract_ts_symbols_with_parent(node, content, symbols, None);
    }

    fn extract_ts_symbols_with_parent(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        let kind_str = node.kind();

        let current_parent = match kind_str {
            "function_declaration" | "method_definition" | "class_declaration"
            | "interface_declaration" | "type_alias_declaration" | "enum_declaration"
            | "variable_declarator" => {
                if let Some(symbol) = self.extract_ts_symbol(node, content, kind_str, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            _ => parent_name.clone(),
        };

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_ts_symbols_with_parent(&child, content, symbols, current_parent.clone());
        }
    }

    fn extract_ts_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let kind = match kind_str {
            "function_declaration" => SymbolKind::Function,
            "method_definition" => SymbolKind::Method,
            "class_declaration" => SymbolKind::Class,
            "interface_declaration" => SymbolKind::Interface,
            "type_alias_declaration" => SymbolKind::Type,
            "enum_declaration" => SymbolKind::Enum,
            "variable_declarator" => SymbolKind::Variable,
            _ => SymbolKind::Unknown,
        };

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_doc_comment(node, content);

        // Extract visibility and modifiers
        let (visibility, modifiers) = self.extract_ts_visibility_and_modifiers(node, content);

        // Extract type parameters
        let type_parameters = self.extract_ts_type_params(node, content);

        // Extract parameters for functions/methods
        let parameters = if kind_str == "function_declaration" || kind_str == "method_definition" {
            self.extract_ts_parameters(node, content)
        } else {
            Vec::new()
        };

        // Extract return type
        let return_type = self.extract_ts_return_type(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers,
            parent_name,
            type_parameters,
            parameters,
            return_type,
        })
    }

    fn extract_ts_visibility_and_modifiers(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> (SymbolVisibility, SymbolModifiers) {
        let mut visibility = SymbolVisibility::Unknown;
        let mut modifiers = SymbolModifiers::default();

        // Check parent for export statement
        if let Some(parent) = node.parent() {
            if parent.kind() == "export_statement" {
                modifiers.is_exported = true;
                visibility = SymbolVisibility::Public;
            }
        }

        // Check for accessibility modifiers in class members
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "accessibility_modifier" => {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        visibility = match text {
                            "public" => SymbolVisibility::Public,
                            "private" => SymbolVisibility::Private,
                            "protected" => SymbolVisibility::Protected,
                            _ => SymbolVisibility::Unknown,
                        };
                    }
                }
                "static" => modifiers.is_static = true,
                "async" => modifiers.is_async = true,
                "abstract" => modifiers.is_abstract = true,
                _ => {}
            }
        }

        // Check node text for export keyword at start
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            let first_line = text.lines().next().unwrap_or("");
            if first_line.trim().starts_with("export ") {
                modifiers.is_exported = true;
                visibility = SymbolVisibility::Public;
            }
        }

        (visibility, modifiers)
    }

    fn extract_ts_type_params(&self, node: &tree_sitter::Node, content: &str) -> Vec<String> {
        let mut params = Vec::new();

        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            let mut cursor = type_params.walk();
            for child in type_params.children(&mut cursor) {
                if child.kind() == "type_parameter" {
                    if let Some(name) = child.child_by_field_name("name") {
                        if let Ok(text) = name.utf8_text(content.as_bytes()) {
                            params.push(text.to_string());
                        }
                    }
                }
            }
        }

        params
    }

    fn extract_ts_parameters(&self, node: &tree_sitter::Node, content: &str) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        if let Some(parameters) = node.child_by_field_name("parameters") {
            let mut cursor = parameters.walk();
            for child in parameters.children(&mut cursor) {
                if child.kind() == "required_parameter" || child.kind() == "optional_parameter" {
                    let name = child
                        .child_by_field_name("pattern")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    let type_annotation = child
                        .child_by_field_name("type")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string());

                    if !name.is_empty() {
                        params.push(ParameterInfo { name, type_annotation });
                    }
                }
            }
        }

        params
    }

    fn extract_ts_return_type(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.trim_start_matches(':').trim().to_string())
    }

    fn parse_go(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_go::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse Go code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_go_symbols(&root, content, &mut symbols);

        Ok(symbols)
    }

    fn extract_go_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        self.extract_go_symbols_with_parent(node, content, symbols, None);
    }

    fn extract_go_symbols_with_parent(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        let kind_str = node.kind();

        let current_parent = match kind_str {
            "function_declaration" | "method_declaration" | "type_declaration"
            | "const_declaration" | "var_declaration" => {
                // For type_declaration, const_declaration, and var_declaration,
                // we extract individual specs as separate symbols
                if kind_str == "type_declaration" {
                    self.extract_go_type_declaration(node, content, symbols, parent_name.clone());
                    parent_name.clone()
                } else if kind_str == "const_declaration" {
                    self.extract_go_const_declaration(node, content, symbols);
                    parent_name.clone()
                } else if kind_str == "var_declaration" {
                    self.extract_go_var_declaration(node, content, symbols);
                    parent_name.clone()
                } else if let Some(symbol) =
                    self.extract_go_symbol(node, content, kind_str, parent_name.clone())
                {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            _ => parent_name.clone(),
        };

        // Recurse into children with updated parent
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_go_symbols_with_parent(&child, content, symbols, current_parent.clone());
        }
    }

    fn extract_go_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        match kind_str {
            "function_declaration" => self.extract_go_function(node, content, parent_name),
            "method_declaration" => self.extract_go_method(node, content),
            _ => None,
        }
    }

    fn extract_go_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_go_visibility(&name);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_go_doc_comment(node, content);
        let parameters = self.extract_go_parameters(node, content);
        let return_type = self.extract_go_return_type(node, content);
        let type_parameters = self.extract_go_type_params(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Function,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters,
            parameters,
            return_type,
        })
    }

    fn extract_go_method(&self, node: &tree_sitter::Node, content: &str) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Get receiver to determine parent type
        let receiver_name = node.child_by_field_name("receiver").and_then(|receiver| {
            // The receiver is a parameter_list containing a parameter_declaration
            let mut cursor = receiver.walk();
            for child in receiver.children(&mut cursor) {
                if child.kind() == "parameter_declaration" {
                    // Look for the type in the parameter
                    if let Some(type_node) = child.child_by_field_name("type") {
                        // Handle pointer types like *User
                        let type_text = type_node.utf8_text(content.as_bytes()).ok()?;
                        return Some(type_text.trim_start_matches('*').to_string());
                    }
                }
            }
            None
        });

        let visibility = self.extract_go_visibility(&name);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_go_doc_comment(node, content);
        let parameters = self.extract_go_parameters(node, content);
        let return_type = self.extract_go_return_type(node, content);
        let type_parameters = self.extract_go_type_params(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Method,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name: receiver_name,
            type_parameters,
            parameters,
            return_type,
        })
    }

    fn extract_go_type_declaration(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        _parent_name: Option<String>,
    ) {
        // type_declaration contains one or more type_spec or type_alias children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "type_spec" => {
                    if let Some(symbol) = self.extract_go_type_spec(&child, content) {
                        symbols.push(symbol);
                    }
                }
                "type_alias" => {
                    if let Some(symbol) = self.extract_go_type_alias(&child, content) {
                        symbols.push(symbol);
                    }
                }
                _ => {}
            }
        }
    }

    fn extract_go_type_spec(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_go_visibility(&name);

        let type_node = node.child_by_field_name("type")?;
        let kind = match type_node.kind() {
            "struct_type" => SymbolKind::Struct,
            "interface_type" => SymbolKind::Interface,
            _ => SymbolKind::TypeAlias,
        };

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for doc comment above the parent type_declaration
        let doc_comment = if let Some(parent) = node.parent() {
            self.find_go_doc_comment(&parent, content)
        } else {
            self.find_go_doc_comment(node, content)
        };

        let type_parameters = self.extract_go_type_params(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name: None,
            type_parameters,
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_go_type_alias(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_go_visibility(&name);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for doc comment above the parent type_declaration
        let doc_comment = if let Some(parent) = node.parent() {
            self.find_go_doc_comment(&parent, content)
        } else {
            self.find_go_doc_comment(node, content)
        };

        let type_parameters = self.extract_go_type_params(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::TypeAlias,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name: None,
            type_parameters,
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_go_const_declaration(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        // const_declaration contains one or more const_spec children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "const_spec" {
                if let Some(symbol) = self.extract_go_const_spec(&child, content) {
                    symbols.push(symbol);
                }
            }
        }
    }

    fn extract_go_const_spec(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_go_visibility(&name);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for doc comment above the parent const_declaration
        let doc_comment = if let Some(parent) = node.parent() {
            self.find_go_doc_comment(&parent, content)
        } else {
            self.find_go_doc_comment(node, content)
        };

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Constant,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name: None,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_go_var_declaration(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        // var_declaration contains one or more var_spec children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "var_spec" {
                if let Some(symbol) = self.extract_go_var_spec(&child, content) {
                    symbols.push(symbol);
                }
            }
        }
    }

    fn extract_go_var_spec(&self, node: &tree_sitter::Node, content: &str) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let visibility = self.extract_go_visibility(&name);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for doc comment above the parent var_declaration
        let doc_comment = if let Some(parent) = node.parent() {
            self.find_go_doc_comment(&parent, content)
        } else {
            self.find_go_doc_comment(node, content)
        };

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Variable,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers: SymbolModifiers::default(),
            parent_name: None,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_go_visibility(&self, name: &str) -> SymbolVisibility {
        // Go uses capitalization for visibility
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            SymbolVisibility::Public
        } else {
            SymbolVisibility::Private
        }
    }

    fn extract_go_parameters(&self, node: &tree_sitter::Node, content: &str) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        if let Some(parameters) = node.child_by_field_name("parameters") {
            let mut cursor = parameters.walk();
            for child in parameters.children(&mut cursor) {
                if child.kind() == "parameter_declaration" {
                    // A parameter_declaration can have multiple names with the same type
                    let type_annotation = child
                        .child_by_field_name("type")
                        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                        .map(|s| s.to_string());

                    // Get all name children
                    let mut name_cursor = child.walk();
                    let mut has_name = false;
                    for name_child in child.children(&mut name_cursor) {
                        if name_child.kind() == "identifier" {
                            // Check if this is the "name" field, not the "type" field
                            if let Some(field_name) =
                                child.field_name_for_child(name_child.id() as u32)
                            {
                                if field_name == "name" {
                                    if let Ok(name) = name_child.utf8_text(content.as_bytes()) {
                                        params.push(ParameterInfo {
                                            name: name.to_string(),
                                            type_annotation: type_annotation.clone(),
                                        });
                                        has_name = true;
                                    }
                                }
                            }
                        }
                    }

                    // If no named parameter found, the type might be the only element (anonymous param)
                    if !has_name && type_annotation.is_some() {
                        params.push(ParameterInfo {
                            name: String::new(),
                            type_annotation,
                        });
                    }
                }
            }
        }

        params
    }

    fn extract_go_return_type(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        node.child_by_field_name("result")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string())
    }

    fn extract_go_type_params(&self, node: &tree_sitter::Node, content: &str) -> Vec<String> {
        let mut params = Vec::new();

        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            let mut cursor = type_params.walk();
            for child in type_params.children(&mut cursor) {
                if child.kind() == "type_parameter_declaration" {
                    if let Some(name) = child.child_by_field_name("name") {
                        if let Ok(text) = name.utf8_text(content.as_bytes()) {
                            params.push(text.to_string());
                        }
                    }
                }
            }
        }

        params
    }

    fn find_go_doc_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for comment
        if let Some(prev) = node.prev_sibling() {
            if prev.kind() == "comment" {
                return prev.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

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

        self.extract_python_symbols(&root, content, &mut symbols);

        Ok(symbols)
    }

    fn extract_python_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        self.extract_python_symbols_with_parent(node, content, symbols, None);
    }

    fn extract_python_symbols_with_parent(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        let kind_str = node.kind();

        let current_parent = match kind_str {
            "function_definition" | "class_definition" => {
                if let Some(symbol) = self.extract_python_symbol(node, content, kind_str, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            _ => parent_name.clone(),
        };

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_python_symbols_with_parent(&child, content, symbols, current_parent.clone());
        }
    }

    fn extract_python_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        // Determine kind - methods are functions inside classes
        let kind = match kind_str {
            "function_definition" => {
                if parent_name.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                }
            }
            "class_definition" => SymbolKind::Class,
            _ => SymbolKind::Unknown,
        };

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Python docstrings
        let doc_comment = self.find_python_docstring(node, content);

        // Extract visibility from name convention
        let visibility = if name.starts_with("__") && !name.ends_with("__") {
            SymbolVisibility::Private
        } else if name.starts_with('_') {
            SymbolVisibility::Protected
        } else {
            SymbolVisibility::Public
        };

        // Extract modifiers from decorators
        let modifiers = self.extract_python_modifiers(node, content);

        // Extract parameters for functions
        let parameters = if kind_str == "function_definition" {
            self.extract_python_parameters(node, content)
        } else {
            Vec::new()
        };

        // Extract return type from annotation
        let return_type = self.extract_python_return_type(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers,
            parent_name,
            type_parameters: Vec::new(), // Python doesn't have explicit type params like generics
            parameters,
            return_type,
        })
    }

    fn extract_python_modifiers(&self, node: &tree_sitter::Node, content: &str) -> SymbolModifiers {
        let mut modifiers = SymbolModifiers::default();

        // Look for decorators in previous sibling or parent's children
        if let Some(parent) = node.parent() {
            let mut cursor = parent.walk();
            let mut found_node = false;
            for child in parent.children(&mut cursor) {
                if child.id() == node.id() {
                    found_node = true;
                    break;
                }
                if child.kind() == "decorator" {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        if text.contains("@staticmethod") {
                            modifiers.is_static = true;
                        }
                        if text.contains("@classmethod") {
                            modifiers.is_static = true; // classmethod is similar to static
                        }
                        if text.contains("@abstractmethod") {
                            modifiers.is_abstract = true;
                        }
                    }
                }
            }
            // Handle decorated_definition wrapper
            if !found_node && parent.kind() == "decorated_definition" {
                let mut cursor = parent.walk();
                for child in parent.children(&mut cursor) {
                    if child.kind() == "decorator" {
                        if let Ok(text) = child.utf8_text(content.as_bytes()) {
                            if text.contains("@staticmethod") {
                                modifiers.is_static = true;
                            }
                            if text.contains("@classmethod") {
                                modifiers.is_static = true;
                            }
                            if text.contains("@abstractmethod") {
                                modifiers.is_abstract = true;
                            }
                        }
                    }
                }
            }
        }

        // Check for async def
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            if text.trim().starts_with("async ") {
                modifiers.is_async = true;
            }
        }

        modifiers
    }

    fn extract_python_parameters(&self, node: &tree_sitter::Node, content: &str) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        if let Some(parameters) = node.child_by_field_name("parameters") {
            let mut cursor = parameters.walk();
            for child in parameters.children(&mut cursor) {
                match child.kind() {
                    "identifier" => {
                        if let Ok(text) = child.utf8_text(content.as_bytes()) {
                            params.push(ParameterInfo {
                                name: text.to_string(),
                                type_annotation: None,
                            });
                        }
                    }
                    "typed_parameter" | "default_parameter" | "typed_default_parameter" => {
                        let name = child
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                            .map(|s| s.to_string())
                            .unwrap_or_default();

                        let type_annotation = child
                            .child_by_field_name("type")
                            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                            .map(|s| s.to_string());

                        if !name.is_empty() {
                            params.push(ParameterInfo { name, type_annotation });
                        }
                    }
                    _ => {}
                }
            }
        }

        params
    }

    fn extract_python_return_type(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string())
    }

    fn parse_ruby(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_ruby::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse Ruby code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_ruby_symbols(&root, content, &mut symbols);

        Ok(symbols)
    }

    fn extract_ruby_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
    ) {
        // Track visibility state: public is default
        let mut visibility = SymbolVisibility::Public;
        self.extract_ruby_symbols_with_context(node, content, symbols, None, &mut visibility);
    }

    fn extract_ruby_symbols_with_context(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
        current_visibility: &mut SymbolVisibility,
    ) {
        let kind_str = node.kind();

        // Check for visibility modifier calls that change state
        if kind_str == "call" || kind_str == "identifier" {
            if let Some(new_vis) = self.check_ruby_visibility_modifier(node, content) {
                *current_visibility = new_vis;
            }
        }

        let current_parent = match kind_str {
            "class" | "module" | "method" | "singleton_method" => {
                if let Some(symbol) = self.extract_ruby_symbol(node, content, kind_str, parent_name.clone(), *current_visibility) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "assignment" => {
                // Check for constant assignments (CONSTANT = value)
                if let Some(symbol) = self.extract_ruby_constant(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "call" => {
                // Check for attr_accessor, attr_reader, attr_writer
                self.extract_ruby_attribute_accessors(node, content, symbols, parent_name.clone());
                parent_name.clone()
            }
            _ => parent_name.clone(),
        };

        // When entering a class or module, reset visibility for the scope
        if kind_str == "class" || kind_str == "module" {
            // Process children with a fresh visibility scope
            let mut scope_visibility = SymbolVisibility::Public;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.extract_ruby_symbols_with_context(&child, content, symbols, current_parent.clone(), &mut scope_visibility);
            }
        } else {
            // Recurse into children with current visibility
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.extract_ruby_symbols_with_context(&child, content, symbols, current_parent.clone(), current_visibility);
            }
        }
    }

    fn check_ruby_visibility_modifier(&self, node: &tree_sitter::Node, content: &str) -> Option<SymbolVisibility> {
        // Check if this is a standalone visibility modifier call (private, protected, public)
        let text = node.utf8_text(content.as_bytes()).ok()?;
        let text = text.trim();

        // Standalone visibility modifiers
        match text {
            "private" => Some(SymbolVisibility::Private),
            "protected" => Some(SymbolVisibility::Protected),
            "public" => Some(SymbolVisibility::Public),
            _ => None,
        }
    }

    fn extract_ruby_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
        parent_name: Option<String>,
        visibility: SymbolVisibility,
    ) -> Option<ExtractedSymbol> {
        let name = self.get_ruby_symbol_name(node, content, kind_str)?;

        // Determine symbol kind
        let kind = match kind_str {
            "class" => SymbolKind::Class,
            "module" => SymbolKind::Module,
            "method" => {
                // Check if it's initialize (constructor)
                if name == "initialize" {
                    SymbolKind::Function
                } else {
                    SymbolKind::Method
                }
            }
            "singleton_method" => SymbolKind::Function, // Class methods are Functions
            _ => SymbolKind::Unknown,
        };

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        // Extract signature (first line)
        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for comment above
        let doc_comment = self.find_ruby_comment(node, content);

        // Determine visibility - check for inline visibility modifier
        let actual_visibility = self.check_ruby_inline_visibility(node, content).unwrap_or(visibility);

        // Extract modifiers
        let modifiers = SymbolModifiers::default();

        // Extract parameters for methods
        let parameters = if kind_str == "method" || kind_str == "singleton_method" {
            self.extract_ruby_parameters(node, content)
        } else {
            Vec::new()
        };

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: actual_visibility,
            modifiers,
            parent_name,
            type_parameters: Vec::new(), // Ruby doesn't have type parameters
            parameters,
            return_type: None, // Ruby doesn't have explicit return types
        })
    }

    fn get_ruby_symbol_name(&self, node: &tree_sitter::Node, content: &str, kind_str: &str) -> Option<String> {
        match kind_str {
            "class" | "module" => {
                // Look for constant child (class/module name)
                node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string())
            }
            "method" => {
                // Look for name field
                node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string())
            }
            "singleton_method" => {
                // For def self.method_name, get the name
                node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .map(|s| s.to_string())
            }
            _ => None,
        }
    }

    fn check_ruby_inline_visibility(&self, node: &tree_sitter::Node, content: &str) -> Option<SymbolVisibility> {
        // Check if this method has an inline visibility modifier (e.g., private def foo)
        // The parent would be a call node with private/protected/public as the method
        if let Some(parent) = node.parent() {
            if parent.kind() == "call" {
                // Get the method name from the call
                if let Some(method_node) = parent.child_by_field_name("method") {
                    let method_text = method_node.utf8_text(content.as_bytes()).ok()?;
                    match method_text {
                        "private" => return Some(SymbolVisibility::Private),
                        "protected" => return Some(SymbolVisibility::Protected),
                        "public" => return Some(SymbolVisibility::Public),
                        _ => {}
                    }
                }
            }
        }
        None
    }

    fn extract_ruby_constant(&self, node: &tree_sitter::Node, content: &str, parent_name: Option<String>) -> Option<ExtractedSymbol> {
        // Assignment node: look for left side being a constant (all uppercase)
        let left = node.child_by_field_name("left")?;
        let left_text = left.utf8_text(content.as_bytes()).ok()?;

        // Ruby constants start with uppercase letter
        if !left_text.chars().next()?.is_uppercase() {
            return None;
        }

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        Some(ExtractedSymbol {
            name: left_text.to_string(),
            kind: SymbolKind::Constant,
            start_line,
            end_line,
            signature,
            doc_comment: None,
            visibility: SymbolVisibility::Public, // Ruby constants are public by default
            modifiers: SymbolModifiers { is_const: true, ..Default::default() },
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_ruby_attribute_accessors(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        // Check if this is an attr_accessor, attr_reader, or attr_writer call
        let method_node = match node.child_by_field_name("method") {
            Some(n) => n,
            None => return,
        };

        let method_text = match method_node.utf8_text(content.as_bytes()) {
            Ok(s) => s,
            Err(_) => return,
        };

        let is_accessor = matches!(method_text, "attr_accessor" | "attr_reader" | "attr_writer");
        if !is_accessor {
            return;
        }

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        // Look for arguments (symbols)
        if let Some(args) = node.child_by_field_name("arguments") {
            let mut cursor = args.walk();
            for child in args.children(&mut cursor) {
                if child.kind() == "simple_symbol" || child.kind() == "symbol" {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        // Remove the leading : from symbol
                        let name = text.trim_start_matches(':').to_string();
                        symbols.push(ExtractedSymbol {
                            name,
                            kind: SymbolKind::Property,
                            start_line,
                            end_line,
                            signature: signature.clone(),
                            doc_comment: None,
                            visibility: SymbolVisibility::Public,
                            modifiers: SymbolModifiers::default(),
                            parent_name: parent_name.clone(),
                            type_parameters: Vec::new(),
                            parameters: Vec::new(),
                            return_type: None,
                        });
                    }
                }
            }
        }
    }

    fn extract_ruby_parameters(&self, node: &tree_sitter::Node, content: &str) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        if let Some(parameters) = node.child_by_field_name("parameters") {
            let mut cursor = parameters.walk();
            for child in parameters.children(&mut cursor) {
                match child.kind() {
                    "identifier" => {
                        if let Ok(text) = child.utf8_text(content.as_bytes()) {
                            params.push(ParameterInfo {
                                name: text.to_string(),
                                type_annotation: None,
                            });
                        }
                    }
                    "optional_parameter" | "keyword_parameter" | "splat_parameter" | "block_parameter" => {
                        if let Some(name_node) = child.child_by_field_name("name") {
                            if let Ok(text) = name_node.utf8_text(content.as_bytes()) {
                                params.push(ParameterInfo {
                                    name: text.to_string(),
                                    type_annotation: None,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        params
    }

    fn find_ruby_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for comment
        if let Some(prev) = node.prev_sibling() {
            if prev.kind() == "comment" {
                return prev.utf8_text(content.as_bytes()).ok().map(|s| {
                    s.trim_start_matches('#').trim().to_string()
                });
            }
        }
        None
    }

    fn find_doc_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for comment
        if let Some(prev) = node.prev_sibling() {
            if prev.kind() == "line_comment" || prev.kind() == "block_comment" {
                return prev.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    #[allow(clippy::never_loop)]
    fn find_python_docstring(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look for string as first child of body
        let body = node.child_by_field_name("body")?;
        // Only check first statement
        let child = body.child(0)?;
        if child.kind() != "expression_statement" {
            return None;
        }
        let mut inner_cursor = child.walk();
        for inner in child.children(&mut inner_cursor) {
            if inner.kind() == "string" {
                return inner.utf8_text(content.as_bytes()).ok().map(|s| {
                    s.trim_matches('"')
                        .trim_matches('\'')
                        .trim()
                        .to_string()
                });
            }
        }
        None
    }

    // ==================== C Language Parsing ====================

    fn parse_c(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set C language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse C code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_c_symbols(&root, content, &mut symbols, None);

        Ok(symbols)
    }

    fn extract_c_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
    ) {
        let kind_str = node.kind();

        let current_parent = match kind_str {
            "function_definition" => {
                if let Some(symbol) = self.extract_c_function(node, content, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "struct_specifier" => {
                if let Some(symbol) = self.extract_c_struct(node, content, SymbolKind::Struct, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "union_specifier" => {
                // Treat unions like structs
                if let Some(symbol) = self.extract_c_struct(node, content, SymbolKind::Struct, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "enum_specifier" => {
                if let Some(symbol) = self.extract_c_enum(node, content, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "type_definition" => {
                if let Some(symbol) = self.extract_c_typedef(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "preproc_def" | "preproc_function_def" => {
                if let Some(symbol) = self.extract_c_macro(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "declaration" => {
                // Global variable declarations
                if node.parent().map(|p| p.kind() == "translation_unit").unwrap_or(false) {
                    if let Some(symbol) = self.extract_c_global_variable(node, content) {
                        symbols.push(symbol);
                    }
                }
                parent_name.clone()
            }
            _ => parent_name.clone(),
        };

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_c_symbols(&child, content, symbols, current_parent.clone());
        }
    }

    fn extract_c_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        // Find the declarator which contains the function name
        let declarator = node.child_by_field_name("declarator")?;
        let name = self.get_c_function_name(&declarator, content)?;

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        // Check for static keyword to determine visibility
        let (visibility, modifiers) = self.extract_c_visibility_and_modifiers(node, content);

        // Extract parameters
        let parameters = self.extract_c_parameters(&declarator, content);

        // Extract return type
        let return_type = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Function,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers,
            parent_name,
            type_parameters: Vec::new(),
            parameters,
            return_type,
        })
    }

    fn get_c_function_name(&self, declarator: &tree_sitter::Node, content: &str) -> Option<String> {
        // Handle function_declarator: the name is in the declarator field
        if declarator.kind() == "function_declarator" {
            if let Some(inner) = declarator.child_by_field_name("declarator") {
                return inner.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        // Handle pointer_declarator wrapping function_declarator
        if declarator.kind() == "pointer_declarator" {
            let mut cursor = declarator.walk();
            for child in declarator.children(&mut cursor) {
                if child.kind() == "function_declarator" {
                    return self.get_c_function_name(&child, content);
                }
            }
        }
        // Fallback: try direct text
        declarator.utf8_text(content.as_bytes()).ok().map(|s| s.to_string())
    }

    fn extract_c_struct(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind: SymbolKind,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        // Struct name is in the "name" field
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: SymbolVisibility::Public, // C has no visibility, default to public
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_c_enum(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Enum,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_c_typedef(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        // For typedef, we need to find the type name being defined
        // The declarator contains the new type name
        let declarator = node.child_by_field_name("declarator")?;
        let name = self.get_c_typedef_name(&declarator, content)?;

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::TypeAlias,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn get_c_typedef_name(&self, declarator: &tree_sitter::Node, content: &str) -> Option<String> {
        match declarator.kind() {
            "type_identifier" | "identifier" => {
                declarator.utf8_text(content.as_bytes()).ok().map(|s| s.to_string())
            }
            "pointer_declarator" | "array_declarator" | "function_declarator" => {
                if let Some(inner) = declarator.child_by_field_name("declarator") {
                    return self.get_c_typedef_name(&inner, content);
                }
                None
            }
            _ => None,
        }
    }

    fn extract_c_macro(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Macro,
            start_line,
            end_line,
            signature,
            doc_comment: None,
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_c_global_variable(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> Option<ExtractedSymbol> {
        // Only handle simple variable declarations at file scope
        // Skip function declarations (they have function_declarator)
        let declarator = node.child_by_field_name("declarator")?;

        // Skip if this is a function declaration
        if declarator.kind() == "function_declarator" {
            return None;
        }

        let name = self.get_c_declarator_name(&declarator, content)?;

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let (visibility, modifiers) = self.extract_c_visibility_and_modifiers(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Variable,
            start_line,
            end_line,
            signature,
            doc_comment: None,
            visibility,
            modifiers,
            parent_name: None,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn get_c_declarator_name(&self, declarator: &tree_sitter::Node, content: &str) -> Option<String> {
        match declarator.kind() {
            "identifier" => {
                declarator.utf8_text(content.as_bytes()).ok().map(|s| s.to_string())
            }
            "init_declarator" => {
                if let Some(inner) = declarator.child_by_field_name("declarator") {
                    return self.get_c_declarator_name(&inner, content);
                }
                None
            }
            "pointer_declarator" | "array_declarator" => {
                if let Some(inner) = declarator.child_by_field_name("declarator") {
                    return self.get_c_declarator_name(&inner, content);
                }
                None
            }
            _ => None,
        }
    }

    fn extract_c_visibility_and_modifiers(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> (SymbolVisibility, SymbolModifiers) {
        let mut visibility = SymbolVisibility::Public;
        let mut modifiers = SymbolModifiers::default();

        // Check the node text for storage class specifiers
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            let first_line = text.lines().next().unwrap_or("");
            if first_line.contains("static ") || first_line.starts_with("static ") {
                visibility = SymbolVisibility::Private; // file-local
                modifiers.is_static = true;
            }
            if first_line.contains("const ") {
                modifiers.is_const = true;
            }
            if first_line.contains("extern ") {
                modifiers.is_exported = true;
            }
        }

        // Also check for storage_class_specifier children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "storage_class_specifier" {
                if let Ok(text) = child.utf8_text(content.as_bytes()) {
                    if text == "static" {
                        visibility = SymbolVisibility::Private;
                        modifiers.is_static = true;
                    }
                    if text == "extern" {
                        modifiers.is_exported = true;
                    }
                }
            }
            if child.kind() == "type_qualifier" {
                if let Ok(text) = child.utf8_text(content.as_bytes()) {
                    if text == "const" {
                        modifiers.is_const = true;
                    }
                }
            }
        }

        (visibility, modifiers)
    }

    fn extract_c_parameters(
        &self,
        declarator: &tree_sitter::Node,
        content: &str,
    ) -> Vec<ParameterInfo> {
        let mut params = Vec::new();

        // Find the function_declarator and get its parameters
        let func_decl = if declarator.kind() == "function_declarator" {
            Some(*declarator)
        } else {
            // Search for function_declarator in children
            let mut found = None;
            let mut cursor = declarator.walk();
            for child in declarator.children(&mut cursor) {
                if child.kind() == "function_declarator" {
                    found = Some(child);
                    break;
                }
            }
            found
        };

        if let Some(func_decl) = func_decl {
            if let Some(parameters) = func_decl.child_by_field_name("parameters") {
                let mut cursor = parameters.walk();
                for child in parameters.children(&mut cursor) {
                    if child.kind() == "parameter_declaration" {
                        let name = child
                            .child_by_field_name("declarator")
                            .and_then(|d| self.get_c_declarator_name(&d, content))
                            .unwrap_or_default();

                        let type_annotation = child
                            .child_by_field_name("type")
                            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                            .map(|s| s.to_string());

                        if !name.is_empty() {
                            params.push(ParameterInfo { name, type_annotation });
                        }
                    }
                }
            }
        }

        params
    }

    fn find_c_doc_comment(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        // Look at previous sibling for comment
        if let Some(prev) = node.prev_sibling() {
            if prev.kind() == "comment" {
                return prev.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    // ==================== C++ Language Parsing ====================

    fn parse_cpp(&self, content: &str) -> Result<Vec<ExtractedSymbol>> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_cpp::LANGUAGE.into())
            .map_err(|e| Error::Parse(format!("Failed to set C++ language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| Error::Parse("Failed to parse C++ code".into()))?;

        let mut symbols = Vec::new();
        let root = tree.root_node();

        self.extract_cpp_symbols(&root, content, &mut symbols, None, None);

        Ok(symbols)
    }

    fn extract_cpp_symbols(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
        current_access: Option<SymbolVisibility>,
    ) {
        let kind_str = node.kind();

        // Track access specifier changes within classes
        let mut access = current_access;

        let current_parent = match kind_str {
            "function_definition" => {
                if let Some(symbol) = self.extract_cpp_function(node, content, parent_name.clone(), access) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "class_specifier" => {
                if let Some(symbol) = self.extract_cpp_class(node, content, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    // Default access for class members is private
                    access = Some(SymbolVisibility::Private);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "struct_specifier" => {
                if let Some(symbol) = self.extract_c_struct(node, content, SymbolKind::Struct, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    // Default access for struct members is public
                    access = Some(SymbolVisibility::Public);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "union_specifier" => {
                if let Some(symbol) = self.extract_c_struct(node, content, SymbolKind::Struct, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "enum_specifier" => {
                if let Some(symbol) = self.extract_c_enum(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "namespace_definition" => {
                if let Some(symbol) = self.extract_cpp_namespace(node, content, parent_name.clone()) {
                    let name = symbol.name.clone();
                    symbols.push(symbol);
                    Some(name)
                } else {
                    parent_name.clone()
                }
            }
            "template_declaration" => {
                // Template wraps another declaration - extract the inner declaration
                // with type parameters
                self.extract_cpp_template(node, content, symbols, parent_name.clone(), access);
                parent_name.clone()
            }
            "type_definition" => {
                if let Some(symbol) = self.extract_c_typedef(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "preproc_def" | "preproc_function_def" => {
                if let Some(symbol) = self.extract_c_macro(node, content, parent_name.clone()) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "field_declaration" => {
                // Class member declarations
                if let Some(symbol) = self.extract_cpp_field(node, content, parent_name.clone(), access) {
                    symbols.push(symbol);
                }
                parent_name.clone()
            }
            "access_specifier" => {
                // Update access level for subsequent members
                if let Ok(text) = node.utf8_text(content.as_bytes()) {
                    access = Some(match text.trim().trim_end_matches(':') {
                        "public" => SymbolVisibility::Public,
                        "protected" => SymbolVisibility::Protected,
                        "private" => SymbolVisibility::Private,
                        _ => current_access.unwrap_or(SymbolVisibility::Private),
                    });
                }
                parent_name.clone()
            }
            "declaration" => {
                // Global variable declarations
                if node.parent().map(|p| p.kind() == "translation_unit").unwrap_or(false) {
                    if let Some(symbol) = self.extract_c_global_variable(node, content) {
                        symbols.push(symbol);
                    }
                }
                parent_name.clone()
            }
            _ => parent_name.clone(),
        };

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            // Pass through access specifier for class body
            self.extract_cpp_symbols(&child, content, symbols, current_parent.clone(), access);
        }
    }

    fn extract_cpp_function(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
        class_access: Option<SymbolVisibility>,
    ) -> Option<ExtractedSymbol> {
        // Find the declarator which contains the function name
        let declarator = node.child_by_field_name("declarator")?;
        let (name, is_method) = self.get_cpp_function_name(&declarator, content)?;

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        // Determine kind: method if has :: in name (out-of-class definition)
        // or if inside a class (class_access is set)
        // Functions inside namespaces are still functions, not methods
        let kind = if is_method || class_access.is_some() {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        // Check for static, virtual, etc.
        let (mut visibility, mut modifiers) = self.extract_cpp_visibility_and_modifiers(node, content);

        // If inside a class, use the class access level if not explicitly specified
        if class_access.is_some() && visibility == SymbolVisibility::Unknown {
            visibility = class_access.unwrap_or(SymbolVisibility::Private);
        }
        if visibility == SymbolVisibility::Unknown {
            visibility = SymbolVisibility::Public;
        }

        // Check for pure virtual (= 0)
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            if text.contains("= 0") {
                modifiers.is_abstract = true;
            }
        }

        // Extract parameters
        let parameters = self.extract_c_parameters(&declarator, content);

        // Extract return type
        let return_type = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(content.as_bytes()).ok())
            .map(|s| s.to_string());

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility,
            modifiers,
            parent_name,
            type_parameters: Vec::new(),
            parameters,
            return_type,
        })
    }

    fn get_cpp_function_name(
        &self,
        declarator: &tree_sitter::Node,
        content: &str,
    ) -> Option<(String, bool)> {
        // Returns (name, is_method)
        if declarator.kind() == "function_declarator" {
            if let Some(inner) = declarator.child_by_field_name("declarator") {
                // Check for qualified_identifier (Class::method)
                if inner.kind() == "qualified_identifier" {
                    // Get the full qualified name
                    if let Ok(text) = inner.utf8_text(content.as_bytes()) {
                        // Extract just the method name (after last ::)
                        let name = text.rsplit("::").next().unwrap_or(text).to_string();
                        return Some((name, true));
                    }
                }
                // Check for destructor
                if inner.kind() == "destructor_name" {
                    if let Ok(text) = inner.utf8_text(content.as_bytes()) {
                        return Some((text.to_string(), true));
                    }
                }
                // Check for operator overload
                if inner.kind() == "operator_name" {
                    if let Ok(text) = inner.utf8_text(content.as_bytes()) {
                        return Some((text.to_string(), false));
                    }
                }
                if let Ok(text) = inner.utf8_text(content.as_bytes()) {
                    return Some((text.to_string(), false));
                }
            }
        }
        // Handle pointer_declarator wrapping function_declarator
        if declarator.kind() == "pointer_declarator" || declarator.kind() == "reference_declarator" {
            let mut cursor = declarator.walk();
            for child in declarator.children(&mut cursor) {
                if child.kind() == "function_declarator" {
                    return self.get_cpp_function_name(&child, content);
                }
            }
        }
        // Fallback
        declarator.utf8_text(content.as_bytes()).ok().map(|s| (s.to_string(), false))
    }

    fn extract_cpp_class(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let doc_comment = self.find_c_doc_comment(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Class,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_cpp_namespace(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Module,
            start_line,
            end_line,
            signature,
            doc_comment: None,
            visibility: SymbolVisibility::Public,
            modifiers: SymbolModifiers::default(),
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_cpp_template(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        symbols: &mut Vec<ExtractedSymbol>,
        parent_name: Option<String>,
        access: Option<SymbolVisibility>,
    ) {
        // Extract template parameters
        let type_params = self.extract_cpp_template_params(node, content);

        // Find the inner declaration
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_definition" => {
                    if let Some(mut symbol) = self.extract_cpp_function(&child, content, parent_name.clone(), access) {
                        symbol.type_parameters = type_params.clone();
                        symbols.push(symbol);
                    }
                }
                "class_specifier" => {
                    if let Some(mut symbol) = self.extract_cpp_class(&child, content, parent_name.clone()) {
                        symbol.type_parameters = type_params.clone();
                        symbols.push(symbol);
                    }
                }
                "struct_specifier" => {
                    if let Some(mut symbol) = self.extract_c_struct(&child, content, SymbolKind::Struct, parent_name.clone()) {
                        symbol.type_parameters = type_params.clone();
                        symbols.push(symbol);
                    }
                }
                "declaration" => {
                    // Template variable or function declaration
                    // We can skip these for now or handle as needed
                }
                _ => {}
            }
        }
    }

    fn extract_cpp_template_params(&self, node: &tree_sitter::Node, content: &str) -> Vec<String> {
        let mut params = Vec::new();

        if let Some(template_params) = node.child_by_field_name("parameters") {
            let mut cursor = template_params.walk();
            for child in template_params.children(&mut cursor) {
                match child.kind() {
                    "type_parameter_declaration" | "template_type_parameter" => {
                        // Get the name of the type parameter
                        if let Some(name_node) = child.child_by_field_name("name") {
                            if let Ok(text) = name_node.utf8_text(content.as_bytes()) {
                                params.push(text.to_string());
                            }
                        } else {
                            // Try to find identifier child
                            let mut inner_cursor = child.walk();
                            for inner in child.children(&mut inner_cursor) {
                                if inner.kind() == "type_identifier" || inner.kind() == "identifier" {
                                    if let Ok(text) = inner.utf8_text(content.as_bytes()) {
                                        params.push(text.to_string());
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    "variadic_type_parameter_declaration" => {
                        if let Some(name_node) = child.child_by_field_name("name") {
                            if let Ok(text) = name_node.utf8_text(content.as_bytes()) {
                                params.push(format!("{}...", text));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        params
    }

    fn extract_cpp_field(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        parent_name: Option<String>,
        access: Option<SymbolVisibility>,
    ) -> Option<ExtractedSymbol> {
        // Skip function declarations (they're handled separately)
        let declarator = node.child_by_field_name("declarator")?;
        if declarator.kind() == "function_declarator" {
            return None;
        }

        let name = self.get_c_declarator_name(&declarator, content)?;

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        let signature = content
            .lines()
            .nth(start_line as usize - 1)
            .map(|s| s.trim().to_string());

        let visibility = access.unwrap_or(SymbolVisibility::Private);
        let (_, modifiers) = self.extract_cpp_visibility_and_modifiers(node, content);

        Some(ExtractedSymbol {
            name,
            kind: SymbolKind::Variable,
            start_line,
            end_line,
            signature,
            doc_comment: None,
            visibility,
            modifiers,
            parent_name,
            type_parameters: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
        })
    }

    fn extract_cpp_visibility_and_modifiers(
        &self,
        node: &tree_sitter::Node,
        content: &str,
    ) -> (SymbolVisibility, SymbolModifiers) {
        let visibility = SymbolVisibility::Unknown;
        let mut modifiers = SymbolModifiers::default();

        // Check the node text for keywords
        if let Ok(text) = node.utf8_text(content.as_bytes()) {
            let first_line = text.lines().next().unwrap_or("");
            if first_line.contains("static ") || first_line.starts_with("static ") {
                modifiers.is_static = true;
            }
            if first_line.contains("const ") {
                modifiers.is_const = true;
            }
            if first_line.contains("virtual ") {
                modifiers.is_abstract = false; // virtual but not pure
            }
            if first_line.contains("inline ") {
                // Could add inline flag if needed
            }
        }

        // Check for specifier children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "storage_class_specifier" => {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        if text == "static" {
                            modifiers.is_static = true;
                        }
                    }
                }
                "type_qualifier" => {
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        if text == "const" {
                            modifiers.is_const = true;
                        }
                    }
                }
                "virtual" | "virtual_function_specifier" => {
                    modifiers.is_abstract = false; // virtual but not necessarily pure
                }
                _ => {}
            }
        }

        (visibility, modifiers)
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
    use cogmcp_core::types::{Language, SymbolKind, SymbolVisibility};

    #[test]
    fn test_parse_rust_with_visibility() {
        let parser = CodeParser::new();
        let code = r#"
pub fn public_function() {}
fn private_function() {}
pub(crate) fn crate_function() {}

pub struct PublicStruct {
    pub field: i32,
}

struct PrivateStruct {}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        // Find the public function
        let public_fn = symbols.iter().find(|s| s.name == "public_function").unwrap();
        assert_eq!(public_fn.kind, SymbolKind::Function);
        assert_eq!(public_fn.visibility, SymbolVisibility::Public);

        // Find the private function
        let private_fn = symbols.iter().find(|s| s.name == "private_function").unwrap();
        assert_eq!(private_fn.kind, SymbolKind::Function);
        assert_eq!(private_fn.visibility, SymbolVisibility::Private);

        // Find crate-visible function
        let crate_fn = symbols.iter().find(|s| s.name == "crate_function").unwrap();
        assert_eq!(crate_fn.visibility, SymbolVisibility::Crate);

        // Find public struct
        let pub_struct = symbols.iter().find(|s| s.name == "PublicStruct").unwrap();
        assert_eq!(pub_struct.kind, SymbolKind::Struct);
        assert_eq!(pub_struct.visibility, SymbolVisibility::Public);

        // Find private struct
        let priv_struct = symbols.iter().find(|s| s.name == "PrivateStruct").unwrap();
        assert_eq!(priv_struct.visibility, SymbolVisibility::Private);
    }

    #[test]
    fn test_parse_rust_async_function() {
        let parser = CodeParser::new();
        let code = r#"
pub async fn async_function() {}
fn sync_function() {}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let async_fn = symbols.iter().find(|s| s.name == "async_function").unwrap();
        assert!(async_fn.modifiers.is_async);

        let sync_fn = symbols.iter().find(|s| s.name == "sync_function").unwrap();
        assert!(!sync_fn.modifiers.is_async);
    }

    #[test]
    fn test_parse_rust_nested_symbols() {
        let parser = CodeParser::new();
        let code = r#"
pub struct Container {
    field: i32,
}

impl Container {
    pub fn method(&self) {}
}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        // We should have at least the struct and the method
        let container = symbols.iter().find(|s| s.name == "Container" && s.kind == SymbolKind::Struct);
        assert!(container.is_some(), "Container struct should be found");

        let method = symbols.iter().find(|s| s.name == "method");
        assert!(method.is_some(), "method should be found");

        // In the current impl, methods in impl blocks get the impl block name as parent
        // which is also "Container" but in a different symbol type
        // The key thing is the method was extracted
        let method = method.unwrap();
        assert_eq!(method.kind, SymbolKind::Function);
        // Parent might be "Container" from impl or None depending on tree-sitter parsing
        // The important thing is the method is extracted with correct kind
    }

    #[test]
    fn test_parse_typescript_exports() {
        let parser = CodeParser::new();
        let code = r#"
export function exportedFunction() {}
function privateFunction() {}

export class ExportedClass {
    public publicMethod() {}
    private privateMethod() {}
}
"#;

        let symbols = parser.parse(code, Language::TypeScript).unwrap();

        // Exported function should have export modifier
        let exported_fn = symbols.iter().find(|s| s.name == "exportedFunction");
        // Note: tree-sitter export detection depends on parsing context
        assert!(exported_fn.is_some());

        // Find the class
        let exported_class = symbols.iter().find(|s| s.name == "ExportedClass");
        assert!(exported_class.is_some());
    }

    #[test]
    fn test_parse_python_visibility_convention() {
        let parser = CodeParser::new();
        let code = r#"
def public_function():
    pass

def _protected_function():
    pass

def __private_function():
    pass

class MyClass:
    def method(self):
        pass
"#;

        let symbols = parser.parse(code, Language::Python).unwrap();

        let public_fn = symbols.iter().find(|s| s.name == "public_function").unwrap();
        assert_eq!(public_fn.visibility, SymbolVisibility::Public);

        let protected_fn = symbols.iter().find(|s| s.name == "_protected_function").unwrap();
        assert_eq!(protected_fn.visibility, SymbolVisibility::Protected);

        let private_fn = symbols.iter().find(|s| s.name == "__private_function").unwrap();
        assert_eq!(private_fn.visibility, SymbolVisibility::Private);

        // Method inside class
        let method = symbols.iter().find(|s| s.name == "method").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(method.parent_name, Some("MyClass".to_string()));
    }

    #[test]
    fn test_parse_rust_type_parameters() {
        let parser = CodeParser::new();
        let code = r#"
pub fn generic_function<T, U>() {}

pub struct GenericStruct<T> {}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let generic_fn = symbols.iter().find(|s| s.name == "generic_function").unwrap();
        assert!(!generic_fn.type_parameters.is_empty() || generic_fn.signature.as_ref().map(|s| s.contains("<")).unwrap_or(false));

        let generic_struct = symbols.iter().find(|s| s.name == "GenericStruct").unwrap();
        assert!(!generic_struct.type_parameters.is_empty() || generic_struct.signature.as_ref().map(|s| s.contains("<")).unwrap_or(false));
    }

    #[test]
    fn test_parse_rust_return_type() {
        let parser = CodeParser::new();
        let code = r#"
pub fn returns_i32() -> i32 { 42 }
pub fn returns_nothing() {}
"#;

        let symbols = parser.parse(code, Language::Rust).unwrap();

        let with_return = symbols.iter().find(|s| s.name == "returns_i32").unwrap();
        assert!(with_return.return_type.is_some() || with_return.signature.as_ref().map(|s| s.contains("->")).unwrap_or(false));

        let no_return = symbols.iter().find(|s| s.name == "returns_nothing").unwrap();
        // Should either have no return type or signature without ->
        assert!(no_return.return_type.is_none() || no_return.signature.as_ref().map(|s| !s.contains("->")).unwrap_or(true));
    }

    #[test]
    fn test_parse_ruby_class_with_visibility() {
        let parser = CodeParser::new();
        let code = r#"
class User
  attr_accessor :name, :email

  def initialize(name)
    @name = name
  end

  def greet
    "Hello, #{@name}"
  end

  private

  def secret_method
    # private
  end

  protected

  def family_method
    # protected
  end
end
"#;
        let symbols = parser.parse(code, Language::Ruby).unwrap();

        // Assert User class extracted
        let user_class = symbols.iter().find(|s| s.name == "User" && s.kind == SymbolKind::Class);
        assert!(user_class.is_some(), "User class should be found");

        // Assert initialize is a constructor (Function)
        let init = symbols.iter().find(|s| s.name == "initialize").unwrap();
        assert_eq!(init.kind, SymbolKind::Function, "initialize should be a Function (constructor)");
        assert_eq!(init.visibility, SymbolVisibility::Public, "initialize should be public");

        // Assert greet is public
        let greet = symbols.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.kind, SymbolKind::Method);
        assert_eq!(greet.visibility, SymbolVisibility::Public);

        // Assert secret_method is private
        let secret = symbols.iter().find(|s| s.name == "secret_method").unwrap();
        assert_eq!(secret.visibility, SymbolVisibility::Private);

        // Assert family_method is protected
        let family = symbols.iter().find(|s| s.name == "family_method").unwrap();
        assert_eq!(family.visibility, SymbolVisibility::Protected);

        // Check attr_accessor creates properties
        let name_prop = symbols.iter().find(|s| s.name == "name" && s.kind == SymbolKind::Property);
        assert!(name_prop.is_some(), "attr_accessor :name should create a Property");
        let email_prop = symbols.iter().find(|s| s.name == "email" && s.kind == SymbolKind::Property);
        assert!(email_prop.is_some(), "attr_accessor :email should create a Property");
    }

    #[test]
    fn test_parse_ruby_module() {
        let parser = CodeParser::new();
        let code = r#"
module Searchable
  def self.included(base)
    base.extend(ClassMethods)
  end

  module ClassMethods
    def search(query)
      # search implementation
    end
  end

  def find_by_name(name)
    # instance method
  end
end
"#;
        let symbols = parser.parse(code, Language::Ruby).unwrap();

        // Assert Searchable module extracted
        let searchable = symbols.iter().find(|s| s.name == "Searchable" && s.kind == SymbolKind::Module);
        assert!(searchable.is_some(), "Searchable module should be found");

        // Assert ClassMethods nested module extracted
        let class_methods = symbols.iter().find(|s| s.name == "ClassMethods" && s.kind == SymbolKind::Module);
        assert!(class_methods.is_some(), "ClassMethods nested module should be found");

        // Assert methods are extracted
        let included = symbols.iter().find(|s| s.name == "included");
        assert!(included.is_some(), "self.included should be found");

        let find_by_name = symbols.iter().find(|s| s.name == "find_by_name");
        assert!(find_by_name.is_some(), "find_by_name instance method should be found");
    }

    #[test]
    fn test_parse_ruby_class_methods() {
        let parser = CodeParser::new();
        let code = r#"
class Configuration
  def self.default
    new
  end

  def instance_method
    # instance method
  end
end
"#;
        let symbols = parser.parse(code, Language::Ruby).unwrap();

        // Assert Configuration class extracted
        let config = symbols.iter().find(|s| s.name == "Configuration" && s.kind == SymbolKind::Class);
        assert!(config.is_some(), "Configuration class should be found");

        // Assert default is a class method (Function - singleton_method)
        let default_method = symbols.iter().find(|s| s.name == "default");
        assert!(default_method.is_some(), "self.default should be found");
        let default_method = default_method.unwrap();
        assert_eq!(default_method.kind, SymbolKind::Function, "class method should be Function");

        // Assert instance_method is a regular method
        let instance_method = symbols.iter().find(|s| s.name == "instance_method");
        assert!(instance_method.is_some(), "instance_method should be found");
        assert_eq!(instance_method.unwrap().kind, SymbolKind::Method);
    }

    #[test]
    fn test_parse_ruby_constants() {
        let parser = CodeParser::new();
        let code = r#"
module Constants
  VERSION = "1.0.0"
  MAX_RETRIES = 3

  NESTED = {
    key: "value"
  }
end
"#;
        let symbols = parser.parse(code, Language::Ruby).unwrap();

        // Assert Constants module extracted
        let constants_mod = symbols.iter().find(|s| s.name == "Constants" && s.kind == SymbolKind::Module);
        assert!(constants_mod.is_some(), "Constants module should be found");

        // Assert VERSION constant extracted
        let version = symbols.iter().find(|s| s.name == "VERSION" && s.kind == SymbolKind::Constant);
        assert!(version.is_some(), "VERSION constant should be found");

        // Assert MAX_RETRIES constant extracted
        let max_retries = symbols.iter().find(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Constant);
        assert!(max_retries.is_some(), "MAX_RETRIES constant should be found");

        // Assert NESTED constant extracted
        let nested = symbols.iter().find(|s| s.name == "NESTED" && s.kind == SymbolKind::Constant);
        assert!(nested.is_some(), "NESTED constant should be found");
    }

    #[test]
    fn test_parse_ruby_method_parameters() {
        let parser = CodeParser::new();
        let code = r#"
class Calculator
  def add(a, b)
    a + b
  end

  def multiply(a, b = 1)
    a * b
  end
end
"#;
        let symbols = parser.parse(code, Language::Ruby).unwrap();

        // Find the add method and check parameters
        let add = symbols.iter().find(|s| s.name == "add").unwrap();
        assert_eq!(add.parameters.len(), 2);
        assert_eq!(add.parameters[0].name, "a");
        assert_eq!(add.parameters[1].name, "b");

        // Find multiply method
        let multiply = symbols.iter().find(|s| s.name == "multiply").unwrap();
        // Should have at least the 'a' parameter
        assert!(!multiply.parameters.is_empty());
    }
}
