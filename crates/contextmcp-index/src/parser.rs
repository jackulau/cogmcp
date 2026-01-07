//! Code parsing with tree-sitter

use cogmcp_core::types::{Language, SymbolKind, SymbolModifiers, SymbolVisibility};
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
    /// Visibility/access modifier
    pub visibility: Option<SymbolVisibility>,
    /// Symbol modifiers (async, static, etc.)
    pub modifiers: SymbolModifiers,
    /// Name of the parent symbol (for nested symbols like methods in a class)
    pub parent_symbol: Option<String>,
    /// Generic type parameters (e.g., ["T", "U"] for fn foo<T, U>)
    pub type_parameters: Vec<String>,
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
        let kind_str = node.kind();

        match kind_str {
            "function_item" | "impl_item" | "struct_item" | "enum_item" | "trait_item"
            | "mod_item" | "const_item" | "static_item" | "type_item" => {
                if let Some(symbol) = self.extract_rust_symbol(node, content, kind_str) {
                    symbols.push(symbol);
                }
            }
            _ => {}
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_rust_symbols(&child, content, symbols);
        }
    }

    fn extract_rust_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
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

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: None,
            modifiers: SymbolModifiers::default(),
            parent_symbol: None,
            type_parameters: Vec::new(),
        })
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
        let kind_str = node.kind();

        match kind_str {
            "function_declaration" | "method_definition" | "class_declaration"
            | "interface_declaration" | "type_alias_declaration" | "enum_declaration"
            | "variable_declarator" => {
                if let Some(symbol) = self.extract_ts_symbol(node, content, kind_str) {
                    symbols.push(symbol);
                }
            }
            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_ts_symbols(&child, content, symbols);
        }
    }

    fn extract_ts_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
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

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: None,
            modifiers: SymbolModifiers::default(),
            parent_symbol: None,
            type_parameters: Vec::new(),
        })
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
        let kind_str = node.kind();

        match kind_str {
            "function_definition" | "class_definition" => {
                if let Some(symbol) = self.extract_python_symbol(node, content, kind_str) {
                    symbols.push(symbol);
                }
            }
            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_python_symbols(&child, content, symbols);
        }
    }

    fn extract_python_symbol(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        kind_str: &str,
    ) -> Option<ExtractedSymbol> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(content.as_bytes()).ok()?.to_string();

        let kind = match kind_str {
            "function_definition" => SymbolKind::Function,
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

        Some(ExtractedSymbol {
            name,
            kind,
            start_line,
            end_line,
            signature,
            doc_comment,
            visibility: None,
            modifiers: SymbolModifiers::default(),
            parent_symbol: None,
            type_parameters: Vec::new(),
        })
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
                                s.trim_matches('"')
                                    .trim_matches('\'')
                                    .trim()
                                    .to_string()
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
