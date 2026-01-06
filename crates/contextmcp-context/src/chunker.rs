//! Smart chunking strategies

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Chunk at semantic boundaries (function/class)
    Semantic,
    /// Recursive splitting at natural boundaries
    Recursive,
    /// Fixed-size chunks
    Fixed,
}

/// A chunk of content
#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub chunk_type: ChunkType,
}

/// Type of chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    Symbol,      // Function, class, etc.
    DocComment,  // Documentation
    CodeBlock,   // Generic code block
    FileHeader,  // Imports, module docs
}

/// Chunker for splitting content into appropriate chunks
pub struct ContextChunker {
    strategy: ChunkingStrategy,
    target_size: usize,
    overlap: f32,
}

impl ContextChunker {
    pub fn new(strategy: ChunkingStrategy, target_size: usize, overlap: f32) -> Self {
        Self {
            strategy,
            target_size,
            overlap,
        }
    }

    /// Chunk content according to the strategy
    pub fn chunk(&self, content: &str) -> Vec<Chunk> {
        match self.strategy {
            ChunkingStrategy::Semantic => self.chunk_semantic(content),
            ChunkingStrategy::Recursive => self.chunk_recursive(content),
            ChunkingStrategy::Fixed => self.chunk_fixed(content),
        }
    }

    fn chunk_semantic(&self, content: &str) -> Vec<Chunk> {
        // For now, treat each function-like block as a chunk
        // This is a simplified implementation
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut current_chunk_start = 0;
        let mut brace_depth = 0;
        let mut in_function = false;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Simple heuristic: function starts
            if (trimmed.starts_with("fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.contains("class "))
                && !in_function
            {
                if i > current_chunk_start {
                    // Save previous chunk if non-empty
                    let chunk_content: String = lines[current_chunk_start..i].join("\n");
                    if !chunk_content.trim().is_empty() {
                        chunks.push(Chunk {
                            content: chunk_content,
                            start_line: current_chunk_start as u32 + 1,
                            end_line: i as u32,
                            chunk_type: ChunkType::CodeBlock,
                        });
                    }
                }
                current_chunk_start = i;
                in_function = true;
            }

            // Track brace depth
            brace_depth += line.matches('{').count() as i32;
            brace_depth -= line.matches('}').count() as i32;

            // Function ends when braces balance
            if in_function && brace_depth == 0 && line.contains('}') {
                let chunk_content: String = lines[current_chunk_start..=i].join("\n");
                chunks.push(Chunk {
                    content: chunk_content,
                    start_line: current_chunk_start as u32 + 1,
                    end_line: i as u32 + 1,
                    chunk_type: ChunkType::Symbol,
                });
                current_chunk_start = i + 1;
                in_function = false;
            }
        }

        // Add remaining content
        if current_chunk_start < lines.len() {
            let chunk_content: String = lines[current_chunk_start..].join("\n");
            if !chunk_content.trim().is_empty() {
                chunks.push(Chunk {
                    content: chunk_content,
                    start_line: current_chunk_start as u32 + 1,
                    end_line: lines.len() as u32,
                    chunk_type: ChunkType::CodeBlock,
                });
            }
        }

        chunks
    }

    fn chunk_recursive(&self, content: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Split at paragraph boundaries (empty lines)
        let mut current_start = 0;
        let mut current_content = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            if line.trim().is_empty() && !current_content.is_empty() {
                let chunk_text = current_content.join("\n");
                if chunk_text.len() >= self.target_size / 2 {
                    chunks.push(Chunk {
                        content: chunk_text,
                        start_line: current_start as u32 + 1,
                        end_line: i as u32,
                        chunk_type: ChunkType::CodeBlock,
                    });
                    current_content.clear();
                    current_start = i + 1;
                }
            } else {
                current_content.push(*line);

                // Check if we've exceeded target size
                let current_size: usize = current_content.iter().map(|l| l.len() + 1).sum();
                if current_size >= self.target_size {
                    chunks.push(Chunk {
                        content: current_content.join("\n"),
                        start_line: current_start as u32 + 1,
                        end_line: i as u32 + 1,
                        chunk_type: ChunkType::CodeBlock,
                    });

                    // Keep overlap
                    let overlap_lines = (current_content.len() as f32 * self.overlap) as usize;
                    current_content = current_content.split_off(current_content.len() - overlap_lines);
                    current_start = i - overlap_lines + 1;
                }
            }
        }

        // Add remaining
        if !current_content.is_empty() {
            chunks.push(Chunk {
                content: current_content.join("\n"),
                start_line: current_start as u32 + 1,
                end_line: lines.len() as u32,
                chunk_type: ChunkType::CodeBlock,
            });
        }

        chunks
    }

    fn chunk_fixed(&self, content: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let lines_per_chunk = (self.target_size / 80).max(10); // Assume ~80 chars per line
        let overlap_lines = (lines_per_chunk as f32 * self.overlap) as usize;

        let mut i = 0;
        while i < lines.len() {
            let end = (i + lines_per_chunk).min(lines.len());
            let chunk_content: String = lines[i..end].join("\n");

            chunks.push(Chunk {
                content: chunk_content,
                start_line: i as u32 + 1,
                end_line: end as u32,
                chunk_type: ChunkType::CodeBlock,
            });

            i = end - overlap_lines;
            if i <= chunks.last().map(|c| c.start_line as usize - 1).unwrap_or(0) {
                i = end;
            }
        }

        chunks
    }
}

impl Default for ContextChunker {
    fn default() -> Self {
        Self::new(ChunkingStrategy::Semantic, 2048, 0.1)
    }
}
