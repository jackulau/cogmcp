//! Context prioritization algorithm

use serde::{Deserialize, Serialize};

/// Weights for priority calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeights {
    pub recency: f32,
    pub relevance: f32,
    pub centrality: f32,
    pub git_activity: f32,
    pub user_focus: f32,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            recency: 0.25,
            relevance: 0.30,
            centrality: 0.20,
            git_activity: 0.15,
            user_focus: 0.10,
        }
    }
}

/// Context item with priority information
#[derive(Debug, Clone)]
pub struct ContextItem {
    pub path: String,
    pub content: String,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub priority_score: f32,
}

/// Context prioritizer for selecting the most relevant context
pub struct ContextPrioritizer {
    weights: PriorityWeights,
}

impl ContextPrioritizer {
    pub fn new(weights: PriorityWeights) -> Self {
        Self { weights }
    }

    /// Calculate priority score for a context item
    pub fn calculate_priority(
        &self,
        recency_score: f32,
        relevance_score: f32,
        centrality_score: f32,
        git_activity_score: f32,
        user_focus_score: f32,
    ) -> f32 {
        self.weights.recency * recency_score
            + self.weights.relevance * relevance_score
            + self.weights.centrality * centrality_score
            + self.weights.git_activity * git_activity_score
            + self.weights.user_focus * user_focus_score
    }

    /// Select context items within a token budget
    pub fn select_within_budget(
        &self,
        mut items: Vec<ContextItem>,
        max_tokens: u32,
    ) -> Vec<ContextItem> {
        // Sort by priority (highest first)
        items.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        let mut selected = Vec::new();
        let mut used_tokens = 0u32;

        for item in items {
            let item_tokens = Self::estimate_tokens(&item.content);
            if used_tokens + item_tokens <= max_tokens {
                used_tokens += item_tokens;
                selected.push(item);
            }
        }

        selected
    }

    /// Rough token estimation (4 chars per token average)
    fn estimate_tokens(text: &str) -> u32 {
        (text.len() / 4) as u32
    }
}

impl Default for ContextPrioritizer {
    fn default() -> Self {
        Self::new(PriorityWeights::default())
    }
}
