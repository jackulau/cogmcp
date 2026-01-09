//! Public API for priority scoring functionality
//!
//! This module exposes the priority calculation logic for external use,
//! allowing callers to query file priorities based on multiple factors.

use crate::prioritizer::{ContextPrioritizer, PriorityWeights};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Input parameters for priority calculation queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PriorityQuery {
    /// Optional list of specific file paths to score
    /// If empty, all indexed files may be considered
    pub file_paths: Option<Vec<String>>,

    /// Optional query text for relevance scoring
    /// Used to compute semantic relevance to the query
    pub query_text: Option<String>,

    /// Maximum number of results to return
    pub limit: Option<usize>,

    /// Minimum priority score threshold (0.0 to 1.0)
    /// Results below this threshold are filtered out
    pub min_score: Option<f32>,

    /// Maximum priority score threshold (0.0 to 1.0)
    /// Results above this threshold are filtered out
    pub max_score: Option<f32>,

    /// Custom weights for priority calculation
    /// If not provided, default weights are used
    pub weights: Option<PriorityWeights>,
}

impl PriorityQuery {
    /// Create a new empty query
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a query for specific file paths
    pub fn for_files(file_paths: Vec<String>) -> Self {
        Self {
            file_paths: Some(file_paths),
            ..Default::default()
        }
    }

    /// Create a query with a search text for relevance scoring
    pub fn for_query(query_text: impl Into<String>) -> Self {
        Self {
            query_text: Some(query_text.into()),
            ..Default::default()
        }
    }

    /// Set the maximum number of results
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the minimum score threshold
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Set the maximum score threshold
    pub fn with_max_score(mut self, max_score: f32) -> Self {
        self.max_score = Some(max_score);
        self
    }

    /// Set custom priority weights
    pub fn with_weights(mut self, weights: PriorityWeights) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Breakdown of individual score components
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Score based on how recently the file was accessed/modified
    pub recency: f32,

    /// Score based on semantic relevance to query
    pub relevance: f32,

    /// Score based on the file's centrality in the codebase (imports/dependencies)
    pub centrality: f32,

    /// Score based on recent git activity
    pub git_activity: f32,

    /// Score based on user focus/interaction patterns
    pub user_focus: f32,
}

impl ScoreBreakdown {
    /// Create a new score breakdown with all scores
    pub fn new(
        recency: f32,
        relevance: f32,
        centrality: f32,
        git_activity: f32,
        user_focus: f32,
    ) -> Self {
        Self {
            recency,
            relevance,
            centrality,
            git_activity,
            user_focus,
        }
    }
}

/// Result of a priority calculation for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityResult {
    /// Path to the file
    pub path: String,

    /// Overall priority score (0.0 to 1.0)
    pub priority_score: f32,

    /// Breakdown of individual score components
    pub breakdown: ScoreBreakdown,
}

impl PriorityResult {
    /// Create a new priority result
    pub fn new(path: impl Into<String>, priority_score: f32, breakdown: ScoreBreakdown) -> Self {
        Self {
            path: path.into(),
            priority_score,
            breakdown,
        }
    }
}

/// Metadata about a file used for priority scoring
#[derive(Debug, Clone, Default)]
pub struct FileScoreMetadata {
    /// Recency score (0.0 to 1.0) based on last access/modification time
    pub recency_score: f32,

    /// Relevance score (0.0 to 1.0) based on query similarity
    pub relevance_score: f32,

    /// Centrality score (0.0 to 1.0) based on import graph position
    pub centrality_score: f32,

    /// Git activity score (0.0 to 1.0) based on recent commits
    pub git_activity_score: f32,

    /// User focus score (0.0 to 1.0) based on interaction patterns
    pub user_focus_score: f32,
}

/// Calculate file priorities based on the provided query and metadata
///
/// This function takes a query specification and a map of file metadata,
/// then returns scored results sorted by priority (highest first).
///
/// # Arguments
///
/// * `query` - The priority query specifying filters and options
/// * `file_metadata` - A map from file paths to their scoring metadata
///
/// # Returns
///
/// A vector of `PriorityResult` items sorted by priority score (highest first)
///
/// # Example
///
/// ```
/// use cogmcp_context::api::{PriorityQuery, FileScoreMetadata, calculate_file_priorities};
/// use std::collections::HashMap;
///
/// let query = PriorityQuery::new().with_limit(10).with_min_score(0.5);
///
/// let mut metadata = HashMap::new();
/// metadata.insert("src/main.rs".to_string(), FileScoreMetadata {
///     recency_score: 0.9,
///     relevance_score: 0.8,
///     centrality_score: 0.7,
///     git_activity_score: 0.6,
///     user_focus_score: 0.5,
/// });
///
/// let results = calculate_file_priorities(&query, &metadata);
/// ```
pub fn calculate_file_priorities(
    query: &PriorityQuery,
    file_metadata: &HashMap<String, FileScoreMetadata>,
) -> Vec<PriorityResult> {
    let weights = query.weights.clone().unwrap_or_default();
    let prioritizer = ContextPrioritizer::new(weights);

    let mut results: Vec<PriorityResult> = file_metadata
        .iter()
        .filter(|(path, _)| {
            // Filter by file_paths if specified
            query
                .file_paths
                .as_ref()
                .map(|paths| paths.contains(path))
                .unwrap_or(true)
        })
        .map(|(path, metadata)| {
            let priority_score = prioritizer.calculate_priority(
                metadata.recency_score,
                metadata.relevance_score,
                metadata.centrality_score,
                metadata.git_activity_score,
                metadata.user_focus_score,
            );

            let breakdown = ScoreBreakdown::new(
                metadata.recency_score,
                metadata.relevance_score,
                metadata.centrality_score,
                metadata.git_activity_score,
                metadata.user_focus_score,
            );

            PriorityResult::new(path.clone(), priority_score, breakdown)
        })
        .filter(|result| {
            // Apply score thresholds
            let above_min = query
                .min_score
                .map(|min| result.priority_score >= min)
                .unwrap_or(true);
            let below_max = query
                .max_score
                .map(|max| result.priority_score <= max)
                .unwrap_or(true);
            above_min && below_max
        })
        .collect();

    // Sort by priority score (highest first)
    results.sort_by(|a, b| {
        b.priority_score
            .partial_cmp(&a.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply limit if specified
    if let Some(limit) = query.limit {
        results.truncate(limit);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metadata() -> HashMap<String, FileScoreMetadata> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "src/main.rs".to_string(),
            FileScoreMetadata {
                recency_score: 0.9,
                relevance_score: 0.8,
                centrality_score: 0.7,
                git_activity_score: 0.6,
                user_focus_score: 0.5,
            },
        );

        metadata.insert(
            "src/lib.rs".to_string(),
            FileScoreMetadata {
                recency_score: 0.5,
                relevance_score: 0.9,
                centrality_score: 0.8,
                git_activity_score: 0.4,
                user_focus_score: 0.3,
            },
        );

        metadata.insert(
            "tests/test.rs".to_string(),
            FileScoreMetadata {
                recency_score: 0.3,
                relevance_score: 0.2,
                centrality_score: 0.1,
                git_activity_score: 0.2,
                user_focus_score: 0.1,
            },
        );

        metadata
    }

    #[test]
    fn test_calculate_file_priorities_basic() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::new();

        let results = calculate_file_priorities(&query, &metadata);

        assert_eq!(results.len(), 3);
        // Results should be sorted by priority (highest first)
        assert!(results[0].priority_score >= results[1].priority_score);
        assert!(results[1].priority_score >= results[2].priority_score);
    }

    #[test]
    fn test_calculate_file_priorities_with_limit() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::new().with_limit(2);

        let results = calculate_file_priorities(&query, &metadata);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_calculate_file_priorities_with_min_score() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::new().with_min_score(0.5);

        let results = calculate_file_priorities(&query, &metadata);

        for result in &results {
            assert!(result.priority_score >= 0.5);
        }
    }

    #[test]
    fn test_calculate_file_priorities_with_max_score() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::new().with_max_score(0.5);

        let results = calculate_file_priorities(&query, &metadata);

        for result in &results {
            assert!(result.priority_score <= 0.5);
        }
    }

    #[test]
    fn test_calculate_file_priorities_with_file_filter() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::for_files(vec!["src/main.rs".to_string()]);

        let results = calculate_file_priorities(&query, &metadata);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "src/main.rs");
    }

    #[test]
    fn test_calculate_file_priorities_empty_input() {
        let metadata = HashMap::new();
        let query = PriorityQuery::new();

        let results = calculate_file_priorities(&query, &metadata);

        assert!(results.is_empty());
    }

    #[test]
    fn test_calculate_file_priorities_with_custom_weights() {
        let metadata = create_test_metadata();

        // Use weights that heavily favor relevance
        let weights = PriorityWeights {
            recency: 0.0,
            relevance: 1.0,
            centrality: 0.0,
            git_activity: 0.0,
            user_focus: 0.0,
        };

        let query = PriorityQuery::new().with_weights(weights);

        let results = calculate_file_priorities(&query, &metadata);

        // With 100% relevance weight, lib.rs (0.9 relevance) should be first
        assert_eq!(results[0].path, "src/lib.rs");
    }

    #[test]
    fn test_score_breakdown_is_populated() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::for_files(vec!["src/main.rs".to_string()]);

        let results = calculate_file_priorities(&query, &metadata);

        assert_eq!(results.len(), 1);
        let breakdown = &results[0].breakdown;

        assert!((breakdown.recency - 0.9).abs() < f32::EPSILON);
        assert!((breakdown.relevance - 0.8).abs() < f32::EPSILON);
        assert!((breakdown.centrality - 0.7).abs() < f32::EPSILON);
        assert!((breakdown.git_activity - 0.6).abs() < f32::EPSILON);
        assert!((breakdown.user_focus - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_priority_query_builder_pattern() {
        let query = PriorityQuery::for_query("search text")
            .with_limit(10)
            .with_min_score(0.3)
            .with_max_score(0.9);

        assert_eq!(query.query_text, Some("search text".to_string()));
        assert_eq!(query.limit, Some(10));
        assert_eq!(query.min_score, Some(0.3));
        assert_eq!(query.max_score, Some(0.9));
    }

    #[test]
    fn test_missing_file_in_filter() {
        let metadata = create_test_metadata();
        let query = PriorityQuery::for_files(vec!["nonexistent.rs".to_string()]);

        let results = calculate_file_priorities(&query, &metadata);

        assert!(results.is_empty());
    }
}
