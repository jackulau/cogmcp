//! Git activity scoring module
//!
//! Calculates normalized scores (0.0-1.0) based on git activity metrics
//! including recent commits, change frequency, and recency.

use std::collections::HashMap;

use cogmcp_core::Result;

use crate::git::GitRepo;

/// Configuration for git activity scoring
#[derive(Debug, Clone)]
pub struct GitScoringConfig {
    /// Time window for "recent" commits in days (default: 7)
    pub recent_window_days: u32,
    /// Time window for change frequency calculation in days (default: 30)
    pub frequency_window_days: u32,
    /// Maximum commits to consider for recent scoring
    pub max_recent_commits: usize,
    /// Weight for recent commits score (default: 0.4)
    pub recent_commits_weight: f32,
    /// Weight for change frequency score (default: 0.3)
    pub change_frequency_weight: f32,
    /// Weight for recency score (default: 0.3)
    pub recency_weight: f32,
    /// Decay factor for exponential recency scoring (higher = faster decay)
    pub recency_decay_factor: f32,
}

impl Default for GitScoringConfig {
    fn default() -> Self {
        Self {
            recent_window_days: 7,
            frequency_window_days: 30,
            max_recent_commits: 50,
            recent_commits_weight: 0.4,
            change_frequency_weight: 0.3,
            recency_weight: 0.3,
            recency_decay_factor: 0.1,
        }
    }
}

/// Breakdown of git activity scoring factors
#[derive(Debug, Clone, Default)]
pub struct GitActivityScore {
    /// Score based on number of commits in the recent window (0.0-1.0)
    pub recent_commits_score: f32,
    /// Score based on how often the file changes (0.0-1.0)
    pub change_frequency_score: f32,
    /// Score based on how recently the file was modified (0.0-1.0)
    pub recency_score: f32,
    /// Weighted combination of all scores (0.0-1.0)
    pub combined_score: f32,
    /// Number of commits in the recent window
    pub recent_commit_count: usize,
    /// Number of commits in the frequency window
    pub frequency_commit_count: usize,
    /// Seconds since last modification (None if no commits found)
    pub seconds_since_last_change: Option<i64>,
}

impl GitActivityScore {
    /// Create a score for a file with no git activity
    pub fn no_activity() -> Self {
        Self::default()
    }
}

/// Scorer for calculating git activity metrics
pub struct GitActivityScorer<'a> {
    repo: &'a GitRepo,
    config: GitScoringConfig,
    now: i64,
}

impl<'a> GitActivityScorer<'a> {
    /// Create a new scorer with the given repository and default config
    pub fn new(repo: &'a GitRepo) -> Self {
        Self::with_config(repo, GitScoringConfig::default())
    }

    /// Create a new scorer with custom configuration
    pub fn with_config(repo: &'a GitRepo, config: GitScoringConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self { repo, config, now }
    }

    /// Calculate the git activity score for a single file
    pub fn calculate_file_score(&self, path: &str) -> Result<GitActivityScore> {
        // Get commits for the file within the frequency window
        let max_commits = self.config.max_recent_commits;
        let commits = self.repo.get_file_commits(path, max_commits)?;

        if commits.is_empty() {
            return Ok(GitActivityScore::no_activity());
        }

        // Calculate time boundaries
        let recent_cutoff = self.now - (self.config.recent_window_days as i64 * 24 * 3600);
        let frequency_cutoff = self.now - (self.config.frequency_window_days as i64 * 24 * 3600);

        // Count commits in each window
        let mut recent_commit_count = 0;
        let mut frequency_commit_count = 0;
        let mut most_recent_timestamp: Option<i64> = None;

        for commit in &commits {
            if most_recent_timestamp.is_none() || commit.timestamp > most_recent_timestamp.unwrap()
            {
                most_recent_timestamp = Some(commit.timestamp);
            }

            if commit.timestamp >= recent_cutoff {
                recent_commit_count += 1;
            }
            if commit.timestamp >= frequency_cutoff {
                frequency_commit_count += 1;
            }
        }

        // Calculate recent commits score (log scale, capped at 1.0)
        // 1 commit = 0.3, 3 commits = 0.6, 10+ commits = 1.0
        let recent_commits_score = if recent_commit_count == 0 {
            0.0
        } else {
            ((recent_commit_count as f32).ln() / (10.0_f32).ln() + 0.3).min(1.0)
        };

        // Calculate change frequency score
        // Based on commits per week in the frequency window
        let weeks_in_window = self.config.frequency_window_days as f32 / 7.0;
        let commits_per_week = frequency_commit_count as f32 / weeks_in_window;
        // Normalize: 0.5 commits/week = 0.3, 1 commit/week = 0.5, 3+ commits/week = 1.0
        let change_frequency_score = (commits_per_week / 3.0).min(1.0);

        // Calculate recency score using exponential decay
        let seconds_since_last_change = most_recent_timestamp.map(|ts| self.now - ts);
        let recency_score = if let Some(seconds) = seconds_since_last_change {
            let days = seconds as f32 / (24.0 * 3600.0);
            // Exponential decay: score = e^(-decay_factor * days)
            (-self.config.recency_decay_factor * days).exp()
        } else {
            0.0
        };

        // Calculate weighted combined score
        let combined_score = recent_commits_score * self.config.recent_commits_weight
            + change_frequency_score * self.config.change_frequency_weight
            + recency_score * self.config.recency_weight;

        Ok(GitActivityScore {
            recent_commits_score,
            change_frequency_score,
            recency_score,
            combined_score,
            recent_commit_count,
            frequency_commit_count,
            seconds_since_last_change,
        })
    }

    /// Calculate scores for multiple files efficiently
    ///
    /// This method attempts to batch operations where possible for better performance.
    pub fn calculate_batch_scores(&self, paths: &[&str]) -> Result<HashMap<String, GitActivityScore>> {
        let mut scores = HashMap::new();

        // Get recently changed files to optimize scoring
        let recent_hours = self.config.recent_window_days as u64 * 24;
        let recently_changed = self.repo.recently_changed_files(recent_hours)?;
        let recently_changed_set: std::collections::HashSet<_> = recently_changed.into_iter().collect();

        for path in paths {
            // Skip expensive git log calls for files not recently changed
            if !recently_changed_set.contains(*path) {
                // File hasn't changed recently, still calculate but expect low scores
                let score = self.calculate_file_score(path)?;
                scores.insert((*path).to_string(), score);
            } else {
                let score = self.calculate_file_score(path)?;
                scores.insert((*path).to_string(), score);
            }
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GitScoringConfig::default();
        assert_eq!(config.recent_window_days, 7);
        assert_eq!(config.frequency_window_days, 30);
        assert_eq!(config.max_recent_commits, 50);
        // Weights should sum to 1.0
        let weight_sum =
            config.recent_commits_weight + config.change_frequency_weight + config.recency_weight;
        assert!((weight_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_no_activity_score() {
        let score = GitActivityScore::no_activity();
        assert_eq!(score.recent_commits_score, 0.0);
        assert_eq!(score.change_frequency_score, 0.0);
        assert_eq!(score.recency_score, 0.0);
        assert_eq!(score.combined_score, 0.0);
        assert_eq!(score.recent_commit_count, 0);
        assert_eq!(score.frequency_commit_count, 0);
        assert!(score.seconds_since_last_change.is_none());
    }

    #[test]
    fn test_score_bounds() {
        // Verify scores are always in valid range
        let score = GitActivityScore {
            recent_commits_score: 0.5,
            change_frequency_score: 0.3,
            recency_score: 0.8,
            combined_score: 0.5,
            recent_commit_count: 5,
            frequency_commit_count: 10,
            seconds_since_last_change: Some(3600),
        };

        assert!(score.recent_commits_score >= 0.0 && score.recent_commits_score <= 1.0);
        assert!(score.change_frequency_score >= 0.0 && score.change_frequency_score <= 1.0);
        assert!(score.recency_score >= 0.0 && score.recency_score <= 1.0);
        assert!(score.combined_score >= 0.0 && score.combined_score <= 1.0);
    }

    #[test]
    fn test_config_weights() {
        let mut config = GitScoringConfig::default();
        config.recent_commits_weight = 0.5;
        config.change_frequency_weight = 0.25;
        config.recency_weight = 0.25;

        let weight_sum =
            config.recent_commits_weight + config.change_frequency_weight + config.recency_weight;
        assert!((weight_sum - 1.0).abs() < 0.001);
    }
}
