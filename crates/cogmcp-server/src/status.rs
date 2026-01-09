//! Server runtime status tracking
//!
//! This module provides status tracking for the MCP server including
//! uptime, request counts, and tool usage statistics.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Server runtime status tracking
pub struct ServerStatus {
    /// When the server started
    started_at: Instant,
    /// Total requests processed
    request_count: AtomicU64,
    /// Calls per tool
    tool_calls: DashMap<String, u64>,
    /// Failed requests
    error_count: AtomicU64,
    /// Timestamp of last request (millis since start)
    last_request_at: AtomicU64,
}

impl ServerStatus {
    /// Create a new server status tracker
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            request_count: AtomicU64::new(0),
            tool_calls: DashMap::new(),
            error_count: AtomicU64::new(0),
            last_request_at: AtomicU64::new(0),
        }
    }

    /// Record a successful tool call
    pub fn record_tool_call(&self, tool_name: &str) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.last_request_at.store(
            self.started_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );

        // Update tool-specific counter
        self.tool_calls
            .entry(tool_name.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Record a failed request
    pub fn record_error(&self, tool_name: &str) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.last_request_at.store(
            self.started_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );

        // Still count the tool call for stats
        self.tool_calls
            .entry(tool_name.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Get server uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Get total request count
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// Get error count
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Get time since last request
    pub fn time_since_last_request(&self) -> Option<std::time::Duration> {
        let last_millis = self.last_request_at.load(Ordering::Relaxed);
        if last_millis == 0 {
            return None;
        }
        let elapsed = self.started_at.elapsed().as_millis() as u64;
        Some(std::time::Duration::from_millis(elapsed - last_millis))
    }

    /// Get tool call counts sorted by count descending
    pub fn tool_call_counts(&self) -> Vec<(String, u64)> {
        let mut counts: Vec<_> = self
            .tool_calls
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts
    }

    /// Format uptime as human-readable string
    pub fn format_uptime(&self) -> String {
        let duration = self.uptime();
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let secs = total_secs % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, secs)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }

    /// Format time since last request as human-readable string
    pub fn format_last_request(&self) -> String {
        match self.time_since_last_request() {
            Some(duration) => {
                let secs = duration.as_secs();
                if secs < 60 {
                    format!("{}s ago", secs)
                } else if secs < 3600 {
                    format!("{}m {}s ago", secs / 60, secs % 60)
                } else {
                    format!("{}h {}m ago", secs / 3600, (secs % 3600) / 60)
                }
            }
            None => "No requests yet".to_string(),
        }
    }

    /// Generate formatted status report
    pub fn format_status(&self) -> String {
        let total_requests = self.request_count();
        let errors = self.error_count();
        let error_rate = if total_requests > 0 {
            (errors as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        let mut output = String::new();
        output.push_str("## Server Status\n\n");
        output.push_str(&format!("- **Uptime:** {}\n", self.format_uptime()));
        output.push_str(&format!("- **Total requests:** {}\n", total_requests));
        output.push_str(&format!(
            "- **Errors:** {} ({:.1}%)\n",
            errors, error_rate
        ));
        output.push_str(&format!("- **Last request:** {}\n", self.format_last_request()));

        let tool_counts = self.tool_call_counts();
        if !tool_counts.is_empty() {
            output.push_str("\n### Tool Usage\n");
            for (tool, count) in tool_counts {
                output.push_str(&format!("- {}: {} calls\n", tool, count));
            }
        }

        output
    }
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_status_new() {
        let status = ServerStatus::new();
        assert_eq!(status.request_count(), 0);
        assert_eq!(status.error_count(), 0);
        assert!(status.tool_call_counts().is_empty());
    }

    #[test]
    fn test_record_tool_call() {
        let status = ServerStatus::new();
        status.record_tool_call("ping");
        status.record_tool_call("ping");
        status.record_tool_call("context_search");

        assert_eq!(status.request_count(), 3);
        assert_eq!(status.error_count(), 0);

        let counts = status.tool_call_counts();
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[0], ("ping".to_string(), 2));
        assert_eq!(counts[1], ("context_search".to_string(), 1));
    }

    #[test]
    fn test_record_error() {
        let status = ServerStatus::new();
        status.record_tool_call("ping");
        status.record_error("unknown_tool");

        assert_eq!(status.request_count(), 2);
        assert_eq!(status.error_count(), 1);
    }

    #[test]
    fn test_format_uptime() {
        let status = ServerStatus::new();
        // Just verify it doesn't panic and returns something reasonable
        let uptime = status.format_uptime();
        assert!(uptime.ends_with('s'));
    }

    #[test]
    fn test_format_status() {
        let status = ServerStatus::new();
        status.record_tool_call("ping");

        let output = status.format_status();
        assert!(output.contains("## Server Status"));
        assert!(output.contains("**Uptime:**"));
        assert!(output.contains("**Total requests:** 1"));
        assert!(output.contains("ping: 1 calls"));
    }
}
