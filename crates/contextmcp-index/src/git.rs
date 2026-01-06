//! Git integration for history and blame

use contextmcp_core::{Error, Result};
use git2::{DiffOptions, Repository};
use std::path::Path;

/// Git repository wrapper
pub struct GitRepo {
    repo: Repository,
}

impl GitRepo {
    /// Open a git repository
    pub fn open(path: &Path) -> Result<Self> {
        let repo = Repository::discover(path)
            .map_err(|e| Error::Git(format!("Failed to open repository: {}", e)))?;
        Ok(Self { repo })
    }

    /// Get recent commits
    pub fn get_commits(&self, limit: usize) -> Result<Vec<CommitInfo>> {
        let mut revwalk = self
            .repo
            .revwalk()
            .map_err(|e| Error::Git(format!("Failed to create revwalk: {}", e)))?;

        revwalk
            .push_head()
            .map_err(|e| Error::Git(format!("Failed to push HEAD: {}", e)))?;

        let mut commits = Vec::new();
        for oid in revwalk.take(limit) {
            let oid = oid.map_err(|e| Error::Git(format!("Failed to get oid: {}", e)))?;
            let commit = self
                .repo
                .find_commit(oid)
                .map_err(|e| Error::Git(format!("Failed to find commit: {}", e)))?;

            let author = commit.author();
            commits.push(CommitInfo {
                hash: oid.to_string(),
                author: author.name().unwrap_or("Unknown").to_string(),
                email: author.email().unwrap_or("").to_string(),
                message: commit.message().unwrap_or("").to_string(),
                timestamp: commit.time().seconds(),
            });
        }

        Ok(commits)
    }

    /// Get commits affecting a specific file
    pub fn get_file_commits(&self, file_path: &str, limit: usize) -> Result<Vec<CommitInfo>> {
        let mut revwalk = self
            .repo
            .revwalk()
            .map_err(|e| Error::Git(format!("Failed to create revwalk: {}", e)))?;

        revwalk
            .push_head()
            .map_err(|e| Error::Git(format!("Failed to push HEAD: {}", e)))?;

        let mut commits = Vec::new();
        let path = Path::new(file_path);

        for oid in revwalk {
            if commits.len() >= limit {
                break;
            }

            let oid = oid.map_err(|e| Error::Git(format!("Failed to get oid: {}", e)))?;
            let commit = self
                .repo
                .find_commit(oid)
                .map_err(|e| Error::Git(format!("Failed to find commit: {}", e)))?;

            // Check if this commit touched the file
            if self.commit_touches_file(&commit, path)? {
                let author = commit.author();
                commits.push(CommitInfo {
                    hash: oid.to_string(),
                    author: author.name().unwrap_or("Unknown").to_string(),
                    email: author.email().unwrap_or("").to_string(),
                    message: commit.message().unwrap_or("").to_string(),
                    timestamp: commit.time().seconds(),
                });
            }
        }

        Ok(commits)
    }

    fn commit_touches_file(&self, commit: &git2::Commit, path: &Path) -> Result<bool> {
        let tree = commit
            .tree()
            .map_err(|e| Error::Git(format!("Failed to get tree: {}", e)))?;

        // Check if file exists in this commit
        if tree.get_path(path).is_ok() {
            // Check parent to see if file changed
            if let Ok(parent) = commit.parent(0) {
                let parent_tree = parent
                    .tree()
                    .map_err(|e| Error::Git(format!("Failed to get parent tree: {}", e)))?;

                let mut diff_opts = DiffOptions::new();
                diff_opts.pathspec(path.to_string_lossy().as_ref());

                let diff = self
                    .repo
                    .diff_tree_to_tree(Some(&parent_tree), Some(&tree), Some(&mut diff_opts))
                    .map_err(|e| Error::Git(format!("Failed to diff: {}", e)))?;

                return Ok(diff.deltas().count() > 0);
            }
            // No parent means initial commit
            return Ok(true);
        }

        Ok(false)
    }

    /// Get blame information for a file
    pub fn blame(&self, file_path: &str) -> Result<Vec<BlameInfo>> {
        let blame = self
            .repo
            .blame_file(Path::new(file_path), None)
            .map_err(|e| Error::Git(format!("Failed to get blame: {}", e)))?;

        let mut result = Vec::new();
        for hunk in blame.iter() {
            let sig = hunk.final_signature();
            result.push(BlameInfo {
                line_start: hunk.final_start_line(),
                line_count: hunk.lines_in_hunk(),
                commit_hash: hunk.final_commit_id().to_string(),
                author: sig.name().unwrap_or("Unknown").to_string(),
                timestamp: sig.when().seconds(),
            });
        }

        Ok(result)
    }

    /// Get uncommitted changes
    pub fn diff_uncommitted(&self) -> Result<String> {
        let head = self
            .repo
            .head()
            .map_err(|e| Error::Git(format!("Failed to get HEAD: {}", e)))?;

        let head_tree = head
            .peel_to_tree()
            .map_err(|e| Error::Git(format!("Failed to get HEAD tree: {}", e)))?;

        let diff = self
            .repo
            .diff_tree_to_workdir_with_index(Some(&head_tree), None)
            .map_err(|e| Error::Git(format!("Failed to get diff: {}", e)))?;

        let mut diff_text = String::new();
        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            if let Ok(content) = std::str::from_utf8(line.content()) {
                let prefix = match line.origin() {
                    '+' => "+",
                    '-' => "-",
                    ' ' => " ",
                    _ => "",
                };
                diff_text.push_str(prefix);
                diff_text.push_str(content);
            }
            true
        })
        .map_err(|e| Error::Git(format!("Failed to print diff: {}", e)))?;

        Ok(diff_text)
    }

    /// Get recently changed files
    pub fn recently_changed_files(&self, hours: u64) -> Result<Vec<String>> {
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - (hours as i64 * 3600);

        let mut revwalk = self
            .repo
            .revwalk()
            .map_err(|e| Error::Git(format!("Failed to create revwalk: {}", e)))?;

        revwalk
            .push_head()
            .map_err(|e| Error::Git(format!("Failed to push HEAD: {}", e)))?;

        let mut files = std::collections::HashSet::new();

        for oid in revwalk {
            let oid = oid.map_err(|e| Error::Git(format!("Failed to get oid: {}", e)))?;
            let commit = self
                .repo
                .find_commit(oid)
                .map_err(|e| Error::Git(format!("Failed to find commit: {}", e)))?;

            if commit.time().seconds() < cutoff {
                break;
            }

            // Get changed files in this commit
            if let Ok(parent) = commit.parent(0) {
                let parent_tree = parent.tree().ok();
                let commit_tree = commit.tree().ok();

                if let (Some(pt), Some(ct)) = (parent_tree, commit_tree) {
                    if let Ok(diff) = self.repo.diff_tree_to_tree(Some(&pt), Some(&ct), None) {
                        for delta in diff.deltas() {
                            if let Some(path) = delta.new_file().path() {
                                files.insert(path.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(files.into_iter().collect())
    }
}

/// Information about a commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub hash: String,
    pub author: String,
    pub email: String,
    pub message: String,
    pub timestamp: i64,
}

/// Blame information for a range of lines
#[derive(Debug, Clone)]
pub struct BlameInfo {
    pub line_start: usize,
    pub line_count: usize,
    pub commit_hash: String,
    pub author: String,
    pub timestamp: i64,
}
