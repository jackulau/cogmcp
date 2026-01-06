//! Dependency parsing for various package managers

use contextmcp_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A project dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: Option<String>,
    pub dep_type: DependencyType,
    pub source: PackageSource,
}

/// Type of dependency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    Runtime,
    Development,
    Build,
    Optional,
    Peer,
}

/// Package source/registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PackageSource {
    Cargo,
    Npm,
    PyPi,
    Go,
    Unknown,
}

/// Parses dependencies from various package manager files
pub struct DependencyParser;

impl DependencyParser {
    /// Parse dependencies from a project directory
    pub fn parse_project(root: &Path) -> Result<Vec<Dependency>> {
        let mut deps = Vec::new();

        // Try Cargo.toml (Rust)
        let cargo_path = root.join("Cargo.toml");
        if cargo_path.exists() {
            deps.extend(Self::parse_cargo(&cargo_path)?);
        }

        // Try package.json (Node.js)
        let npm_path = root.join("package.json");
        if npm_path.exists() {
            deps.extend(Self::parse_package_json(&npm_path)?);
        }

        // Try requirements.txt (Python)
        let pip_path = root.join("requirements.txt");
        if pip_path.exists() {
            deps.extend(Self::parse_requirements_txt(&pip_path)?);
        }

        // Try pyproject.toml (Python)
        let pyproject_path = root.join("pyproject.toml");
        if pyproject_path.exists() {
            deps.extend(Self::parse_pyproject(&pyproject_path)?);
        }

        Ok(deps)
    }

    fn parse_cargo(path: &Path) -> Result<Vec<Dependency>> {
        let content = fs::read_to_string(path)?;
        let manifest: cargo_toml::Manifest = cargo_toml::Manifest::from_str(&content)
            .map_err(|e| Error::Parse(format!("Failed to parse Cargo.toml: {}", e)))?;

        let mut deps = Vec::new();

        for (name, dep) in manifest.dependencies {
            let version = match &dep {
                cargo_toml::Dependency::Simple(v) => Some(v.clone()),
                cargo_toml::Dependency::Detailed(d) => d.version.clone(),
                cargo_toml::Dependency::Inherited(_) => None,
            };

            deps.push(Dependency {
                name,
                version,
                dep_type: DependencyType::Runtime,
                source: PackageSource::Cargo,
            });
        }

        for (name, dep) in manifest.dev_dependencies {
            let version = match &dep {
                cargo_toml::Dependency::Simple(v) => Some(v.clone()),
                cargo_toml::Dependency::Detailed(d) => d.version.clone(),
                cargo_toml::Dependency::Inherited(_) => None,
            };

            deps.push(Dependency {
                name,
                version,
                dep_type: DependencyType::Development,
                source: PackageSource::Cargo,
            });
        }

        for (name, dep) in manifest.build_dependencies {
            let version = match &dep {
                cargo_toml::Dependency::Simple(v) => Some(v.clone()),
                cargo_toml::Dependency::Detailed(d) => d.version.clone(),
                cargo_toml::Dependency::Inherited(_) => None,
            };

            deps.push(Dependency {
                name,
                version,
                dep_type: DependencyType::Build,
                source: PackageSource::Cargo,
            });
        }

        Ok(deps)
    }

    fn parse_package_json(path: &Path) -> Result<Vec<Dependency>> {
        let content = fs::read_to_string(path)?;
        let pkg: serde_json::Value = serde_json::from_str(&content)?;

        let mut deps = Vec::new();

        if let Some(dependencies) = pkg.get("dependencies").and_then(|d| d.as_object()) {
            for (name, version) in dependencies {
                deps.push(Dependency {
                    name: name.clone(),
                    version: version.as_str().map(|s| s.to_string()),
                    dep_type: DependencyType::Runtime,
                    source: PackageSource::Npm,
                });
            }
        }

        if let Some(dev_deps) = pkg.get("devDependencies").and_then(|d| d.as_object()) {
            for (name, version) in dev_deps {
                deps.push(Dependency {
                    name: name.clone(),
                    version: version.as_str().map(|s| s.to_string()),
                    dep_type: DependencyType::Development,
                    source: PackageSource::Npm,
                });
            }
        }

        if let Some(peer_deps) = pkg.get("peerDependencies").and_then(|d| d.as_object()) {
            for (name, version) in peer_deps {
                deps.push(Dependency {
                    name: name.clone(),
                    version: version.as_str().map(|s| s.to_string()),
                    dep_type: DependencyType::Peer,
                    source: PackageSource::Npm,
                });
            }
        }

        Ok(deps)
    }

    fn parse_requirements_txt(path: &Path) -> Result<Vec<Dependency>> {
        let content = fs::read_to_string(path)?;
        let mut deps = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
                continue;
            }

            // Parse requirement line (e.g., "requests>=2.28.0" or "flask==2.0.1")
            let (name, version) = if let Some(pos) = line.find(|c| c == '=' || c == '>' || c == '<' || c == '!' || c == '~') {
                (line[..pos].to_string(), Some(line[pos..].to_string()))
            } else {
                (line.to_string(), None)
            };

            deps.push(Dependency {
                name,
                version,
                dep_type: DependencyType::Runtime,
                source: PackageSource::PyPi,
            });
        }

        Ok(deps)
    }

    fn parse_pyproject(path: &Path) -> Result<Vec<Dependency>> {
        let content = fs::read_to_string(path)?;
        let pyproject: toml::Value = toml::from_str(&content)?;

        let mut deps = Vec::new();

        // Try [project.dependencies]
        if let Some(project_deps) = pyproject
            .get("project")
            .and_then(|p| p.get("dependencies"))
            .and_then(|d| d.as_array())
        {
            for dep in project_deps {
                if let Some(dep_str) = dep.as_str() {
                    let (name, version) = Self::parse_pep508(dep_str);
                    deps.push(Dependency {
                        name,
                        version,
                        dep_type: DependencyType::Runtime,
                        source: PackageSource::PyPi,
                    });
                }
            }
        }

        // Try [tool.poetry.dependencies]
        if let Some(poetry_deps) = pyproject
            .get("tool")
            .and_then(|t| t.get("poetry"))
            .and_then(|p| p.get("dependencies"))
            .and_then(|d| d.as_table())
        {
            for (name, version) in poetry_deps {
                if name == "python" {
                    continue;
                }
                let version_str = match version {
                    toml::Value::String(s) => Some(s.clone()),
                    _ => None,
                };
                deps.push(Dependency {
                    name: name.clone(),
                    version: version_str,
                    dep_type: DependencyType::Runtime,
                    source: PackageSource::PyPi,
                });
            }
        }

        Ok(deps)
    }

    fn parse_pep508(spec: &str) -> (String, Option<String>) {
        // Simple PEP 508 parsing (e.g., "requests>=2.28.0")
        let spec = spec.trim();
        if let Some(pos) = spec.find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
            (spec[..pos].to_string(), Some(spec[pos..].trim().to_string()))
        } else {
            (spec.to_string(), None)
        }
    }
}
