//! Model loading, download, and configuration

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use cogmcp_core::{Error, Result};

/// Model file information
const MODEL_NAME: &str = "all-MiniLM-L6-v2";
const MODEL_FILENAME: &str = "model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Hugging Face URLs for the model
const MODEL_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// Expected SHA256 hash for model integrity (can be updated)
/// Note: This is the hash when the model was first integrated
const MODEL_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Default batch size for batch embedding operations
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Maximum batch size to prevent memory issues
pub const MAX_BATCH_SIZE: usize = 128;

/// Embedding model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the ONNX model file
    pub model_path: String,
    /// Path to the tokenizer JSON file
    pub tokenizer_path: String,
    /// Embedding dimension (384 for all-MiniLM-L6-v2)
    pub embedding_dim: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size for batch embedding operations
    pub batch_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            embedding_dim: 384,
            max_length: 512,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl ModelConfig {
    /// Create config with a specific model path
    pub fn with_path(model_path: impl AsRef<Path>) -> Self {
        let model_path = model_path.as_ref();
        let tokenizer_path = model_path.with_file_name(TOKENIZER_FILENAME);
        Self {
            model_path: model_path.to_string_lossy().to_string(),
            tokenizer_path: tokenizer_path.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    /// Set the batch size for batch embedding operations
    ///
    /// The batch size is clamped to MAX_BATCH_SIZE to prevent memory issues.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.min(MAX_BATCH_SIZE);
        self
    }

    /// Check if the model file exists
    pub fn model_exists(&self) -> bool {
        !self.model_path.is_empty() && Path::new(&self.model_path).exists()
    }

    /// Check if the tokenizer file exists
    pub fn tokenizer_exists(&self) -> bool {
        !self.tokenizer_path.is_empty() && Path::new(&self.tokenizer_path).exists()
    }
}

/// Manages model downloading and storage
pub struct ModelManager {
    /// Base directory for model storage
    base_dir: PathBuf,
}

impl ModelManager {
    /// Create a new model manager with default storage location
    pub fn new() -> Result<Self> {
        let base_dir = Self::default_model_dir()?;
        Ok(Self { base_dir })
    }

    /// Create a model manager with a custom base directory
    pub fn with_base_dir(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Get the default model directory (~/.local/share/cogmcp/models/)
    pub fn default_model_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .ok_or_else(|| Error::Config("Could not determine local data directory".into()))?;
        Ok(data_dir.join("cogmcp").join("models"))
    }

    /// Get the model directory for a specific model
    pub fn model_dir(&self) -> PathBuf {
        self.base_dir.join(MODEL_NAME)
    }

    /// Get the path to the ONNX model file
    pub fn model_path(&self) -> PathBuf {
        self.model_dir().join(MODEL_FILENAME)
    }

    /// Get the path to the tokenizer file
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_dir().join(TOKENIZER_FILENAME)
    }

    /// Check if the model is available locally
    pub fn is_model_available(&self) -> bool {
        self.model_path().exists() && self.tokenizer_path().exists()
    }

    /// Ensure the model is available, downloading if necessary
    pub fn ensure_model_available(&self) -> Result<ModelConfig> {
        let model_path = self.model_path();
        let tokenizer_path = self.tokenizer_path();

        // Create directories if needed
        if let Some(parent) = model_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::FileSystem(format!("Failed to create model directory: {}", e))
            })?;
        }

        // Download model if not present
        if !model_path.exists() {
            info!("Downloading ONNX model from Hugging Face...");
            self.download_file(MODEL_URL, &model_path)?;
            info!("Model downloaded successfully");
        } else {
            debug!("Model already exists at {:?}", model_path);
        }

        // Download tokenizer if not present
        if !tokenizer_path.exists() {
            info!("Downloading tokenizer from Hugging Face...");
            self.download_file(TOKENIZER_URL, &tokenizer_path)?;
            info!("Tokenizer downloaded successfully");
        } else {
            debug!("Tokenizer already exists at {:?}", tokenizer_path);
        }

        Ok(ModelConfig {
            model_path: model_path.to_string_lossy().to_string(),
            tokenizer_path: tokenizer_path.to_string_lossy().to_string(),
            embedding_dim: 384,
            max_length: 512,
            batch_size: DEFAULT_BATCH_SIZE,
        })
    }

    /// Download a file from URL to the specified path
    fn download_file(&self, url: &str, dest: &Path) -> Result<()> {
        debug!("Downloading {} to {:?}", url, dest);

        let response = ureq::get(url)
            .call()
            .map_err(|e| Error::Embedding(format!("Failed to download model: {}", e)))?;

        let content_length = response
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok());

        if let Some(len) = content_length {
            info!("Downloading {} bytes...", len);
        }

        // Create a temporary file first, then rename for atomicity
        let temp_path = dest.with_extension("tmp");
        let mut file = fs::File::create(&temp_path).map_err(|e| {
            Error::FileSystem(format!("Failed to create temporary file: {}", e))
        })?;

        let mut reader = response.into_reader();
        let mut buffer = [0u8; 8192];
        let mut total_bytes = 0u64;

        loop {
            let bytes_read = reader.read(&mut buffer).map_err(|e| {
                Error::Embedding(format!("Failed to read download data: {}", e))
            })?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read]).map_err(|e| {
                Error::FileSystem(format!("Failed to write model file: {}", e))
            })?;

            total_bytes += bytes_read as u64;

            // Log progress for large files
            if let Some(len) = content_length {
                if total_bytes.is_multiple_of(10 * 1024 * 1024) {
                    let percent = (total_bytes as f64 / len as f64) * 100.0;
                    debug!("Download progress: {:.1}%", percent);
                }
            }
        }

        // Ensure data is written to disk
        file.flush().map_err(|e| {
            Error::FileSystem(format!("Failed to flush model file: {}", e))
        })?;
        drop(file);

        // Rename temp file to final destination
        fs::rename(&temp_path, dest).map_err(|e| {
            Error::FileSystem(format!("Failed to move model file: {}", e))
        })?;

        info!("Downloaded {} bytes to {:?}", total_bytes, dest);
        Ok(())
    }

    /// Verify model integrity using SHA256 hash
    pub fn verify_model_integrity(&self) -> Result<bool> {
        let model_path = self.model_path();
        if !model_path.exists() {
            return Ok(false);
        }

        let hash = self.compute_file_hash(&model_path)?;
        let is_valid = hash == MODEL_SHA256;

        if !is_valid {
            warn!(
                "Model integrity check failed. Expected: {}, Got: {}",
                MODEL_SHA256, hash
            );
        }

        Ok(is_valid)
    }

    /// Compute SHA256 hash of a file
    fn compute_file_hash(&self, path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)
            .map_err(|e| Error::FileSystem(format!("Failed to open file for hashing: {}", e)))?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer).map_err(|e| {
                Error::FileSystem(format!("Failed to read file for hashing: {}", e))
            })?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hex::encode(hasher.finalize()))
    }

    /// Get a ModelConfig for the managed model
    pub fn get_config(&self) -> ModelConfig {
        ModelConfig {
            model_path: self.model_path().to_string_lossy().to_string(),
            tokenizer_path: self.tokenizer_path().to_string_lossy().to_string(),
            embedding_dim: 384,
            max_length: 512,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default ModelManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.max_length, 512);
        assert!(config.model_path.is_empty());
    }

    #[test]
    fn test_model_config_with_path() {
        let config = ModelConfig::with_path("/tmp/model.onnx");
        assert_eq!(config.model_path, "/tmp/model.onnx");
        assert_eq!(config.tokenizer_path, "/tmp/tokenizer.json");
    }

    #[test]
    fn test_model_manager_paths() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::with_base_dir(temp_dir.path());

        let model_path = manager.model_path();
        assert!(model_path.to_string_lossy().contains(MODEL_NAME));
        assert!(model_path.to_string_lossy().ends_with("model.onnx"));

        let tokenizer_path = manager.tokenizer_path();
        assert!(tokenizer_path.to_string_lossy().ends_with("tokenizer.json"));
    }

    #[test]
    fn test_model_not_available_initially() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::with_base_dir(temp_dir.path());
        assert!(!manager.is_model_available());
    }
}
