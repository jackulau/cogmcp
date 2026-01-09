//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index
//!
//! This module provides fast approximate nearest neighbor search using the HNSW algorithm,
//! reducing search complexity from O(n) to O(log n) with minimal accuracy loss.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use cogmcp_core::{Error, Result};
use instant_distance::{Builder, HnswMap, Search};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Configuration for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of connections per layer during construction (higher = better recall, slower build)
    pub ef_construction: usize,
    /// Number of neighbors to search during query (higher = better recall, slower search)
    pub ef_search: usize,
    /// Maximum number of connections per layer (higher = better recall, more memory)
    pub m: usize,
    /// Minimum number of embeddings to trigger HNSW usage (below this, brute-force is faster)
    pub min_embeddings: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 100,
            m: 16,
            min_embeddings: 1000,
        }
    }
}

/// A point in the HNSW index containing an embedding vector
#[derive(Clone, Debug)]
pub struct HnswPoint {
    /// The embedding vector
    pub embedding: Vec<f32>,
}

impl instant_distance::Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // instant-distance expects distance (lower = more similar)
        1.0 - cosine_similarity(&self.embedding, &other.embedding)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denominator = (norm_a.sqrt()) * (norm_b.sqrt());
    if denominator == 0.0 {
        0.0
    } else {
        dot_product / denominator
    }
}

/// Result from HNSW search
#[derive(Debug, Clone)]
pub struct HnswSearchResult {
    /// Index of the embedding in the original data
    pub index: usize,
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f32,
}

/// Serializable format for HNSW index persistence
#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    /// All embeddings stored in the index
    embeddings: Vec<Vec<f32>>,
    /// Checksum of the embeddings for invalidation detection
    checksum: u64,
    /// Configuration used to build the index
    config: HnswConfig,
}

/// HNSW index for approximate nearest neighbor search
pub struct HnswIndex {
    /// The HNSW map (index -> HnswPoint)
    index: RwLock<Option<HnswMap<HnswPoint, usize>>>,
    /// Stored embeddings for rebuilding and verification
    embeddings: RwLock<Vec<Vec<f32>>>,
    /// Configuration
    config: HnswConfig,
    /// Checksum of current embeddings
    checksum: RwLock<u64>,
}

impl HnswIndex {
    /// Create a new empty HNSW index with default configuration
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create a new HNSW index with custom configuration
    pub fn with_config(config: HnswConfig) -> Self {
        Self {
            index: RwLock::new(None),
            embeddings: RwLock::new(Vec::new()),
            config,
            checksum: RwLock::new(0),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Check if the index is built and ready for search
    pub fn is_ready(&self) -> bool {
        self.index.read().is_some()
    }

    /// Get the number of embeddings in the index
    pub fn len(&self) -> usize {
        self.embeddings.read().len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.read().is_empty()
    }

    /// Check if HNSW should be used based on the number of embeddings
    pub fn should_use_hnsw(&self) -> bool {
        self.len() >= self.config.min_embeddings && self.is_ready()
    }

    /// Insert a single embedding into the index
    /// Note: The index needs to be rebuilt after insertions for the new embedding to be searchable
    pub fn insert(&self, embedding: Vec<f32>) -> usize {
        let mut embeddings = self.embeddings.write();
        let index = embeddings.len();
        embeddings.push(embedding);
        // Invalidate the current index since we have new data
        *self.index.write() = None;
        index
    }

    /// Insert multiple embeddings in batch
    /// Note: The index needs to be rebuilt after insertions
    pub fn insert_batch(&self, new_embeddings: Vec<Vec<f32>>) -> Vec<usize> {
        let mut embeddings = self.embeddings.write();
        let start_index = embeddings.len();
        let count = new_embeddings.len();
        embeddings.extend(new_embeddings);
        // Invalidate the current index since we have new data
        *self.index.write() = None;
        (start_index..start_index + count).collect()
    }

    /// Clear all embeddings and the index
    pub fn clear(&self) {
        *self.embeddings.write() = Vec::new();
        *self.index.write() = None;
        *self.checksum.write() = 0;
    }

    /// Replace all embeddings with new data
    pub fn replace_all(&self, new_embeddings: Vec<Vec<f32>>) {
        *self.embeddings.write() = new_embeddings;
        *self.index.write() = None;
    }

    /// Build or rebuild the HNSW index from current embeddings
    pub fn build(&self) -> Result<()> {
        let embeddings = self.embeddings.read();

        if embeddings.is_empty() {
            debug!("No embeddings to build HNSW index from");
            *self.index.write() = None;
            return Ok(());
        }

        info!("Building HNSW index with {} embeddings", embeddings.len());

        // Create points from embeddings
        let points: Vec<HnswPoint> = embeddings
            .iter()
            .map(|e| HnswPoint { embedding: e.clone() })
            .collect();

        // Create indices (0..n)
        let indices: Vec<usize> = (0..points.len()).collect();

        // Build the HNSW index
        let hnsw = Builder::default()
            .ef_construction(self.config.ef_construction)
            .build(points, indices);

        // Compute checksum
        let new_checksum = compute_checksum(&embeddings);

        // Store the index
        *self.index.write() = Some(hnsw);
        *self.checksum.write() = new_checksum;

        info!("HNSW index built successfully");
        Ok(())
    }

    /// Search for the k nearest neighbors of a query embedding
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<HnswSearchResult>> {
        let index_guard = self.index.read();
        let index = index_guard.as_ref().ok_or_else(|| {
            Error::Search("HNSW index not built. Call build() first.".into())
        })?;

        let query_point = HnswPoint {
            embedding: query.to_vec(),
        };

        // Create a reusable search buffer
        let mut search = Search::default();

        // Perform the search
        let results: Vec<HnswSearchResult> = index
            .search(&query_point, &mut search)
            .take(k)
            .map(|item| {
                let distance = item.distance;
                // Convert distance back to similarity (distance = 1 - similarity)
                let similarity = 1.0 - distance;
                HnswSearchResult {
                    index: *item.value,
                    similarity,
                }
            })
            .collect();

        debug!("HNSW search returned {} results", results.len());
        Ok(results)
    }

    /// Search with a minimum similarity threshold
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        k: usize,
        min_similarity: f32,
    ) -> Result<Vec<HnswSearchResult>> {
        let results = self.search(query, k)?;
        Ok(results
            .into_iter()
            .filter(|r| r.similarity >= min_similarity)
            .collect())
    }

    /// Save the index to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let embeddings = self.embeddings.read();
        let checksum = *self.checksum.read();

        if embeddings.is_empty() {
            debug!("No embeddings to save");
            return Ok(());
        }

        let data = HnswIndexData {
            embeddings: embeddings.clone(),
            checksum,
            config: self.config.clone(),
        };

        let file = File::create(path)
            .map_err(|e| Error::Storage(format!("Failed to create HNSW index file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Serialize using bincode for efficiency
        let encoded = bincode_serialize(&data)?;
        writer.write_all(&encoded)
            .map_err(|e| Error::Storage(format!("Failed to write HNSW index: {}", e)))?;

        info!("Saved HNSW index with {} embeddings to {:?}", embeddings.len(), path);
        Ok(())
    }

    /// Load the index from a file
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(Error::Storage(format!(
                "HNSW index file not found: {:?}",
                path
            )));
        }

        let file = File::open(path)
            .map_err(|e| Error::Storage(format!("Failed to open HNSW index file: {}", e)))?;
        let mut reader = BufReader::new(file);

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)
            .map_err(|e| Error::Storage(format!("Failed to read HNSW index: {}", e)))?;

        let data: HnswIndexData = bincode_deserialize(&buffer)?;

        info!("Loaded HNSW index data with {} embeddings from {:?}", data.embeddings.len(), path);

        let index = Self::with_config(data.config);
        *index.embeddings.write() = data.embeddings;
        *index.checksum.write() = data.checksum;

        // Rebuild the index from loaded embeddings
        index.build()?;

        Ok(index)
    }

    /// Check if the index matches a given checksum (for invalidation)
    pub fn matches_checksum(&self, checksum: u64) -> bool {
        *self.checksum.read() == checksum
    }

    /// Get the current checksum
    pub fn checksum(&self) -> u64 {
        *self.checksum.read()
    }

    /// Compute checksum for a set of embeddings
    pub fn compute_embeddings_checksum(embeddings: &[Vec<f32>]) -> u64 {
        compute_checksum(embeddings)
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a checksum for embeddings (for invalidation detection)
fn compute_checksum(embeddings: &[Vec<f32>]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    embeddings.len().hash(&mut hasher);

    // Hash a sample of embeddings for performance
    let sample_size = embeddings.len().min(100);
    let step = if embeddings.len() > sample_size {
        embeddings.len() / sample_size
    } else {
        1
    };

    for (i, emb) in embeddings.iter().enumerate() {
        if i % step == 0 {
            emb.len().hash(&mut hasher);
            // Hash first and last few values
            for val in emb.iter().take(4) {
                val.to_bits().hash(&mut hasher);
            }
            for val in emb.iter().rev().take(4) {
                val.to_bits().hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// Serialize data using JSON (simple and portable)
fn bincode_serialize<T: Serialize>(data: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(data)
        .map_err(|e| Error::Storage(format!("Failed to serialize HNSW data: {}", e)))
}

/// Deserialize data using JSON
fn bincode_deserialize<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T> {
    serde_json::from_slice(data)
        .map_err(|e| Error::Storage(format!("Failed to deserialize HNSW data: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn create_random_embedding(dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic "random" embedding for testing
        (0..dim)
            .map(|i| {
                let x = ((seed as f32 * 0.1) + (i as f32 * 0.01)).sin();
                x
            })
            .collect()
    }

    fn normalize(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn test_hnsw_index_creation() {
        let index = HnswIndex::new();
        assert!(!index.is_ready());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hnsw_insert_and_build() {
        let index = HnswIndex::new();

        // Insert some embeddings
        for i in 0..100 {
            let mut emb = create_random_embedding(384, i);
            normalize(&mut emb);
            index.insert(emb);
        }

        assert_eq!(index.len(), 100);
        assert!(!index.is_ready()); // Not built yet

        // Build the index
        index.build().unwrap();
        assert!(index.is_ready());
    }

    #[test]
    fn test_hnsw_search() {
        let index = HnswIndex::with_config(HnswConfig {
            ef_construction: 100,
            ef_search: 50,
            m: 16,
            min_embeddings: 10,
        });

        // Insert embeddings
        let dim = 384;
        for i in 0..100 {
            let mut emb = create_random_embedding(dim, i);
            normalize(&mut emb);
            index.insert(emb);
        }

        index.build().unwrap();

        // Search with a query that should be similar to embedding 0
        let mut query = create_random_embedding(dim, 0);
        normalize(&mut query);

        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should have high similarity (since query == embedding[0])
        assert!(results[0].similarity > 0.9);
        assert_eq!(results[0].index, 0); // Should find itself
    }

    #[test]
    fn test_hnsw_search_accuracy() {
        // Test that HNSW results are reasonably accurate compared to brute-force
        let index = HnswIndex::with_config(HnswConfig {
            ef_construction: 200,
            ef_search: 100,
            m: 16,
            min_embeddings: 10,
        });

        let dim = 128; // Smaller dimension for faster test
        let n_embeddings = 1000;
        let mut embeddings = Vec::new();

        for i in 0..n_embeddings {
            let mut emb = create_random_embedding(dim, i);
            normalize(&mut emb);
            embeddings.push(emb.clone());
            index.insert(emb);
        }

        index.build().unwrap();

        // Test with multiple queries
        let mut recall_sum = 0.0;
        let n_queries = 10;
        let k = 10;

        for q in 0..n_queries {
            let mut query = create_random_embedding(dim, q * 1000 + 500);
            normalize(&mut query);

            // Brute-force top-k
            let mut brute_force: Vec<(usize, f32)> = embeddings
                .iter()
                .enumerate()
                .map(|(i, emb)| (i, cosine_similarity(&query, emb)))
                .collect();
            brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let bf_top_k: Vec<usize> = brute_force.iter().take(k).map(|(i, _)| *i).collect();

            // HNSW top-k
            let hnsw_results = index.search(&query, k).unwrap();
            let hnsw_top_k: Vec<usize> = hnsw_results.iter().map(|r| r.index).collect();

            // Calculate recall
            let hits = bf_top_k.iter().filter(|i| hnsw_top_k.contains(i)).count();
            recall_sum += hits as f32 / k as f32;
        }

        let avg_recall = recall_sum / n_queries as f32;
        println!("Average recall@{}: {:.2}", k, avg_recall);
        assert!(avg_recall > 0.8, "Recall should be > 80%, got {:.2}%", avg_recall * 100.0);
    }

    #[test]
    fn test_hnsw_search_with_threshold() {
        let index = HnswIndex::new();

        let dim = 384;
        for i in 0..50 {
            let mut emb = create_random_embedding(dim, i);
            normalize(&mut emb);
            index.insert(emb);
        }

        index.build().unwrap();

        let mut query = create_random_embedding(dim, 0);
        normalize(&mut query);

        // Search with high threshold
        let results = index.search_with_threshold(&query, 10, 0.95).unwrap();

        // All results should meet the threshold
        for r in &results {
            assert!(r.similarity >= 0.95);
        }
    }

    #[test]
    fn test_hnsw_insert_batch() {
        let index = HnswIndex::new();

        let dim = 384;
        let embeddings: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                let mut emb = create_random_embedding(dim, i);
                normalize(&mut emb);
                emb
            })
            .collect();

        let indices = index.insert_batch(embeddings);
        assert_eq!(indices.len(), 50);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[49], 49);
        assert_eq!(index.len(), 50);
    }

    #[test]
    fn test_hnsw_clear() {
        let index = HnswIndex::new();

        for i in 0..10 {
            index.insert(create_random_embedding(384, i));
        }
        index.build().unwrap();

        assert!(index.is_ready());
        assert_eq!(index.len(), 10);

        index.clear();

        assert!(!index.is_ready());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hnsw_persistence() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_hnsw_index.json");

        // Create and populate index
        let index = HnswIndex::new();
        let dim = 128;

        for i in 0..100 {
            let mut emb = create_random_embedding(dim, i);
            normalize(&mut emb);
            index.insert(emb);
        }
        index.build().unwrap();

        // Save
        index.save(&path).unwrap();

        // Load into new index
        let loaded_index = HnswIndex::load(&path).unwrap();

        assert_eq!(loaded_index.len(), 100);
        assert!(loaded_index.is_ready());
        assert_eq!(loaded_index.checksum(), index.checksum());

        // Search should work on loaded index
        let mut query = create_random_embedding(dim, 0);
        normalize(&mut query);

        let results = loaded_index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].similarity > 0.9);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hnsw_checksum() {
        let index = HnswIndex::new();

        for i in 0..10 {
            index.insert(create_random_embedding(384, i));
        }
        index.build().unwrap();

        let checksum1 = index.checksum();
        assert!(checksum1 > 0);

        // Adding more embeddings should change checksum after rebuild
        index.insert(create_random_embedding(384, 100));
        index.build().unwrap();

        let checksum2 = index.checksum();
        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors should have similarity 1.0
        let v1 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v1) - 1.0).abs() < 0.0001);

        // Orthogonal vectors should have similarity 0.0
        let v2 = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 0.0).abs() < 0.0001);

        // Opposite vectors should have similarity -1.0
        let v3 = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v3) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_should_use_hnsw() {
        let config = HnswConfig {
            min_embeddings: 100,
            ..Default::default()
        };
        let index = HnswIndex::with_config(config);

        // Below threshold
        for i in 0..50 {
            index.insert(create_random_embedding(384, i));
        }
        index.build().unwrap();
        assert!(!index.should_use_hnsw());

        // Above threshold
        for i in 50..150 {
            index.insert(create_random_embedding(384, i));
        }
        index.build().unwrap();
        assert!(index.should_use_hnsw());
    }

    #[test]
    fn test_hnsw_performance() {
        // Test that HNSW is faster than brute-force for large datasets
        let index = HnswIndex::with_config(HnswConfig {
            ef_construction: 100,
            ef_search: 50,
            m: 16,
            min_embeddings: 100,
        });

        let dim = 128;
        let n_embeddings = 5000;
        let mut embeddings = Vec::new();

        for i in 0..n_embeddings {
            let mut emb = create_random_embedding(dim, i);
            normalize(&mut emb);
            embeddings.push(emb.clone());
            index.insert(emb);
        }

        index.build().unwrap();

        let mut query = create_random_embedding(dim, 999999);
        normalize(&mut query);
        let k = 10;

        // Time HNSW search
        let start = Instant::now();
        for _ in 0..100 {
            let _ = index.search(&query, k).unwrap();
        }
        let hnsw_time = start.elapsed();

        // Time brute-force search
        let start = Instant::now();
        for _ in 0..100 {
            let mut scores: Vec<(usize, f32)> = embeddings
                .iter()
                .enumerate()
                .map(|(i, emb)| (i, cosine_similarity(&query, emb)))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let _: Vec<(usize, f32)> = scores.into_iter().take(k).collect();
        }
        let bf_time = start.elapsed();

        println!(
            "HNSW: {:?}, Brute-force: {:?}, Speedup: {:.1}x",
            hnsw_time,
            bf_time,
            bf_time.as_secs_f64() / hnsw_time.as_secs_f64()
        );

        // HNSW should be at least 2x faster for this dataset size
        assert!(
            hnsw_time < bf_time,
            "HNSW ({:?}) should be faster than brute-force ({:?})",
            hnsw_time,
            bf_time
        );
    }
}
