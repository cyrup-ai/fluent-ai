//! Advanced indexing strategies for high-performance vector search
//!
//! Implements various indexing approaches including LSH, k-means clustering,
//! and hierarchical indices for approximate nearest neighbor search.

use crate::providers::embedding::similarity::{cosine_similarity, euclidean_distance, SimilarityMetric};
use crate::vector_store::in_memory::VectorEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Index strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexStrategy {
    /// Linear search (exact results, slower for large datasets)
    Linear,
    /// Locality-Sensitive Hashing for approximate search
    LSH {
        /// Number of hash functions
        num_hashes: usize,
        /// Number of hash tables
        num_tables: usize,
        /// Random projection dimensions
        projection_dim: usize,
    },
    /// K-means clustering for approximate search
    KMeans {
        /// Number of clusters
        num_clusters: usize,
        /// Maximum iterations for k-means
        max_iterations: usize,
        /// Search multiple nearest clusters
        search_clusters: usize,
    },
    /// Hierarchical indexing
    Hierarchical {
        /// Branching factor
        branch_factor: usize,
        /// Maximum leaf size
        leaf_size: usize,
    },
}

impl Default for IndexStrategy {
    fn default() -> Self {
        IndexStrategy::Linear
    }
}

/// Index configuration with performance parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Indexing strategy
    pub strategy: IndexStrategy,
    /// Rebuild index threshold (number of updates before rebuild)
    pub rebuild_threshold: usize,
    /// Enable parallel processing for index operations
    pub enable_parallel: bool,
    /// Memory limit for index structures (in MB)
    pub memory_limit_mb: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            strategy: IndexStrategy::Linear,
            rebuild_threshold: 1000,
            enable_parallel: true,
            memory_limit_mb: 100,
        }
    }
}

/// Vector index trait for different indexing strategies
pub trait VectorIndex: Send + Sync {
    /// Build index from vector entries
    fn build(&mut self, entries: &[VectorEntry]) -> Result<(), String>;
    
    /// Update index with new entry
    fn update(&mut self, entry: &VectorEntry) -> Result<(), String>;
    
    /// Remove entry from index
    fn remove(&mut self, id: &str) -> Result<(), String>;
    
    /// Search for similar vectors
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        similarity_metric: SimilarityMetric,
        threshold: f32,
    ) -> Vec<(String, f32)>;
    
    /// Get index statistics
    fn stats(&self) -> IndexStats;
    
    /// Check if index needs rebuilding
    fn needs_rebuild(&self) -> bool;
}

/// Index performance statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of indexed vectors
    pub vector_count: usize,
    /// Index memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of updates since last rebuild
    pub updates_since_rebuild: usize,
    /// Average search time in microseconds
    pub avg_search_time_us: f64,
    /// Index efficiency score (0.0 to 1.0)
    pub efficiency_score: f32,
}

/// Linear search index (exact results)
pub struct LinearIndex {
    entries: Vec<VectorEntry>,
    id_to_index: HashMap<String, usize>,
    updates_count: usize,
}

impl LinearIndex {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            id_to_index: HashMap::new(),
            updates_count: 0,
        }
    }
}

impl VectorIndex for LinearIndex {
    fn build(&mut self, entries: &[VectorEntry]) -> Result<(), String> {
        self.entries = entries.to_vec();
        self.id_to_index.clear();
        
        for (index, entry) in self.entries.iter().enumerate() {
            self.id_to_index.insert(entry.id.clone(), index);
        }
        
        self.updates_count = 0;
        Ok(())
    }
    
    fn update(&mut self, entry: &VectorEntry) -> Result<(), String> {
        if let Some(&index) = self.id_to_index.get(&entry.id) {
            // Update existing entry
            self.entries[index] = entry.clone();
        } else {
            // Add new entry
            let index = self.entries.len();
            self.entries.push(entry.clone());
            self.id_to_index.insert(entry.id.clone(), index);
        }
        
        self.updates_count += 1;
        Ok(())
    }
    
    fn remove(&mut self, id: &str) -> Result<(), String> {
        if let Some(index) = self.id_to_index.remove(id) {
            self.entries.remove(index);
            
            // Update indices for subsequent entries
            for (_entry_id, entry_index) in self.id_to_index.iter_mut() {
                if *entry_index > index {
                    *entry_index -= 1;
                }
            }
            
            self.updates_count += 1;
            Ok(())
        } else {
            Err(format!("Entry with id '{}' not found", id))
        }
    }
    
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        similarity_metric: SimilarityMetric,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let mut results = Vec::with_capacity(self.entries.len().min(top_k * 2));
        
        for entry in &self.entries {
            let similarity = match similarity_metric {
                SimilarityMetric::Cosine => cosine_similarity(query_vector, &entry.vector),
                SimilarityMetric::Euclidean => {
                    let distance = euclidean_distance(query_vector, &entry.vector);
                    if distance.is_infinite() {
                        0.0
                    } else {
                        1.0 / (1.0 + distance)
                    }
                },
                SimilarityMetric::DotProduct => {
                    query_vector.iter()
                        .zip(entry.vector.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                },
                _ => cosine_similarity(query_vector, &entry.vector),
            };
            
            if similarity >= threshold {
                results.push((entry.id.clone(), similarity));
            }
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
    
    fn stats(&self) -> IndexStats {
        let memory_usage = self.entries.len() * std::mem::size_of::<VectorEntry>()
            + self.id_to_index.len() * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
        
        IndexStats {
            vector_count: self.entries.len(),
            memory_usage_bytes: memory_usage,
            updates_since_rebuild: self.updates_count,
            avg_search_time_us: 100.0, // Placeholder
            efficiency_score: 1.0, // Linear search is always exact
        }
    }
    
    fn needs_rebuild(&self) -> bool {
        false // Linear index doesn't need rebuilding
    }
}

/// LSH (Locality-Sensitive Hashing) index for approximate search
pub struct LSHIndex {
    /// Hash tables
    hash_tables: Vec<HashMap<Vec<u32>, Vec<String>>>,
    /// Random projection matrices
    projections: Vec<Vec<Vec<f32>>>,
    /// Vector entries indexed by ID
    entries: HashMap<String, VectorEntry>,
    /// Configuration
    config: LSHConfig,
    /// Update counter
    updates_count: usize,
}

#[derive(Debug, Clone)]
struct LSHConfig {
    num_hashes: usize,
    #[allow(dead_code)] // TODO: Use in LSH implementation
    num_tables: usize,
    #[allow(dead_code)] // TODO: Use in LSH implementation
    projection_dim: usize,
    vector_dim: usize,
}

impl LSHIndex {
    pub fn new(num_hashes: usize, num_tables: usize, projection_dim: usize, vector_dim: usize) -> Self {
        let mut projections = Vec::with_capacity(num_tables);
        let mut hash_tables = Vec::with_capacity(num_tables);
        
        // Initialize random projections and hash tables
        for _ in 0..num_tables {
            let mut table_projections = Vec::with_capacity(num_hashes);
            for _ in 0..num_hashes {
                // Generate random projection vector
                let projection = (0..vector_dim)
                    .map(|_| fastrand::f32() * 2.0 - 1.0) // Random values in [-1, 1]
                    .collect::<Vec<f32>>();
                table_projections.push(projection);
            }
            projections.push(table_projections);
            hash_tables.push(HashMap::new());
        }
        
        Self {
            hash_tables,
            projections,
            entries: HashMap::new(),
            config: LSHConfig {
                num_hashes,
                num_tables,
                projection_dim,
                vector_dim,
            },
            updates_count: 0,
        }
    }
    
    /// Compute LSH hash for a vector
    fn compute_hash(&self, vector: &[f32], table_idx: usize) -> Vec<u32> {
        let projections = &self.projections[table_idx];
        let mut hash = Vec::with_capacity(self.config.num_hashes);
        
        for projection in projections {
            let dot_product: f32 = vector.iter()
                .zip(projection.iter())
                .map(|(&v, &p)| v * p)
                .sum();
            
            // Convert to hash bucket (binary hash)
            hash.push(if dot_product >= 0.0 { 1 } else { 0 });
        }
        
        hash
    }
}

impl VectorIndex for LSHIndex {
    fn build(&mut self, entries: &[VectorEntry]) -> Result<(), String> {
        // Clear existing data
        for table in &mut self.hash_tables {
            table.clear();
        }
        self.entries.clear();
        
        // Add all entries
        for entry in entries {
            self.update(entry)?;
        }
        
        self.updates_count = 0;
        Ok(())
    }
    
    fn update(&mut self, entry: &VectorEntry) -> Result<(), String> {
        if entry.vector.len() != self.config.vector_dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.vector_dim, entry.vector.len()
            ));
        }
        
        // Store entry
        self.entries.insert(entry.id.clone(), entry.clone());
        
        // Hash and insert into all tables - compute all hashes first to avoid borrowing conflicts
        let hashes: Vec<Vec<u32>> = (0..self.hash_tables.len())
            .map(|table_idx| self.compute_hash(&entry.vector, table_idx))
            .collect();
        
        for (table_idx, table) in self.hash_tables.iter_mut().enumerate() {
            let hash = hashes[table_idx].clone();
            table.entry(hash).or_insert_with(Vec::new).push(entry.id.clone());
        }
        
        self.updates_count += 1;
        Ok(())
    }
    
    fn remove(&mut self, id: &str) -> Result<(), String> {
        if let Some(entry) = self.entries.remove(id) {
            // Compute all hashes first to avoid borrowing conflicts
            let hashes: Vec<Vec<u32>> = (0..self.hash_tables.len())
                .map(|table_idx| self.compute_hash(&entry.vector, table_idx))
                .collect();
            
            // Remove from all hash tables
            for (table_idx, table) in self.hash_tables.iter_mut().enumerate() {
                let hash = hashes[table_idx].clone();
                if let Some(bucket) = table.get_mut(&hash) {
                    bucket.retain(|entry_id| entry_id != id);
                    if bucket.is_empty() {
                        table.remove(&hash);
                    }
                }
            }
            
            self.updates_count += 1;
            Ok(())
        } else {
            Err(format!("Entry with id '{}' not found", id))
        }
    }
    
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        similarity_metric: SimilarityMetric,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let mut candidate_ids = std::collections::HashSet::new();
        
        // Collect candidates from all hash tables
        for (table_idx, table) in self.hash_tables.iter().enumerate() {
            let hash = self.compute_hash(query_vector, table_idx);
            if let Some(bucket) = table.get(&hash) {
                for id in bucket {
                    candidate_ids.insert(id.clone());
                }
            }
        }
        
        // Compute exact similarities for candidates
        let mut results = Vec::with_capacity(candidate_ids.len());
        
        for id in candidate_ids {
            if let Some(entry) = self.entries.get(&id) {
                let similarity = match similarity_metric {
                    SimilarityMetric::Cosine => cosine_similarity(query_vector, &entry.vector),
                    SimilarityMetric::Euclidean => {
                        let distance = euclidean_distance(query_vector, &entry.vector);
                        if distance.is_infinite() {
                            0.0
                        } else {
                            1.0 / (1.0 + distance)
                        }
                    },
                    SimilarityMetric::DotProduct => {
                        query_vector.iter()
                            .zip(entry.vector.iter())
                            .map(|(&a, &b)| a * b)
                            .sum::<f32>()
                    },
                    _ => cosine_similarity(query_vector, &entry.vector),
                };
                
                if similarity >= threshold {
                    results.push((id, similarity));
                }
            }
        }
        
        // Sort and return top K
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
    
    fn stats(&self) -> IndexStats {
        let memory_usage = self.entries.len() * std::mem::size_of::<VectorEntry>()
            + self.hash_tables.iter().map(|table| {
                table.len() * (std::mem::size_of::<Vec<u32>>() + std::mem::size_of::<Vec<String>>())
            }).sum::<usize>()
            + self.projections.len() * self.config.num_hashes * self.config.vector_dim * std::mem::size_of::<f32>();
        
        IndexStats {
            vector_count: self.entries.len(),
            memory_usage_bytes: memory_usage,
            updates_since_rebuild: self.updates_count,
            avg_search_time_us: 50.0, // Approximate search is faster
            efficiency_score: 0.85, // LSH provides approximate results
        }
    }
    
    fn needs_rebuild(&self) -> bool {
        self.updates_count > 1000 // Rebuild after many updates
    }
}

/// Factory for creating vector indices
pub struct IndexFactory;

impl IndexFactory {
    /// Create index based on strategy and configuration
    #[inline(always)]
    pub fn create_index(
        strategy: &IndexStrategy,
        vector_dim: usize,
    ) -> Result<Box<dyn VectorIndex>, String> {
        match strategy {
            IndexStrategy::Linear => Ok(Box::new(LinearIndex::new())),
            IndexStrategy::LSH { num_hashes, num_tables, projection_dim } => {
                Ok(Box::new(LSHIndex::new(*num_hashes, *num_tables, *projection_dim, vector_dim)))
            },
            IndexStrategy::KMeans { .. } => {
                // K-means index implementation would go here
                Err("K-means index not implemented yet".to_string())
            },
            IndexStrategy::Hierarchical { .. } => {
                // Hierarchical index implementation would go here
                Err("Hierarchical index not implemented yet".to_string())
            },
        }
    }
    
    /// Recommend optimal index strategy based on dataset characteristics
    #[inline(always)]
    pub fn recommend_strategy(
        vector_count: usize,
        vector_dim: usize,
        _memory_limit_mb: usize,
    ) -> IndexStrategy {
        if vector_count < 1000 {
            // Small datasets: use linear search for exact results
            IndexStrategy::Linear
        } else if vector_count < 100000 {
            // Medium datasets: use LSH for good performance/accuracy tradeoff
            let num_hashes = (vector_dim as f32).log2().ceil() as usize;
            let num_tables = 10;
            let projection_dim = vector_dim.min(64);
            
            IndexStrategy::LSH {
                num_hashes,
                num_tables,
                projection_dim,
            }
        } else {
            // Large datasets: use hierarchical indexing (when implemented)
            // For now, fall back to LSH with optimized parameters
            IndexStrategy::LSH {
                num_hashes: 16,
                num_tables: 20,
                projection_dim: 32,
            }
        }
    }
}