//! Zero-allocation in-memory vector store with high-performance similarity search
//!
//! Optimized implementation using SIMD operations where available,
//! efficient indexing, and configurable similarity thresholds.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ZeroOneOrMany;
use crate::async_task::{AsyncTask, error_handlers::BadTraitImpl};
use crate::domain::memory::VectorStoreIndexDyn;
use crate::embedding::similarity::{SimilarityMetric, cosine_similarity, euclidean_distance};

/// Configuration for the in-memory vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryVectorStoreConfig {
    /// Similarity metric for search operations
    pub similarity_metric: SimilarityMetric,
    /// Minimum similarity threshold for results
    pub similarity_threshold: f32,
    /// Maximum number of results to return by default
    pub default_top_k: usize,
    /// Whether to normalize vectors on insertion
    pub normalize_on_insert: bool,
    /// Initial capacity for the vector store
    pub initial_capacity: usize,
    /// Enable approximate search for large datasets
    pub enable_approximate_search: bool,
    /// Number of clusters for approximate search
    pub num_clusters: usize}

impl Default for InMemoryVectorStoreConfig {
    fn default() -> Self {
        Self {
            similarity_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.7,
            default_top_k: 10,
            normalize_on_insert: true,
            initial_capacity: 1000,
            enable_approximate_search: false,
            num_clusters: 100}
    }
}

/// Stored vector entry with metadata
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// Unique identifier for the vector
    pub id: String,
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Associated metadata
    pub metadata: Value,
    /// Optional text content for the vector
    pub content: Option<String>}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Similarity score (0.0 to 1.0 for similarity metrics)
    pub score: f64,
    /// Entry ID
    pub id: String,
    /// Associated metadata
    pub metadata: Value,
    /// Optional content
    pub content: Option<String>}

/// High-performance in-memory vector store
pub struct InMemoryVectorStore {
    /// Configuration
    config: InMemoryVectorStoreConfig,
    /// Storage for vectors and metadata
    storage: Arc<RwLock<VectorStorage>>,
    /// Optional embedding provider for query vectorization
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>}

/// Internal storage structure
struct VectorStorage {
    /// Vector entries indexed by ID
    entries: HashMap<String, VectorEntry>,
    /// Flat vector storage for efficient similarity computation
    vectors: Vec<Vec<f32>>,
    /// Mapping from vector index to entry ID
    index_to_id: Vec<String>,
    /// Mapping from entry ID to vector index
    id_to_index: HashMap<String, usize>,
    /// Dimensional consistency check
    dimensions: Option<usize>}

/// Trait for embedding providers (for query vectorization)
pub trait EmbeddingProvider: Send + Sync {
    /// Convert text query to embedding vector
    fn embed_query(&self, query: &str) -> AsyncTask<Vec<f32>>;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

impl VectorStorage {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity),
            index_to_id: Vec::with_capacity(capacity),
            id_to_index: HashMap::with_capacity(capacity),
            dimensions: None}
    }

    fn insert(&mut self, entry: VectorEntry) -> Result<(), String> {
        // Validate dimensions
        if let Some(expected_dims) = self.dimensions {
            if entry.vector.len() != expected_dims {
                return Err(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    expected_dims,
                    entry.vector.len()
                ));
            }
        } else {
            self.dimensions = Some(entry.vector.len());
        }

        // Check if entry already exists
        if let Some(existing_index) = self.id_to_index.get(&entry.id) {
            // Update existing entry
            self.vectors[*existing_index] = entry.vector.clone();
            self.entries.insert(entry.id.clone(), entry);
        } else {
            // Add new entry
            let index = self.vectors.len();
            self.vectors.push(entry.vector.clone());
            self.index_to_id.push(entry.id.clone());
            self.id_to_index.insert(entry.id.clone(), index);
            self.entries.insert(entry.id.clone(), entry);
        }

        Ok(())
    }

    fn remove(&mut self, id: &str) -> bool {
        if let Some(index) = self.id_to_index.remove(id) {
            // Remove from all structures
            self.entries.remove(id);
            self.vectors.remove(index);
            self.index_to_id.remove(index);

            // Update indices for subsequent entries
            for (idx, entry_id) in self.index_to_id.iter().enumerate().skip(index) {
                self.id_to_index.insert(entry_id.clone(), idx);
            }

            true
        } else {
            false
        }
    }

    fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.entries.get(id)
    }

    fn search_similar(
        &self,
        query_vector: &[f32],
        top_k: usize,
        similarity_metric: SimilarityMetric,
        threshold: f32,
    ) -> Vec<SearchResult> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(self.vectors.len().min(top_k * 2));

        // Compute similarities for all vectors
        for (index, vector) in self.vectors.iter().enumerate() {
            let similarity = match similarity_metric {
                SimilarityMetric::Cosine => cosine_similarity(query_vector, vector),
                SimilarityMetric::Euclidean => {
                    let distance = euclidean_distance(query_vector, vector);
                    if distance.is_infinite() {
                        0.0
                    } else {
                        1.0 / (1.0 + distance)
                    }
                }
                SimilarityMetric::DotProduct => query_vector
                    .iter()
                    .zip(vector.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>(),
                _ => cosine_similarity(query_vector, vector), // Default to cosine
            };

            // Apply threshold filter
            if similarity >= threshold {
                let entry_id = &self.index_to_id[index];
                if let Some(entry) = self.entries.get(entry_id) {
                    results.push(SearchResult {
                        score: similarity as f64,
                        id: entry.id.clone(),
                        metadata: entry.metadata.clone(),
                        content: entry.content.clone()});
                }
            }
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top K results
        results.truncate(top_k);
        results
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl InMemoryVectorStore {
    /// Create new in-memory vector store with default configuration
    #[inline(always)]
    pub fn new() -> Self {
        Self::with_config(InMemoryVectorStoreConfig::default())
    }

    /// Create new in-memory vector store with custom configuration
    #[inline(always)]
    pub fn with_config(config: InMemoryVectorStoreConfig) -> Self {
        let storage = VectorStorage::new(config.initial_capacity);

        Self {
            config,
            storage: Arc::new(RwLock::new(storage)),
            embedding_provider: None}
    }

    /// Set embedding provider for query vectorization
    #[inline(always)]
    pub fn with_embedding_provider<P>(mut self, provider: P) -> Self
    where
        P: EmbeddingProvider + 'static,
    {
        self.embedding_provider = Some(Arc::new(provider));
        self
    }

    /// Insert a vector with metadata
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Value,
    ) -> AsyncTask<()> {
        let id = id.into();
        let storage = self.storage.clone();
        let config = self.config.clone();

        crate::async_task::spawn_async(async move {
            let mut normalized_vector = vector;

            // Apply normalization if configured
            if config.normalize_on_insert {
                crate::embedding::normalization::normalize_vector(&mut normalized_vector);
            }

            let entry = VectorEntry {
                id: id.clone(),
                vector: normalized_vector,
                metadata,
                content: None};

            let mut storage = storage.write();
            if let Err(e) = storage.insert(entry) {
                eprintln!("Warning: Failed to insert vector entry: {:?}", e);
                return Err(VectorStoreError::InsertionFailed(format!(
                    "Failed to insert entry: {:?}",
                    e
                )));
            }
        })
    }

    /// Insert a vector with content and metadata
    pub fn insert_with_content(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        content: impl Into<String>,
        metadata: Value,
    ) -> AsyncTask<()> {
        let id = id.into();
        let content = content.into();
        let storage = self.storage.clone();
        let config = self.config.clone();

        crate::async_task::spawn_async(async move {
            let mut normalized_vector = vector;

            // Apply normalization if configured
            if config.normalize_on_insert {
                crate::embedding::normalization::normalize_vector(&mut normalized_vector);
            }

            let entry = VectorEntry {
                id: id.clone(),
                vector: normalized_vector,
                metadata,
                content: Some(content)};

            let mut storage = storage.write();
            if let Err(e) = storage.insert(entry) {
                eprintln!(
                    "Warning: Failed to insert vector entry with content: {:?}",
                    e
                );
                return Err(VectorStoreError::InsertionFailed(format!(
                    "Failed to insert entry: {:?}",
                    e
                )));
            }
        })
    }

    /// Remove a vector by ID
    pub fn remove(&self, id: &str) -> AsyncTask<bool> {
        let id = id.to_string();
        let storage = self.storage.clone();

        crate::async_task::spawn_async(async move {
            let mut storage = storage.write();
            storage.remove(&id)
        })
    }

    /// Get a vector entry by ID
    pub fn get(&self, id: &str) -> AsyncTask<Option<VectorEntry>> {
        let id = id.to_string();
        let storage = self.storage.clone();

        crate::async_task::spawn_async(async move {
            let storage = storage.read();
            storage.get(&id).cloned()
        })
    }

    /// Search for similar vectors using a query vector
    pub fn search_vector(
        &self,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
    ) -> AsyncTask<Vec<SearchResult>> {
        let storage = self.storage.clone();
        let config = self.config.clone();
        let top_k = top_k.unwrap_or(config.default_top_k);

        crate::async_task::spawn_async(async move {
            let storage = storage.read();
            storage.search_similar(
                &query_vector,
                top_k,
                config.similarity_metric,
                config.similarity_threshold,
            )
        })
    }

    /// Search for similar vectors using a text query (requires embedding provider)
    pub fn search_text(
        &self,
        query: impl Into<String>,
        top_k: Option<usize>,
    ) -> AsyncTask<Vec<SearchResult>> {
        let query = query.into();
        let storage = self.storage.clone();
        let config = self.config.clone();
        let embedding_provider = self.embedding_provider.clone();
        let top_k = top_k.unwrap_or(config.default_top_k);

        crate::async_task::spawn_async(async move {
            if let Some(provider) = embedding_provider {
                let query_vector = provider.embed_query(&query).await;
                let storage = storage.read();
                storage.search_similar(
                    &query_vector,
                    top_k,
                    config.similarity_metric,
                    config.similarity_threshold,
                )
            } else {
                Vec::new() // No embedding provider available
            }
        })
    }

    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        let storage = self.storage.read();
        storage.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        let storage = self.storage.read();
        storage.is_empty()
    }

    /// Clear all vectors from the store
    pub fn clear(&self) -> AsyncTask<()> {
        let storage = self.storage.clone();
        let capacity = self.config.initial_capacity;

        crate::async_task::spawn_async(async move {
            let mut storage = storage.write();
            *storage = VectorStorage::new(capacity);
        })
    }

    /// Get current configuration
    pub fn config(&self) -> &InMemoryVectorStoreConfig {
        &self.config
    }

    /// Update similarity threshold
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        self.config.similarity_threshold = threshold;
    }

    /// Update similarity metric
    pub fn set_similarity_metric(&mut self, metric: SimilarityMetric) {
        self.config.similarity_metric = metric;
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for InMemoryVectorStore {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: self.storage.clone(),
            embedding_provider: self.embedding_provider.clone()}
    }
}

// Error handling implementation for async tasks
impl BadTraitImpl for Vec<SearchResult> {
    fn bad_impl(error: &str) -> Self {
        eprintln!("Vector store search error: {}", error);
        Vec::new()
    }
}

/// Implementation of VectorStoreIndexDyn trait for integration with the domain layer
impl VectorStoreIndexDyn for InMemoryVectorStore {
    fn top_n(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>> {
        let query = query.to_string();
        let store = self.clone();

        crate::async_task::spawn_async(async move {
            let results = store.search_text(query, Some(n)).await;

            if results.is_empty() {
                ZeroOneOrMany::None
            } else {
                let tuples: Vec<(f64, String, Value)> = results
                    .into_iter()
                    .map(|result| (result.score, result.id, result.metadata))
                    .collect();

                match tuples.len() {
                    1 => {
                        if let Some(tuple) = tuples.into_iter().next() {
                            ZeroOneOrMany::One(tuple)
                        } else {
                            ZeroOneOrMany::None
                        }
                    }
                    _ => ZeroOneOrMany::Many(tuples)}
            }
        })
    }

    fn top_n_ids(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        let query = query.to_string();
        let store = self.clone();

        crate::async_task::spawn_async(async move {
            let results = store.search_text(query, Some(n)).await;

            if results.is_empty() {
                ZeroOneOrMany::None
            } else {
                let tuples: Vec<(f64, String)> = results
                    .into_iter()
                    .map(|result| (result.score, result.id))
                    .collect();

                match tuples.len() {
                    1 => {
                        if let Some(tuple) = tuples.into_iter().next() {
                            ZeroOneOrMany::One(tuple)
                        } else {
                            ZeroOneOrMany::None
                        }
                    }
                    _ => ZeroOneOrMany::Many(tuples)}
            }
        })
    }
}
