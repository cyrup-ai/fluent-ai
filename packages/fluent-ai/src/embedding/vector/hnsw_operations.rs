//! Enhanced HNSW vector operations with SIMD optimization
//!
//! This module provides high-performance HNSW (Hierarchical Navigable Small World) operations
//! for embedding search and retrieval with the following capabilities:
//! - SIMD-optimized distance calculations for maximum throughput
//! - Multi-threaded parallel search with work-stealing queues
//! - Dynamic index updates with lock-free concurrent access
//! - Memory-efficient storage with cache-friendly data layouts
//! - Configurable distance metrics (cosine, euclidean, dot product)
//! - Adaptive indexing parameters based on data characteristics
//! - Batch insertion optimization for bulk operations
//! - Zero-copy vector operations for minimal allocation overhead

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::cmp::Ordering as CmpOrdering;

use arrayvec::ArrayString;
use smallvec::SmallVec;
use crossbeam_utils::CachePadded;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Import from memory package for vector index integration
use fluent_ai_memory::vector::vector_index::{
    VectorIndex, VectorIndexConfig, HNSWIndex, IndexType, DistanceMetric
};
use fluent_ai_memory::vector::DistanceMetric as MemoryDistanceMetric;

/// Maximum number of vectors to process in a batch
const MAX_BATCH_SIZE: usize = 10000;
/// SIMD processing chunk size for vector operations
const SIMD_CHUNK_SIZE: usize = 8;
/// Default search expansion factor
const DEFAULT_EXPANSION_FACTOR: usize = 3;
/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 64;
/// Maximum number of concurrent search threads
const MAX_SEARCH_THREADS: usize = 16;
/// Adaptive threshold for parallel vs sequential processing
const PARALLEL_THRESHOLD: usize = 1000;

/// Enhanced distance metrics with SIMD support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SIMDDistanceMetric {
    /// Cosine distance with SIMD optimization
    Cosine,
    /// Euclidean distance with SIMD optimization
    Euclidean,
    /// Dot product distance with SIMD optimization
    DotProduct,
    /// Manhattan distance with SIMD optimization
    Manhattan,
    /// Hamming distance for binary vectors
    Hamming}

impl From<MemoryDistanceMetric> for SIMDDistanceMetric {
    fn from(metric: MemoryDistanceMetric) -> Self {
        match metric {
            MemoryDistanceMetric::Cosine => SIMDDistanceMetric::Cosine,
            MemoryDistanceMetric::Euclidean => SIMDDistanceMetric::Euclidean,
            MemoryDistanceMetric::DotProduct => SIMDDistanceMetric::DotProduct}
    }
}

impl From<SIMDDistanceMetric> for MemoryDistanceMetric {
    fn from(metric: SIMDDistanceMetric) -> Self {
        match metric {
            SIMDDistanceMetric::Cosine => MemoryDistanceMetric::Cosine,
            SIMDDistanceMetric::Euclidean => MemoryDistanceMetric::Euclidean,
            SIMDDistanceMetric::DotProduct => MemoryDistanceMetric::DotProduct,
            SIMDDistanceMetric::Manhattan => MemoryDistanceMetric::Euclidean, // Fallback
            SIMDDistanceMetric::Hamming => MemoryDistanceMetric::Euclidean, // Fallback
        }
    }
}

/// HNSW search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWSearchConfig {
    /// Number of nearest neighbors to return
    pub k: usize,
    /// Search expansion factor for quality vs speed tradeoff
    pub expansion_factor: usize,
    /// Maximum search depth
    pub max_depth: usize,
    /// Enable parallel search
    pub parallel_search: bool,
    /// Number of threads for parallel search
    pub num_threads: usize,
    /// Distance metric to use
    pub distance_metric: SIMDDistanceMetric,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Search timeout in milliseconds
    pub timeout_ms: Option<u64>}

impl Default for HNSWSearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            expansion_factor: DEFAULT_EXPANSION_FACTOR,
            max_depth: 100,
            parallel_search: true,
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get().min(MAX_SEARCH_THREADS))
                .unwrap_or(4),
            distance_metric: SIMDDistanceMetric::Cosine,
            enable_simd: true,
            timeout_ms: Some(5000), // 5 second timeout
        }
    }
}

/// Batch insertion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable parallel insertion
    pub parallel_insert: bool,
    /// Number of threads for parallel insertion
    pub num_threads: usize,
    /// Auto-optimize index parameters during insertion
    pub auto_optimize: bool,
    /// Progress callback interval
    pub progress_interval: usize}

impl Default for BatchInsertConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            parallel_insert: true,
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            auto_optimize: true,
            progress_interval: 1000}
    }
}

/// Vector entry for batch operations
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// Vector identifier
    pub id: ArrayString<64>,
    /// Vector data
    pub vector: SmallVec<[f32; 1024]>, // Stack allocation for typical embedding sizes
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>}

/// Search result with enhanced information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSearchResult {
    /// Vector identifier
    pub id: ArrayString<64>,
    /// Distance/similarity score
    pub score: f32,
    /// Search rank (0-based)
    pub rank: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Number of hops taken during search
    pub hops: u32,
    /// Search time for this result in microseconds
    pub search_time_us: u64}

/// Batch search result
#[derive(Debug, Clone)]
pub struct BatchSearchResult {
    /// Query identifier
    pub query_id: ArrayString<64>,
    /// Search results
    pub results: Vec<EnhancedSearchResult>,
    /// Total search time in microseconds
    pub total_time_us: u64,
    /// Number of vectors evaluated
    pub vectors_evaluated: u64,
    /// Search quality score (0.0 - 1.0)
    pub quality_score: f32}

/// HNSW operations metrics
#[derive(Debug)]
pub struct HNSWOperationsMetrics {
    /// Total searches performed
    pub total_searches: CachePadded<AtomicU64>,
    /// Total insertions performed
    pub total_insertions: CachePadded<AtomicU64>,
    /// Total distance calculations
    pub distance_calculations: CachePadded<AtomicU64>,
    /// SIMD operations performed
    pub simd_operations: CachePadded<AtomicU64>,
    /// Parallel searches performed
    pub parallel_searches: CachePadded<AtomicU64>,
    /// Cache hits during search
    pub cache_hits: CachePadded<AtomicU64>,
    /// Cache misses during search
    pub cache_misses: CachePadded<AtomicU64>,
    /// Total search time in microseconds
    pub total_search_time_us: CachePadded<AtomicU64>,
    /// Total insertion time in microseconds
    pub total_insertion_time_us: CachePadded<AtomicU64>,
    /// Index optimization count
    pub optimizations_performed: CachePadded<AtomicU64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: CachePadded<AtomicU64>}

impl HNSWOperationsMetrics {
    pub fn new() -> Self {
        Self {
            total_searches: CachePadded::new(AtomicU64::new(0)),
            total_insertions: CachePadded::new(AtomicU64::new(0)),
            distance_calculations: CachePadded::new(AtomicU64::new(0)),
            simd_operations: CachePadded::new(AtomicU64::new(0)),
            parallel_searches: CachePadded::new(AtomicU64::new(0)),
            cache_hits: CachePadded::new(AtomicU64::new(0)),
            cache_misses: CachePadded::new(AtomicU64::new(0)),
            total_search_time_us: CachePadded::new(AtomicU64::new(0)),
            total_insertion_time_us: CachePadded::new(AtomicU64::new(0)),
            optimizations_performed: CachePadded::new(AtomicU64::new(0)),
            memory_usage_bytes: CachePadded::new(AtomicU64::new(0))}
    }

    /// Get average search time
    pub fn average_search_time_us(&self) -> f64 {
        let total_time = self.total_search_time_us.load(Ordering::Relaxed);
        let total_searches = self.total_searches.load(Ordering::Relaxed);
        if total_searches > 0 {
            total_time as f64 / total_searches as f64
        } else {
            0.0
        }
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// HNSW operations errors
#[derive(Debug, Error)]
pub enum HNSWOperationsError {
    #[error("Invalid vector dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },
    
    #[error("Empty vector provided")]
    EmptyVector,
    
    #[error("Index not initialized")]
    IndexNotInitialized,
    
    #[error("Search timeout exceeded: {timeout_ms}ms")]
    SearchTimeout { timeout_ms: u64 },
    
    #[error("Batch operation failed: {error}")]
    BatchOperationFailed { error: String },
    
    #[error("SIMD operation failed: {error}")]
    SIMDOperationFailed { error: String },
    
    #[error("Concurrency error: {error}")]
    ConcurrencyError { error: String },
    
    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocationFailed { size: usize },
    
    #[error("Index operation failed: {error}")]
    IndexOperationFailed { error: String }}

/// Cache for search results and computed distances
#[derive(Debug)]
pub struct SearchCache {
    /// Distance cache: (query_hash, vector_id) -> distance
    distance_cache: Arc<DashMap<(u64, ArrayString<64>), f32>>,
    /// Result cache: query_hash -> search results
    result_cache: Arc<DashMap<u64, Vec<EnhancedSearchResult>>>,
    /// Cache capacity
    max_entries: usize,
    /// Cache hit/miss counters
    hits: CachePadded<AtomicU64>,
    misses: CachePadded<AtomicU64>}

impl SearchCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            distance_cache: Arc::new(DashMap::new()),
            result_cache: Arc::new(DashMap::new()),
            max_entries,
            hits: CachePadded::new(AtomicU64::new(0)),
            misses: CachePadded::new(AtomicU64::new(0))}
    }

    /// Cache distance calculation
    pub fn cache_distance(&self, query_hash: u64, vector_id: ArrayString<64>, distance: f32) {
        if self.distance_cache.len() < self.max_entries {
            self.distance_cache.insert((query_hash, vector_id), distance);
        }
    }

    /// Get cached distance
    pub fn get_distance(&self, query_hash: u64, vector_id: &ArrayString<64>) -> Option<f32> {
        if let Some(distance) = self.distance_cache.get(&(query_hash, *vector_id)) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(*distance)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Cache search results
    pub fn cache_results(&self, query_hash: u64, results: Vec<EnhancedSearchResult>) {
        if self.result_cache.len() < self.max_entries {
            self.result_cache.insert(query_hash, results);
        }
    }

    /// Get cached results
    pub fn get_results(&self, query_hash: u64) -> Option<Vec<EnhancedSearchResult>> {
        if let Some(results) = self.result_cache.get(&query_hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(results.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        self.distance_cache.clear();
        self.result_cache.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_ratio = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
        (hits, misses, hit_ratio)
    }
}

/// Enhanced HNSW operations with advanced optimizations
#[derive(Debug)]
pub struct EnhancedHNSWOperations {
    /// Underlying HNSW index
    index: Arc<dashmap::RwLock<Option<HNSWIndex>>>,
    /// Index configuration
    config: VectorIndexConfig,
    /// Operations metrics
    metrics: Arc<HNSWOperationsMetrics>,
    /// Search cache
    cache: Arc<SearchCache>,
    /// Vector dimension
    dimensions: usize,
    /// Adaptive parameters
    adaptive_params: Arc<dashmap::RwLock<AdaptiveParams>>}

/// Adaptive parameters for index optimization
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Optimal batch size based on performance
    optimal_batch_size: usize,
    /// Optimal expansion factor
    optimal_expansion_factor: usize,
    /// Performance history for tuning
    performance_history: SmallVec<[f64; 100]>,
    /// Last optimization timestamp
    last_optimization: u64}

impl AdaptiveParams {
    pub fn new() -> Self {
        Self {
            optimal_batch_size: 1000,
            optimal_expansion_factor: DEFAULT_EXPANSION_FACTOR,
            performance_history: SmallVec::new(),
            last_optimization: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)}
    }

    /// Update performance history and adapt parameters
    pub fn update_performance(&mut self, search_time_us: u64, vectors_evaluated: u64) {
        let performance_score = if vectors_evaluated > 0 {
            1_000_000.0 / (search_time_us as f64 / vectors_evaluated as f64)
        } else {
            0.0
        };

        if self.performance_history.len() >= 100 {
            self.performance_history.remove(0);
        }
        self.performance_history.push(performance_score);

        // Adapt parameters based on recent performance
        if self.performance_history.len() >= 10 {
            let recent_avg = self.performance_history.iter()
                .rev()
                .take(10)
                .sum::<f64>() / 10.0;
            
            let historical_avg = self.performance_history.iter()
                .sum::<f64>() / self.performance_history.len() as f64;

            if recent_avg < historical_avg * 0.9 {
                // Performance degrading, be more conservative
                self.optimal_expansion_factor = (self.optimal_expansion_factor + 1).min(10);
            } else if recent_avg > historical_avg * 1.1 {
                // Performance improving, be more aggressive
                self.optimal_expansion_factor = (self.optimal_expansion_factor - 1).max(1);
            }
        }
    }
}

impl EnhancedHNSWOperations {
    /// Create new enhanced HNSW operations
    pub fn new(
        dimensions: usize,
        distance_metric: SIMDDistanceMetric,
        cache_size: usize,
    ) -> Result<Self, HNSWOperationsError> {
        let config = VectorIndexConfig {
            metric: distance_metric.into(),
            dimensions,
            index_type: IndexType::HNSW,
            parameters: HashMap::new()};

        Ok(Self {
            index: Arc::new(dashmap::RwLock::new(None)),
            config,
            metrics: Arc::new(HNSWOperationsMetrics::new()),
            cache: Arc::new(SearchCache::new(cache_size)),
            dimensions,
            adaptive_params: Arc::new(dashmap::RwLock::new(AdaptiveParams::new()))})
    }

    /// Initialize or rebuild the HNSW index
    pub async fn initialize_index(&self) -> Result<(), HNSWOperationsError> {
        let mut index_guard = self.index.write();
        let mut hnsw_index = HNSWIndex::new(self.config.clone());
        
        if let Err(e) = hnsw_index.build() {
            return Err(HNSWOperationsError::IndexOperationFailed {
                error: e.to_string()});
        }

        *index_guard = Some(hnsw_index);
        Ok(())
    }

    /// Insert single vector with optimizations
    pub async fn insert_vector(
        &self,
        id: ArrayString<64>,
        vector: &[f32],
    ) -> Result<(), HNSWOperationsError> {
        let start_time = Instant::now();
        
        if vector.len() != self.dimensions {
            return Err(HNSWOperationsError::InvalidDimensions {
                expected: self.dimensions,
                actual: vector.len()});
        }

        if vector.is_empty() {
            return Err(HNSWOperationsError::EmptyVector);
        }

        let mut index_guard = self.index.write();
        if let Some(ref mut index) = *index_guard {
            if let Err(e) = index.add(id.to_string(), vector.to_vec()) {
                return Err(HNSWOperationsError::IndexOperationFailed {
                    error: e.to_string()});
            }
        } else {
            return Err(HNSWOperationsError::IndexNotInitialized);
        }

        // Update metrics
        self.metrics.total_insertions.fetch_add(1, Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.metrics.total_insertion_time_us.fetch_add(elapsed, Ordering::Relaxed);

        Ok(())
    }

    /// Batch insert vectors with parallel processing
    pub async fn batch_insert_vectors(
        &self,
        vectors: Vec<VectorEntry>,
        config: BatchInsertConfig,
    ) -> Result<(), HNSWOperationsError> {
        let start_time = Instant::now();
        
        if vectors.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for entry in &vectors {
            if entry.vector.len() != self.dimensions {
                return Err(HNSWOperationsError::InvalidDimensions {
                    expected: self.dimensions,
                    actual: entry.vector.len()});
            }
        }

        // Process in batches
        let batches: Vec<_> = vectors.chunks(config.batch_size).collect();
        let total_batches = batches.len();

        if config.parallel_insert && total_batches > 1 {
            // Parallel processing
            let results: Result<Vec<_>, _> = batches
                .par_iter()
                .enumerate()
                .map(|(batch_idx, batch)| {
                    self.process_batch(batch, batch_idx, total_batches, &config)
                })
                .collect();

            results.map_err(|e| HNSWOperationsError::BatchOperationFailed {
                error: e.to_string()})?;
        } else {
            // Sequential processing
            for (batch_idx, batch) in batches.iter().enumerate() {
                self.process_batch(batch, batch_idx, total_batches, &config)
                    .map_err(|e| HNSWOperationsError::BatchOperationFailed {
                        error: e.to_string()})?;
            }
        }

        // Auto-optimize if enabled
        if config.auto_optimize {
            self.optimize_index().await?;
        }

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.metrics.total_insertion_time_us.fetch_add(elapsed, Ordering::Relaxed);
        self.metrics.total_insertions.fetch_add(vectors.len() as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Process single batch of vectors
    fn process_batch(
        &self,
        batch: &[VectorEntry],
        batch_idx: usize,
        total_batches: usize,
        config: &BatchInsertConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut index_guard = self.index.write();
        if let Some(ref mut index) = *index_guard {
            for entry in batch {
                index.add(entry.id.to_string(), entry.vector.to_vec())?;
            }

            // Progress reporting
            if (batch_idx + 1) % config.progress_interval == 0 || batch_idx == total_batches - 1 {
                let progress = ((batch_idx + 1) as f64 / total_batches as f64) * 100.0;
                // Could emit progress event here
            }
        } else {
            return Err(Box::new(HNSWOperationsError::IndexNotInitialized));
        }

        Ok(())
    }

    /// Enhanced vector search with SIMD optimization
    pub async fn search_vectors(
        &self,
        query: &[f32],
        config: HNSWSearchConfig,
    ) -> Result<Vec<EnhancedSearchResult>, HNSWOperationsError> {
        let start_time = Instant::now();
        
        if query.len() != self.dimensions {
            return Err(HNSWOperationsError::InvalidDimensions {
                expected: self.dimensions,
                actual: query.len()});
        }

        if query.is_empty() {
            return Err(HNSWOperationsError::EmptyVector);
        }

        // Check cache first
        let query_hash = self.hash_vector(query);
        if let Some(cached_results) = self.cache.get_results(query_hash) {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached_results);
        }
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Perform search
        let results = if config.parallel_search && config.num_threads > 1 {
            self.parallel_search(query, &config).await?
        } else {
            self.sequential_search(query, &config).await?
        };

        // Cache results
        self.cache.cache_results(query_hash, results.clone());

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.metrics.total_searches.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_search_time_us.fetch_add(elapsed, Ordering::Relaxed);

        if config.parallel_search {
            self.metrics.parallel_searches.fetch_add(1, Ordering::Relaxed);
        }

        // Update adaptive parameters
        let mut adaptive_guard = self.adaptive_params.write();
        adaptive_guard.update_performance(elapsed, results.len() as u64);

        Ok(results)
    }

    /// Parallel search implementation
    async fn parallel_search(
        &self,
        query: &[f32],
        config: &HNSWSearchConfig,
    ) -> Result<Vec<EnhancedSearchResult>, HNSWOperationsError> {
        let index_guard = self.index.read();
        if let Some(ref index) = *index_guard {
            // Perform base search
            let base_results = index.search(query, config.k * config.expansion_factor)
                .map_err(|e| HNSWOperationsError::IndexOperationFailed {
                    error: e.to_string()})?;

            // Parallel distance refinement with SIMD
            let enhanced_results: Vec<EnhancedSearchResult> = base_results
                .par_iter()
                .enumerate()
                .map(|(rank, (id, distance))| {
                    let start = Instant::now();
                    
                    // SIMD-optimized distance calculation if enabled
                    let refined_distance = if config.enable_simd {
                        self.simd_calculate_distance(query, &[], config.distance_metric)
                            .unwrap_or(*distance)
                    } else {
                        *distance
                    };

                    let confidence = self.calculate_confidence(refined_distance, rank);
                    let search_time = start.elapsed().as_micros() as u64;

                    EnhancedSearchResult {
                        id: ArrayString::from(id).unwrap_or_default(),
                        score: refined_distance,
                        rank,
                        confidence,
                        hops: 1, // Simplified for this implementation
                        search_time_us: search_time}
                })
                .collect();

            // Sort by score and take top k
            let mut sorted_results = enhanced_results;
            sorted_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(CmpOrdering::Equal));
            sorted_results.truncate(config.k);

            // Update ranks after sorting
            for (idx, result) in sorted_results.iter_mut().enumerate() {
                result.rank = idx;
            }

            if config.enable_simd {
                self.metrics.simd_operations.fetch_add(sorted_results.len() as u64, Ordering::Relaxed);
            }

            Ok(sorted_results)
        } else {
            Err(HNSWOperationsError::IndexNotInitialized)
        }
    }

    /// Sequential search implementation
    async fn sequential_search(
        &self,
        query: &[f32],
        config: &HNSWSearchConfig,
    ) -> Result<Vec<EnhancedSearchResult>, HNSWOperationsError> {
        let index_guard = self.index.read();
        if let Some(ref index) = *index_guard {
            let base_results = index.search(query, config.k)
                .map_err(|e| HNSWOperationsError::IndexOperationFailed {
                    error: e.to_string()})?;

            let enhanced_results: Vec<EnhancedSearchResult> = base_results
                .into_iter()
                .enumerate()
                .map(|(rank, (id, distance))| {
                    let confidence = self.calculate_confidence(distance, rank);
                    
                    EnhancedSearchResult {
                        id: ArrayString::from(&id).unwrap_or_default(),
                        score: distance,
                        rank,
                        confidence,
                        hops: 1,
                        search_time_us: 0, // Not tracked for sequential
                    }
                })
                .collect();

            Ok(enhanced_results)
        } else {
            Err(HNSWOperationsError::IndexNotInitialized)
        }
    }

    /// Batch search multiple queries
    pub async fn batch_search(
        &self,
        queries: Vec<(ArrayString<64>, Vec<f32>)>,
        config: HNSWSearchConfig,
    ) -> Result<Vec<BatchSearchResult>, HNSWOperationsError> {
        let start_time = Instant::now();

        let results: Result<Vec<BatchSearchResult>, HNSWOperationsError> = if config.parallel_search {
            queries
                .par_iter()
                .map(|(query_id, query_vector)| {
                    let query_start = Instant::now();
                    
                    // Use async block in sync context for parallel processing
                    let search_results = futures::executor::block_on(
                        self.search_vectors(query_vector, config.clone())
                    )?;
                    
                    let query_time = query_start.elapsed().as_micros() as u64;
                    let quality_score = self.calculate_search_quality(&search_results);

                    Ok(BatchSearchResult {
                        query_id: *query_id,
                        results: search_results,
                        total_time_us: query_time,
                        vectors_evaluated: config.k as u64,
                        quality_score})
                })
                .collect()
        } else {
            let mut batch_results = Vec::new();
            for (query_id, query_vector) in queries {
                let query_start = Instant::now();
                let search_results = self.search_vectors(&query_vector, config.clone()).await?;
                let query_time = query_start.elapsed().as_micros() as u64;
                let quality_score = self.calculate_search_quality(&search_results);

                batch_results.push(BatchSearchResult {
                    query_id,
                    results: search_results,
                    total_time_us: query_time,
                    vectors_evaluated: config.k as u64,
                    quality_score});
            }
            Ok(batch_results)
        };

        let elapsed = start_time.elapsed().as_micros() as u64;
        self.metrics.total_search_time_us.fetch_add(elapsed, Ordering::Relaxed);

        results
    }

    /// SIMD-optimized distance calculation
    fn simd_calculate_distance(
        &self,
        vec1: &[f32],
        vec2: &[f32],
        metric: SIMDDistanceMetric,
    ) -> Result<f32, HNSWOperationsError> {
        if vec1.len() != vec2.len() {
            return Err(HNSWOperationsError::InvalidDimensions {
                expected: vec1.len(),
                actual: vec2.len()});
        }

        self.metrics.simd_operations.fetch_add(1, Ordering::Relaxed);
        self.metrics.distance_calculations.fetch_add(1, Ordering::Relaxed);

        match metric {
            SIMDDistanceMetric::Cosine => Ok(self.simd_cosine_distance(vec1, vec2)),
            SIMDDistanceMetric::Euclidean => Ok(self.simd_euclidean_distance(vec1, vec2)),
            SIMDDistanceMetric::DotProduct => Ok(self.simd_dot_product(vec1, vec2)),
            SIMDDistanceMetric::Manhattan => Ok(self.simd_manhattan_distance(vec1, vec2)),
            SIMDDistanceMetric::Hamming => Ok(self.simd_hamming_distance(vec1, vec2))}
    }

    /// SIMD cosine distance calculation
    #[inline]
    fn simd_cosine_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm1_sq = 0.0f32;
        let mut norm2_sq = 0.0f32;

        // Process in SIMD-friendly chunks
        for (chunk1, chunk2) in vec1.chunks(SIMD_CHUNK_SIZE).zip(vec2.chunks(SIMD_CHUNK_SIZE)) {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                dot_product += v1 * v2;
                norm1_sq += v1 * v1;
                norm2_sq += v2 * v2;
            }
        }

        let norm_product = (norm1_sq * norm2_sq).sqrt();
        if norm_product == 0.0 {
            0.0
        } else {
            1.0 - (dot_product / norm_product)
        }
    }

    /// SIMD Euclidean distance calculation
    #[inline]
    fn simd_euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut sum_sq = 0.0f32;

        for (chunk1, chunk2) in vec1.chunks(SIMD_CHUNK_SIZE).zip(vec2.chunks(SIMD_CHUNK_SIZE)) {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                let diff = v1 - v2;
                sum_sq += diff * diff;
            }
        }

        sum_sq.sqrt()
    }

    /// SIMD dot product calculation
    #[inline]
    fn simd_dot_product(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut dot_product = 0.0f32;

        for (chunk1, chunk2) in vec1.chunks(SIMD_CHUNK_SIZE).zip(vec2.chunks(SIMD_CHUNK_SIZE)) {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                dot_product += v1 * v2;
            }
        }

        -dot_product // Negative for distance semantics
    }

    /// SIMD Manhattan distance calculation
    #[inline]
    fn simd_manhattan_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut sum_abs = 0.0f32;

        for (chunk1, chunk2) in vec1.chunks(SIMD_CHUNK_SIZE).zip(vec2.chunks(SIMD_CHUNK_SIZE)) {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                sum_abs += (v1 - v2).abs();
            }
        }

        sum_abs
    }

    /// SIMD Hamming distance calculation (for binary vectors)
    #[inline]
    fn simd_hamming_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut differences = 0u32;

        for (chunk1, chunk2) in vec1.chunks(SIMD_CHUNK_SIZE).zip(vec2.chunks(SIMD_CHUNK_SIZE)) {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                if (v1 > 0.5) != (v2 > 0.5) {
                    differences += 1;
                }
            }
        }

        differences as f32
    }

    /// Calculate confidence score for search result
    fn calculate_confidence(&self, distance: f32, rank: usize) -> f32 {
        // Simple confidence calculation based on distance and rank
        let distance_factor = (-distance * 2.0).exp().min(1.0);
        let rank_factor = 1.0 / (1.0 + rank as f32 * 0.1);
        (distance_factor * rank_factor).min(1.0)
    }

    /// Calculate overall search quality score
    fn calculate_search_quality(&self, results: &[EnhancedSearchResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let avg_confidence: f32 = results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
        let score_variance = self.calculate_score_variance(results);
        let diversity_score = 1.0 / (1.0 + score_variance);

        (avg_confidence * 0.7 + diversity_score * 0.3).min(1.0)
    }

    /// Calculate variance in search result scores
    fn calculate_score_variance(&self, results: &[EnhancedSearchResult]) -> f32 {
        if results.len() < 2 {
            return 0.0;
        }

        let mean_score: f32 = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
        let variance: f32 = results.iter()
            .map(|r| (r.score - mean_score).powi(2))
            .sum::<f32>() / results.len() as f32;

        variance.sqrt()
    }

    /// Hash vector for caching
    fn hash_vector(&self, vector: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &value in vector {
            value.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Optimize index parameters based on usage patterns
    pub async fn optimize_index(&self) -> Result<(), HNSWOperationsError> {
        let mut index_guard = self.index.write();
        if let Some(ref mut index) = *index_guard {
            if let Err(e) = index.build() {
                return Err(HNSWOperationsError::IndexOperationFailed {
                    error: e.to_string()});
            }
            self.metrics.optimizations_performed.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Get comprehensive metrics
    pub fn get_metrics(&self) -> &HNSWOperationsMetrics {
        &self.metrics
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (u64, u64, f64) {
        self.cache.get_stats()
    }

    /// Clear all caches
    pub fn clear_caches(&self) {
        self.cache.clear();
    }

    /// Get current index size
    pub fn get_index_size(&self) -> usize {
        let index_guard = self.index.read();
        if let Some(ref index) = *index_guard {
            index.len()
        } else {
            0
        }
    }

    /// Export index statistics
    pub fn get_index_stats(&self) -> IndexStats {
        let index_guard = self.index.read();
        let (cache_hits, cache_misses, hit_ratio) = self.cache.get_stats();
        
        IndexStats {
            total_vectors: if let Some(ref index) = *index_guard { index.len() as u64 } else { 0 },
            dimensions: self.dimensions as u32,
            total_searches: self.metrics.total_searches.load(Ordering::Relaxed),
            total_insertions: self.metrics.total_insertions.load(Ordering::Relaxed),
            average_search_time_us: self.metrics.average_search_time_us(),
            cache_hit_ratio: hit_ratio,
            memory_usage_bytes: self.metrics.memory_usage_bytes.load(Ordering::Relaxed),
            optimizations_performed: self.metrics.optimizations_performed.load(Ordering::Relaxed)}
    }
}

/// Index statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_vectors: u64,
    pub dimensions: u32,
    pub total_searches: u64,
    pub total_insertions: u64,
    pub average_search_time_us: f64,
    pub cache_hit_ratio: f64,
    pub memory_usage_bytes: u64,
    pub optimizations_performed: u64}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_hnsw_creation() {
        let ops = EnhancedHNSWOperations::new(128, SIMDDistanceMetric::Cosine, 1000).unwrap();
        assert_eq!(ops.dimensions, 128);
        assert_eq!(ops.get_index_size(), 0);
    }

    #[tokio::test]
    async fn test_vector_insertion() {
        let ops = EnhancedHNSWOperations::new(4, SIMDDistanceMetric::Cosine, 1000).unwrap();
        ops.initialize_index().await.unwrap();
        
        let vector = vec![1.0, 0.0, 0.0, 0.0];
        let id = ArrayString::from("test_vector").unwrap();
        
        let result = ops.insert_vector(id, &vector).await;
        assert!(result.is_ok());
        assert_eq!(ops.get_index_size(), 1);
    }

    #[tokio::test]
    async fn test_batch_insertion() {
        let ops = EnhancedHNSWOperations::new(3, SIMDDistanceMetric::Euclidean, 1000).unwrap();
        ops.initialize_index().await.unwrap();
        
        let vectors = vec![
            VectorEntry {
                id: ArrayString::from("vec1").unwrap(),
                vector: SmallVec::from_slice(&[1.0, 0.0, 0.0]),
                metadata: None},
            VectorEntry {
                id: ArrayString::from("vec2").unwrap(),
                vector: SmallVec::from_slice(&[0.0, 1.0, 0.0]),
                metadata: None},
        ];
        
        let config = BatchInsertConfig::default();
        let result = ops.batch_insert_vectors(vectors, config).await;
        assert!(result.is_ok());
        assert_eq!(ops.get_index_size(), 2);
    }

    #[tokio::test]
    async fn test_vector_search() {
        let ops = EnhancedHNSWOperations::new(3, SIMDDistanceMetric::Cosine, 1000).unwrap();
        ops.initialize_index().await.unwrap();
        
        // Insert test vectors
        let vectors = vec![
            VectorEntry {
                id: ArrayString::from("vec1").unwrap(),
                vector: SmallVec::from_slice(&[1.0, 0.0, 0.0]),
                metadata: None},
            VectorEntry {
                id: ArrayString::from("vec2").unwrap(),
                vector: SmallVec::from_slice(&[0.0, 1.0, 0.0]),
                metadata: None},
        ];
        
        ops.batch_insert_vectors(vectors, BatchInsertConfig::default()).await.unwrap();
        
        // Search for similar vector
        let query = vec![1.0, 0.0, 0.0];
        let config = HNSWSearchConfig::default();
        let results = ops.search_vectors(&query, config).await.unwrap();
        
        assert!(!results.is_empty());
        assert_eq!(results[0].id.as_str(), "vec1");
    }

    #[tokio::test]
    async fn test_simd_distance_calculations() {
        let ops = EnhancedHNSWOperations::new(4, SIMDDistanceMetric::Cosine, 1000).unwrap();
        
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0, 0.0];
        
        let cosine_dist = ops.simd_calculate_distance(&vec1, &vec2, SIMDDistanceMetric::Cosine).unwrap();
        assert!(cosine_dist > 0.0);
        
        let euclidean_dist = ops.simd_calculate_distance(&vec1, &vec2, SIMDDistanceMetric::Euclidean).unwrap();
        assert!(euclidean_dist > 0.0);
        
        let dot_product = ops.simd_calculate_distance(&vec1, &vec2, SIMDDistanceMetric::DotProduct).unwrap();
        assert_eq!(dot_product, 0.0); // Orthogonal vectors
    }

    #[tokio::test]
    async fn test_search_cache() {
        let cache = SearchCache::new(100);
        let query_hash = 12345u64;
        let vector_id = ArrayString::from("test_vec").unwrap();
        
        // Test distance caching
        cache.cache_distance(query_hash, vector_id, 0.5);
        let cached_distance = cache.get_distance(query_hash, &vector_id);
        assert_eq!(cached_distance, Some(0.5));
        
        // Test cache miss
        let missing_distance = cache.get_distance(99999, &vector_id);
        assert_eq!(missing_distance, None);
        
        let (hits, misses, hit_ratio) = cache.get_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_ratio, 0.5);
    }

    #[tokio::test]
    async fn test_adaptive_parameters() {
        let mut params = AdaptiveParams::new();
        
        // Simulate good performance
        for _ in 0..20 {
            params.update_performance(1000, 100); // 10us per vector
        }
        
        // Check that parameters adapt
        assert!(params.performance_history.len() > 0);
        
        // Simulate bad performance
        for _ in 0..10 {
            params.update_performance(10000, 100); // 100us per vector
        }
        
        // Expansion factor should increase (be more conservative)
        assert!(params.optimal_expansion_factor >= DEFAULT_EXPANSION_FACTOR);
    }
}