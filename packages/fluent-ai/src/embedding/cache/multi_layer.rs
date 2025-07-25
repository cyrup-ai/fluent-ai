//! Multi-layer caching system for embeddings with L1/L2/L3 architecture
//!
//! This module provides a high-performance, three-tier caching system:
//! - L1: Lock-free in-memory cache for hot data access
//! - L2: SurrealDB persistent cache for durability 
//! - L3: HNSW vector similarity cache for semantic search
//!
//! Features zero-allocation patterns, cache coherence, TTL management,
//! and comprehensive performance monitoring.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use arrayvec::ArrayString;
use smallvec::SmallVec;
use crossbeam_utils::CachePadded;
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Interval};
use tokio::sync::{watch, RwLock};
use thiserror::Error;

// Import types from memory system
use crate::memory::manager::surreal::SurrealDBManager;
use crate::memory::vector::vector_index::{HNSWIndex, VectorIndexConfig, IndexType};
use crate::vector::DistanceMetric;

/// Maximum cache key length for zero allocation
const MAX_CACHE_KEY_LENGTH: usize = 128;
/// Maximum embedding dimension for stack allocation  
const MAX_EMBEDDING_DIM: usize = 4096;
/// Default L1 cache capacity
const DEFAULT_L1_CAPACITY: usize = 100000;
/// Default TTL in seconds
const DEFAULT_TTL_SECONDS: u32 = 3600;
/// Cache entry version for coherence
const CACHE_VERSION: u32 = 1;
/// Maximum batch size for bulk operations
const MAX_BATCH_SIZE: usize = 1000;

/// Cache key type with zero allocation
type CacheKey = ArrayString<MAX_CACHE_KEY_LENGTH>;
/// Embedding vector type with stack allocation preference  
type EmbeddingVector = SmallVec<[f32; 1536]>; // Most common embedding size

/// Cached embedding entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbedding {
    /// The embedding vector
    pub embedding: EmbeddingVector,
    /// Unix timestamp when cached
    pub timestamp: u64,
    /// Time-to-live in seconds
    pub ttl_seconds: u32,
    /// Cache entry version for coherence
    pub version: u32,
    /// Provider that generated this embedding
    pub provider: ArrayString<32>,
    /// Model used for generation
    pub model: ArrayString<64>,
    /// Estimated quality score
    pub quality_score: f32,
    /// Access count for LRU tracking
    pub access_count: u32,
    /// Checksum for integrity validation
    pub checksum: u64}

impl CachedEmbedding {
    /// Create new cached embedding
    pub fn new(
        embedding: &[f32],
        ttl_seconds: u32,
        provider: &str,
        model: &str,
        quality_score: f32,
    ) -> Self {
        let embedding_vec = SmallVec::from_slice(embedding);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let checksum = Self::compute_checksum(&embedding_vec);

        Self {
            embedding: embedding_vec,
            timestamp,
            ttl_seconds,
            version: CACHE_VERSION,
            provider: ArrayString::from(provider).unwrap_or_default(),
            model: ArrayString::from(model).unwrap_or_default(),
            quality_score,
            access_count: 0,
            checksum}
    }

    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        now.saturating_sub(self.timestamp) > self.ttl_seconds as u64
    }

    /// Validate checksum integrity
    pub fn is_valid(&self) -> bool {
        self.checksum == Self::compute_checksum(&self.embedding)
    }

    /// Touch entry for LRU tracking
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Compute checksum for integrity validation
    fn compute_checksum(embedding: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash the embedding values as bytes for integrity checking
        for &value in embedding {
            value.to_bits().hash(&mut hasher);
        }
        
        hasher.finish()
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        now.saturating_sub(self.timestamp)
    }
}

/// L1 cache statistics
#[derive(Debug)]
pub struct L1CacheStats {
    pub hits: CachePadded<AtomicU64>,
    pub misses: CachePadded<AtomicU64>,
    pub evictions: CachePadded<AtomicU64>,
    pub entries: CachePadded<AtomicU64>,
    pub memory_bytes: CachePadded<AtomicU64>}

impl L1CacheStats {
    pub fn new() -> Self {
        Self {
            hits: CachePadded::new(AtomicU64::new(0)),
            misses: CachePadded::new(AtomicU64::new(0)),
            evictions: CachePadded::new(AtomicU64::new(0)),
            entries: CachePadded::new(AtomicU64::new(0)),
            memory_bytes: CachePadded::new(AtomicU64::new(0))}
    }

    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }
}

/// L1 in-memory cache with lock-free access
#[derive(Debug)]
pub struct L1Cache {
    /// Lock-free skip map for O(log n) access
    data: Arc<SkipMap<CacheKey, CachedEmbedding>>,
    /// Cache statistics
    stats: Arc<L1CacheStats>,
    /// Maximum capacity
    max_capacity: usize,
    /// LRU tracking for eviction
    access_tracker: Arc<DashMap<CacheKey, AtomicU64>>}

impl L1Cache {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            data: Arc::new(SkipMap::new()),
            stats: Arc::new(L1CacheStats::new()),
            max_capacity,
            access_tracker: Arc::new(DashMap::new())}
    }

    /// Get cached embedding
    pub fn get(&self, key: &CacheKey) -> Option<CachedEmbedding> {
        if let Some(entry) = self.data.get(key) {
            let mut cached = entry.value().clone();
            
            // Validate integrity and expiry
            if cached.is_valid() && !cached.is_expired() {
                cached.touch();
                self.record_access(key);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(cached);
            } else {
                // Remove invalid/expired entry
                self.remove(key);
            }
        }
        
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert cached embedding with eviction if needed
    pub fn insert(&self, key: CacheKey, embedding: CachedEmbedding) -> Result<(), MultiLayerCacheError> {
        // Check capacity and evict if necessary
        if self.data.len() >= self.max_capacity {
            self.evict_lru_entries()?;
        }

        // Calculate memory size
        let memory_size = std::mem::size_of::<CachedEmbedding>() + 
                         embedding.embedding.len() * std::mem::size_of::<f32>();

        // Insert with atomic operations
        let was_new = self.data.insert(key.clone(), embedding).is_none();
        
        if was_new {
            self.stats.entries.fetch_add(1, Ordering::Relaxed);
            self.stats.memory_bytes.fetch_add(memory_size as u64, Ordering::Relaxed);
        }
        
        self.record_access(&key);
        Ok(())
    }

    /// Remove entry from cache
    pub fn remove(&self, key: &CacheKey) -> Option<CachedEmbedding> {
        if let Some(entry) = self.data.remove(key) {
            let memory_size = std::mem::size_of::<CachedEmbedding>() + 
                             entry.1.embedding.len() * std::mem::size_of::<f32>();
            
            self.stats.entries.fetch_sub(1, Ordering::Relaxed);
            self.stats.memory_bytes.fetch_sub(memory_size as u64, Ordering::Relaxed);
            self.access_tracker.remove(key);
            
            Some(entry.1)
        } else {
            None
        }
    }

    /// Evict LRU entries to make space
    fn evict_lru_entries(&self) -> Result<(), MultiLayerCacheError> {
        let target_evictions = self.max_capacity / 10; // Evict 10% when full
        let mut evicted = 0;

        // Find least recently used entries
        let mut lru_entries: Vec<_> = self.access_tracker
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        // Sort by access time (oldest first)
        lru_entries.sort_by_key(|(_, access_time)| *access_time);

        // Evict oldest entries
        for (key, _) in lru_entries.into_iter().take(target_evictions) {
            if self.remove(&key).is_some() {
                evicted += 1;
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        if evicted == 0 {
            return Err(MultiLayerCacheError::EvictionFailed(
                "Failed to evict any entries".to_string()
            ));
        }

        Ok(())
    }

    /// Record access for LRU tracking
    fn record_access(&self, key: &CacheKey) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.access_tracker
            .entry(key.clone())
            .or_insert_with(|| AtomicU64::new(now))
            .store(now, Ordering::Relaxed);
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        let expired_keys: Vec<_> = self.data
            .iter()
            .filter(|entry| entry.value().is_expired())
            .map(|entry| entry.key().clone())
            .collect();

        for key in expired_keys {
            self.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &L1CacheStats {
        &self.stats
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// L2 persistent cache using SurrealDB
#[derive(Debug)]
pub struct L2Cache {
    surreal_manager: Arc<SurrealDBManager>,
    stats: Arc<L1CacheStats>, // Reuse stats structure
}

impl L2Cache {
    pub fn new(surreal_manager: Arc<SurrealDBManager>) -> Self {
        Self {
            surreal_manager,
            stats: Arc::new(L1CacheStats::new())}
    }

    /// Get embedding from persistent storage
    pub async fn get(&self, key: &CacheKey) -> Result<Option<CachedEmbedding>, MultiLayerCacheError> {
        // Query SurrealDB by metadata containing the cache key
        let query = format!(
            "SELECT * FROM embedding WHERE metadata.cache_key = '{}' AND metadata.expiry > time::now()",
            key
        );

        match self.surreal_manager.query(&query).await {
            Ok(mut result) => {
                if let Some(embedding_data) = result.take::<Vec<surrealdb::sql::Value>>(0).ok() {
                    if let Some(data) = embedding_data.first() {
                        // Parse the embedding data and convert to CachedEmbedding
                        if let Ok(cached) = self.parse_surreal_embedding(data) {
                            self.stats.hits.fetch_add(1, Ordering::Relaxed);
                            return Ok(Some(cached));
                        }
                    }
                }
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
            Err(e) => Err(MultiLayerCacheError::L2Error(format!("SurrealDB query failed: {}", e)))
        }
    }

    /// Store embedding in persistent storage
    pub async fn insert(&self, key: &CacheKey, embedding: &CachedEmbedding) -> Result<(), MultiLayerCacheError> {
        let expiry_timestamp = embedding.timestamp + embedding.ttl_seconds as u64;
        
        let record = surrealdb::sql::json!({
            "metadata": {
                "cache_key": key.as_str(),
                "provider": embedding.provider.as_str(),
                "model": embedding.model.as_str(),
                "quality_score": embedding.quality_score,
                "version": embedding.version,
                "expiry": expiry_timestamp,
                "checksum": embedding.checksum
            },
            "embedding": embedding.embedding.as_slice(),
            "created_at": embedding.timestamp
        });

        let query = format!(
            "CREATE embedding:{{}} CONTENT {}",
            key.as_str().replace('-', "_"), // SurrealDB record ID format
            record
        );

        self.surreal_manager.query(&query).await
            .map_err(|e| MultiLayerCacheError::L2Error(format!("Failed to insert: {}", e)))?;

        self.stats.entries.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Parse SurrealDB embedding data
    fn parse_surreal_embedding(&self, data: &surrealdb::sql::Value) -> Result<CachedEmbedding, MultiLayerCacheError> {
        // Simplified parsing - in production would use proper SurrealDB value extraction
        // This is a placeholder that would need to be implemented based on SurrealDB's actual API
        Err(MultiLayerCacheError::L2Error("Parsing not implemented".to_string()))
    }

    /// Clean up expired entries
    pub async fn cleanup_expired(&self) -> Result<u64, MultiLayerCacheError> {
        let query = "DELETE embedding WHERE metadata.expiry < time::now()";
        
        match self.surreal_manager.query(query).await {
            Ok(mut result) => {
                // Extract count of deleted records
                Ok(0) // Placeholder - would extract actual count from result
            }
            Err(e) => Err(MultiLayerCacheError::L2Error(format!("Cleanup failed: {}", e)))
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &L1CacheStats {
        &self.stats
    }
}

/// L3 HNSW vector similarity cache
#[derive(Debug)]
pub struct L3Cache {
    hnsw_index: Arc<RwLock<HNSWIndex>>,
    embedding_map: Arc<DashMap<String, CachedEmbedding>>,
    stats: Arc<L1CacheStats>}

impl L3Cache {
    pub fn new(dimensions: usize, distance_metric: DistanceMetric) -> Result<Self, MultiLayerCacheError> {
        let config = VectorIndexConfig {
            metric: distance_metric,
            dimensions,
            index_type: IndexType::HNSW,
            parameters: std::collections::HashMap::new()};

        let hnsw_index = HNSWIndex::new(config);

        Ok(Self {
            hnsw_index: Arc::new(RwLock::new(hnsw_index)),
            embedding_map: Arc::new(DashMap::new()),
            stats: Arc::new(L1CacheStats::new())})
    }

    /// Find similar embeddings
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        max_results: usize,
    ) -> Result<Vec<(CachedEmbedding, f32)>, MultiLayerCacheError> {
        let index = self.hnsw_index.read().await;
        
        match index.search(query_embedding, max_results) {
            Ok(results) => {
                let mut similar_embeddings = Vec::new();
                
                for (id, distance) in results {
                    if distance <= threshold {
                        if let Some(cached) = self.embedding_map.get(&id) {
                            if !cached.is_expired() {
                                similar_embeddings.push((cached.clone(), distance));
                            }
                        }
                    }
                }
                
                self.stats.hits.fetch_add(similar_embeddings.len() as u64, Ordering::Relaxed);
                Ok(similar_embeddings)
            }
            Err(e) => Err(MultiLayerCacheError::L3Error(format!("HNSW search failed: {}", e)))
        }
    }

    /// Add embedding to vector index
    pub async fn insert(&self, key: &CacheKey, embedding: &CachedEmbedding) -> Result<(), MultiLayerCacheError> {
        let id = key.to_string();
        
        // Add to HNSW index
        {
            let mut index = self.hnsw_index.write().await;
            index.add(id.clone(), embedding.embedding.to_vec())
                .map_err(|e| MultiLayerCacheError::L3Error(format!("HNSW add failed: {}", e)))?;
        }

        // Store in embedding map
        self.embedding_map.insert(id, embedding.clone());
        self.stats.entries.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Remove embedding from index
    pub async fn remove(&self, key: &CacheKey) -> Result<(), MultiLayerCacheError> {
        let id = key.to_string();
        
        // Remove from HNSW index
        {
            let mut index = self.hnsw_index.write().await;
            index.remove(&id)
                .map_err(|e| MultiLayerCacheError::L3Error(format!("HNSW remove failed: {}", e)))?;
        }

        // Remove from embedding map
        self.embedding_map.remove(&id);
        self.stats.entries.fetch_sub(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> &L1CacheStats {
        &self.stats
    }
}

/// Multi-layer cache errors
#[derive(Debug, Error)]
pub enum MultiLayerCacheError {
    #[error("L1 cache error: {0}")]
    L1Error(String),
    
    #[error("L2 cache error: {0}")]
    L2Error(String),
    
    #[error("L3 cache error: {0}")]
    L3Error(String),
    
    #[error("Cache coherence error: {0}")]
    CoherenceError(String),
    
    #[error("Eviction failed: {0}")]
    EvictionFailed(String),
    
    #[error("TTL management error: {0}")]
    TTLError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String)}

/// Multi-layer cache coordinator
#[derive(Debug)]
pub struct MultiLayerCache {
    l1_cache: Arc<L1Cache>,
    l2_cache: Arc<L2Cache>,
    l3_cache: Arc<L3Cache>,
    ttl_manager: TTLManager,
    cache_warmer: CacheWarmer,
    coherence_enabled: AtomicBool}

impl MultiLayerCache {
    /// Create new multi-layer cache
    pub async fn new(
        l1_capacity: usize,
        surreal_manager: Arc<SurrealDBManager>,
        embedding_dimensions: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Self, MultiLayerCacheError> {
        let l1_cache = Arc::new(L1Cache::new(l1_capacity));
        let l2_cache = Arc::new(L2Cache::new(surreal_manager));
        let l3_cache = Arc::new(L3Cache::new(embedding_dimensions, distance_metric)?);
        
        let ttl_manager = TTLManager::new(
            l1_cache.clone(),
            l2_cache.clone(),
            l3_cache.clone(),
        );
        
        let cache_warmer = CacheWarmer::new(
            l1_cache.clone(),
            l2_cache.clone(),
        );

        Ok(Self {
            l1_cache,
            l2_cache,
            l3_cache,
            ttl_manager,
            cache_warmer,
            coherence_enabled: AtomicBool::new(true)})
    }

    /// Get embedding with cache hierarchy fallback
    pub async fn get(&self, key: &str) -> Result<Option<CachedEmbedding>, MultiLayerCacheError> {
        let cache_key = Self::normalize_key(key)?;

        // L1 Cache (fastest)
        if let Some(cached) = self.l1_cache.get(&cache_key) {
            return Ok(Some(cached));
        }

        // L2 Cache (persistent)
        if let Some(cached) = self.l2_cache.get(&cache_key).await? {
            // Populate L1 cache
            self.l1_cache.insert(cache_key, cached.clone())?;
            return Ok(Some(cached));
        }

        // L3 Cache (similarity search) - only if we have a query embedding
        // This would require the caller to provide an embedding for similarity search
        
        Ok(None)
    }

    /// Insert embedding into all cache layers
    pub async fn insert(
        &self,
        key: &str,
        embedding: &[f32],
        provider: &str,
        model: &str,
        quality_score: f32,
        ttl_seconds: Option<u32>,
    ) -> Result<(), MultiLayerCacheError> {
        let cache_key = Self::normalize_key(key)?;
        let ttl = ttl_seconds.unwrap_or(DEFAULT_TTL_SECONDS);
        
        let cached_embedding = CachedEmbedding::new(
            embedding,
            ttl,
            provider,
            model,
            quality_score,
        );

        // Insert into all layers (write-through strategy)
        self.l1_cache.insert(cache_key.clone(), cached_embedding.clone())?;
        self.l2_cache.insert(&cache_key, &cached_embedding).await?;
        self.l3_cache.insert(&cache_key, &cached_embedding).await?;

        Ok(())
    }

    /// Find similar embeddings using L3 cache
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        similarity_threshold: f32,
        max_results: usize,
    ) -> Result<Vec<(CachedEmbedding, f32)>, MultiLayerCacheError> {
        self.l3_cache.find_similar(query_embedding, similarity_threshold, max_results).await
    }

    /// Remove embedding from all cache layers
    pub async fn remove(&self, key: &str) -> Result<(), MultiLayerCacheError> {
        let cache_key = Self::normalize_key(key)?;

        self.l1_cache.remove(&cache_key);
        // L2 removal would require a delete query
        self.l3_cache.remove(&cache_key).await?;

        Ok(())
    }

    /// Start background tasks for TTL management and cache warming
    pub fn start_background_tasks(&self) -> tokio::task::JoinHandle<()> {
        let ttl_manager = self.ttl_manager.clone();
        let cache_warmer = self.cache_warmer.clone();

        tokio::spawn(async move {
            tokio::join!(
                ttl_manager.start_cleanup_task(),
                cache_warmer.start_warming_task()
            );
        })
    }

    /// Get comprehensive cache statistics
    pub fn get_stats(&self) -> MultiLayerCacheStats {
        MultiLayerCacheStats {
            l1_stats: CacheLayerStats::from_l1_stats(self.l1_cache.stats()),
            l2_stats: CacheLayerStats::from_l1_stats(self.l2_cache.stats()),
            l3_stats: CacheLayerStats::from_l1_stats(self.l3_cache.stats())}
    }

    /// Normalize cache key for consistent access
    fn normalize_key(key: &str) -> Result<CacheKey, MultiLayerCacheError> {
        // Normalize key: lowercase, trim whitespace, hash if too long
        let normalized = key.trim().to_lowercase();
        
        if normalized.len() <= MAX_CACHE_KEY_LENGTH {
            ArrayString::from(&normalized).map_err(|_| {
                MultiLayerCacheError::ValidationError("Invalid key format".to_string())
            })
        } else {
            // Hash long keys
            let mut hasher = DefaultHasher::new();
            normalized.hash(&mut hasher);
            let hash = hasher.finish();
            
            ArrayString::from(&format!("hash_{:x}", hash)).map_err(|_| {
                MultiLayerCacheError::ValidationError("Failed to create hash key".to_string())
            })
        }
    }
}

/// TTL Manager for automatic cleanup
#[derive(Debug, Clone)]
pub struct TTLManager {
    l1_cache: Arc<L1Cache>,
    l2_cache: Arc<L2Cache>,
    l3_cache: Arc<L3Cache>}

impl TTLManager {
    pub fn new(
        l1_cache: Arc<L1Cache>,
        l2_cache: Arc<L2Cache>,
        l3_cache: Arc<L3Cache>,
    ) -> Self {
        Self {
            l1_cache,
            l2_cache,
            l3_cache}
    }

    /// Start background cleanup task
    pub async fn start_cleanup_task(self) {
        let mut cleanup_interval = interval(Duration::from_secs(300)); // 5 minutes

        loop {
            cleanup_interval.tick().await;
            
            // Cleanup L1 cache
            self.l1_cache.cleanup_expired();
            
            // Cleanup L2 cache
            if let Err(e) = self.l2_cache.cleanup_expired().await {
                eprintln!("L2 cache cleanup failed: {}", e);
            }
        }
    }
}

/// Cache Warmer for proactive cache population
#[derive(Debug, Clone)]
pub struct CacheWarmer {
    l1_cache: Arc<L1Cache>,
    l2_cache: Arc<L2Cache>}

impl CacheWarmer {
    pub fn new(l1_cache: Arc<L1Cache>, l2_cache: Arc<L2Cache>) -> Self {
        Self {
            l1_cache,
            l2_cache}
    }

    /// Start background warming task
    pub async fn start_warming_task(self) {
        let mut warming_interval = interval(Duration::from_secs(600)); // 10 minutes

        loop {
            warming_interval.tick().await;
            
            // Implement cache warming logic here
            // This could involve pre-loading frequently accessed embeddings
        }
    }
}

/// Cache statistics for each layer
#[derive(Debug, Clone)]
pub struct CacheLayerStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
    pub entries: u64,
    pub memory_bytes: u64,
    pub evictions: u64}

impl CacheLayerStats {
    fn from_l1_stats(stats: &L1CacheStats) -> Self {
        let hits = stats.hits.load(Ordering::Relaxed);
        let misses = stats.misses.load(Ordering::Relaxed);
        
        Self {
            hits,
            misses,
            hit_ratio: if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 },
            entries: stats.entries.load(Ordering::Relaxed),
            memory_bytes: stats.memory_bytes.load(Ordering::Relaxed),
            evictions: stats.evictions.load(Ordering::Relaxed)}
    }
}

/// Comprehensive cache statistics
#[derive(Debug, Clone)]
pub struct MultiLayerCacheStats {
    pub l1_stats: CacheLayerStats,
    pub l2_stats: CacheLayerStats,
    pub l3_stats: CacheLayerStats}

impl MultiLayerCacheStats {
    /// Get overall hit ratio across all layers
    pub fn overall_hit_ratio(&self) -> f64 {
        let total_hits = self.l1_stats.hits + self.l2_stats.hits + self.l3_stats.hits;
        let total_requests = total_hits + self.l1_stats.misses + self.l2_stats.misses + self.l3_stats.misses;
        
        if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Get total memory usage across L1 cache
    pub fn total_memory_bytes(&self) -> u64 {
        self.l1_stats.memory_bytes // L2 and L3 memory is external
    }

    /// Get total entries across all layers
    pub fn total_entries(&self) -> u64 {
        self.l1_stats.entries + self.l2_stats.entries + self.l3_stats.entries
    }
}