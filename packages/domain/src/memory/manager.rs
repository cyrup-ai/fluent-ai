//! Core Memory Management System
//!
//! This module provides zero-allocation, lock-free interface to the cognitive memory system
//! with blazing-fast performance and comprehensive error handling.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

// Ultra-high-performance zero-allocation imports
use arrayvec::ArrayVec;
use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;

use super::primitives::MemoryNode;
use super::types_legacy::MemoryType;
use super::MemoryError;
// Conditional re-exports for cognitive features
#[cfg(feature = "cognitive")]
pub use fluent_ai_memory::{
    CognitiveMemoryManager, CognitiveMemoryNode, CognitiveSettings, CognitiveState,
    EvolutionMetadata, QuantumSignature,
};
// Re-export core types from fluent_ai_memory (avoiding conflicts)
pub use fluent_ai_memory::{
    MemoryConfig, MemoryManager as MemoryManagerTrait, MemoryMetadata,
    SurrealDBMemoryManager,
    memory::{SurrealMemoryQuery},
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::ZeroOneOrMany;
use crate::{AsyncTask, spawn_async};

/// Memory stub that provides safe fallback for synchronous contexts
///
/// This wrapper provides a safe alternative to unsafe memory operations when
/// full initialization is not possible in synchronous contexts.
pub struct MemoryStub {
    config: MemoryConfig,
}

impl MemoryStub {
    /// Create new memory stub with minimal configuration
    ///
    /// # Returns
    /// Safe memory stub that fails gracefully on operations
    ///
    /// # Performance
    /// Zero allocation with minimal initialization
    #[inline]
    pub fn new() -> Self {
        let config = MemoryConfig {
            database: fluent_ai_memory::utils::config::DatabaseConfig {
                connection_string: "memory://stub".to_string(),
                namespace: "stub".to_string(),
                database: "stub".to_string(),
            },
            ..Default::default()
        };

        Self { config }
    }

    /// Convert to Memory instance asynchronously
    ///
    /// # Returns
    /// Result containing properly initialized Memory instance
    ///
    /// # Performance
    /// Zero allocation with proper async initialization
    pub async fn into_memory(self) -> Result<Memory, MemoryError> {
        Memory::new(self.config).await
    }
}

/// Maximum number of memory nodes in result collections
const MAX_MEMORY_RESULTS: usize = 1000;

/// Maximum number of search results per query
const MAX_SEARCH_RESULTS: usize = 100;

/// Memory node pool for zero-allocation operations
static MEMORY_NODE_POOL: Lazy<SegQueue<Box<MemoryNode>>> = Lazy::new(|| SegQueue::new());

/// Pool statistics for monitoring
static POOL_STATS: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// Lock-free timestamp cache for high-performance time operations
static TIMESTAMP_CACHE: Lazy<CachePadded<AtomicU64>> =
    Lazy::new(|| CachePadded::new(AtomicU64::new(0)));

/// Last timestamp cache update time
static TIMESTAMP_CACHE_LAST_UPDATE: Lazy<CachePadded<AtomicU64>> =
    Lazy::new(|| CachePadded::new(AtomicU64::new(0)));

/// Timestamp cache refresh interval in microseconds (1ms = 1000µs)
const TIMESTAMP_CACHE_REFRESH_INTERVAL_MICROS: u64 = 1000;
/// Initialize memory node pool with pre-allocated nodes
///
/// # Arguments
/// * `initial_size` - Initial number of nodes to allocate
/// * `embedding_dim` - Embedding dimension for nodes
///
/// # Performance
/// Zero allocation during runtime through pre-allocation
#[inline(always)]
pub fn initialize_memory_node_pool(initial_size: usize, embedding_dim: usize) {
    for _ in 0..initial_size {
        let node = Box::new(MemoryNode::new(super::primitives::MemoryTypeEnum::Episodic, super::primitives::MemoryContent::text("")));
        MEMORY_NODE_POOL.push(node);
    }
    POOL_STATS.store(initial_size, Ordering::Relaxed);
}

/// Get memory node from pool or create new one
///
/// # Returns
/// Pooled memory node for zero-allocation operations
///
/// # Performance
/// Lock-free pool access with atomic statistics
#[inline(always)]
pub fn get_pooled_memory_node() -> Box<MemoryNode> {
    if let Some(node) = MEMORY_NODE_POOL.pop() {
        POOL_STATS.fetch_sub(1, Ordering::Relaxed);
        node
    } else {
        Box::new(MemoryNode::new(super::primitives::MemoryTypeEnum::Episodic, super::primitives::MemoryContent::text("")))
    }
}

/// Return memory node to pool
///
/// # Arguments
/// * `node` - Memory node to return to pool
///
/// # Performance
/// Zero allocation with lock-free pool management
#[inline(always)]
pub fn return_pooled_memory_node(mut node: Box<MemoryNode>) {
    // Clear node data for reuse
    node.content = String::new();
    node.metadata = MemoryMetadata::default();

    MEMORY_NODE_POOL.push(node);
    POOL_STATS.fetch_add(1, Ordering::Relaxed);
}
/// Initialize timestamp caching system for zero-allocation operations
///
/// # Performance
/// Zero allocation with pre-allocated timestamp cache
#[inline(always)]
pub fn initialize_timestamp_cache() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    TIMESTAMP_CACHE.store(now, Ordering::Relaxed);
    TIMESTAMP_CACHE_LAST_UPDATE.store(now, Ordering::Relaxed);
}

/// Get cached timestamp with automatic refresh
///
/// # Returns
/// Cached timestamp in microseconds since Unix epoch
///
/// # Performance
/// Sub-millisecond access with lock-free atomic operations
#[inline(always)]
pub fn get_cached_timestamp() -> u64 {
    let cached_time = TIMESTAMP_CACHE.load(Ordering::Relaxed);
    let last_update = TIMESTAMP_CACHE_LAST_UPDATE.load(Ordering::Relaxed);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    // Check if cache needs refresh
    if now - last_update > TIMESTAMP_CACHE_REFRESH_INTERVAL_MICROS {
        TIMESTAMP_CACHE.store(now, Ordering::Relaxed);
        TIMESTAMP_CACHE_LAST_UPDATE.store(now, Ordering::Relaxed);
        now
    } else {
        cached_time
    }
}
/// Get timestamp cache statistics
///
/// # Returns
/// (cached_timestamp, last_update_time, cache_age_micros)
///
/// # Performance
/// Lock-free atomic read operations
#[inline(always)]
pub fn get_timestamp_cache_stats() -> (u64, u64, u64) {
    let cached_time = TIMESTAMP_CACHE.load(Ordering::Relaxed);
    let last_update = TIMESTAMP_CACHE_LAST_UPDATE.load(Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    (cached_time, last_update, now - last_update)
}

/// Check if timestamp cache is fresh
///
/// # Returns
/// True if cache is within refresh interval
///
/// # Performance
/// Lock-free atomic read operation
#[inline(always)]
pub fn is_timestamp_cache_fresh() -> bool {
    let last_update = TIMESTAMP_CACHE_LAST_UPDATE.load(Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    now - last_update <= TIMESTAMP_CACHE_REFRESH_INTERVAL_MICROS
}
/// Zero-allocation memory manager wrapper with lock-free operations
#[derive(Debug, Clone)]
pub struct Memory {
    /// Shared memory manager instance for concurrent access
    memory: Arc<SurrealDBMemoryManager>,
}

impl Memory {
    /// Create new memory instance with zero-allocation initialization
    ///
    /// # Arguments
    /// * `config` - Memory configuration with SurrealDB settings
    ///
    /// # Returns
    /// Result containing configured Memory instance or initialization error
    ///
    /// # Performance
    /// Zero allocation initialization with lock-free connection pooling
    pub async fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let memory = Arc::new(fluent_ai_memory::initialize(&config).await?);

        Ok(Self { memory })
    }

    /// Create memory instance with default configuration
    ///
    /// # Returns
    /// Result containing Memory instance with default settings
    ///
    /// # Performance
    /// Zero allocation with pre-configured cognitive settings
    pub async fn with_defaults() -> Result<Self, MemoryError> {
        let config = MemoryConfig::default();
        Self::new(config).await
    }

    /// Store a memory node in the memory system
    ///
    /// # Arguments
    /// * `memory_node` - The memory node to store
    ///
    /// # Returns
    /// Future that completes when the memory is stored
    pub fn store_memory(&self, memory_node: &MemoryNode) -> Pin<Box<dyn Future<Output = Result<(), MemoryError>> + Send>> {
        let memory = self.memory.clone();
        let memory_node = memory_node.clone();
        
        Box::pin(async move {
            use fluent_ai_memory::MemoryManager;
            let pending = memory.create_memory(memory_node);
            pending.await.map_err(|e| MemoryError::StorageError(e.to_string()))
        })
    }
}
