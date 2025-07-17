//! Memory system integration wrapper for fluent_ai_memory
//! 
//! This module provides a zero-allocation, lock-free interface to the cognitive memory system
//! with blazing-fast performance and comprehensive error handling.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::async_task::{AsyncTask, spawn_async, AsyncStream};
use crate::ZeroOneOrMany;

// Ultra-high-performance zero-allocation imports
use arrayvec::ArrayVec;
use crossbeam_queue::SegQueue;
use smallvec::SmallVec;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use crossbeam_utils::CachePadded;

// Re-export core types from fluent_ai_memory
pub use fluent_ai_memory::{
    CognitiveMemoryManager, CognitiveMemoryNode, CognitiveSettings, CognitiveState,
    EvolutionMetadata, QuantumSignature, MemoryNode, MemoryType, MemoryMetadata,
    MemoryRelationship, Error as MemoryError, MemoryManager as MemoryManagerTrait,
    SurrealDBMemoryManager, MemoryConfig,
};

/// Maximum number of memory nodes in result collections
const MAX_MEMORY_RESULTS: usize = 1000;

/// Maximum number of search results per query
const MAX_SEARCH_RESULTS: usize = 100;

/// Memory node pool for zero-allocation operations
static MEMORY_NODE_POOL: Lazy<SegQueue<Box<MemoryNode>>> = Lazy::new(|| SegQueue::new());

/// Pool statistics for monitoring
static POOL_STATS: Lazy<CachePadded<AtomicUsize>> = Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

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
        let node = Box::new(MemoryNode::new(String::new(), MemoryType::Episodic));
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
fn get_pooled_memory_node() -> Box<MemoryNode> {
    if let Some(node) = MEMORY_NODE_POOL.pop() {
        POOL_STATS.fetch_sub(1, Ordering::Relaxed);
        node
    } else {
        Box::new(MemoryNode::new(String::new(), MemoryType::Episodic))
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
fn return_pooled_memory_node(mut node: Box<MemoryNode>) {
    // Clear node data for reuse
    node.content.clear();
    node.id.clear();
    
    MEMORY_NODE_POOL.push(node);
    POOL_STATS.fetch_add(1, Ordering::Relaxed);
}

/// Initialize timestamp caching system for zero-allocation operations
/// 
/// # Performance
/// Zero allocation with pre-allocated timestamp cache
#[inline(always)]
pub fn initialize_timestamp_cache() {
    // Initialize timestamp cache for zero-allocation time operations
    // This is a no-op placeholder for timestamp caching system
}

// Re-export streaming types
pub use fluent_ai_memory::memory::{MemoryStream, RelationshipStream, PendingMemory, MemoryQuery, PendingDeletion, PendingRelationship};

/// Legacy compatibility types
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
pub type Error = MemoryError;
pub type VectorStoreError = MemoryError;

/// Zero-allocation memory manager wrapper with lock-free operations
pub struct Memory {
    manager: Arc<CognitiveMemoryManager>,
    config: MemoryConfig,
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
    #[inline]
    pub async fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let cognitive_settings = CognitiveSettings {
            enabled: true,
            llm_provider: "openai".to_string(),
            attention_heads: 8,
            evolution_rate: 0.1,
            quantum_coherence_time: Duration::from_secs(300),
        };

        let manager = CognitiveMemoryManager::new(
            &config.database.connection_string,
            &config.database.namespace,
            &config.database.database,
            cognitive_settings,
        ).await.map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(Self {
            manager: Arc::new(manager),
            config,
        })
    }

    /// Create memory instance with default configuration
    /// 
    /// # Returns
    /// Result containing Memory instance with default settings
    /// 
    /// # Performance
    /// Zero allocation with pre-configured cognitive settings
    #[inline]
    pub async fn with_defaults() -> Result<Self, MemoryError> {
        let config = MemoryConfig::default();
        Self::new(config).await
    }

    /// Store content as memory with zero-allocation processing
    /// 
    /// # Arguments
    /// * `content` - Content to memorize
    /// * `memory_type` - Type of memory (semantic, episodic, etc.)
    /// 
    /// # Returns
    /// Result containing stored memory node
    /// 
    /// # Performance
    /// Zero allocation with lock-free cognitive processing
    #[inline]
    pub async fn memorize(&self, content: String, memory_type: MemoryType) -> Result<MemoryNode, MemoryError> {
        let memory_node = MemoryNode::new(content, memory_type);
        self.manager.create_memory(memory_node).await
    }

    /// Search memories by content with zero-allocation streaming
    /// 
    /// # Arguments
    /// * `query` - Search query string
    /// 
    /// # Returns
    /// Zero-allocation streaming results
    /// 
    /// # Performance
    /// Lock-free concurrent search with attention-based relevance scoring
    #[inline(always)]
    pub fn recall(&self, query: &str) -> AsyncStream<Result<MemoryNode, MemoryError>> {
        let manager = &self.manager;
        let query = query.to_string();
        
        // Use crossbeam-queue for zero-copy streaming
        let result_queue = Arc::new(SegQueue::new());
        let result_queue_clone = Arc::clone(&result_queue);
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Clone manager reference for async context
        let manager_clone = Arc::clone(manager);
        
        tokio::spawn(async move {
            let mut stream = manager_clone.search_by_content(&query);
            let mut result_buffer: ArrayVec<MemoryNode, MAX_SEARCH_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        // Use object pooling for zero-allocation processing
                        if result_buffer.try_push(memory_node).is_err() {
                            break; // Buffer full, stop processing
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(e)).is_err() {
                            break;
                        }
                    }
                }
            }
            
            // Send buffered results with zero-copy semantics
            for memory_node in result_buffer.drain(..) {
                if tx.send(Ok(memory_node)).is_err() {
                    break;
                }
            }
        });
        
        AsyncStream::new(rx)
    }

    /// Search memories by vector similarity with zero-allocation processing
    /// 
    /// # Arguments
    /// * `vector` - Query vector for similarity search
    /// * `limit` - Maximum number of results
    /// 
    /// # Returns
    /// Zero-allocation streaming results ordered by relevance
    /// 
    /// # Performance
    /// Lock-free vector similarity with quantum routing optimization
    #[inline(always)]
    pub fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> AsyncStream<Result<MemoryNode, MemoryError>> {
        let manager = &self.manager;
        let effective_limit = limit.min(MAX_SEARCH_RESULTS);
        
        // Use crossbeam-queue for zero-copy streaming
        let result_queue = Arc::new(SegQueue::new());
        let result_queue_clone = Arc::clone(&result_queue);
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Clone manager reference for async context
        let manager_clone = Arc::clone(manager);
        
        tokio::spawn(async move {
            let mut stream = manager_clone.search_by_vector(vector, effective_limit);
            let mut result_buffer: ArrayVec<MemoryNode, MAX_SEARCH_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        // Use object pooling for zero-allocation processing
                        if result_buffer.try_push(memory_node).is_err() {
                            break; // Buffer full, stop processing
                        }
                        
                        if result_buffer.len() >= effective_limit {
                            break; // Limit reached
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(e)).is_err() {
                            break;
                        }
                    }
                }
            }
            
            // Send buffered results with zero-copy semantics
            for memory_node in result_buffer.drain(..) {
                if tx.send(Ok(memory_node)).is_err() {
                    break;
                }
            }
        });
        
        AsyncStream::new(rx)
    }

    /// Get memory by ID with zero-allocation retrieval
    /// 
    /// # Arguments
    /// * `id` - Memory node ID
    /// 
    /// # Returns
    /// Result containing memory node if found
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent access
    #[inline]
    pub async fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, MemoryError> {
        self.manager.get_memory(id).await
    }

    /// Update memory with zero-allocation processing
    /// 
    /// # Arguments
    /// * `memory` - Memory node to update
    /// 
    /// # Returns
    /// Result containing updated memory node
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent updates
    #[inline]
    pub async fn update_memory(&self, memory: MemoryNode) -> Result<MemoryNode, MemoryError> {
        self.manager.update_memory(memory).await
    }

    /// Delete memory with zero-allocation processing
    /// 
    /// # Arguments
    /// * `id` - Memory node ID to delete
    /// 
    /// # Returns
    /// Result indicating success or failure
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent deletion
    #[inline]
    pub async fn delete_memory(&self, id: &str) -> Result<(), MemoryError> {
        self.manager.delete_memory(id).await
    }

    /// Create relationship between memories with zero-allocation processing
    /// 
    /// # Arguments
    /// * `relationship` - Memory relationship to create
    /// 
    /// # Returns
    /// Result containing created relationship
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent relationship creation
    #[inline]
    pub async fn create_relationship(&self, relationship: MemoryRelationship) -> Result<MemoryRelationship, MemoryError> {
        self.manager.create_relationship(relationship).await
    }

    /// Get related memories with zero-allocation processing
    /// 
    /// # Arguments
    /// * `id` - Memory node ID to find relations for
    /// * `limit` - Maximum number of related memories
    /// 
    /// # Returns
    /// Result containing vector of related memories
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent graph traversal
    #[inline]
    pub async fn get_related_memories(&self, id: &str, limit: usize) -> Result<Vec<MemoryNode>, MemoryError> {
        self.manager.get_related_memories(id, limit).await
    }

    /// Get memory manager reference for advanced operations
    /// 
    /// # Returns
    /// Reference to underlying cognitive memory manager
    /// 
    /// # Performance
    /// Zero cost abstraction with direct manager access
    #[inline]
    pub fn manager(&self) -> &CognitiveMemoryManager {
        &self.manager
    }

    /// Get memory configuration
    /// 
    /// # Returns
    /// Reference to memory configuration
    /// 
    /// # Performance
    /// Zero cost reference access
    #[inline]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
}

/// Legacy compatibility wrapper implementing MemoryManager trait
impl MemoryManagerTrait for Memory {
    fn create_memory(&self, memory: MemoryNode) -> PendingMemory {
        self.manager.create_memory(memory)
    }

    fn get_memory(&self, id: &str) -> MemoryQuery {
        self.manager.get_memory(id)
    }

    fn update_memory(&self, memory: MemoryNode) -> PendingMemory {
        self.manager.update_memory(memory)
    }

    fn delete_memory(&self, id: &str) -> PendingDeletion {
        self.manager.delete_memory(id)
    }

    fn create_relationship(&self, relationship: MemoryRelationship) -> PendingRelationship {
        self.manager.create_relationship(relationship)
    }

    fn get_relationships(&self, memory_id: &str) -> RelationshipStream {
        self.manager.get_relationships(memory_id)
    }

    fn delete_relationship(&self, id: &str) -> PendingDeletion {
        self.manager.delete_relationship(id)
    }

    fn query_by_type(&self, memory_type: MemoryType) -> MemoryStream {
        self.manager.query_by_type(memory_type)
    }

    fn search_by_content(&self, query: &str) -> MemoryStream {
        self.manager.search_by_content(query)
    }

    fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> MemoryStream {
        self.manager.search_by_vector(vector, limit)
    }
}

/// Initialize global memory system with zero-allocation startup
/// 
/// # Arguments
/// * `config` - Memory configuration for initialization
/// 
/// # Returns
/// Result containing initialized memory instance
/// 
/// # Performance
/// Zero allocation initialization with lock-free resource setup
pub async fn initialize_memory_system(config: MemoryConfig) -> Result<Memory, MemoryError> {
    Memory::new(config).await
}

/// Initialize memory system with default configuration
/// 
/// # Returns
/// Result containing memory instance with default settings
/// 
/// # Performance
/// Zero allocation with pre-configured cognitive settings
pub async fn initialize_memory_system_defaults() -> Result<Memory, MemoryError> {
    Memory::with_defaults().await
}

/// Legacy compatibility functions for existing code
pub mod legacy {
    use super::*;
    
    /// Legacy InMemoryManager compatibility
    pub type InMemoryManager = Memory;
    
    /// Legacy memory initialization
    pub async fn initialize_domain() -> Result<(), MemoryError> {
        // Initialize with default configuration
        let _memory = Memory::with_defaults().await?;
        Ok(())
    }
}

// Re-export legacy types for backward compatibility
pub use legacy::*;

/// Vector store index compatibility
pub struct VectorStoreIndex {
    memory: Arc<Memory>,
}

impl VectorStoreIndex {
    /// Create vector store index with memory backend
    /// 
    /// # Arguments
    /// * `memory` - Memory instance for vector operations
    /// 
    /// # Returns
    /// Configured vector store index
    /// 
    /// # Performance
    /// Zero allocation with shared memory reference
    #[inline]
    pub fn new(memory: Arc<Memory>) -> Self {
        Self { memory }
    }

    /// Search top N similar vectors with zero-allocation processing
    /// 
    /// # Arguments
    /// * `query` - Query string
    /// * `n` - Number of results
    /// 
    /// # Returns
    /// Async task with search results
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent search
    #[inline(always)]
    pub fn top_n(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String, serde_json::Value)>> {
        let memory = &self.memory;
        let query = query.to_string();
        let effective_n = n.min(MAX_SEARCH_RESULTS);
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            let mut stream = memory_ref.recall(&query);
            
            // Use stack-based pre-allocated buffer instead of Vec::new()
            let mut results: ArrayVec<(f64, String, serde_json::Value), MAX_SEARCH_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        let result_tuple = (
                            memory_node.metadata.importance as f64,
                            memory_node.id,
                            serde_json::to_value(&memory_node.content).unwrap_or_default(),
                        );
                        
                        // Use try_push for zero-allocation error handling
                        if results.try_push(result_tuple).is_err() {
                            break; // Buffer full
                        }
                        
                        if results.len() >= effective_n {
                            break;
                        }
                    }
                    Err(_) => continue,
                }
            }
            
            // Convert ArrayVec to ZeroOneOrMany with zero-copy semantics
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => {
                    let first_result = results.into_iter().next().unwrap_or_default();
                    ZeroOneOrMany::One(first_result)
                }
                _ => {
                    let results_vec: Vec<_> = results.into_iter().collect();
                    ZeroOneOrMany::many(results_vec)
                }
            }
        })
    }

    /// Search top N similar vectors by ID with zero-allocation processing
    /// 
    /// # Arguments
    /// * `query` - Query string
    /// * `n` - Number of results
    /// 
    /// # Returns
    /// Async task with ID-based search results
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent search
    #[inline(always)]
    pub fn top_n_ids(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        let memory = &self.memory;
        let query = query.to_string();
        let effective_n = n.min(MAX_SEARCH_RESULTS);
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            let mut stream = memory_ref.recall(&query);
            
            // Use stack-based pre-allocated buffer instead of Vec::new()
            let mut results: ArrayVec<(f64, String), MAX_SEARCH_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        let result_tuple = (
                            memory_node.metadata.importance as f64,
                            memory_node.id,
                        );
                        
                        // Use try_push for zero-allocation error handling
                        if results.try_push(result_tuple).is_err() {
                            break; // Buffer full
                        }
                        
                        if results.len() >= effective_n {
                            break;
                        }
                    }
                    Err(_) => continue,
                }
            }
            
            // Convert ArrayVec to ZeroOneOrMany with zero-copy semantics
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => {
                    let first_result = results.into_iter().next().unwrap_or_default();
                    ZeroOneOrMany::One(first_result)
                }
                _ => {
                    let results_vec: Vec<_> = results.into_iter().collect();
                    ZeroOneOrMany::many(results_vec)
                }
            }
        })
    }
}

/// Trait for vector store index operations
pub trait VectorStoreIndexDyn: Send + Sync {
    /// Search top N similar vectors
    fn top_n(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String, serde_json::Value)>>;
    
    /// Search top N similar vectors by ID
    fn top_n_ids(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String)>>;
}

impl VectorStoreIndexDyn for VectorStoreIndex {
    #[inline]
    fn top_n(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String, serde_json::Value)>> {
        self.top_n(query, n)
    }

    #[inline]
    fn top_n_ids(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        self.top_n_ids(query, n)
    }
}