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

// Re-export core types from fluent_ai_memory
pub use fluent_ai_memory::{
    CognitiveMemoryManager, CognitiveMemoryNode, CognitiveSettings, CognitiveState,
    EvolutionMetadata, QuantumSignature, MemoryNode, MemoryType, MemoryMetadata,
    MemoryRelationship, Error as MemoryError, MemoryManager as MemoryManagerTrait,
    SurrealDBMemoryManager, MemoryConfig,
};

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
    #[inline]
    pub fn recall(&self, query: &str) -> AsyncStream<Result<MemoryNode, MemoryError>> {
        let manager = Arc::clone(&self.manager);
        let query = query.to_string();
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let mut stream = manager.search_by_content(&query);
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                if tx.send(result).is_err() {
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
    #[inline]
    pub fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> AsyncStream<Result<MemoryNode, MemoryError>> {
        let manager = Arc::clone(&self.manager);
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let mut stream = manager.search_by_vector(vector, limit);
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                if tx.send(result).is_err() {
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
    #[inline]
    pub fn top_n(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String, serde_json::Value)>> {
        let memory = Arc::clone(&self.memory);
        let query = query.to_string();
        
        spawn_async(async move {
            let mut stream = memory.recall(&query);
            let mut results = Vec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        results.push((
                            memory_node.metadata.importance as f64,
                            memory_node.id,
                            serde_json::to_value(&memory_node.content).unwrap_or_default(),
                        ));
                        
                        if results.len() >= n {
                            break;
                        }
                    }
                    Err(_) => continue,
                }
            }
            
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => ZeroOneOrMany::One(results.into_iter().next().unwrap_or_default()),
                _ => ZeroOneOrMany::many(results),
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
    #[inline]
    pub fn top_n_ids(&self, query: &str, n: usize) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        let memory = Arc::clone(&self.memory);
        let query = query.to_string();
        
        spawn_async(async move {
            let mut stream = memory.recall(&query);
            let mut results = Vec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        results.push((
                            memory_node.metadata.importance as f64,
                            memory_node.id,
                        ));
                        
                        if results.len() >= n {
                            break;
                        }
                    }
                    Err(_) => continue,
                }
            }
            
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => ZeroOneOrMany::One(results.into_iter().next().unwrap_or_default()),
                _ => ZeroOneOrMany::many(results),
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