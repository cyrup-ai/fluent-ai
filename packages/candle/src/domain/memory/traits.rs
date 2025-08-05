//! CandleMemory trait definition - mirrors fluent-ai-domain Memory trait exactly
//!
//! This trait provides the core memory interface for Candle-backed memory implementations,
//! enabling trait composition, testability, and 'room to move' architecture benefits.

use fluent_ai_async::AsyncStream;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use super::primitives::MemoryNode;

/// CandleMemory trait - mirrors fluent-ai-domain::Memory exactly with Candle prefix
/// 
/// This trait enables:
/// - Trait composition for flexible memory architectures  
/// - Testability with mock implementations
/// - Zero-cost abstractions via static dispatch
/// - 'Room to move' for future memory backend changes
pub trait CandleMemory: Send + Sync + 'static {
    /// Store a memory node in the memory system
    ///
    /// # Arguments
    /// * `memory_node` - The memory node to store
    ///
    /// # Returns
    /// AsyncStream that completes when the memory is stored
    fn store_memory(&self, memory_node: &MemoryNode) -> AsyncStream<()>;
    
    /// Retrieve memory nodes by similarity search
    ///
    /// # Arguments
    /// * `query` - The query text to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// AsyncStream of memory nodes ranked by similarity
    fn search_memory(&self, query: &str, limit: usize) -> AsyncStream<MemoryNode>;
    
    /// Get memory node by ID
    ///
    /// # Arguments
    /// * `id` - The unique ID of the memory node
    ///
    /// # Returns
    /// AsyncStream containing the memory node if found
    fn get_memory(&self, id: &str) -> AsyncStream<Option<MemoryNode>>;
    
    /// Delete memory node by ID
    ///
    /// # Arguments
    /// * `id` - The unique ID of the memory node to delete
    ///
    /// # Returns
    /// AsyncStream that completes when the memory is deleted
    fn delete_memory(&self, id: &str) -> AsyncStream<bool>;
    
    /// Get memory statistics
    ///
    /// # Returns
    /// AsyncStream containing memory system statistics
    fn get_stats(&self) -> AsyncStream<CandleMemoryStats>;
}

/// Memory statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleMemoryStats {
    /// Total number of memory nodes stored
    pub total_memories: u64,
    /// Total size of memory data in bytes
    pub total_size_bytes: u64,
    /// Average embedding dimension
    pub avg_embedding_dimension: f32,
    /// Memory system uptime in seconds
    pub uptime_seconds: u64,
    /// Cache hit ratio (0.0 to 1.0)
    pub cache_hit_ratio: f32,
}

impl Default for CandleMemoryStats {
    fn default() -> Self {
        Self {
            total_memories: 0,
            total_size_bytes: 0,
            avg_embedding_dimension: 768.0,
            uptime_seconds: 0,
            cache_hit_ratio: 0.0,
        }
    }
}

/// Concrete memory implementation using the existing Memory struct
/// This enables trait-backed architecture while preserving existing functionality
#[derive(Debug, Clone)]
pub struct CandleMemoryImpl {
    /// Inner memory instance
    inner: crate::domain::memory::manager::Memory,
}

impl CandleMemoryImpl {
    /// Create new memory implementation with configuration
    ///
    /// # Arguments
    /// * `config` - Memory configuration
    ///
    /// # Returns
    /// AsyncStream containing configured memory implementation
    pub fn new(config: super::manager::MemoryConfig) -> AsyncStream<Self> {
        AsyncStream::with_channel(move |sender| {
            // Spawn task to handle the memory creation
            std::thread::spawn(move || {
                let memory_stream = crate::domain::memory::manager::Memory::new(config);
                let mut memory_stream = Box::pin(memory_stream);
                
                // Collect the memory instance and wrap it

                if let Some(memory) = memory_stream.next() {
                    let implementation = Self { inner: memory };
                    let _ = sender.send(implementation);
                }
            });
        })
    }
    
    /// Create memory implementation with default configuration
    ///
    /// # Returns
    /// AsyncStream containing memory implementation with default settings
    pub fn with_defaults() -> AsyncStream<Self> {
        let config = super::manager::MemoryConfig::default();
        Self::new(config)
    }
}

impl CandleMemory for CandleMemoryImpl {
    fn store_memory(&self, memory_node: &MemoryNode) -> AsyncStream<()> {
        self.inner.store_memory(memory_node)
    }
    
    fn search_memory(&self, _query: &str, _limit: usize) -> AsyncStream<MemoryNode> {
        // TODO: Implement search functionality
        AsyncStream::with_channel(move |_sender| {
            // Stub implementation for now
        })
    }
    
    fn get_memory(&self, _id: &str) -> AsyncStream<Option<MemoryNode>> {
        // TODO: Implement get functionality
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(None);
        })
    }
    
    fn delete_memory(&self, _id: &str) -> AsyncStream<bool> {
        // TODO: Implement delete functionality
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(false);
        })
    }
    
    fn get_stats(&self) -> AsyncStream<CandleMemoryStats> {
        AsyncStream::with_channel(move |sender| {
            let stats = CandleMemoryStats::default();
            let _ = sender.send(stats);
        })
    }
}

/// Mock memory implementation for testing
#[derive(Debug, Clone, Default)]
pub struct MockCandleMemory {
    /// Stored memory nodes for testing
    pub stored_nodes: std::sync::Arc<std::sync::Mutex<Vec<MemoryNode>>>,
}

impl MockCandleMemory {
    /// Create new mock memory instance
    pub fn new() -> Self {
        Self::default()
    }
}

impl CandleMemory for MockCandleMemory {
    fn store_memory(&self, memory_node: &MemoryNode) -> AsyncStream<()> {
        let node = memory_node.clone();
        let stored = self.stored_nodes.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(mut nodes) = stored.lock() {
                nodes.push(node);
            }
            let _ = sender.send(());
        })
    }
    
    fn search_memory(&self, _query: &str, limit: usize) -> AsyncStream<MemoryNode> {
        let stored = self.stored_nodes.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(nodes) = stored.lock() {
                for node in nodes.iter().take(limit) {
                    let _ = sender.send(node.clone());
                }
            }
        })
    }
    
    fn get_memory(&self, id: &str) -> AsyncStream<Option<MemoryNode>> {
        let search_uuid = match Uuid::parse_str(id) {
            Ok(uuid) => uuid,
            Err(_) => {
                return AsyncStream::with_channel(move |sender| {
                    let _ = sender.send(None);
                });
            }
        };
        let stored = self.stored_nodes.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(nodes) = stored.lock() {
                let found = nodes.iter().find(|node| node.base_memory.id == search_uuid).cloned();
                let _ = sender.send(found);
            } else {
                let _ = sender.send(None);
            }
        })
    }
    
    fn delete_memory(&self, id: &str) -> AsyncStream<bool> {
        let search_uuid = match Uuid::parse_str(id) {
            Ok(uuid) => uuid,
            Err(_) => {
                return AsyncStream::with_channel(move |sender| {
                    let _ = sender.send(false);
                });
            }
        };
        let stored = self.stored_nodes.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(mut nodes) = stored.lock() {
                let initial_len = nodes.len();
                nodes.retain(|node| node.base_memory.id != search_uuid);
                let deleted = nodes.len() < initial_len;
                let _ = sender.send(deleted);
            } else {
                let _ = sender.send(false);
            }
        })
    }
    
    fn get_stats(&self) -> AsyncStream<CandleMemoryStats> {
        let stored = self.stored_nodes.clone();
        
        AsyncStream::with_channel(move |sender| {
            let stats = if let Ok(nodes) = stored.lock() {
                CandleMemoryStats {
                    total_memories: nodes.len() as u64,
                    total_size_bytes: nodes.iter().map(|n| n.base_memory.content.to_string().len() as u64).sum(),
                    avg_embedding_dimension: 768.0,
                    uptime_seconds: 0,
                    cache_hit_ratio: 1.0,
                }
            } else {
                CandleMemoryStats::default()
            };
            let _ = sender.send(stats);
        })
    }
}