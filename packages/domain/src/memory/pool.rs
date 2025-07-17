use super::types::{MemoryNode, MemoryType, ImportanceContext, MemoryMetadata, calculate_importance, next_memory_id};
use super::cache::get_cached_system_time;

/// Lock-free memory node pool for zero-allocation MemoryNode reuse
pub struct MemoryNodePool {
    available: crossbeam_queue::ArrayQueue<MemoryNode>,
    embedding_dimension: usize,
    max_capacity: usize,
}

impl MemoryNodePool {
    /// Create new memory node pool with specified capacity and embedding dimension
    #[inline]
    pub fn new(capacity: usize, embedding_dimension: usize) -> Self {
        let pool = Self {
            available: crossbeam_queue::ArrayQueue::new(capacity),
            embedding_dimension,
            max_capacity: capacity,
        };
        
        // Pre-allocate nodes to avoid allocations during runtime
        for _ in 0..capacity {
            let node = MemoryNode {
                id: 0, // Will be set when acquired
                content: String::with_capacity(1024), // Pre-allocate string capacity
                memory_type: MemoryType::ShortTerm,
                metadata: MemoryMetadata {
                    importance: 0.0,
                    last_accessed: get_cached_system_time(),
                    creation_time: get_cached_system_time(),
                },
                embedding: Some(vec![0.0; embedding_dimension]), // Pre-allocate embedding
            };
            let _ = pool.available.push(node);
        }
        
        pool
    }
    
    /// Acquire a node from the pool (zero-allocation in common case)
    #[inline(always)]
    pub fn acquire(&self) -> PooledMemoryNode {
        let node = self.available.pop().unwrap_or_else(|| {
            // Fallback: create new node if pool is empty
            MemoryNode {
                id: 0,
                content: String::with_capacity(1024),
                memory_type: MemoryType::ShortTerm,
                metadata: MemoryMetadata {
                    importance: 0.0,
                    last_accessed: get_cached_system_time(),
                    creation_time: get_cached_system_time(),
                },
                embedding: Some(vec![0.0; self.embedding_dimension]),
            }
        });
        
        PooledMemoryNode {
            node: std::mem::ManuallyDrop::new(node),
            pool: self,
            taken: false,
        }
    }
    
    /// Release a node back to the pool for reuse
    #[inline(always)]
    fn release(&self, mut node: MemoryNode) {
        // Clear the node data for reuse
        node.id = 0;
        node.content.clear();
        node.memory_type = MemoryType::ShortTerm;
        node.metadata.importance = 0.0;
        node.metadata.last_accessed = get_cached_system_time();
        node.metadata.creation_time = get_cached_system_time();
        
        // Clear embedding vector but keep allocation
        if let Some(ref mut embedding) = node.embedding {
            embedding.fill(0.0);
        }
        
        // Return to pool (ignore if pool is full)
        let _ = self.available.push(node);
    }
    
    /// Get pool statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> (usize, usize) {
        (self.available.len(), self.max_capacity)
    }
}

/// Pooled memory node that automatically returns to pool on drop
pub struct PooledMemoryNode<'a> {
    node: std::mem::ManuallyDrop<MemoryNode>,
    pool: &'a MemoryNodePool,
    taken: bool,
}

impl<'a> PooledMemoryNode<'a> {
    /// Initialize the pooled node with content and context
    #[inline(always)]
    pub fn initialize(
        &mut self,
        content: String,
        memory_type: MemoryType,
        context: ImportanceContext,
    ) {
        if !self.taken {
            self.node.id = next_memory_id();
            self.node.content = content;
            self.node.memory_type = memory_type;
            self.node.metadata.importance = calculate_importance(&self.node.memory_type, context, self.node.content.len());
            self.node.metadata.last_accessed = get_cached_system_time();
            self.node.metadata.creation_time = get_cached_system_time();
        }
    }
    
    /// Set embedding for the pooled node
    #[inline(always)]
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        if !self.taken {
            self.node.embedding = Some(embedding);
        }
    }
    
    /// Get immutable reference to the inner node
    #[inline(always)]
    pub fn as_ref(&self) -> Option<&MemoryNode> {
        if self.taken {
            None
        } else {
            Some(&self.node)
        }
    }
    
    /// Get mutable reference to the inner node
    #[inline(always)]
    pub fn as_mut(&mut self) -> Option<&mut MemoryNode> {
        if self.taken {
            None
        } else {
            Some(&mut self.node)
        }
    }
    
    /// Take ownership of the inner node (prevents return to pool)
    #[inline(always)]
    pub fn take(mut self) -> Option<MemoryNode> {
        if self.taken {
            None
        } else {
            self.taken = true;
            Some(std::mem::ManuallyDrop::into_inner(
                std::mem::replace(&mut self.node, std::mem::ManuallyDrop::new(MemoryNode {
                    id: 0,
                    content: String::new(),
                    memory_type: MemoryType::ShortTerm,
                    metadata: MemoryMetadata {
                        importance: 0.0,
                        last_accessed: get_cached_system_time(),
                        creation_time: get_cached_system_time(),
                    },
                    embedding: None,
                }))
            ))
        }
    }
}

impl<'a> std::ops::Deref for PooledMemoryNode<'a> {
    type Target = MemoryNode;
    
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // PooledMemoryNode always contains a valid node unless taken
        &self.node
    }
}

impl<'a> std::ops::DerefMut for PooledMemoryNode<'a> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // PooledMemoryNode always contains a valid node unless taken
        &mut self.node
    }
}

impl<'a> Drop for PooledMemoryNode<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.taken {
            let node = std::mem::ManuallyDrop::into_inner(
                std::mem::replace(&mut self.node, std::mem::ManuallyDrop::new(MemoryNode {
                    id: 0,
                    content: String::new(),
                    memory_type: MemoryType::ShortTerm,
                    metadata: MemoryMetadata {
                        importance: 0.0,
                        last_accessed: get_cached_system_time(),
                        creation_time: get_cached_system_time(),
                    },
                    embedding: None,
                }))
            );
            self.pool.release(node);
        }
    }
}

/// Global memory node pool for zero-allocation operations
static MEMORY_NODE_POOL: std::sync::OnceLock<MemoryNodePool> = std::sync::OnceLock::new();

/// Initialize the global memory node pool
#[inline]
pub fn initialize_memory_node_pool(capacity: usize, embedding_dimension: usize) {
    let _ = MEMORY_NODE_POOL.set(MemoryNodePool::new(capacity, embedding_dimension));
}

/// Get a node from the global pool
#[inline(always)]
pub fn acquire_pooled_node() -> Option<PooledMemoryNode<'static>> {
    MEMORY_NODE_POOL.get().map(|pool| pool.acquire())
}

/// Get pool statistics from the global pool
#[inline]
#[must_use]
pub fn memory_node_pool_stats() -> Option<(usize, usize)> {
    MEMORY_NODE_POOL.get().map(|pool| pool.stats())
}