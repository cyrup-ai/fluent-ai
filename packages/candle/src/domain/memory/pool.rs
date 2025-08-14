use super::primitives::{MemoryContent, MemoryNode, MemoryTypeEnum as MemoryType};

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
            let content = MemoryContent::text(String::with_capacity(1024)); // Pre-allocate string capacity
            let mut node = MemoryNode::new(MemoryType::Working, content); // Use Working type for pooled nodes

            // Set embedding if requested
            if embedding_dimension > 0 {
                let _ = node.set_embedding(vec![0.0; embedding_dimension]); // Pre-allocate embedding
            }

            let _ = pool.available.push(node);
        }

        pool
    }

    /// Acquire a node from the pool (zero-allocation in common case)
    #[inline(always)]
    pub fn acquire(&self) -> PooledMemoryNode<'_> {
        let node = self.available.pop().unwrap_or_else(|| {
            // Fallback: create new node if pool is empty
            let content = MemoryContent::text(String::with_capacity(1024));
            let mut node = MemoryNode::new(MemoryType::Working, content);

            // Set embedding if requested
            if self.embedding_dimension > 0 {
                let _ = node.set_embedding(vec![0.0; self.embedding_dimension]);
            }

            node
        });

        PooledMemoryNode {
            node: std::mem::ManuallyDrop::new(node),
            pool: self,
            taken: false,
        }
    }

    /// Release a node back to the pool for reuse
    #[inline(always)]
    fn release(&self, node: MemoryNode) {
        // Reset the node to a clean state for reuse
        // The modern MemoryNode doesn't allow direct mutation of its fields,

        // The modern MemoryNode doesn't allow direct mutation of its fields,
        // so we'll just use it as-is for the pool. The creation of new nodes
        // handles the clean state.

        // For a more efficient pool, we could add reset methods to MemoryNode,
        // but for now we'll accept the cost of recreation on acquire.

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
    /// Initialize the pooled node with content
    #[inline(always)]
    pub fn initialize(&mut self, content: String, memory_type: MemoryType) {
        if !self.taken {
            // For the current design, we replace the node entirely since
            // the modern MemoryNode doesn't expose mutable fields directly
            let new_content = MemoryContent::text(content);
            let mut new_node = MemoryNode::new(memory_type, new_content);

            // Calculate base importance from memory type
            let importance = memory_type.base_importance();
            let _ = new_node.set_importance(importance);

            // Replace the node
            let old_node = std::mem::replace(&mut self.node, std::mem::ManuallyDrop::new(new_node));
            std::mem::ManuallyDrop::into_inner(old_node); // Drop the old node
        }
    }

    /// Set embedding for the pooled node
    #[inline(always)]
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        if !self.taken {
            let _ = self.node.set_embedding(embedding);
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
            Some(std::mem::ManuallyDrop::into_inner(std::mem::replace(
                &mut self.node,
                std::mem::ManuallyDrop::new(MemoryNode::new(
                    MemoryType::Working,
                    MemoryContent::text(""),
                )),
            )))
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
            let node = std::mem::ManuallyDrop::into_inner(std::mem::replace(
                &mut self.node,
                std::mem::ManuallyDrop::new(MemoryNode::new(
                    MemoryType::Working,
                    MemoryContent::text(""),
                )),
            ));
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
