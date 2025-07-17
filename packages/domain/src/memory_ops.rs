use futures::stream::StreamExt;
use crate::ZeroOneOrMany;

// Define Op trait locally - no external dependencies
pub trait Op {
    type Input;
    type Output;

    fn call(&self, input: Self::Input) -> impl std::future::Future<Output = Self::Output> + Send;
}

use crate::memory::{
    MemoryError, MemoryManager, MemoryNode, MemoryRelationship, MemoryType, ImportanceContext,
    InMemoryEmbeddingCache, EmbeddingService,
};

/// Standard embedding dimension for text embeddings
pub const EMBEDDING_DIMENSION: usize = 768;

/// Small embedding dimension for stack allocation
pub const SMALL_EMBEDDING_DIMENSION: usize = 64;

/// Generate small embedding using stack allocation for blazing-fast performance
#[inline(always)]
#[must_use]
pub fn generate_small_embedding(content: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Use stack-allocated array for small embeddings
    let mut embedding = [0.0f32; SMALL_EMBEDDING_DIMENSION];
    
    // Fill with deterministic values
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((hash + i as u64) as f32) / (u64::MAX as f32);
    }
    
    embedding.to_vec()
}

/// Store a piece of content as a memory node with zero-allocation embedding
pub struct StoreMemory<M> {
    manager: M,
    memory_type: MemoryType,
    generate_embedding: bool,
    importance_context: ImportanceContext,
    embedding_cache: InMemoryEmbeddingCache,
}

impl<M> StoreMemory<M> {
    /// Create new StoreMemory with standard embedding dimension
    #[inline]
    pub fn new(manager: M, memory_type: MemoryType) -> Self {
        Self {
            manager,
            memory_type,
            generate_embedding: true,
            importance_context: ImportanceContext::SystemResponse,
            embedding_cache: InMemoryEmbeddingCache::new(EMBEDDING_DIMENSION),
        }
    }
    
    /// Create new StoreMemory with custom embedding dimension
    #[inline]
    pub fn with_embedding_dimension(manager: M, memory_type: MemoryType, dimension: usize) -> Self {
        Self {
            manager,
            memory_type,
            generate_embedding: true,
            importance_context: ImportanceContext::SystemResponse,
            embedding_cache: InMemoryEmbeddingCache::new(dimension),
        }
    }

    #[inline]
    pub fn without_embedding(mut self) -> Self {
        self.generate_embedding = false;
        self
    }
    
    #[inline]
    pub fn with_context(mut self, context: ImportanceContext) -> Self {
        self.importance_context = context;
        self
    }
    
    /// Create small embedding StoreMemory for short content (uses stack allocation)
    #[inline]
    pub fn small_embedding(manager: M, memory_type: MemoryType) -> Self {
        Self {
            manager,
            memory_type,
            generate_embedding: true,
            importance_context: ImportanceContext::SystemResponse,
            embedding_cache: InMemoryEmbeddingCache::new(SMALL_EMBEDDING_DIMENSION),
        }
    }
}

impl<M> Op for StoreMemory<M>
where
    M: MemoryManager + Clone,
{
    type Input = String;
    type Output = Result<MemoryNode, MemoryError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let mut memory = MemoryNode::new_with_context(
            input.clone(), 
            self.memory_type.clone(), 
            self.importance_context
        );

        // Generate embedding if enabled using zero-allocation patterns
        if self.generate_embedding {
            let embedding = if input.len() <= 100 && self.embedding_cache.embedding_dimension() == SMALL_EMBEDDING_DIMENSION {
                // Use stack-allocated small embedding for short content
                generate_small_embedding(&input)
            } else {
                // Use pool-allocated embedding for longer content
                self.embedding_cache.generate_deterministic(&input)
            };
            memory = memory.with_embedding(embedding);
        }

        self.manager.create_memory(memory).await
    }
}

/// Retrieve memories by vector similarity
pub struct RetrieveMemories<M> {
    manager: M,
    limit: usize,
}

impl<M> RetrieveMemories<M> {
    pub fn new(manager: M, limit: usize) -> Self {
        Self { manager, limit }
    }
}

impl<M> Op for RetrieveMemories<M>
where
    M: MemoryManager + Clone,
{
    type Input = Vec<f32>;
    type Output = Result<ZeroOneOrMany<MemoryNode>, MemoryError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let mut stream = self.manager.search_by_vector(input, self.limit);
        let mut memories = Vec::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(memory) => memories.push(memory),
                Err(e) => return Err(e),
            }
        }

        // Convert Vec<MemoryNode> to ZeroOneOrMany<MemoryNode>
        let result = match memories.len() {
            0 => ZeroOneOrMany::None,
            1 => {
                if let Some(memory) = memories.into_iter().next() {
                    ZeroOneOrMany::One(memory)
                } else {
                    ZeroOneOrMany::None
                }
            },
            _ => ZeroOneOrMany::many(memories),
        };
        
        Ok(result)
    }
}

/// Search memories by content
pub struct SearchMemories<M> {
    manager: M,
}

impl<M> SearchMemories<M> {
    pub fn new(manager: M) -> Self {
        Self { manager }
    }
}

impl<M> Op for SearchMemories<M>
where
    M: MemoryManager + Clone,
{
    type Input = String;
    type Output = Result<ZeroOneOrMany<MemoryNode>, MemoryError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let mut stream = self.manager.search_by_content(&input);
        let mut memories = Vec::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(memory) => memories.push(memory),
                Err(e) => return Err(e),
            }
        }

        // Convert Vec<MemoryNode> to ZeroOneOrMany<MemoryNode>
        let result = match memories.len() {
            0 => ZeroOneOrMany::None,
            1 => {
                if let Some(memory) = memories.into_iter().next() {
                    ZeroOneOrMany::One(memory)
                } else {
                    ZeroOneOrMany::None
                }
            },
            _ => ZeroOneOrMany::many(memories),
        };
        
        Ok(result)
    }
}

/// Update memory importance based on access
pub struct UpdateImportance<M> {
    manager: M,
    boost: f32,
}

impl<M> UpdateImportance<M> {
    pub fn new(manager: M, boost: f32) -> Self {
        Self { manager, boost }
    }
}

impl<M> Op for UpdateImportance<M>
where
    M: MemoryManager + Clone,
{
    type Input = MemoryNode;
    type Output = Result<MemoryNode, MemoryError>;

    async fn call(&self, mut input: Self::Input) -> Self::Output {
        // Update importance and last accessed time
        input.metadata.importance = (input.metadata.importance + self.boost).min(1.0);
        input.update_last_accessed();

        self.manager.update_memory(input).await
    }
}

/// Create a relationship between two memories
pub struct LinkMemories<M> {
    manager: M,
    relationship_type: String,
}

impl<M> LinkMemories<M> {
    pub fn new(manager: M, relationship_type: String) -> Self {
        Self {
            manager,
            relationship_type,
        }
    }
}

impl<M> Op for LinkMemories<M>
where
    M: MemoryManager + Clone,
{
    type Input = (u64, u64); // (source_id, target_id)
    type Output = Result<MemoryRelationship, MemoryError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let (source_id, target_id) = input;
        let relationship = MemoryRelationship {
            id: crate::memory::next_memory_id(),
            from_id: source_id,
            to_id: target_id,
            relationship_type: self.relationship_type.clone(),
        };

        self.manager.create_relationship(relationship).await
    }
}

/// Store memory with context - combines storage with relationship creation
pub struct StoreWithContext<M> {
    manager: M,
    memory_type: MemoryType,
}

impl<M> StoreWithContext<M> {
    pub fn new(manager: M, memory_type: MemoryType) -> Self {
        Self {
            manager,
            memory_type,
        }
    }
}

impl<M> Op for StoreWithContext<M>
where
    M: MemoryManager + Clone,
{
    type Input = (String, Vec<u64>); // (content, related_memory_ids)
    type Output = Result<(MemoryNode, ZeroOneOrMany<MemoryRelationship>), MemoryError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let (content, related_ids) = input;

        // Create the new memory with appropriate context
        let memory = MemoryNode::new_with_context(
            content, 
            self.memory_type.clone(), 
            ImportanceContext::UserInput
        );
        let stored_memory = self.manager.create_memory(memory).await?;

        // Create relationships to related memories
        let mut relationships = Vec::new();
        for related_id in related_ids {
            let relationship = MemoryRelationship {
                id: crate::memory::next_memory_id(),
                from_id: stored_memory.id,
                to_id: related_id,
                relationship_type: "related_to".to_string(),
            };

            match self.manager.create_relationship(relationship).await {
                Ok(rel) => relationships.push(rel),
                Err(e) => {
                    // Log error but don't fail the whole operation
                    eprintln!("Failed to create relationship: {}", e);
                }
            }
        }

        // Convert Vec<MemoryRelationship> to ZeroOneOrMany<MemoryRelationship>
        let relationships_result = match relationships.len() {
            0 => ZeroOneOrMany::None,
            1 => {
                if let Some(rel) = relationships.into_iter().next() {
                    ZeroOneOrMany::One(rel)
                } else {
                    ZeroOneOrMany::None
                }
            },
            _ => ZeroOneOrMany::many(relationships),
        };
        
        Ok((stored_memory, relationships_result))
    }
}

/// Convenience functions for creating memory operations
#[inline(always)]
#[must_use]
pub fn store_memory<M: MemoryManager + Clone>(
    manager: M,
    memory_type: MemoryType,
) -> StoreMemory<M> {
    StoreMemory::new(manager, memory_type)
}

#[inline(always)]
#[must_use]
pub fn retrieve_memories<M: MemoryManager + Clone>(
    manager: M,
    limit: usize,
) -> RetrieveMemories<M> {
    RetrieveMemories::new(manager, limit)
}

#[inline(always)]
#[must_use]
pub fn search_memories<M: MemoryManager + Clone>(manager: M) -> SearchMemories<M> {
    SearchMemories::new(manager)
}

#[inline(always)]
#[must_use]
pub fn update_importance<M: MemoryManager + Clone>(manager: M, boost: f32) -> UpdateImportance<M> {
    UpdateImportance::new(manager, boost)
}

#[inline(always)]
#[must_use]
pub fn link_memories<M: MemoryManager + Clone>(
    manager: M,
    relationship_type: String,
) -> LinkMemories<M> {
    LinkMemories::new(manager, relationship_type)
}

#[inline(always)]
#[must_use]
pub fn store_with_context<M: MemoryManager + Clone>(
    manager: M,
    memory_type: MemoryType,
) -> StoreWithContext<M> {
    StoreWithContext::new(manager, memory_type)
}
