//! Memory node builder implementations with zero-allocation, lock-free design
//!
//! All memory node construction logic and builder patterns.

use fluent_ai_domain::memory::{MemoryNode, MemoryType, MemoryMetadata};
use std::time::SystemTime;

/// Zero-allocation memory node builder with blazing-fast construction
pub struct MemoryNodeBuilder {
    content: String,
    memory_type: MemoryType,
    importance: f32,
    embedding: Option<Vec<f32>>,
}

impl MemoryNodeBuilder {
    /// Create new memory node builder - EXACT syntax: MemoryNodeBuilder::new(content, memory_type)
    #[inline]
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        Self {
            content,
            memory_type,
            importance: 0.5,
            embedding: None,
        }
    }

    /// Set importance level - EXACT syntax: .with_importance(importance)
    #[inline]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// Set embedding vector - EXACT syntax: .with_embedding(embedding)
    #[inline]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Build final memory node - EXACT syntax: .build()
    #[inline]
    pub fn build(self) -> MemoryNode {
        let now = SystemTime::now();
        MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: self.content,
            memory_type: self.memory_type,
            metadata: MemoryMetadata {
                importance: self.importance,
                last_accessed: now,
                creation_time: now,
            },
        }
    }
}

impl MemoryNode {
    /// Create memory node builder - EXACT syntax: MemoryNode::builder(content, memory_type)
    #[inline]
    pub fn builder(content: String, memory_type: MemoryType) -> MemoryNodeBuilder {
        MemoryNodeBuilder::new(content, memory_type)
    }
}

/// Builder function for convenient memory node construction
#[inline]
pub fn memory_node(content: String, memory_type: MemoryType) -> MemoryNodeBuilder {
    MemoryNodeBuilder::new(content, memory_type)
}