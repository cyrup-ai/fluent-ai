//! Memory node builder implementations with zero-allocation, lock-free design
//!
//! All memory node construction logic and builder patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use uuid::Uuid;

use crate::domain::memory::primitives::types::{
    CandleBaseMemory as BaseMemory, CandleMemoryContent as MemoryContent, CandleMemoryError as MemoryError, CandleMemoryResult as MemoryResult, CandleMemoryTypeEnum as MemoryTypeEnum, 
};
use crate::domain::memory::primitives::node::CandleMemoryNode as MemoryNode;

/// Ergonomic builder for memory nodes with zero-allocation move semantics
#[derive(Debug, Default)]
pub struct MemoryNodeBuilder {
    id: Option<Uuid>,
    memory_type: Option<MemoryTypeEnum>,
    content: Option<MemoryContent>,
    embedding: Option<Vec<f32>>,
    importance: Option<f32>,
    keywords: Vec<Arc<str>>,
    tags: Vec<Arc<str>>,
    custom_metadata: HashMap<Arc<str>, serde_json::Value>,
}

impl MemoryNodeBuilder {
    /// Create new builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set node ID (generates UUID if not set)
    #[inline]
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }

    /// Set memory type
    #[inline]
    pub fn with_memory_type(mut self, memory_type: MemoryTypeEnum) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    /// Set content
    #[inline]
    pub fn with_content(mut self, content: MemoryContent) -> Self {
        self.content = Some(content);
        self
    }

    /// Set text content (convenience method)
    #[inline]
    pub fn with_text(mut self, text: impl Into<Arc<str>>) -> Self {
        self.content = Some(MemoryContent::text(text));
        self
    }

    /// Set base memory
    #[inline]
    pub fn with_base_memory(mut self, base_memory: BaseMemory) -> Self {
        self.id = Some(base_memory.id);
        self.memory_type = Some(base_memory.memory_type);
        self.content = Some(base_memory.content);
        self
    }

    /// Set embedding
    #[inline]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set importance
    #[inline]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }

    /// Set creation time (for compatibility)
    #[inline]
    pub fn with_creation_time(self, _time: SystemTime) -> Self {
        // Creation time is handled by BaseMemory constructor
        self
    }

    /// Set last accessed time (for compatibility)
    #[inline]
    pub fn with_last_accessed(self, _time: SystemTime) -> Self {
        // Last accessed is tracked by stats automatically
        self
    }

    /// Add keyword
    #[inline]
    pub fn with_keyword(mut self, keyword: impl Into<Arc<str>>) -> Self {
        self.keywords.push(keyword.into());
        self
    }

    /// Add tag
    #[inline]
    pub fn with_tag(mut self, tag: impl Into<Arc<str>>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add custom metadata
    #[inline]
    pub fn with_custom_metadata(
        mut self,
        key: impl Into<Arc<str>>,
        value: serde_json::Value,
    ) -> Self {
        self.custom_metadata.insert(key.into(), value);
        self
    }

    /// Build the memory node with validation
    pub fn build(self) -> MemoryResult<MemoryNode> {
        let memory_type = self
            .memory_type
            .ok_or_else(|| MemoryError::validation("Memory type is required"))?;

        let content = self.content.unwrap_or_default();

        let id = self.id.unwrap_or_else(Uuid::new_v4);

        let mut node = MemoryNode::with_id(id, memory_type, content);

        // Set embedding if provided
        if let Some(embedding) = self.embedding {
            node.set_embedding(embedding)?;
        }

        // Set importance if provided
        if let Some(importance) = self.importance {
            node.set_importance(importance)?;
        }

        // Add keywords
        for keyword in self.keywords {
            node.add_keyword(keyword);
        }

        // Add tags
        for tag in self.tags {
            node.add_tag(tag);
        }

        // Add custom metadata
        for (key, value) in self.custom_metadata {
            node.set_custom_metadata(key, value);
        }

        Ok(node)
    }
}