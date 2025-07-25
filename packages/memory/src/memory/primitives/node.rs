// src/memory/primitives/node.rs
//! Memory node implementation for the memory system.
//! This module defines the core data structures for memory nodes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;

use super::metadata::MemoryMetadata;
use super::types::MemoryTypeEnum;

/// A memory node in the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique identifier for the memory
    pub id: String,
    /// Content of the memory
    pub content: String,
    /// Type of memory
    pub memory_type: MemoryTypeEnum,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Metadata associated with the memory
    pub metadata: MemoryMetadata}

impl MemoryNode {
    /// Create a new memory node
    pub fn new(content: String, memory_type: MemoryTypeEnum) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        Self {
            id,
            content,
            memory_type,
            created_at: now,
            updated_at: now,
            embedding: None,
            metadata: MemoryMetadata::with_memory_type(memory_type)}
    }

    /// Create a new memory node with a specific ID
    pub fn with_id(id: String, content: String, memory_type: MemoryTypeEnum) -> Self {
        let now = Utc::now();

        Self {
            id,
            content,
            memory_type,
            created_at: now,
            updated_at: now,
            embedding: None,
            metadata: MemoryMetadata::with_memory_type(memory_type)}
    }

    /// Set the embedding for this memory
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding.clone());
        self
    }

    /// Set the importance for this memory
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.metadata.importance = importance;
        self
    }

    /// Add custom metadata to this memory
    pub fn with_custom_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        if let Err(_) = self.metadata.set_custom(&key, value.clone()) {
            // If setting custom fails, create a new object and insert
            if let serde_json::Value::Object(ref mut map) = self.metadata.custom {
                map.insert(key, value);
            } else {
                let mut map = serde_json::Map::new();
                map.insert(key, value);
                self.metadata.custom = serde_json::Value::Object(map);
            }
        }
        self
    }

    /// Update the last accessed time
    pub fn update_last_accessed(&mut self) {
        self.metadata.last_accessed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
}
