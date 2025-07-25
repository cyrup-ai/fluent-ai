//! Embedding and Memory Manager Traits
//!
//! Immutable trait definitions for streaming operations with zero-allocation patterns.

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use super::memory::MemoryNode;
use super::errors::ValidationError;

/// Embedding model information with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    pub name: String,
    pub version: String,
    pub vector_dimension: usize,
    pub max_input_length: usize,
    pub supported_languages: Vec<String>,
}

impl EmbeddingModelInfo {
    /// Create new embedding model info
    pub fn new(
        name: String,
        version: String,
        vector_dimension: usize,
        max_input_length: usize,
    ) -> Self {
        Self {
            name,
            version,
            vector_dimension,
            max_input_length,
            supported_languages: Vec::new(),
        }
    }

    /// Add supported language
    pub fn with_language(mut self, language: String) -> Self {
        self.supported_languages.push(language);
        self
    }

    /// Check if language is supported
    pub fn supports_language(&self, language: &str) -> bool {
        self.supported_languages.is_empty() || self.supported_languages.contains(&language.to_string())
    }
}

/// Immutable embedding model with streaming operations
pub trait ImmutableEmbeddingModel: Send + Sync + 'static {
    /// Generate embeddings for text with streaming results - returns unwrapped values
    fn embed(&self, text: &str, context: Option<String>) -> AsyncStream<Vec<f32>>;

    /// Get model information
    fn model_info(&self) -> EmbeddingModelInfo;

    /// Validate input text
    fn validate_input(&self, text: &str) -> Result<(), ValidationError>;
}

/// Memory manager information with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagerInfo {
    pub name: String,
    pub version: String,
    pub storage_type: String,
    pub max_memory_nodes: Option<usize>,
    pub supported_operations: Vec<String>,
}

impl MemoryManagerInfo {
    /// Create new memory manager info
    pub fn new(name: String, version: String, storage_type: String) -> Self {
        Self {
            name,
            version,
            storage_type,
            max_memory_nodes: None,
            supported_operations: Vec::new(),
        }
    }

    /// Set maximum memory nodes
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_memory_nodes = Some(max_nodes);
        self
    }

    /// Add supported operation
    pub fn with_operation(mut self, operation: String) -> Self {
        self.supported_operations.push(operation);
        self
    }

    /// Check if operation is supported
    pub fn supports_operation(&self, operation: &str) -> bool {
        self.supported_operations.contains(&operation.to_string())
    }
}

/// Immutable memory manager with streaming operations
pub trait ImmutableMemoryManager: Send + Sync + 'static {
    /// Create memory with streaming confirmation - returns unwrapped values
    fn create_memory(&self, node: MemoryNode) -> AsyncStream<()>;

    /// Search by vector with streaming results - returns unwrapped values
    fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> AsyncStream<MemoryNode>;

    /// Search by text with streaming results - returns unwrapped values
    fn search_by_text(&self, query: &str, limit: usize) -> AsyncStream<MemoryNode>;

    /// Update memory with streaming confirmation - returns unwrapped values
    fn update_memory(&self, memory_id: &str, node: MemoryNode) -> AsyncStream<()>;

    /// Delete memory with streaming confirmation - returns unwrapped values
    fn delete_memory(&self, memory_id: &str) -> AsyncStream<()>;

    /// Get memory manager information
    fn manager_info(&self) -> MemoryManagerInfo;
}

/// Backward compatibility aliases (deprecated)
#[deprecated(note = "Use ImmutableEmbeddingModel instead")]
pub trait EmbeddingModel: ImmutableEmbeddingModel {}

#[deprecated(note = "Use ImmutableMemoryManager instead")]
pub trait MemoryManager: ImmutableMemoryManager {}