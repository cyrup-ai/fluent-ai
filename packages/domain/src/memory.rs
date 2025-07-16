use crate::async_task::{AsyncTask, spawn_async};
use crate::ZeroOneOrMany;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

#[derive(Debug)]
pub enum VectorStoreError {
    NotFound,
    ConnectionError(String),
    InvalidQuery(String),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::NotFound => write!(f, "Vector store item not found"),
            VectorStoreError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            VectorStoreError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
        }
    }
}

impl std::error::Error for VectorStoreError {}

pub type Error = VectorStoreError;

#[derive(Debug)]
pub enum MemoryError {
    NotFound,
    StorageError(String),
    ValidationError(String),
    NetworkError(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::NotFound => write!(f, "Memory not found"),
            MemoryError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            MemoryError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            MemoryError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl From<MemoryError> for VectorStoreError {
    fn from(error: MemoryError) -> Self {
        match error {
            MemoryError::NotFound => VectorStoreError::NotFound,
            MemoryError::StorageError(msg) => VectorStoreError::ConnectionError(msg),
            MemoryError::ValidationError(msg) => VectorStoreError::InvalidQuery(msg),
            MemoryError::NetworkError(msg) => VectorStoreError::ConnectionError(msg),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Semantic,
    Episodic,
}

#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub metadata: MemoryMetadata,
}

#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub last_accessed: std::time::SystemTime,
    pub creation_time: std::time::SystemTime,
}

impl MemoryNode {
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            memory_type,
            metadata: MemoryMetadata {
                importance: 0.5,
                last_accessed: std::time::SystemTime::now(),
                creation_time: std::time::SystemTime::now(),
            },
        }
    }
    
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.metadata.importance = importance;
        self
    }
    
    pub fn with_embedding(self, _embedding: Vec<f32>) -> Self {
        // TODO: Add embedding field to MemoryNode or metadata
        self
    }
    
    pub fn update_last_accessed(&mut self) {
        self.metadata.last_accessed = std::time::SystemTime::now();
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRelationship {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relationship_type: String,
}

/// Trait for managing memory operations
pub trait MemoryManager: Send + Sync {
    /// Create a new memory node
    fn create_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>>;
    
    /// Update an existing memory node
    fn update_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>>;
    
    /// Get a memory node by ID
    fn get_memory(&self, id: &str) -> BoxFuture<Result<Option<MemoryNode>, MemoryError>>;
    
    /// Search memories by vector similarity - returns a stream of results
    fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>>;
    
    /// Search memories by content - returns a stream of results
    fn search_by_content(&self, content: &str) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>>;
    
    /// Create a memory relationship
    fn create_relationship(&self, relationship: MemoryRelationship) -> BoxFuture<Result<MemoryRelationship, MemoryError>>;
}

/// Default in-memory implementation of MemoryManager
#[derive(Debug, Clone)]
pub struct InMemoryManager {
    pub nodes: Vec<MemoryNode>,
    pub relationships: Vec<MemoryRelationship>,
}

impl InMemoryManager {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            relationships: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, node: MemoryNode) {
        self.nodes.push(node);
    }
    
    pub fn add_relationship(&mut self, relationship: MemoryRelationship) {
        self.relationships.push(relationship);
    }
}


impl MemoryManager for InMemoryManager {
    fn create_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would persist to storage
            Ok(memory)
        })
    }
    
    fn update_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would update in storage
            Ok(memory)
        })
    }
    
    fn get_memory(&self, id: &str) -> BoxFuture<Result<Option<MemoryNode>, MemoryError>> {
        let id = id.to_string();
        Box::pin(async move {
            // In a real implementation, this would search in storage
            // For now, return None
            let _ = id;
            Ok(None)
        })
    }
    
    fn search_by_vector(&self, _vector: Vec<f32>, _limit: usize) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>> {
        // Return empty stream for now
        crate::async_task::AsyncStream::empty()
    }
    
    fn search_by_content(&self, _content: &str) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>> {
        // Return empty stream for now
        crate::async_task::AsyncStream::empty()
    }
    
    fn create_relationship(&self, relationship: MemoryRelationship) -> BoxFuture<Result<MemoryRelationship, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would persist to storage
            Ok(relationship)
        })
    }
}

pub type Memory = InMemoryManager;

pub trait VectorStoreIndexDyn: Send + Sync {
    fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>>;
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<ZeroOneOrMany<(f64, String)>>;
}

pub struct VectorStoreIndex {
    backend: Box<dyn VectorStoreIndexDyn>,
}

pub struct VectorQueryBuilder<'a> {
    index: &'a VectorStoreIndex,
    query: String,
    n: usize,
}

impl VectorStoreIndex {
    // Direct creation from backend
    pub fn with_backend<B: VectorStoreIndexDyn + 'static>(backend: B) -> Self {
        VectorStoreIndex {
            backend: Box::new(backend),
        }
    }

    // Semantic query entry point
    pub fn search(&self, query: impl Into<String>) -> VectorQueryBuilder<'_> {
        VectorQueryBuilder {
            index: self,
            query: query.into(),
            n: 10, // default
        }
    }
}

impl<'a> VectorQueryBuilder<'a> {
    pub fn top(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    // Terminal method - returns full results with metadata
    pub fn retrieve(self) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>> {
        let _future = self.index.backend.top_n(&self.query, self.n);
        spawn_async(async move {
            // This would properly await the future
            ZeroOneOrMany::None
        })
    }

    // Terminal method - returns just IDs
    pub fn retrieve_ids(self) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        let _future = self.index.backend.top_n_ids(&self.query, self.n);
        spawn_async(async move {
            // This would properly await the future
            ZeroOneOrMany::None
        })
    }

    // Terminal method with result handler
    pub fn on_results<F, T>(self, handler: F) -> AsyncTask<T>
    where
        F: FnOnce(ZeroOneOrMany<(f64, String, Value)>) -> T + Send + 'static,
        T: Send + 'static + crate::async_task::NotResult,
    {
        let _future = self.index.backend.top_n(&self.query, self.n);
        spawn_async(async move {
            // This would properly await the future and pass to handler
            let result = ZeroOneOrMany::None;
            handler(result)
        })
    }
}
