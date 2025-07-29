//! Memory system integration for fluent-ai
//! Implements the sweetmcp-memory traits and library memory services

use crate::prelude::*;

// Library memory integration service
pub mod library;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
pub use library::{LibraryExt, LibraryMemoryService, LibraryServiceError, LibraryServiceResult};
use parking_lot::RwLock;
// TODO: Re-export key types from sweetmcp-memory when available
// pub use sweetmcp_memory::{
//     MemoryManager as MemoryManagerTrait, MemoryMetadata, MemoryNode, MemoryRelationship, MemoryType};

// Placeholder types until sweetmcp_memory is added
#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub metadata: MemoryMetadata}

impl MemoryNode {
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            memory_type,
            metadata: MemoryMetadata::default()}
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.metadata.embedding = Some(embedding);
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.metadata.importance = importance;
        self
    }

    pub fn update_last_accessed(&mut self) {
        // Placeholder for updating last accessed time
    }
}

#[derive(Debug, Clone, Default)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub embedding: Option<Vec<f32>>}

#[derive(Debug, Clone)]
pub enum MemoryType {
    Semantic,
    Episodic,
    Procedural}

#[derive(Debug, Clone)]
pub struct MemoryRelationship {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relationship_type: String,
    pub metadata: Option<serde_json::Value>}

// Error type for memory operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Memory not found")]
    NotFound,
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Memory manager not configured - call with_manager() first")]
    ManagerNotConfigured}

pub trait MemoryManagerTrait: Send + Sync {
    fn get_memory(
        &self,
        _id: &str,
    ) -> fluent_ai_domain::AsyncStream<Option<MemoryNode>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        fluent_ai_domain::AsyncStream::with_channel(move |sender| {
            // Use handle_error! for proper internal error handling
            let _ = sender.send(None);
        })
    }
    fn delete_memory(&self, _id: &str) -> fluent_ai_domain::AsyncStream<bool> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = tx.send(false);
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
    fn create_memory(
        &self,
        node: MemoryNode,
    ) -> fluent_ai_domain::AsyncStream<MemoryNode> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = tx.send(Ok(node));
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
    fn update_memory(
        &self,
        node: MemoryNode,
    ) -> fluent_ai_domain::AsyncStream<MemoryNode> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = tx.send(Ok(node));
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
    fn create_relationship(
        &self,
        rel: MemoryRelationship,
    ) -> fluent_ai_domain::AsyncStream<MemoryRelationship> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = tx.send(Ok(rel));
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
    fn query_by_type(
        &self,
        _memory_type: MemoryType,
    ) -> Box<dyn Stream<Item = Result<MemoryNode, Error>> + Send + Unpin> {
        Box::new(futures::stream::empty())
    }
    fn search_by_content(
        &self,
        _query: &str,
    ) -> Box<dyn Stream<Item = Result<MemoryNode, Error>> + Send + Unpin> {
        Box::new(futures::stream::empty())
    }
    fn search_by_vector(
        &self,
        _vector: Vec<f32>,
        _limit: usize,
    ) -> Box<dyn Stream<Item = Result<MemoryNode, Error>> + Send + Unpin> {
        Box::new(futures::stream::empty())
    }
}

// Alias for compatibility
pub trait MemoryManager: MemoryManagerTrait {}

/// Fluent memory builder
pub struct Memory {
    inner: Arc<RwLock<MemoryBuilder>>}

struct MemoryBuilder {
    manager: Option<Arc<dyn MemoryManagerTrait>>,
    default_type: MemoryType}

impl Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("inner", &"Arc<RwLock<MemoryBuilder>>")
            .finish()
    }
}

impl Clone for Memory {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner)}
    }
}

impl Memory {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MemoryBuilder {
                manager: None,
                default_type: MemoryType::Semantic}))}
    }

    /// Set the memory manager backend
    pub fn with_manager(self, manager: impl MemoryManagerTrait + 'static) -> Self {
        self.inner.write().manager = Some(Arc::new(manager));
        self
    }

    /// Set default memory type for new memories
    pub fn default_type(self, memory_type: MemoryType) -> Self {
        self.inner.write().default_type = memory_type;
        self
    }

    /// Store a memory
    pub fn memorize(self, content: impl Into<String>) -> StoreMemory {
        StoreMemory {
            memory: self,
            content: content.into(),
            memory_type: None,
            metadata: None}
    }

    /// Query memories by content
    pub fn recall(self, query: impl Into<String>) -> QueryMemory {
        QueryMemory {
            memory: self,
            query_type: QueryType::ByContent(query.into()),
            limit: 10}
    }

    /// Recall all memories
    pub fn recall_all(self) -> QueryMemory {
        QueryMemory {
            memory: self,
            query_type: QueryType::All,
            limit: 10}
    }

    /// Get a specific memory by ID
    pub fn get(self, id: impl Into<String>) -> AsyncTask<Option<MemoryNode>> {
        let id = id.into();
        let manager = match self.inner.read().manager.clone() {
            Some(manager) => manager,
            None => return crate::async_task::spawn_async(async move { None })};

        crate::async_task::spawn_async(async move {
            match manager.get_memory(&id).await {
                Ok(result) => result,
                Err(_) => None}
        })
    }

    /// Delete a memory
    pub fn delete(self, id: impl Into<String>) -> AsyncTask<bool> {
        let id = id.into();
        let manager = match self.inner.read().manager.clone() {
            Some(manager) => manager,
            None => return crate::async_task::spawn_async(async move { false })};

        crate::async_task::spawn_async(async move { manager.delete_memory(&id).await })
    }
}

/// Builder for storing memories
pub struct StoreMemory {
    memory: Memory,
    content: String,
    memory_type: Option<MemoryType>,
    metadata: Option<MemoryMetadata>}

impl StoreMemory {
    pub fn memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    pub fn metadata(mut self, metadata: MemoryMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn importance(mut self, importance: f32) -> Self {
        let mut metadata = self.metadata.unwrap_or_default();
        metadata.importance = importance;
        self.metadata = Some(metadata);
        self
    }

    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        let mut metadata = self.metadata.unwrap_or_default();
        metadata.embedding = Some(embedding);
        self.metadata = Some(metadata);
        self
    }

    /// Memorize the content
    pub fn memorize(self) -> AsyncTask<MemoryNode> {
        let memory_type = self
            .memory_type
            .unwrap_or_else(|| self.memory.inner.read().default_type.clone());

        let mut node = MemoryNode::new(self.content, memory_type);
        if let Some(metadata) = self.metadata {
            node.metadata = metadata;
        }

        let manager = match self.memory.inner.read().manager.clone() {
            Some(manager) => manager,
            None => return crate::async_task::spawn_async(async move { node })};

        crate::async_task::spawn_async(async move {
            match manager.create_memory(node.clone()).await {
                Ok(result) => result,
                Err(_) => node, // Return the original node if storage fails
            }
        })
    }
}

/// Query builder for memories
pub struct QueryMemory {
    memory: Memory,
    query_type: QueryType,
    limit: usize}

enum QueryType {
    All,
    ByType(MemoryType),
    ByContent(String),
    ByVector(Vec<f32>)}

impl QueryMemory {
    pub fn by_type(mut self, memory_type: MemoryType) -> Self {
        self.query_type = QueryType::ByType(memory_type);
        self
    }

    pub fn by_content(mut self, query: impl Into<String>) -> Self {
        self.query_type = QueryType::ByContent(query.into());
        self
    }

    pub fn by_vector(mut self, vector: Vec<f32>) -> Self {
        self.query_type = QueryType::ByVector(vector);
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Recall memories based on the query
    pub fn recall(self) -> AsyncStream<MemoryNode> {
        let manager = match self.memory.inner.read().manager.clone() {
            Some(manager) => manager,
            None => {
                let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
                // No manager configured - return empty stream
                return AsyncStream::new(rx);
            }
        };

        let limit = self.limit;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        match self.query_type {
            QueryType::All => {
                tokio::spawn(async move {
                    let stream = manager.query_by_type(MemoryType::Semantic);
                    let results = stream.collect();
                    for result in results {
                        match result {
                            Ok(node) => {
                                if tx.send(node).is_err() {
                                    break;
                                }
                            }
                            Err(_) => {
                                // Skip errors, continue with next items
                                continue;
                            }
                        }
                    }
                });
            }
            QueryType::ByType(memory_type) => {
                tokio::spawn(async move {
                    let stream = manager.query_by_type(memory_type);
                    let results = stream.collect();
                    for result in results {
                        match result {
                            Ok(node) => {
                                if tx.send(node).is_err() {
                                    break;
                                }
                            }
                            Err(_) => {
                                // Skip errors, continue with next items
                                continue;
                            }
                        }
                    }
                });
            }
            QueryType::ByContent(query) => {
                tokio::spawn(async move {
                    let stream = manager.search_by_content(&query);
                    let results = stream.collect();
                    for result in results {
                        match result {
                            Ok(node) => {
                                if tx.send(node).is_err() {
                                    break;
                                }
                            }
                            Err(_) => {
                                // Skip errors, continue with next items
                                continue;
                            }
                        }
                    }
                });
            }
            QueryType::ByVector(vector) => {
                tokio::spawn(async move {
                    let stream = manager.search_by_vector(vector, limit);
                    let results = stream.collect();
                    for result in results {
                        match result {
                            Ok(node) => {
                                if tx.send(node).is_err() {
                                    break;
                                }
                            }
                            Err(_) => {
                                // Skip errors, continue with next items
                                continue;
                            }
                        }
                    }
                });
            }
        };

        AsyncStream::new(rx)
    }

    /// Collect all results into a ZeroOneOrMany
    pub fn collect(self) -> AsyncTask<ZeroOneOrMany<MemoryNode>> {
        let mut stream = self.recall();
        crate::async_task::spawn_async(async move {
            use futures_util::StreamExt;
            let mut pinned_stream = std::pin::Pin::new(&mut stream);

            let mut results = Vec::new();
            while let Some(node) = pinned_stream.next().await {
                results.push(node);
            }

            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => {
                    if let Some(result) = results.into_iter().next() {
                        ZeroOneOrMany::One(result)
                    } else {
                        ZeroOneOrMany::None
                    }
                }
                _ => ZeroOneOrMany::from_vec(results)}
        })
    }
}

/// Create relationships between memories
pub struct Relationship {
    source_id: String,
    target_id: String,
    relationship_type: String,
    metadata: Option<serde_json::Value>}

impl Relationship {
    pub fn new(source_id: impl Into<String>, target_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            relationship_type: "related".to_string(),
            metadata: None}
    }

    pub fn relationship_type(mut self, rel_type: impl Into<String>) -> Self {
        self.relationship_type = rel_type.into();
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn create(self, memory: &Memory) -> AsyncTask<MemoryRelationship> {
        let manager = match memory.inner.read().manager.clone() {
            Some(manager) => manager,
            None => {
                return crate::async_task::spawn_async(async move {
                    // Return default relationship when manager not configured
                    MemoryRelationship {
                        id: uuid::Uuid::new_v4().to_string(),
                        source_id: String::new(),
                        target_id: String::new(),
                        relationship_type: "error".to_string(),
                        metadata: None}
                });
            }
        };

        let relationship = MemoryRelationship {
            id: uuid::Uuid::new_v4().to_string(),
            source_id: self.source_id,
            target_id: self.target_id,
            relationship_type: self.relationship_type,
            metadata: self.metadata};

        crate::async_task::spawn_async(async move {
            match manager.create_relationship(relationship.clone()).await {
                Ok(rel) => rel,
                Err(_) => relationship, // Return original if creation fails
            }
        })
    }
}
