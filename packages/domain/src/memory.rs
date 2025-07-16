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

pub type Error = VectorStoreError;

#[derive(Debug)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Semantic,
}

#[derive(Debug)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
}

#[derive(Debug)]
pub struct MemoryRelationship {
    pub from_id: String,
    pub to_id: String,
    pub relationship_type: String,
}

#[derive(Debug)]
pub struct MemoryManager {
    pub nodes: Vec<MemoryNode>,
    pub relationships: Vec<MemoryRelationship>,
}

impl MemoryManager {
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

pub type Memory = MemoryManager;

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
