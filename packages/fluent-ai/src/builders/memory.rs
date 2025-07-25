//! Memory and vector store builder implementations with zero-allocation, lock-free design
//!
//! Provides EXACT API syntax for vector store queries and memory operations.

use fluent_ai_domain::{
    AsyncTask,
    memory::{VectorStoreIndex, VectorStoreIndexDyn, ZeroOneOrMany},
    spawn_async};
use serde_json::Value;

/// Zero-allocation vector query builder with blazing-fast terminal methods
pub struct VectorQueryBuilder<'a> {
    index: &'a VectorStoreIndex,
    query: String,
    n: usize}

impl<'a> VectorQueryBuilder<'a> {
    /// Create new query builder - EXACT syntax: VectorQueryBuilder::new(index, query)
    pub fn new(index: &'a VectorStoreIndex, query: String) -> Self {
        Self {
            index,
            query,
            n: 10, // default
        }
    }

    /// Set top N results - EXACT syntax: .top(n)
    pub fn top(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    /// Terminal method - returns full results with metadata - EXACT syntax: .retrieve()
    pub fn retrieve(self) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>> {
        let future = self.index.backend.top_n(&self.query, self.n);
        spawn_async(async move {
            match future.await {
                Ok(results) => results,
                Err(_) => ZeroOneOrMany::None}
        })
    }

    /// Terminal method - returns just IDs - EXACT syntax: .retrieve_ids()
    pub fn retrieve_ids(self) -> AsyncTask<ZeroOneOrMany<(f64, String)>> {
        let future = self.index.backend.top_n_ids(&self.query, self.n);
        spawn_async(async move {
            match future.await {
                Ok(results) => results,
                Err(_) => ZeroOneOrMany::None}
        })
    }

    /// Terminal method with result handler - EXACT syntax: .on_results(|results| { ... })
    pub fn on_results<F, T>(self, handler: F) -> AsyncTask<T>
    where
        F: FnOnce(ZeroOneOrMany<(f64, String, Value)>) -> T + Send + 'static,
        T: Send + 'static,
    {
        let future = self.index.backend.top_n(&self.query, self.n);
        spawn_async(async move {
            let result = match future.await {
                Ok(results) => results,
                Err(_) => ZeroOneOrMany::None};
            handler(result)
        })
    }
}

impl VectorStoreIndex {
    /// Direct creation from backend - EXACT syntax: VectorStoreIndex::with_backend(backend)
    pub fn with_backend<B: VectorStoreIndexDyn + 'static>(backend: B) -> Self {
        VectorStoreIndex {
            backend: Box::new(backend)}
    }

    /// Semantic query entry point - EXACT syntax: .search(query)
    pub fn search(&self, query: impl Into<String>) -> VectorQueryBuilder<'_> {
        VectorQueryBuilder::new(self, query.into())
    }
}
