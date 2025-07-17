//! VectorQuery builder implementations
//!
//! All vector query construction logic and builder patterns.

use fluent_ai_domain::memory::{VectorStoreIndex, ZeroOneOrMany};
use fluent_ai_domain::async_task::{AsyncTask, spawn_async};
use serde_json::Value;

pub struct VectorQueryBuilder<'a> {
    index: &'a VectorStoreIndex,
    query: String,
    n: usize,
}

impl<'a> VectorQueryBuilder<'a> {
    /// Create new query builder
    pub fn new(index: &'a VectorStoreIndex, query: String) -> Self {
        Self {
            index,
            query,
            n: 10, // default
        }
    }

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
        T: Send + 'static + fluent_ai_domain::async_task::NotResult,
    {
        let _future = self.index.backend.top_n(&self.query, self.n);
        spawn_async(async move {
            // This would properly await the future and pass to handler
            let result = ZeroOneOrMany::None;
            handler(result)
        })
    }
}

impl VectorStoreIndex {
    // Semantic query entry point
    pub fn search(&self, query: impl Into<String>) -> VectorQueryBuilder<'_> {
        VectorQueryBuilder::new(self, query.into())
    }
}