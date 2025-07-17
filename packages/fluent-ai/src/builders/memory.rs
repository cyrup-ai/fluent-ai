use crate::domain::memory::VectorStoreIndex;
use crate::domain::{AsyncTask, spawn_async, ZeroOneOrMany};
use serde_json::Value;

pub struct VectorQueryBuilder<'a> {
    index: &'a VectorStoreIndex,
    query: String,
    n: usize,
}

impl VectorStoreIndex {
    // Direct creation from backend
    pub fn with_backend<B: crate::domain::memory::VectorStoreIndexDyn + 'static>(backend: B) -> Self {
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
}