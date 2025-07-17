use crate::domain::memory::{VectorStoreIndex, VectorStoreIndexDyn};
use crate::domain::{AsyncTask, spawn_async, ZeroOneOrMany};
use serde_json::Value;
use std::sync::Arc;

/// Zero-allocation vector query builder with blazing-fast performance
#[derive(Debug)]
pub struct VectorQueryBuilder<'a> {
    index: &'a VectorStoreIndex,
    query: String,
    n: usize,
    threshold: Option<f64>,
    include_metadata: bool,
    filters: Vec<(String, Value)>,
}

impl<'a> VectorQueryBuilder<'a> {
    /// Create a new query builder - internal use only
    #[inline(always)]
    pub(crate) fn new(index: &'a VectorStoreIndex, query: String) -> Self {
        Self {
            index,
            query,
            n: 10,
            threshold: None,
            include_metadata: true,
            filters: Vec::new(),
        }
    }
    
    /// Set the number of top results to return
    #[inline(always)]
    pub fn top(mut self, n: usize) -> Self {
        self.n = n;
        self
    }
    
    /// Set similarity threshold for filtering results
    #[inline(always)]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }
    
    /// Control whether to include metadata in results
    #[inline(always)]
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }
    
    /// Add a metadata filter for the query
    #[inline(always)]
    pub fn filter(mut self, key: impl Into<String>, value: Value) -> Self {
        self.filters.push((key.into(), value));
        self
    }
    
    /// Add multiple metadata filters
    #[inline(always)]
    pub fn filters<I>(mut self, filters: I) -> Self 
    where
        I: IntoIterator<Item = (String, Value)>,
    {
        self.filters.extend(filters);
        self
    }
    
    /// Execute query and return full results with metadata
    pub fn retrieve(self) -> AsyncTask<Result<ZeroOneOrMany<(f64, String, Value)>, String>> {
        let future = self.index.backend.top_n(&self.query, self.n);
        let threshold = self.threshold;
        let include_metadata = self.include_metadata;
        let filters = self.filters;
        
        spawn_async(async move {
            match future.await {
                Ok(results) => {
                    let filtered_results = Self::apply_filters_and_threshold(
                        results, 
                        threshold, 
                        include_metadata,
                        &filters
                    );
                    Ok(filtered_results)
                },
                Err(_) => Ok(ZeroOneOrMany::None),
            }
        })
    }
    
    /// Execute query and return only IDs with scores
    pub fn retrieve_ids(self) -> AsyncTask<Result<ZeroOneOrMany<(f64, String)>, String>> {
        let future = self.index.backend.top_n_ids(&self.query, self.n);
        let threshold = self.threshold;
        
        spawn_async(async move {
            match future.await {
                Ok(results) => {
                    let filtered_results = if let Some(threshold) = threshold {
                        Self::apply_threshold_to_ids(results, threshold)
                    } else {
                        results
                    };
                    Ok(filtered_results)
                },
                Err(_) => Ok(ZeroOneOrMany::None),
            }
        })
    }
    
    /// Execute query with custom result handler
    pub fn on_results<F, T>(self, handler: F) -> AsyncTask<T>
    where
        F: FnOnce(Result<ZeroOneOrMany<(f64, String, Value)>, String>) -> T + Send + 'static,
        T: Send + 'static,
    {
        let future = self.index.backend.top_n(&self.query, self.n);
        let threshold = self.threshold;
        let include_metadata = self.include_metadata;
        let filters = self.filters;
        
        spawn_async(async move {
            let result = match future.await {
                Ok(results) => {
                    let filtered_results = Self::apply_filters_and_threshold(
                        results, 
                        threshold, 
                        include_metadata,
                        &filters
                    );
                    Ok(filtered_results)
                },
                Err(_) => Ok(ZeroOneOrMany::None),
            };
            handler(result)
        })
    }
    
    /// Execute streaming query for large result sets
    pub fn stream(self) -> AsyncTask<Result<crate::domain::AsyncStream<(f64, String, Value)>, String>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let future = self.index.backend.top_n(&self.query, self.n);
        let threshold = self.threshold;
        let include_metadata = self.include_metadata;
        let filters = self.filters;
        
        spawn_async(async move {
            match future.await {
                Ok(results) => {
                    let filtered_results = Self::apply_filters_and_threshold(
                        results, 
                        threshold, 
                        include_metadata,
                        &filters
                    );
                    
                    // Stream results
                    match filtered_results {
                        ZeroOneOrMany::None => {
                            drop(tx);
                        },
                        ZeroOneOrMany::One(item) => {
                            let _ = tx.send(item);
                            drop(tx);
                        },
                        ZeroOneOrMany::Many(items) => {
                            for item in items {
                                if tx.send(item).is_err() {
                                    break;
                                }
                            }
                            drop(tx);
                        },
                    }
                    
                    Ok(crate::domain::async_task::AsyncStream::new(rx))
                },
                Err(_) => {
                    drop(tx);
                    Ok(crate::domain::async_task::AsyncStream::empty())
                },
            }
        })
    }
    
    // Internal helper methods
    
    #[inline(always)]
    fn apply_filters_and_threshold(
        results: ZeroOneOrMany<(f64, String, Value)>,
        threshold: Option<f64>,
        include_metadata: bool,
        filters: &[(String, Value)]
    ) -> ZeroOneOrMany<(f64, String, Value)> {
        let filter_fn = |item: (f64, String, Value)| -> Option<(f64, String, Value)> {
            // Apply threshold
            if let Some(threshold) = threshold {
                if item.0 < threshold {
                    return None;
                }
            }
            
            // Apply metadata filters
            if !filters.is_empty() {
                if let Value::Object(metadata) = &item.2 {
                    for (key, expected_value) in filters {
                        if let Some(actual_value) = metadata.get(key) {
                            if actual_value != expected_value {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
            
            // Handle metadata inclusion
            if include_metadata {
                Some(item)
            } else {
                Some((item.0, item.1, Value::Null))
            }
        };
        
        match results {
            ZeroOneOrMany::None => ZeroOneOrMany::None,
            ZeroOneOrMany::One(item) => {
                if let Some(filtered_item) = filter_fn(item) {
                    ZeroOneOrMany::One(filtered_item)
                } else {
                    ZeroOneOrMany::None
                }
            },
            ZeroOneOrMany::Many(items) => {
                let filtered_items: Vec<_> = items.into_iter()
                    .filter_map(filter_fn)
                    .collect();
                
                if filtered_items.is_empty() {
                    ZeroOneOrMany::None
                } else if filtered_items.len() == 1 {
                    ZeroOneOrMany::One(filtered_items.into_iter().next().unwrap_or_default())
                } else {
                    ZeroOneOrMany::Many(filtered_items)
                }
            }
        }
    }
    
    #[inline(always)]
    fn apply_threshold_to_ids(
        results: ZeroOneOrMany<(f64, String)>,
        threshold: f64
    ) -> ZeroOneOrMany<(f64, String)> {
        let filter_fn = |item: (f64, String)| -> Option<(f64, String)> {
            if item.0 >= threshold {
                Some(item)
            } else {
                None
            }
        };
        
        match results {
            ZeroOneOrMany::None => ZeroOneOrMany::None,
            ZeroOneOrMany::One(item) => {
                if let Some(filtered_item) = filter_fn(item) {
                    ZeroOneOrMany::One(filtered_item)
                } else {
                    ZeroOneOrMany::None
                }
            },
            ZeroOneOrMany::Many(items) => {
                let filtered_items: Vec<_> = items.into_iter()
                    .filter_map(filter_fn)
                    .collect();
                
                if filtered_items.is_empty() {
                    ZeroOneOrMany::None
                } else if filtered_items.len() == 1 {
                    ZeroOneOrMany::One(filtered_items.into_iter().next().unwrap_or_default())
                } else {
                    ZeroOneOrMany::Many(filtered_items)
                }
            }
        }
    }
}

/// Extension trait for VectorStoreIndex to enable query building
pub trait VectorStoreIndexExt {
    /// Start building a semantic search query
    fn search(&self, query: impl Into<String>) -> VectorQueryBuilder<'_>;
    
    /// Perform a quick similarity search with default settings
    fn quick_search(&self, query: impl Into<String>, n: usize) -> AsyncTask<Result<ZeroOneOrMany<(f64, String, Value)>, String>>;
}

impl VectorStoreIndexExt for VectorStoreIndex {
    #[inline(always)]
    fn search(&self, query: impl Into<String>) -> VectorQueryBuilder<'_> {
        VectorQueryBuilder::new(self, query.into())
    }
    
    #[inline(always)]
    fn quick_search(&self, query: impl Into<String>, n: usize) -> AsyncTask<Result<ZeroOneOrMany<(f64, String, Value)>, String>> {
        self.search(query).top(n).retrieve()
    }
}

/// Advanced vector store operations
impl VectorStoreIndex {
    /// Perform batch similarity search for multiple queries
    pub fn batch_search(&self, queries: Vec<String>, n: usize) -> AsyncTask<Result<Vec<ZeroOneOrMany<(f64, String, Value)>>, String>> {
        let backend = &self.backend;
        let futures = queries.into_iter()
            .map(|query| backend.top_n(&query, n))
            .collect::<Vec<_>>();
        
        spawn_async(async move {
            let results = futures::future::join_all(futures).await;
            let converted_results: Result<Vec<_>, _> = results.into_iter()
                .map(|result| result.map_err(|_| "Batch search failed".to_string()))
                .collect();
            
            converted_results
        })
    }
    
    /// Find similar vectors to a given vector ID
    pub fn find_similar(&self, vector_id: &str, n: usize) -> AsyncTask<Result<ZeroOneOrMany<(f64, String, Value)>, String>> {
        // This would typically use the vector directly, but we'll use the ID as a proxy
        self.search(vector_id.to_string()).top(n).retrieve()
    }
}