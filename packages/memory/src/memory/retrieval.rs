//! Memory retrieval strategies and algorithms

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;
use futures::stream::StreamExt;

use crate::memory::filter::MemoryFilter;
use crate::utils::Result;
use crate::vector::VectorStore;

/// Retrieval method used to find the memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalMethod {
    VectorSimilarity,
    Semantic,
    Temporal,
    Keyword,
    Hybrid,
}

/// A pending retrieval operation
pub struct PendingRetrieval {
    rx: oneshot::Receiver<Result<Vec<RetrievalResult>>>,
}

impl PendingRetrieval {
    pub fn new(rx: oneshot::Receiver<Result<Vec<RetrievalResult>>>) -> Self {
        Self { rx }
    }
}

impl Future for PendingRetrieval {
    type Output = Result<Vec<RetrievalResult>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(crate::utils::error::Error::Internal(
                "Retrieval task failed".to_string(),
            ))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Memory retrieval strategy trait
pub trait RetrievalStrategy: Send + Sync {
    /// Retrieve memories based on the strategy
    fn retrieve(
        &self,
        query: String,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> PendingRetrieval;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Result from memory retrieval
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Memory ID
    pub id: String,

    /// Relevance score
    pub score: f32,

    /// Retrieval method used
    pub method: RetrievalMethod,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Hybrid retrieval strategy combining multiple approaches
pub struct HybridRetrieval<V: VectorStore> {
    vector_store: V,
    strategies: std::sync::Arc<Vec<std::sync::Arc<dyn RetrievalStrategy>>>,
    weights: std::sync::Arc<HashMap<String, f32>>,
}

impl<V: VectorStore> HybridRetrieval<V> {
    /// Create a new hybrid retrieval strategy
    pub fn new(vector_store: V) -> Self {
        let mut weights = HashMap::new();
        weights.insert("semantic".to_string(), 0.6);
        weights.insert("keyword".to_string(), 0.2);
        weights.insert("temporal".to_string(), 0.2);

        Self {
            vector_store,
            strategies: std::sync::Arc::new(Vec::new()),
            weights: std::sync::Arc::new(weights),
        }
    }

    /// Add a retrieval strategy
    pub fn add_strategy(mut self, strategy: std::sync::Arc<dyn RetrievalStrategy>) -> Self {
        std::sync::Arc::make_mut(&mut self.strategies).push(strategy);
        self
    }

    /// Set weight for a strategy
    pub fn set_weight(mut self, strategy_name: &str, weight: f32) -> Self {
        std::sync::Arc::make_mut(&mut self.weights).insert(strategy_name.to_string(), weight);
        self
    }

    /// Get vector similarity results from the vector store
    pub async fn get_vector_similarity(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let filter = crate::memory::filter::MemoryFilter::new();
        let results = self
            .vector_store
            .search(query_vector, limit, Some(filter))
            .await?;
        let retrieval_results = results
            .into_iter()
            .map(|result| RetrievalResult {
                id: result.id,
                method: RetrievalMethod::VectorSimilarity,
                score: result.score,
                metadata: HashMap::new(),
            })
            .collect();
        Ok(retrieval_results)
    }
}

impl<V: VectorStore + Send + Sync + 'static> RetrievalStrategy for HybridRetrieval<V> {
    fn retrieve(
        &self,
        query: String,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> PendingRetrieval {
        let (tx, rx) = oneshot::channel();
        let strategies = self.strategies.clone();
        let weights = self.weights.clone();

        tokio::spawn(async move {
            let result: Result<Vec<RetrievalResult>> = (async {
                let mut all_results: HashMap<String, (f32, RetrievalResult)> = HashMap::new();

                // Get results from each strategy
                for strategy in &*strategies {
                    let results = strategy
                        .retrieve(query.clone(), limit * 2, filter.clone())
                        .await?;
                    let weight = weights.get(strategy.name()).unwrap_or(&1.0);

                    for result in results {
                        let weighted_score = result.score * weight;

                        all_results
                            .entry(result.id.clone())
                            .and_modify(|(score, _)| *score += weighted_score)
                            .or_insert((weighted_score, result));
                    }
                }

                // Sort by combined score and take top results
                let mut sorted_results: Vec<_> = all_results
                    .into_iter()
                    .map(|(_, (score, mut result))| {
                        result.score = score;
                        result
                    })
                    .collect();

                sorted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                sorted_results.truncate(limit);

                Ok(sorted_results)
            })
            .await;

            let _ = tx.send(result);
        });

        PendingRetrieval::new(rx)
    }

    fn name(&self) -> &str {
        "hybrid"
    }
}

/// Semantic similarity retrieval using vector embeddings
pub struct SemanticRetrieval<V: VectorStore> {
    vector_store: std::sync::Arc<V>,
}

impl<V: VectorStore> SemanticRetrieval<V> {
    pub fn new(vector_store: V) -> Self {
        Self {
            vector_store: std::sync::Arc::new(vector_store),
        }
    }
}

impl<V: VectorStore + Send + Sync + 'static> RetrievalStrategy for SemanticRetrieval<V> {
    fn retrieve(
        &self,
        query: String,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> PendingRetrieval {
        let (tx, rx) = oneshot::channel();
        let vector_store = self.vector_store.clone();

        tokio::spawn(async move {
            let result: Result<Vec<RetrievalResult>> = (async {
                // Generate query embedding
                let query_embedding = vector_store.embed(query).await?;

                // Search in vector store
                let results = vector_store.search(query_embedding, limit, filter).await?;

                let retrieval_results = results
                    .into_iter()
                    .map(|r| RetrievalResult {
                        id: r.id,
                        score: r.score,
                        method: RetrievalMethod::Semantic,
                        metadata: HashMap::new(), // VectorSearchResult doesn't include metadata
                    })
                    .collect();

                Ok(retrieval_results)
            })
            .await;

            let _ = tx.send(result);
        });

        PendingRetrieval::new(rx)
    }

    fn name(&self) -> &str {
        "semantic"
    }
}

/// Temporal proximity retrieval with production-ready database integration
pub struct TemporalRetrieval {
    time_decay_factor: f32,
    memory_manager: std::sync::Arc<dyn crate::memory::MemoryManager>,
}

impl TemporalRetrieval {
    /// Create a new temporal retrieval strategy with memory manager
    pub fn new(time_decay_factor: f32, memory_manager: std::sync::Arc<dyn crate::memory::MemoryManager>) -> Self {
        Self { 
            time_decay_factor,
            memory_manager,
        }
    }
}

impl RetrievalStrategy for TemporalRetrieval {
    fn retrieve(
        &self,
        query: String,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> PendingRetrieval {
        let (tx, rx) = oneshot::channel();
        let time_decay = self.time_decay_factor;
        let memory_manager = self.memory_manager.clone();

        tokio::spawn(async move {
            let result: Result<Vec<RetrievalResult>> = (async {
                // Create temporal filter - prioritize recent memories
                let mut temporal_filter = filter.unwrap_or_default();
                
                // Set time window to last 30 days if not specified
                if temporal_filter.time_range.is_none() {
                    let now = chrono::Utc::now();
                    let thirty_days_ago = now - chrono::Duration::days(30);
                    temporal_filter.time_range = Some(crate::memory::filter::TimeRange {
                        start: Some(thirty_days_ago),
                        end: Some(now),
                    });
                }

                // Apply limit multiplier to get enough results for filtering
                temporal_filter.limit = Some(limit * 3);

                // Query content-based memories first if query is provided
                let mut all_memories = Vec::new();
                
                if !query.is_empty() {
                    // Use content search as primary filter
                    let content_stream = memory_manager.search_by_content(&query);
                    let mut content_memories = Vec::new();
                    
                    tokio::pin!(content_stream);
                    while let Some(memory_result) = content_stream.next().await {
                        match memory_result {
                            Ok(memory) => {
                                // Apply time range filter manually
                                if let Some(ref time_range) = temporal_filter.time_range {
                                    let created_at = memory.created_at;
                                    let in_range = time_range.start.is_none_or(|start| created_at >= start) 
                                        && time_range.end.is_none_or(|end| created_at < end);
                                    
                                    if in_range {
                                        content_memories.push(memory);
                                    }
                                } else {
                                    content_memories.push(memory);
                                }
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                        
                        if content_memories.len() >= temporal_filter.limit.unwrap_or(limit * 3) {
                            break;
                        }
                    }
                    
                    all_memories.extend(content_memories);
                }

                // If we don't have enough memories, get recent ones by type
                if all_memories.len() < limit {
                    let remaining = limit * 3 - all_memories.len();
                    let memory_types = temporal_filter.memory_types.clone()
                        .unwrap_or_else(|| vec![
                            crate::memory::MemoryType::Episodic,
                            crate::memory::MemoryType::Semantic,
                            crate::memory::MemoryType::Procedural,
                        ]);
                    
                    for memory_type in memory_types {
                        let type_stream = memory_manager.query_by_type(memory_type);
                        let mut type_memories = Vec::new();
                        
                        tokio::pin!(type_stream);
                        while let Some(memory_result) = type_stream.next().await {
                            match memory_result {
                                Ok(memory) => {
                                    // Apply time range filter manually
                                    if let Some(ref time_range) = temporal_filter.time_range {
                                        let created_at = memory.created_at;
                                        let in_range = time_range.start.is_none_or(|start| created_at >= start) 
                                            && time_range.end.is_none_or(|end| created_at < end);
                                        
                                        if in_range {
                                            type_memories.push(memory);
                                        }
                                    } else {
                                        type_memories.push(memory);
                                    }
                                }
                                Err(e) => {
                                    return Err(e);
                                }
                            }
                            
                            if type_memories.len() >= remaining {
                                break;
                            }
                        }
                        
                        all_memories.extend(type_memories);
                        
                        if all_memories.len() >= limit * 3 {
                            break;
                        }
                    }
                }

                // Apply temporal scoring with decay factor
                let now = chrono::Utc::now();
                let now_timestamp = now.timestamp() as f64;
                
                let mut scored_results: Vec<RetrievalResult> = all_memories
                    .into_iter()
                    .map(|memory| {
                        // Calculate age in hours
                        let memory_timestamp = memory.created_at.timestamp() as f64;
                        let age_hours = (now_timestamp - memory_timestamp) / 3600.0;
                        
                        // Apply exponential decay based on age
                        let temporal_score = if age_hours >= 0.0 {
                            -(age_hours * time_decay as f64).exp() as f32
                        } else {
                            1.0 // Future memories get max score
                        };
                        
                        // Base relevance score (could be enhanced with content matching)
                        let relevance_score = if !query.is_empty() {
                            // Simple content matching score
                            let content_lower = memory.content.to_lowercase();
                            let query_lower = query.to_lowercase();
                            
                            if content_lower.contains(&query_lower) {
                                let match_ratio = query_lower.len() as f32 / content_lower.len() as f32;
                                (match_ratio * 2.0).min(1.0)
                            } else {
                                0.1 // Minimal base score for type-based matches
                            }
                        } else {
                            0.5 // Default relevance for temporal-only queries
                        };
                        
                        // Combine temporal and relevance scores
                        let final_score = (temporal_score * 0.7) + (relevance_score * 0.3);
                        
                        // Create metadata with temporal information
                        let mut metadata = HashMap::new();
                        metadata.insert("age_hours".to_string(), serde_json::Value::Number(
                            serde_json::Number::from_f64(age_hours).unwrap_or_else(|| serde_json::Number::from(0))
                        ));
                        metadata.insert("temporal_score".to_string(), serde_json::Value::Number(
                            serde_json::Number::from_f64(temporal_score as f64).unwrap_or_else(|| serde_json::Number::from(0))
                        ));
                        metadata.insert("relevance_score".to_string(), serde_json::Value::Number(
                            serde_json::Number::from_f64(relevance_score as f64).unwrap_or_else(|| serde_json::Number::from(0))
                        ));
                        metadata.insert("created_at".to_string(), serde_json::Value::String(
                            memory.created_at.to_rfc3339()
                        ));
                        
                        RetrievalResult {
                            id: memory.id,
                            score: final_score,
                            method: RetrievalMethod::Temporal,
                            metadata,
                        }
                    })
                    .collect();

                // Sort by score (descending) and take top results
                scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                scored_results.truncate(limit);

                Ok(scored_results)
            })
            .await;

            let _ = tx.send(result);
        });

        PendingRetrieval::new(rx)
    }

    fn name(&self) -> &str {
        "temporal"
    }
}

/// Memory retrieval manager
pub struct RetrievalManager<V: VectorStore> {
    strategies: HashMap<String, std::sync::Arc<dyn RetrievalStrategy>>,
    default_strategy: String,
    vector_store: V,
}

impl<V: VectorStore + Clone + Send + Sync + 'static> RetrievalManager<V> {
    /// Create a new retrieval manager
    pub fn new(vector_store: V) -> Self {
        let mut strategies: HashMap<String, std::sync::Arc<dyn RetrievalStrategy>> = HashMap::new();

        // Add default strategies
        strategies.insert(
            "semantic".to_string(),
            std::sync::Arc::new(SemanticRetrieval::new(vector_store.clone())),
        );

        // Note: Temporal retrieval requires a memory manager to be added later
        // via add_temporal_strategy method

        Self {
            strategies,
            default_strategy: "semantic".to_string(),
            vector_store,
        }
    }

    /// Add temporal retrieval strategy with a memory manager
    pub fn add_temporal_strategy(
        &mut self,
        memory_manager: std::sync::Arc<dyn crate::memory::MemoryManager>,
        time_decay_factor: f32,
    ) {
        self.strategies.insert(
            "temporal".to_string(),
            std::sync::Arc::new(TemporalRetrieval::new(time_decay_factor, memory_manager)),
        );
    }

    /// Set the default retrieval strategy
    pub fn set_default_strategy(&mut self, strategy_name: String) {
        self.default_strategy = strategy_name;
    }

    /// Add a custom retrieval strategy
    pub fn add_strategy(&mut self, name: String, strategy: std::sync::Arc<dyn RetrievalStrategy>) {
        self.strategies.insert(name, strategy);
    }

    /// Direct vector search using the managed vector store
    pub async fn direct_vector_search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<crate::vector::VectorSearchResult>> {
        let filter = crate::memory::filter::MemoryFilter::new();
        self.vector_store
            .search(query_vector, limit, Some(filter))
            .await
    }

    /// Retrieve memories using the specified strategy
    pub async fn retrieve(
        &self,
        query: &str,
        strategy_name: Option<&str>,
        limit: usize,
        filter: Option<&MemoryFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        let strategy_name = strategy_name.unwrap_or(&self.default_strategy);

        if let Some(strategy) = self.strategies.get(strategy_name) {
            strategy
                .retrieve(query.to_string(), limit, filter.cloned())
                .await
        } else {
            Err(crate::utils::error::Error::InvalidInput(format!(
                "Unknown retrieval strategy: {strategy_name}"
            )))
        }
    }

    /// Retrieve using multiple strategies and combine results
    pub async fn multi_strategy_retrieve(
        &self,
        query: &str,
        strategy_names: Vec<&str>,
        limit: usize,
        filter: Option<&MemoryFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        let mut all_results = Vec::new();

        for strategy_name in strategy_names {
            if let Some(strategy) = self.strategies.get(strategy_name) {
                let results = strategy
                    .retrieve(query.to_string(), limit, filter.cloned())
                    .await?;
                all_results.extend(results);
            }
        }

        // Deduplicate and sort by score
        let mut unique_results: HashMap<String, RetrievalResult> = HashMap::new();
        for result in all_results {
            unique_results
                .entry(result.id.clone())
                .and_modify(|r| {
                    if result.score > r.score {
                        r.score = result.score;
                    }
                })
                .or_insert(result);
        }

        let mut sorted_results: Vec<_> = unique_results.into_values().collect();
        sorted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sorted_results.truncate(limit);

        Ok(sorted_results)
    }
}
