//! Advanced similarity search algorithms and optimization strategies
//!
//! Implements various search algorithms including exact search, approximate search,
//! multi-vector queries, and search result optimization techniques.

use crate::async_task::{AsyncTask, AsyncStream};
use crate::domain::chunk::EmbeddingChunk;
use crate::providers::embedding::similarity::{
    SimilarityMetric, cosine_similarity, euclidean_distance, manhattan_distance
};
use crate::vector_store::in_memory::{SearchResult, InMemoryVectorStore};
use crate::vector_store::index::VectorIndex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

/// Advanced search configuration with optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Similarity metric for primary search
    pub similarity_metric: SimilarityMetric,
    /// Minimum similarity threshold
    pub similarity_threshold: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Enable result diversification
    pub enable_diversification: bool,
    /// Diversification factor (0.0 to 1.0)
    pub diversification_factor: f32,
    /// Enable re-ranking based on multiple metrics
    pub enable_reranking: bool,
    /// Re-ranking metrics and weights
    pub reranking_weights: HashMap<SimilarityMetric, f32>,
    /// Enable result filtering based on metadata
    pub enable_metadata_filtering: bool,
    /// Metadata filters
    pub metadata_filters: HashMap<String, serde_json::Value>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            similarity_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.7,
            max_results: 10,
            enable_diversification: false,
            diversification_factor: 0.3,
            enable_reranking: false,
            reranking_weights: HashMap::new(),
            enable_metadata_filtering: false,
            metadata_filters: HashMap::new(),
        }
    }
}

/// Multi-vector query for complex search scenarios
#[derive(Debug, Clone)]
pub struct MultiVectorQuery {
    /// Primary query vector
    pub primary_vector: Vec<f32>,
    /// Additional query vectors with weights
    pub secondary_vectors: Vec<(Vec<f32>, f32)>,
    /// Combination strategy
    pub combination_strategy: CombinationStrategy,
}

/// Strategy for combining multiple vector similarities
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Weighted average of similarities
    WeightedAverage,
    /// Maximum similarity across all vectors
    Maximum,
    /// Minimum similarity across all vectors
    Minimum,
    /// Product of all similarities
    Product,
    /// Custom combination function
    Custom,
}

/// Search result with enhanced metadata
#[derive(Debug, Clone)]
pub struct EnhancedSearchResult {
    /// Basic search result
    pub result: SearchResult,
    /// Similarity scores for different metrics
    pub metric_scores: HashMap<SimilarityMetric, f32>,
    /// Diversification score
    pub diversification_score: Option<f32>,
    /// Re-ranking score
    pub reranking_score: Option<f32>,
    /// Distance to query in embedding space
    pub embedding_distance: f32,
}

/// Batch search query for processing multiple queries efficiently
#[derive(Debug, Clone)]
pub struct BatchSearchQuery {
    /// Query vectors
    pub queries: Vec<Vec<f32>>,
    /// Per-query configurations (optional)
    pub configs: Option<Vec<SearchConfig>>,
    /// Global configuration (used if per-query configs not provided)
    pub global_config: SearchConfig,
}

/// Advanced similarity search engine
pub struct SimilaritySearchEngine {
    /// Vector store reference
    store: Arc<InMemoryVectorStore>,
    /// Optional custom index
    index: Arc<RwLock<Option<Box<dyn VectorIndex>>>>,
    /// Default search configuration
    default_config: SearchConfig,
    /// Search performance statistics
    stats: Arc<RwLock<SearchStats>>,
}

/// Search performance statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total number of searches performed
    pub total_searches: u64,
    /// Average search time in microseconds
    pub avg_search_time_us: f64,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Index usage rate
    pub index_usage_rate: f32,
    /// Result accuracy (for approximate searches)
    pub result_accuracy: f32,
}

impl SimilaritySearchEngine {
    /// Create new similarity search engine
    #[inline(always)]
    pub fn new(store: Arc<InMemoryVectorStore>) -> Self {
        Self {
            store,
            index: Arc::new(RwLock::new(None)),
            default_config: SearchConfig::default(),
            stats: Arc::new(RwLock::new(SearchStats::default())),
        }
    }
    
    /// Create search engine with custom configuration
    #[inline(always)]
    pub fn with_config(store: Arc<InMemoryVectorStore>, config: SearchConfig) -> Self {
        Self {
            store,
            index: Arc::new(RwLock::new(None)),
            default_config: config,
            stats: Arc::new(RwLock::new(SearchStats::default())),
        }
    }
    
    /// Set custom index for approximate search
    pub fn set_index(&self, index: Box<dyn VectorIndex>) {
        let mut index_lock = self.index.write();
        *index_lock = Some(index);
    }
    
    /// Perform enhanced similarity search with advanced features
    pub fn search_enhanced(
        &self,
        query_vector: Vec<f32>,
        config: Option<SearchConfig>,
    ) -> AsyncTask<Vec<EnhancedSearchResult>> {
        let engine = self.clone();
        let config = config.unwrap_or_else(|| engine.default_config.clone());
        
        crate::async_task::spawn_async(async move {
            let start_time = std::time::Instant::now();
            
            // Perform primary search
            let mut results = engine.perform_primary_search(&query_vector, &config).await;
            
            // Apply metadata filtering if enabled
            if config.enable_metadata_filtering {
                results = engine.apply_metadata_filters(results, &config.metadata_filters);
            }
            
            // Apply re-ranking if enabled
            if config.enable_reranking {
                results = engine.apply_reranking(&query_vector, results, &config).await;
            }
            
            // Apply diversification if enabled
            if config.enable_diversification {
                results = engine.apply_diversification(results, config.diversification_factor);
            }
            
            // Convert to enhanced results
            let enhanced_results = engine.create_enhanced_results(results, &query_vector, &config);
            
            // Update statistics
            let search_time = start_time.elapsed().as_micros() as f64;
            engine.update_search_stats(search_time);
            
            enhanced_results
        })
    }
    
    /// Perform multi-vector search
    pub fn search_multi_vector(
        &self,
        query: MultiVectorQuery,
        config: Option<SearchConfig>,
    ) -> AsyncTask<Vec<EnhancedSearchResult>> {
        let engine = self.clone();
        let config = config.unwrap_or_else(|| engine.default_config.clone());
        
        crate::async_task::spawn_async(async move {
            let mut combined_results = HashMap::new();
            
            // Search with primary vector
            let primary_results = engine.store.search_vector(query.primary_vector.clone(), Some(config.max_results * 2)).await;
            
            for result in primary_results {
                combined_results.insert(result.id.clone(), vec![(result.score as f32, 1.0)]);
            }
            
            // Search with secondary vectors
            for (vector, weight) in &query.secondary_vectors {
                let secondary_results = engine.store.search_vector(vector.clone(), Some(config.max_results * 2)).await;
                
                for result in secondary_results {
                    combined_results.entry(result.id.clone())
                        .or_insert_with(Vec::new)
                        .push((result.score as f32, *weight));
                }
            }
            
            // Combine scores based on strategy
            let mut final_results = Vec::new();
            
            for (id, scores) in combined_results {
                let combined_score = engine.combine_scores(&scores, query.combination_strategy);
                
                if combined_score >= config.similarity_threshold {
                    if let Some(entry) = engine.store.get(&id).await {
                        final_results.push(SearchResult {
                            score: combined_score as f64,
                            id: entry.id,
                            metadata: entry.metadata,
                            content: entry.content,
                        });
                    }
                }
            }
            
            // Sort and limit results
            final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            final_results.truncate(config.max_results);
            
            // Convert to enhanced results
            engine.create_enhanced_results(final_results, &query.primary_vector, &config)
        })
    }
    
    /// Perform batch search for multiple queries
    pub fn search_batch(
        &self,
        batch_query: BatchSearchQuery,
    ) -> AsyncTask<Vec<Vec<EnhancedSearchResult>>> {
        let engine = self.clone();
        
        crate::async_task::spawn_async(async move {
            let mut all_results = Vec::with_capacity(batch_query.queries.len());
            
            for (i, query_vector) in batch_query.queries.into_iter().enumerate() {
                let config = if let Some(ref configs) = batch_query.configs {
                    configs.get(i).cloned().unwrap_or_else(|| batch_query.global_config.clone())
                } else {
                    batch_query.global_config.clone()
                };
                
                let results = engine.search_enhanced(query_vector, Some(config)).await;
                all_results.push(results);
            }
            
            all_results
        })
    }
    
    /// Stream search results for large result sets
    pub fn search_stream(
        &self,
        query_vector: Vec<f32>,
        config: Option<SearchConfig>,
    ) -> AsyncStream<EmbeddingChunk> {
        let engine = self.clone();
        let config = config.unwrap_or_else(|| engine.default_config.clone());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let results = engine.search_enhanced(query_vector, Some(config)).await;
            
            for (index, result) in results.into_iter().enumerate() {
                let mut metadata = HashMap::new();
                metadata.insert("similarity_score".to_string(), serde_json::json!(result.result.score));
                metadata.insert("id".to_string(), serde_json::json!(result.result.id));
                
                if let Some(content) = result.result.content {
                    metadata.insert("content".to_string(), serde_json::json!(content));
                }
                
                for (metric, score) in result.metric_scores {
                    metadata.insert(format!("{:?}_score", metric).to_lowercase(), serde_json::json!(score));
                }
                
                let chunk = EmbeddingChunk {
                    embeddings: crate::ZeroOneOrMany::from_vec(Vec::new()), // Would contain the result vector if needed
                    index,
                    metadata,
                };
                
                if tx.send(chunk).is_err() {
                    break;
                }
            }
        });
        
        AsyncStream::new(rx)
    }
    
    /// Internal method to perform primary search
    async fn perform_primary_search(
        &self,
        query_vector: &[f32],
        config: &SearchConfig,
    ) -> Vec<SearchResult> {
        // Check if we have a custom index and extract candidates first
        let candidates = {
            if let Some(ref index) = *self.index.read() {
                // Use custom index for approximate search
                Some(index.search(
                    query_vector,
                    config.max_results * 2, // Get more candidates for post-processing
                    config.similarity_metric,
                    config.similarity_threshold,
                ))
            } else {
                None
            }
        }; // Read guard is dropped here
        
        if let Some(candidates) = candidates {
            // Convert index results to SearchResult format
            let mut results = Vec::new();
            for (id, score) in candidates {
                if let Some(entry) = self.store.get(&id).await {
                    results.push(SearchResult {
                        score: score as f64,
                        id: entry.id,
                        metadata: entry.metadata,
                        content: entry.content,
                    });
                }
            }
            results
        } else {
            // Use linear search from vector store
            self.store.search_vector(query_vector.to_vec(), Some(config.max_results * 2)).await
        }
    }
    
    /// Apply metadata filtering to search results
    fn apply_metadata_filters(
        &self,
        results: Vec<SearchResult>,
        filters: &HashMap<String, serde_json::Value>,
    ) -> Vec<SearchResult> {
        if filters.is_empty() {
            return results;
        }
        
        results.into_iter()
            .filter(|result| {
                filters.iter().all(|(key, expected_value)| {
                    result.metadata.get(key).map_or(false, |actual_value| {
                        actual_value == expected_value
                    })
                })
            })
            .collect()
    }
    
    /// Apply re-ranking based on multiple metrics
    async fn apply_reranking(
        &self,
        query_vector: &[f32],
        mut results: Vec<SearchResult>,
        config: &SearchConfig,
    ) -> Vec<SearchResult> {
        if config.reranking_weights.is_empty() {
            return results;
        }
        
        // Compute additional similarity scores
        for result in &mut results {
            if let Some(entry) = self.store.get(&result.id).await {
                let mut rerank_score = 0.0;
                let mut total_weight = 0.0;
                
                for (&metric, &weight) in &config.reranking_weights {
                    let score = match metric {
                        SimilarityMetric::Cosine => cosine_similarity(query_vector, &entry.vector),
                        SimilarityMetric::Euclidean => {
                            let distance = euclidean_distance(query_vector, &entry.vector);
                            1.0 / (1.0 + distance)
                        },
                        SimilarityMetric::Manhattan => {
                            let distance = manhattan_distance(query_vector, &entry.vector);
                            1.0 / (1.0 + distance)
                        },
                        _ => result.score as f32,
                    };
                    
                    rerank_score += score * weight;
                    total_weight += weight;
                }
                
                if total_weight > 0.0 {
                    result.score = (rerank_score / total_weight) as f64;
                }
            }
        }
        
        // Re-sort based on new scores
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
    
    /// Apply diversification to reduce result redundancy
    fn apply_diversification(
        &self,
        results: Vec<SearchResult>,
        diversification_factor: f32,
    ) -> Vec<SearchResult> {
        if results.len() <= 1 || diversification_factor <= 0.0 {
            return results;
        }
        
        let mut diversified = Vec::new();
        let mut selected_indices = HashSet::new();
        
        // Always include the top result
        if !results.is_empty() {
            diversified.push(results[0].clone());
            selected_indices.insert(0);
        }
        
        // Select remaining results with diversification
        while diversified.len() < results.len() && diversified.len() < 50 { // Limit for performance
            let mut best_idx = None;
            let mut best_score = f32::NEG_INFINITY;
            
            for (i, candidate) in results.iter().enumerate() {
                if selected_indices.contains(&i) {
                    continue;
                }
                
                // Calculate diversification penalty
                let mut min_similarity = f32::INFINITY;
                for selected in &diversified {
                    // Simple content-based diversification (placeholder)
                    let similarity = if candidate.id == selected.id { 1.0 } else { 0.0 };
                    min_similarity = min_similarity.min(similarity);
                }
                
                // Combine relevance and diversity
                let relevance_score = candidate.score as f32;
                let diversity_bonus = (1.0 - min_similarity) * diversification_factor;
                let combined_score = relevance_score + diversity_bonus;
                
                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = Some(i);
                }
            }
            
            if let Some(idx) = best_idx {
                diversified.push(results[idx].clone());
                selected_indices.insert(idx);
            } else {
                break;
            }
        }
        
        diversified
    }
    
    /// Create enhanced results with additional metadata
    fn create_enhanced_results(
        &self,
        results: Vec<SearchResult>,
        _query_vector: &[f32],
        config: &SearchConfig,
    ) -> Vec<EnhancedSearchResult> {
        results.into_iter()
            .enumerate()
            .map(|(rank, result)| {
                let mut metric_scores = HashMap::new();
                
                // Compute scores for different metrics (placeholder implementation)
                metric_scores.insert(SimilarityMetric::Cosine, result.score as f32);
                
                EnhancedSearchResult {
                    embedding_distance: 1.0 - result.score as f32, // Simple distance approximation
                    metric_scores,
                    diversification_score: if config.enable_diversification {
                        Some(1.0 / (rank as f32 + 1.0)) // Simple diversification score
                    } else {
                        None
                    },
                    reranking_score: if config.enable_reranking {
                        Some(result.score as f32)
                    } else {
                        None
                    },
                    result,
                }
            })
            .collect()
    }
    
    /// Combine multiple similarity scores
    fn combine_scores(&self, scores: &[(f32, f32)], strategy: CombinationStrategy) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        
        match strategy {
            CombinationStrategy::WeightedAverage => {
                let weighted_sum: f32 = scores.iter().map(|(score, weight)| score * weight).sum();
                let total_weight: f32 = scores.iter().map(|(_, weight)| weight).sum();
                if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 }
            },
            CombinationStrategy::Maximum => {
                scores.iter().map(|(score, _)| *score).fold(f32::NEG_INFINITY, f32::max)
            },
            CombinationStrategy::Minimum => {
                scores.iter().map(|(score, _)| *score).fold(f32::INFINITY, f32::min)
            },
            CombinationStrategy::Product => {
                scores.iter().map(|(score, _)| *score).product()
            },
            CombinationStrategy::Custom => {
                // Custom combination would be implemented here
                scores.iter().map(|(score, weight)| score * weight).sum::<f32>() / scores.len() as f32
            },
        }
    }
    
    /// Update search performance statistics
    fn update_search_stats(&self, search_time_us: f64) {
        let mut stats = self.stats.write();
        stats.total_searches += 1;
        
        // Update running average
        if stats.total_searches == 1 {
            stats.avg_search_time_us = search_time_us;
        } else {
            let alpha = 0.1; // Exponential moving average factor
            stats.avg_search_time_us = alpha * search_time_us + (1.0 - alpha) * stats.avg_search_time_us;
        }
    }
    
    /// Get search performance statistics
    pub fn get_stats(&self) -> SearchStats {
        self.stats.read().clone()
    }
}

impl Clone for SimilaritySearchEngine {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            index: self.index.clone(),
            default_config: self.default_config.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Utility functions for similarity search
pub mod utils {
    use super::*;
    
    /// Create optimized search configuration for specific use cases
    #[inline(always)]
    pub fn create_semantic_search_config() -> SearchConfig {
        SearchConfig {
            similarity_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.75,
            max_results: 20,
            enable_diversification: true,
            diversification_factor: 0.3,
            enable_reranking: true,
            reranking_weights: {
                let mut weights = HashMap::new();
                weights.insert(SimilarityMetric::Cosine, 0.7);
                weights.insert(SimilarityMetric::DotProduct, 0.3);
                weights
            },
            enable_metadata_filtering: false,
            metadata_filters: HashMap::new(),
        }
    }
    
    /// Create configuration optimized for document retrieval
    #[inline(always)]
    pub fn create_document_retrieval_config() -> SearchConfig {
        SearchConfig {
            similarity_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.6,
            max_results: 10,
            enable_diversification: true,
            diversification_factor: 0.5,
            enable_reranking: false,
            reranking_weights: HashMap::new(),
            enable_metadata_filtering: true,
            metadata_filters: HashMap::new(),
        }
    }
    
    /// Create configuration for high-precision search
    #[inline(always)]
    pub fn create_high_precision_config() -> SearchConfig {
        SearchConfig {
            similarity_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.9,
            max_results: 5,
            enable_diversification: false,
            diversification_factor: 0.0,
            enable_reranking: true,
            reranking_weights: {
                let mut weights = HashMap::new();
                weights.insert(SimilarityMetric::Cosine, 0.8);
                weights.insert(SimilarityMetric::Euclidean, 0.2);
                weights
            },
            enable_metadata_filtering: false,
            metadata_filters: HashMap::new(),
        }
    }
}