//! Attention mechanism for cognitive memory management

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::cognitive::types::{RoutingDecision, RoutingStrategy};
use crate::cognitive::types::EnhancedQuery;

/// Attention mechanism for relevance scoring and focus management
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f32,
    pub attention_scores: HashMap<String, f32>,
}

/// Multi-head attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub dropout_rate: f32,
    pub use_causal_mask: bool,
}

/// Attention weights for memory nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub query_weights: Vec<f32>,
    pub key_weights: Vec<f32>,
    pub value_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
}

/// Attention output
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    pub weighted_values: Vec<f32>,
    pub attention_scores: Vec<Vec<f32>>,
    pub context_vector: Vec<f32>,
}

impl AttentionMechanism {
    /// Create a new attention mechanism
    pub fn new(config: AttentionConfig) -> Self {
        let head_dim = config.hidden_dim / config.num_heads;

        Self {
            num_heads: config.num_heads,
            head_dim,
            dropout_rate: config.dropout_rate,
            attention_scores: HashMap::new(),
        }
    }

    /// Calculate attention weights for a query
    pub async fn calculate_attention_weights(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> crate::cognitive::quantum::types::CognitiveResult<AttentionOutput> {
        if keys.len() != values.len() {
            return Err(
                crate::cognitive::quantum::types::CognitiveError::ContextProcessingError(
                    "Keys and values must have same length".to_string(),
                ),
            );
        }

        let _seq_len = keys.len();
        let mut all_attention_scores = Vec::with_capacity(self.num_heads);
        let mut all_weighted_values = Vec::with_capacity(self.num_heads * self.head_dim);

        // Process each attention head
        for head in 0..self.num_heads {
            let head_scores = self.compute_head_attention(query, keys, values, head)?;

            all_attention_scores.push(head_scores.attention_scores.clone());
            all_weighted_values.extend(&head_scores.weighted_values);
        }

        // Concatenate heads and create output
        let context_vector = self.merge_heads(&all_weighted_values);

        Ok(AttentionOutput {
            weighted_values: all_weighted_values,
            attention_scores: all_attention_scores,
            context_vector,
        })
    }

    /// Compute attention for a single head
    fn compute_head_attention(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        head_idx: usize,
    ) -> crate::cognitive::quantum::types::CognitiveResult<HeadAttentionOutput> {
        let seq_len = keys.len();
        let mut scores = vec![0.0; seq_len];

        // Project query for this head
        let head_query = self.project_to_head(query, head_idx);

        // Compute attention scores
        for (i, key) in keys.iter().enumerate() {
            let head_key = self.project_to_head(key, head_idx);
            scores[i] = self.scaled_dot_product(&head_query, &head_key);
        }

        // Apply softmax
        let attention_weights = self.softmax(&scores);

        // Apply attention to values
        let mut weighted_values = vec![0.0; self.head_dim];
        for (i, value) in values.iter().enumerate() {
            let head_value = self.project_to_head(value, head_idx);
            for j in 0..self.head_dim {
                weighted_values[j] += attention_weights[i] * head_value[j];
            }
        }

        Ok(HeadAttentionOutput {
            weighted_values,
            attention_scores: attention_weights,
        })
    }

    /// Project vector to attention head subspace
    fn project_to_head(&self, vector: &[f32], head_idx: usize) -> Vec<f32> {
        let start = head_idx * self.head_dim;
        let end = start + self.head_dim;

        if end <= vector.len() {
            vector[start..end].to_vec()
        } else {
            // Pad with zeros if vector is too short
            let mut result = vec![0.0; self.head_dim];
            let available = vector.len() - start;
            if available > 0 {
                result[..available].copy_from_slice(&vector[start..]);
            }
            result
        }
    }

    /// Scaled dot product attention score
    fn scaled_dot_product(&self, query: &[f32], key: &[f32]) -> f32 {
        let dot_product: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();

        dot_product / (self.head_dim as f32).sqrt()
    }

    /// Softmax normalization
    fn softmax(&self, scores: &[f32]) -> Vec<f32> {
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();

        let sum: f32 = exp_scores.iter().sum();

        exp_scores.iter().map(|&e| e / sum).collect()
    }

    /// Merge attention heads
    fn merge_heads(&self, all_head_outputs: &[f32]) -> Vec<f32> {
        // Simple concatenation for now
        all_head_outputs.to_vec()
    }

    /// Calculate attention scores for memory retrieval
    pub async fn score_memories(
        &mut self,
        query_embedding: &[f32],
        memory_embeddings: &[(String, Vec<f32>)],
    ) -> Vec<(String, f32)> {
        let mut scores = Vec::new();

        for (memory_id, embedding) in memory_embeddings {
            let score = self.calculate_similarity(query_embedding, embedding);
            scores.push((memory_id.clone(), score));
            self.attention_scores.insert(memory_id.clone(), score);
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
    }

    /// Calculate similarity between embeddings
    fn calculate_similarity(&self, query: &[f32], memory: &[f32]) -> f32 {
        // Cosine similarity
        let dot_product: f32 = query.iter().zip(memory.iter()).map(|(q, m)| q * m).sum();

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let memory_norm: f32 = memory.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_norm > 0.0 && memory_norm > 0.0 {
            dot_product / (query_norm * memory_norm)
        } else {
            0.0
        }
    }

    /// Update attention scores based on feedback
    pub fn update_scores(&mut self, memory_id: &str, feedback: f32) {
        if let Some(score) = self.attention_scores.get_mut(memory_id) {
            // Exponential moving average update
            let alpha = 0.1;
            *score = (1.0 - alpha) * *score + alpha * feedback;
        }
    }

    /// Get top-k attended memories
    pub fn get_top_k(&self, k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<_> = self
            .attention_scores
            .iter()
            .map(|(id, score)| (id.clone(), *score))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        scores
    }
}

/// Output from a single attention head
struct HeadAttentionOutput {
    weighted_values: Vec<f32>,
    attention_scores: Vec<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            hidden_dim: 512,
            dropout_rate: 0.1,
            use_causal_mask: false,
        }
    }
}

impl AttentionWeights {
    /// Create random attention weights
    pub fn random(hidden_dim: usize) -> Self {
        Self {
            query_weights: Self::random_vector(hidden_dim),
            key_weights: Self::random_vector(hidden_dim),
            value_weights: Self::random_vector(hidden_dim),
            output_weights: Self::random_vector(hidden_dim),
        }
    }

    /// Generate random vector
    fn random_vector(size: usize) -> Vec<f32> {
        (0..size)
            .map(|_| rand::random::<f32>() * 0.1 - 0.05)
            .collect()
    }
}

/// Production attention router for cognitive memory management
///
/// The AttentionRouter orchestrates attention mechanisms across the cognitive system,
/// providing routing logic for attention-based operations similar to how QuantumRouter
/// handles quantum operations.
#[derive(Debug)]
pub struct AttentionRouter {
    /// Core attention mechanism
    pub attention_mechanism: Arc<RwLock<AttentionMechanism>>,
    /// Attention configuration
    pub config: AttentionConfig,
    /// Attention state cache
    pub attention_cache: RwLock<HashMap<String, AttentionOutput>>,
    /// Metrics for attention operations
    pub metrics: RwLock<AttentionMetrics>,
}

/// Metrics for attention operations
#[derive(Debug, Clone, Default)]
pub struct AttentionMetrics {
    /// Total number of attention operations
    pub total_operations: u64,
    /// Average attention computation time
    pub avg_computation_time: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
}

impl AttentionRouter {
    /// Create a new attention router
    pub fn new(config: AttentionConfig) -> Self {
        let attention_mechanism = Arc::new(RwLock::new(AttentionMechanism::new(config.clone())));

        Self {
            attention_mechanism,
            config,
            attention_cache: RwLock::new(HashMap::new()),
            metrics: RwLock::new(AttentionMetrics::default()),
        }
    }

    /// Compute attention between two text inputs
    pub async fn compute_attention(&self, query: &str, key: &str) -> f32 {
        // In a real implementation, we would:
        // 1. Get or create embeddings for query and key
        // 2. Use the attention mechanism to compute attention scores
        // 3. Return the attention score

        // For now, return a dummy score based on string similarity
        let similarity = if query == key {
            1.0
        } else if !query.is_empty() && !key.is_empty() {
            let common_chars = query.chars().filter(|c| key.contains(*c)).count();
            common_chars as f32 / query.len().max(key.len()) as f32
        } else {
            0.0
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;

        similarity
    }

    /// Route attention computation with caching
    pub async fn route_attention(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        cache_key: Option<String>,
    ) -> crate::cognitive::quantum::types::CognitiveResult<AttentionOutput> {
        // Check cache if key provided
        if let Some(ref key) = cache_key {
            if let Some(cached) = self.attention_cache.read().await.get(key) {
                let mut metrics = self.metrics.write().await;
                metrics.cache_hits += 1;
                metrics.cache_hit_rate =
                    metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64;
                return Ok(cached.clone());
            }
        }

        // Compute attention
        let start_time = std::time::Instant::now();
        let mechanism = self.attention_mechanism.read().await;
        let result = mechanism
            .calculate_attention_weights(query, keys, values)
            .await?;
        let computation_time = start_time.elapsed().as_secs_f64();

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.avg_computation_time = (metrics.avg_computation_time
            * (metrics.total_operations - 1) as f64
            + computation_time)
            / metrics.total_operations as f64;

        if cache_key.is_some() {
            metrics.cache_misses += 1;
            metrics.cache_hit_rate =
                metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64;
        }

        // Cache result if key provided
        if let Some(key) = cache_key {
            self.attention_cache
                .write()
                .await
                .insert(key, result.clone());
        }

        Ok(result)
    }

    /// Route with attention using contextual information
    pub async fn route_with_attention(
        &self,
        query: &EnhancedQuery,
        contexts: &[String],
    ) -> Result<RoutingDecision, Box<dyn std::error::Error + Send + Sync>> {
        // Convert contexts to attention vectors
        let context_vectors: Vec<Vec<f32>> = contexts
            .iter()
            .map(|context| {
                // Simple conversion - in practice this would use embeddings
                context
                    .chars()
                    .map(|c| c as u8 as f32 / 255.0)
                    .take(self.config.hidden_dim / self.config.num_heads)
                    .chain(std::iter::repeat(0.0))
                    .take(self.config.hidden_dim / self.config.num_heads)
                    .collect()
            })
            .collect();

        // Convert query to vector
        let query_vector: Vec<f32> = query
            .original
            .chars()
            .map(|c| c as u8 as f32 / 255.0)
            .take(self.config.hidden_dim / self.config.num_heads)
            .chain(std::iter::repeat(0.0))
            .take(self.config.hidden_dim / self.config.num_heads)
            .collect();

        // Use attention mechanism
        let attention_result = self
            .route_attention(
                &query_vector,
                &context_vectors,
                &context_vectors,
                Some(format!("route_{}", uuid::Uuid::new_v4())),
            )
            .await?;

        // Find the best context based on attention scores
        let best_context_idx = attention_result
            .attention_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best_context = contexts
            .get(best_context_idx)
            .cloned()
            .unwrap_or_else(|| "default_context".to_string());

        Ok(RoutingDecision {
            strategy: RoutingStrategy::Attention,
            target_context: best_context.clone(),
            confidence: attention_result.attention_scores[best_context_idx][0].min(1.0),
            alternatives: Vec::new(),
            reasoning: format!(
                "Attention-based routing selected context '{}' with confidence {:.3}",
                best_context, attention_result.attention_scores[best_context_idx][0]
            ),
        })
    }

    /// Get attention metrics
    pub async fn get_metrics(&self) -> AttentionMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear attention cache
    pub async fn clear_cache(&self) {
        self.attention_cache.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_mechanism_creation() {
        let config = AttentionConfig::default();
        let attention = AttentionMechanism::new(config);

        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.head_dim, 64); // 512 / 8
    }

    #[tokio::test]
    async fn test_attention_calculation() {
        let config = AttentionConfig {
            num_heads: 2,
            hidden_dim: 4,
            dropout_rate: 0.0,
            use_causal_mask: false,
        };

        let attention = AttentionMechanism::new(config);

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let keys = vec![vec![0.1, 0.1, 0.1, 0.1], vec![0.2, 0.2, 0.2, 0.2]];
        let values = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let output = attention
            .calculate_attention_weights(&query, &keys, &values)
            .await
            .unwrap();

        assert_eq!(output.attention_scores.len(), 2); // 2 heads
        assert!(!output.context_vector.is_empty());
    }

    #[test]
    fn test_softmax() {
        let attention = AttentionMechanism::new(AttentionConfig::default());

        let scores = vec![1.0, 2.0, 3.0];
        let softmax = attention.softmax(&scores);

        // Check that softmax sums to 1
        let sum: f32 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that higher scores get higher probabilities
        assert!(softmax[2] > softmax[1]);
        assert!(softmax[1] > softmax[0]);
    }

    #[tokio::test]
    async fn test_memory_scoring() {
        let mut attention = AttentionMechanism::new(AttentionConfig::default());

        let query = vec![0.1, 0.2, 0.3];
        let memories = vec![
            ("mem1".to_string(), vec![0.1, 0.2, 0.3]), // Same as query
            ("mem2".to_string(), vec![0.3, 0.2, 0.1]), // Different
            ("mem3".to_string(), vec![0.2, 0.4, 0.6]), // Scaled version
        ];

        let scores = attention.score_memories(&query, &memories).await;

        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].0, "mem1"); // Should be highest similarity
        assert!(scores[0].1 > 0.99); // Should be ~1.0 for identical vectors
    }
}
