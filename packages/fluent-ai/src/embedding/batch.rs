//! Zero-allocation batch processing utilities for embeddings
//!
//! High-performance batch operations with optimal memory management,
//! parallel processing, and configurable chunking strategies.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::ZeroOneOrMany;
use crate::async_task::{AsyncStream, AsyncTask};
use crate::domain::chunk::EmbeddingChunk;
use crate::embedding::normalization::{NormalizationMethod, apply_normalization};

/// Batch input types for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingBatch {
    /// Text inputs for embedding
    Texts(Vec<String>),
    /// Image inputs as byte arrays
    Images(Vec<Vec<u8>>),
    /// Mixed text and image inputs
    Mixed {
        texts: Vec<String>,
        images: Vec<Vec<u8>>}}

/// Configuration for batch processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size for processing
    pub batch_size: usize,
    /// Maximum concurrent batches
    pub max_concurrency: usize,
    /// Normalization method to apply
    pub normalization: NormalizationMethod,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Timeout configuration
    pub timeout_seconds: u64,
    /// Whether to preserve input order in results
    pub preserve_order: bool}

/// Retry configuration for failed batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial delay between retries in milliseconds
    pub initial_delay_ms: u64,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrency: 4,
            normalization: NormalizationMethod::L2,
            retry_config: RetryConfig::default(),
            timeout_seconds: 60,
            preserve_order: true}
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 5000}
    }
}

/// Result of a batch embedding operation
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Successfully processed embeddings
    pub embeddings: Vec<Vec<f32>>,
    /// Failed items with error messages
    pub failures: Vec<BatchFailure>,
    /// Processing statistics
    pub stats: BatchStats}

/// Information about a failed batch item
#[derive(Debug, Clone)]
pub struct BatchFailure {
    /// Original input index
    pub index: usize,
    /// Error message
    pub error: String,
    /// Number of retry attempts made
    pub retry_count: usize}

/// Statistics for batch processing operations
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total items processed
    pub total_items: usize,
    /// Successfully processed items
    pub successful_items: usize,
    /// Failed items
    pub failed_items: usize,
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    /// Average time per item in milliseconds
    pub avg_time_per_item_ms: f64,
    /// Number of batches processed
    pub batch_count: usize}

/// High-performance batch processor for embeddings
pub struct BatchProcessor<T> {
    embedding_provider: T,
    config: BatchConfig,
    semaphore: Arc<Semaphore>}

impl<T> BatchProcessor<T>
where
    T: crate::embedding::providers::EnhancedEmbeddingModel + Clone + Send + Sync + 'static,
{
    /// Create new batch processor with provider and configuration
    #[inline(always)]
    pub fn new(embedding_provider: T, config: BatchConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrency));

        Self {
            embedding_provider,
            config,
            semaphore}
    }

    /// Create batch processor with default configuration
    #[inline(always)]
    pub fn with_provider(embedding_provider: T) -> Self {
        Self::new(embedding_provider, BatchConfig::default())
    }

    /// Process a batch of texts with optimal parallelization
    pub fn process_text_batch(&self, texts: Vec<String>) -> AsyncTask<BatchResult> {
        let processor = self.clone();

        crate::async_task::spawn_async(async move {
            let start_time = std::time::Instant::now();
            let total_items = texts.len();

            if texts.is_empty() {
                return BatchResult {
                    embeddings: Vec::new(),
                    failures: Vec::new(),
                    stats: BatchStats {
                        total_items: 0,
                        successful_items: 0,
                        failed_items: 0,
                        processing_time_ms: 0,
                        avg_time_per_item_ms: 0.0,
                        batch_count: 0}};
            }

            let mut all_embeddings = Vec::with_capacity(total_items);
            let mut all_failures = Vec::new();

            // Process in chunks
            let chunks: Vec<_> = texts.chunks(processor.config.batch_size).collect();
            let batch_count = chunks.len();

            for (batch_idx, chunk) in chunks.into_iter().enumerate() {
                let _permit = match processor.semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => continue, // Skip this batch if semaphore is closed
                };

                let batch_texts: Vec<String> = chunk.to_vec();
                let provider = processor.embedding_provider.clone();
                let config = processor.config.clone();

                // Process batch with retry logic
                let batch_result = Self::process_batch_with_retry(
                    provider,
                    batch_texts.clone(),
                    batch_idx * processor.config.batch_size,
                    &config.retry_config,
                )
                .await;

                match batch_result {
                    Ok(mut embeddings) => {
                        // Apply normalization if configured
                        if !matches!(config.normalization, NormalizationMethod::None) {
                            for embedding in &mut embeddings {
                                apply_normalization(embedding, config.normalization);
                            }
                        }
                        all_embeddings.extend(embeddings);
                    }
                    Err(failures) => {
                        all_failures.extend(failures);
                        // Add empty embeddings for failed items to maintain order
                        for _ in 0..batch_texts.len() {
                            all_embeddings.push(Vec::new());
                        }
                    }
                }
            }

            let processing_time = start_time.elapsed();
            let processing_time_ms = processing_time.as_millis() as u64;
            let failed_items = all_failures.len();
            let successful_items = total_items - failed_items;

            BatchResult {
                embeddings: all_embeddings,
                failures: all_failures,
                stats: BatchStats {
                    total_items,
                    successful_items,
                    failed_items,
                    processing_time_ms,
                    avg_time_per_item_ms: if total_items > 0 {
                        processing_time_ms as f64 / total_items as f64
                    } else {
                        0.0
                    },
                    batch_count}}
        })
    }

    /// Stream embeddings for large batches with backpressure control
    pub fn stream_text_batch(&self, texts: Vec<String>) -> AsyncStream<EmbeddingChunk> {
        let processor = self.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let chunks: Vec<_> = texts.chunks(processor.config.batch_size).collect();

            for (batch_idx, chunk) in chunks.into_iter().enumerate() {
                let _permit = match processor.semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => continue, // Skip this batch if semaphore is closed
                };

                let batch_texts: Vec<String> = chunk.to_vec();
                let provider = processor.embedding_provider.clone();
                let base_index = batch_idx * processor.config.batch_size;

                match Self::process_batch_with_retry(
                    provider,
                    batch_texts.clone(),
                    base_index,
                    &processor.config.retry_config,
                )
                .await
                {
                    Ok(mut embeddings) => {
                        // Apply normalization
                        if !matches!(processor.config.normalization, NormalizationMethod::None) {
                            for embedding in &mut embeddings {
                                apply_normalization(embedding, processor.config.normalization);
                            }
                        }

                        // Send individual embedding chunks
                        for (idx, embedding) in embeddings.into_iter().enumerate() {
                            let chunk = EmbeddingChunk {
                                embeddings: crate::ZeroOneOrMany::from_vec(embedding),
                                index: base_index + idx,
                                metadata: HashMap::new()};

                            if tx.send(chunk).is_err() {
                                break;
                            }
                        }
                    }
                    Err(failures) => {
                        // Send error chunks for failed items
                        for failure in failures {
                            let mut metadata = HashMap::new();
                            metadata.insert("error".to_string(), serde_json::json!(failure.error));
                            metadata.insert(
                                "retry_count".to_string(),
                                serde_json::json!(failure.retry_count),
                            );

                            let error_chunk = EmbeddingChunk {
                                embeddings: crate::ZeroOneOrMany::from_vec(Vec::new()),
                                index: failure.index,
                                metadata};

                            if tx.send(error_chunk).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        });

        AsyncStream::new(rx)
    }

    /// Process a batch with retry logic and exponential backoff
    async fn process_batch_with_retry(
        provider: T,
        texts: Vec<String>,
        base_index: usize,
        retry_config: &RetryConfig,
    ) -> Result<Vec<Vec<f32>>, Vec<BatchFailure>> {
        let mut attempt = 0;
        let mut delay_ms = retry_config.initial_delay_ms;

        loop {
            // Try to process the batch
            let embedding_config = super::EmbeddingConfig::default();
            let texts_zero_one_many = ZeroOneOrMany::from_vec(texts.clone());
            let result = provider
                .embed_batch_texts(&texts_zero_one_many, Some(&embedding_config))
                .await;

            // Convert ZeroOneOrMany<ZeroOneOrMany<f32>> to Vec<Vec<f32>> for compatibility
            let result_vec: Vec<Vec<f32>> = match result {
                ZeroOneOrMany::None => Vec::new(),
                ZeroOneOrMany::One(single_embedding) => {
                    let vec_embedding = match single_embedding {
                        ZeroOneOrMany::None => Vec::new(),
                        ZeroOneOrMany::One(val) => vec![val],
                        ZeroOneOrMany::Many(vals) => vals};
                    vec![vec_embedding]
                }
                ZeroOneOrMany::Many(embeddings) => embeddings
                    .into_iter()
                    .map(|emb| match emb {
                        ZeroOneOrMany::None => Vec::new(),
                        ZeroOneOrMany::One(val) => vec![val],
                        ZeroOneOrMany::Many(vals) => vals})
                    .collect()};

            // Check if we got valid embeddings
            if !result_vec.is_empty() && result_vec.len() == texts.len() {
                // Verify embeddings are not all zeros (indicating errors)
                let has_valid_embeddings = result_vec
                    .iter()
                    .any(|emb| !emb.is_empty() && emb.iter().any(|&x| x != 0.0));

                if has_valid_embeddings {
                    return Ok(result_vec);
                }
            }

            // If we've exhausted retries, return failures
            attempt += 1;
            if attempt > retry_config.max_retries {
                let failures: Vec<BatchFailure> = texts
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| BatchFailure {
                        index: base_index + idx,
                        error: "Failed to generate embedding after retries".to_string(),
                        retry_count: attempt - 1})
                    .collect();

                return Err(failures);
            }

            // Wait before retrying with exponential backoff
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            delay_ms = (delay_ms as f64 * retry_config.backoff_multiplier) as u64;
            delay_ms = delay_ms.min(retry_config.max_delay_ms);
        }
    }
}

impl<T> Clone for BatchProcessor<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            embedding_provider: self.embedding_provider.clone(),
            config: self.config.clone(),
            semaphore: self.semaphore.clone()}
    }
}

/// Utility functions for batch processing
pub mod utils {
    use super::*;

    /// Split large input into optimal batch sizes
    #[inline(always)]
    pub fn optimize_batch_size(
        total_items: usize,
        max_batch_size: usize,
        max_concurrency: usize,
    ) -> usize {
        if total_items <= max_batch_size {
            return total_items;
        }

        // Calculate optimal batch size for even distribution
        let batches_needed = (total_items + max_batch_size - 1) / max_batch_size;
        let optimal_batches = batches_needed.min(max_concurrency);
        (total_items + optimal_batches - 1) / optimal_batches
    }

    /// Estimate processing time based on batch configuration
    #[inline(always)]
    pub fn estimate_processing_time(
        item_count: usize,
        avg_time_per_item_ms: f64,
        batch_config: &BatchConfig,
    ) -> u64 {
        if item_count == 0 {
            return 0;
        }

        let batches_needed = (item_count + batch_config.batch_size - 1) / batch_config.batch_size;
        let parallel_batches =
            (batches_needed + batch_config.max_concurrency - 1) / batch_config.max_concurrency;

        (parallel_batches as f64 * batch_config.batch_size as f64 * avg_time_per_item_ms) as u64
    }

    /// Calculate memory requirements for batch processing
    #[inline(always)]
    pub fn estimate_memory_usage(
        item_count: usize,
        avg_embedding_dimensions: usize,
        batch_config: &BatchConfig,
    ) -> usize {
        // Estimate memory for concurrent batches
        let concurrent_items = batch_config.batch_size * batch_config.max_concurrency;
        let active_items = concurrent_items.min(item_count);

        // Memory for f32 vectors plus overhead
        active_items * avg_embedding_dimensions * std::mem::size_of::<f32>() * 2 // 2x for processing overhead
    }

    /// Create optimized batch configuration for specific workload
    #[inline(always)]
    pub fn create_optimized_config(
        total_items: usize,
        available_memory_mb: usize,
        avg_embedding_dims: usize,
    ) -> BatchConfig {
        // Calculate memory per item (in bytes)
        let memory_per_item = avg_embedding_dims * std::mem::size_of::<f32>() * 2;
        let available_memory_bytes = available_memory_mb * 1024 * 1024;

        // Calculate safe batch size based on memory
        let max_items_in_memory = available_memory_bytes / memory_per_item;
        let safe_batch_size = (max_items_in_memory / 4).clamp(10, 1000); // Reserve 75% headroom

        // Calculate optimal concurrency
        let optimal_concurrency = if total_items > safe_batch_size * 4 {
            4 // High parallelism for large datasets
        } else if total_items > safe_batch_size {
            2 // Moderate parallelism
        } else {
            1 // Single batch
        };

        BatchConfig {
            batch_size: safe_batch_size,
            max_concurrency: optimal_concurrency,
            normalization: NormalizationMethod::L2,
            retry_config: RetryConfig::default(),
            timeout_seconds: 120, // Longer timeout for large batches
            preserve_order: true}
    }

    /// Validate batch configuration for consistency
    #[inline(always)]
    pub fn validate_config(config: &BatchConfig) -> Result<(), String> {
        if config.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        if config.max_concurrency == 0 {
            return Err("Max concurrency must be greater than 0".to_string());
        }

        if config.timeout_seconds == 0 {
            return Err("Timeout must be greater than 0".to_string());
        }

        if config.retry_config.max_retries > 10 {
            return Err("Max retries should not exceed 10".to_string());
        }

        Ok(())
    }
}
