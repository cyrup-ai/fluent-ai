//! Embedding model interface for generating vector embeddings - THREAD-SAFE SYNCHRONOUS OPERATIONS
//!
//! This module provides a comprehensive interface for text-to-vector embedding generation:
//! - Synchronous embedding operations for maximum performance
//! - Thread-safe implementations for concurrent access
//! - Batch processing capabilities for efficiency  
//! - Comprehensive error handling and validation
//! - Support for different embedding models and providers

use crate::utils::error::Result;

/// Trait for embedding model implementations - SYNCHRONOUS OPERATIONS ONLY
///
/// This trait defines a thread-safe interface for generating vector embeddings
/// from text. All operations are synchronous for optimal performance and
/// can be used safely from multiple threads.
///
/// Implementations should be:
/// - Thread-safe (Send + Sync)
/// - Deterministic (same input produces same output)
/// - Efficient (optimized for batch processing)
/// - Robust (comprehensive error handling)
#[cfg_attr(test, mockall::automock)]
pub trait EmbeddingModel: Send + Sync + std::fmt::Debug {
    /// Generate a vector embedding for a single text input
    ///
    /// # Arguments
    /// * `text` - Input text to generate embedding for
    /// * `task` - Optional task hint for the embedding model (e.g., "search", "classification")
    ///
    /// # Returns
    /// Result containing the embedding vector as Vec<f32>
    ///
    /// # Errors
    /// - `InvalidInput` if text is empty or contains invalid characters
    /// - `ModelError` if the embedding model fails to process the input
    /// - `NetworkError` if using a remote embedding service
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
    ///
    /// fn generate_embedding(model: &dyn EmbeddingModel) -> Result<(), Box<dyn std::error::Error>> {
    ///     let embedding = model.embed("Hello world", Some("search".to_string()))?;
    ///     println!("Generated embedding with {} dimensions", embedding.len());
    ///     Ok(())
    /// }
    /// ```
    fn embed(&self, text: &str, task: Option<String>) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts in a single operation
    ///
    /// This method is typically more efficient than calling `embed` multiple times
    /// as it can batch the operations and reduce overhead.
    ///
    /// # Arguments
    /// * `texts` - Vector of input texts to generate embeddings for
    /// * `task` - Optional task hint applied to all texts
    ///
    /// # Returns
    /// Result containing vector of embedding vectors (Vec<Vec<f32>>)
    ///
    /// # Errors
    /// - `InvalidInput` if any text is empty or the vector is empty
    /// - `ModelError` if the embedding model fails to process any input
    /// - `NetworkError` if using a remote embedding service
    ///
    /// # Performance Notes
    /// - Implementations should process texts in parallel when possible
    /// - Memory usage scales with batch size - consider chunking large batches
    /// - Network-based models benefit significantly from batching
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
    ///
    /// fn batch_process(model: &dyn EmbeddingModel) -> Result<(), Box<dyn std::error::Error>> {
    ///     let texts = vec!["First text".to_string(), "Second text".to_string()];
    ///     let embeddings = model.batch_embed(&texts, Some("search".to_string()))?;
    ///     println!("Generated {} embeddings", embeddings.len());
    ///     Ok(())
    /// }
    /// ```
    fn batch_embed(&self, texts: &[String], task: Option<String>) -> Result<Vec<Vec<f32>>>;

    /// Get the dimension of the embedding vectors produced by this model
    ///
    /// This is a constant for any given model and is useful for:
    /// - Pre-allocating vector storage
    /// - Validating embedding compatibility
    /// - Configuring downstream components
    ///
    /// # Returns
    /// The number of dimensions in embeddings produced by this model
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
    ///
    /// fn check_dimensions(model: &dyn EmbeddingModel) {
    ///     let dim = model.dimension();
    ///     println!("Model produces {}-dimensional embeddings", dim);
    /// }
    /// ```
    fn dimension(&self) -> usize;

    /// Get the name/identifier of this embedding model
    ///
    /// This should be a human-readable identifier that uniquely identifies
    /// the model variant, useful for:
    /// - Logging and debugging
    /// - Model compatibility checking
    /// - Configuration management
    ///
    /// # Returns
    /// String slice containing the model name
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
    ///
    /// fn log_model_info(model: &dyn EmbeddingModel) {
    ///     println!("Using embedding model: {}", model.name());
    ///     println!("Dimension: {}", model.dimension());
    /// }
    /// ```
    fn name(&self) -> &str;

    /// Validate input text before processing (optional override)
    ///
    /// Default implementation performs basic validation. Models can override
    /// this for more specific validation requirements.
    ///
    /// # Arguments
    /// * `text` - Text to validate
    ///
    /// # Returns
    /// Result indicating validation success or failure
    ///
    /// # Errors
    /// - `InvalidInput` if text fails model-specific validation
    fn validate_input(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Err(crate::utils::error::Error::InvalidInput(
                "Input text cannot be empty".to_string(),
            ));
        }

        if text.len() > 1_000_000 {
            return Err(crate::utils::error::Error::InvalidInput(
                "Input text exceeds maximum length (1M characters)".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate a batch of texts (optional override)
    ///
    /// Default implementation validates each text individually and checks batch size.
    ///
    /// # Arguments
    /// * `texts` - Batch of texts to validate
    ///
    /// # Returns
    /// Result indicating validation success or failure
    ///
    /// # Errors
    /// - `InvalidInput` if batch is empty or any text fails validation
    fn validate_batch(&self, texts: &[String]) -> Result<()> {
        if texts.is_empty() {
            return Err(crate::utils::error::Error::InvalidInput(
                "Batch cannot be empty".to_string(),
            ));
        }

        if texts.len() > 10_000 {
            return Err(crate::utils::error::Error::InvalidInput(
                "Batch size exceeds maximum (10,000 texts)".to_string(),
            ));
        }

        for (index, text) in texts.iter().enumerate() {
            self.validate_input(text).map_err(|e| {
                crate::utils::error::Error::InvalidInput(format!(
                    "Text at index {} failed validation: {}",
                    index, e
                ))
            })?;
        }

        Ok(())
    }

    /// Get model configuration information (optional override)
    ///
    /// Returns a map of configuration parameters for this model.
    /// Useful for debugging, monitoring, and configuration management.
    ///
    /// # Returns
    /// Map of configuration key-value pairs
    fn config_info(&self) -> std::collections::HashMap<String, String> {
        let mut info = std::collections::HashMap::new();
        info.insert("name".to_string(), self.name().to_string());
        info.insert("dimension".to_string(), self.dimension().to_string());
        info
    }

    /// Check if this model supports the specified task type (optional override)
    ///
    /// # Arguments
    /// * `task` - Task type to check (e.g., "search", "classification", "clustering")
    ///
    /// # Returns
    /// True if the task is supported, false otherwise
    fn supports_task(&self, _task: &str) -> bool {
        true // Default: support all tasks
    }

    /// Get recommended batch size for optimal performance (optional override)
    ///
    /// # Returns
    /// Recommended number of texts to process in a single batch
    fn recommended_batch_size(&self) -> usize {
        32 // Default batch size
    }

    /// Get maximum supported batch size (optional override)
    ///
    /// # Returns
    /// Maximum number of texts that can be processed in a single batch
    fn max_batch_size(&self) -> usize {
        1000 // Default maximum
    }

    /// Check if the model is ready for processing (optional override)
    ///
    /// This can be used to verify that:
    /// - Model files are loaded
    /// - Network connections are established
    /// - Required resources are available
    ///
    /// # Returns
    /// Result indicating readiness status
    fn health_check(&self) -> Result<()> {
        Ok(()) // Default: always ready
    }

    /// Estimate memory usage for a batch operation (optional override)
    ///
    /// # Arguments
    /// * `batch_size` - Number of texts to process
    ///
    /// # Returns
    /// Estimated memory usage in bytes
    fn estimate_memory_usage(&self, batch_size: usize) -> usize {
        // Rough estimate: 1KB per text + embedding storage
        let text_memory = batch_size * 1024;
        let embedding_memory = batch_size * self.dimension() * std::mem::size_of::<f32>();
        text_memory + embedding_memory
    }

    /// Process texts in chunks to avoid memory/performance issues (optional override)
    ///
    /// This is a convenience method that automatically chunks large batches
    /// into smaller ones based on the model's recommended batch size.
    ///
    /// # Arguments
    /// * `texts` - Large batch of texts to process
    /// * `task` - Task hint for all texts
    ///
    /// # Returns
    /// Result containing all embeddings in original order
    fn chunked_batch_embed(&self, texts: &[String], task: Option<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size = self.recommended_batch_size();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(chunk_size) {
            let chunk_embeddings = self.batch_embed(chunk, task.clone())?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }
}
