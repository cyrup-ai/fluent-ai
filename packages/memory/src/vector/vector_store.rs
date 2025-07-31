//! Vector store interface for storing and retrieving vectors - THREAD-BASED ARCHITECTURE

use std::collections::HashMap;
use surrealdb::sql::Value;

use crate::utils::error::Result;

/// Type alias for vector metadata
pub type VectorMetadata = Option<HashMap<String, Value>>;

/// Type alias for vector search results
pub type VectorSearchResult = (String, Vec<f32>, f32, VectorMetadata);

/// Trait for vector store implementations - SYNCHRONOUS OPERATIONS ONLY
///
/// This trait provides a thread-safe, synchronous interface for vector storage operations.
/// All implementations must be thread-safe (Send + Sync) but use synchronous methods.
/// For concurrent operations, use thread pools and channels externally.
pub trait VectorStore: Send + Sync {
    /// Add a vector to the store
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data to store
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn add_vector(&mut self, id: &str, vector: Vec<f32>) -> Result<()>;

    /// Get a vector by ID
    ///
    /// # Arguments  
    /// * `id` - Unique identifier of the vector to retrieve
    ///
    /// # Returns
    /// Result containing the vector data and optional metadata
    fn get_vector(&self, id: &str) -> Result<(Vec<f32>, VectorMetadata)>;

    /// Update a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector to update
    /// * `vector` - New vector data
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn update_vector(&mut self, id: &str, vector: Vec<f32>) -> Result<()>;

    /// Delete a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector to delete
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn delete_vector(&mut self, id: &str) -> Result<()>;

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `query_vector` - Vector to find similar vectors for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Result containing vector IDs sorted by similarity (descending)
    fn search_similar(&self, query_vector: &[f32], limit: usize) -> Result<Vec<String>>;

    /// Advanced search with filtering
    ///
    /// # Arguments
    /// * `query_vector` - Vector to find similar vectors for
    /// * `limit` - Optional maximum number of results
    /// * `filters` - Optional metadata filters to apply
    ///
    /// # Returns
    /// Result containing detailed search results with similarity scores
    fn search(
        &self,
        query_vector: &[f32],
        limit: Option<usize>,
        filters: Option<HashMap<String, Value>>,
    ) -> Result<Vec<VectorSearchResult>>;

    /// Get total count of vectors in the store
    ///
    /// # Returns
    /// Result containing the number of vectors stored
    fn count(&self) -> Result<usize>;

    /// Clear all vectors from the store
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn clear(&mut self) -> Result<()>;

    /// Add metadata for a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector
    /// * `metadata` - Metadata to associate with the vector
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn add_metadata(&mut self, id: &str, metadata: HashMap<String, Value>) -> Result<()> {
        // Default implementation returns error - implementations should override
        Err(crate::utils::error::Error::Other(
            "Metadata operations not supported by this vector store".to_string(),
        ))
    }

    /// Get metadata for a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector
    ///
    /// # Returns
    /// Result containing the metadata if it exists
    fn get_metadata(&self, id: &str) -> Result<VectorMetadata> {
        // Default implementation returns None
        Ok(None)
    }

    /// Update metadata for a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector
    /// * `metadata` - New metadata to replace existing metadata
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn update_metadata(&mut self, id: &str, metadata: HashMap<String, Value>) -> Result<()> {
        // Default implementation returns error - implementations should override
        Err(crate::utils::error::Error::Other(
            "Metadata operations not supported by this vector store".to_string(),
        ))
    }

    /// Remove metadata for a vector
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the vector
    ///
    /// # Returns
    /// Result indicating success or failure of the operation
    fn remove_metadata(&mut self, id: &str) -> Result<()> {
        // Default implementation returns error - implementations should override
        Err(crate::utils::error::Error::Other(
            "Metadata operations not supported by this vector store".to_string(),
        ))
    }
}
