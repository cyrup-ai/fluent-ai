//! Embedding Configuration Types and Traits
//!
//! This module provides core configuration types for working with embedding models.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for embedding models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    /// Model identifier (e.g., "text-embedding-ada-002")
    pub model: Option<String>,

    /// Embedding dimensions (if configurable)
    pub dimensions: Option<usize>,

    /// Whether to normalize embeddings to unit length
    #[serde(default = "default_normalize")]
    pub normalize: bool,

    /// Batch size for processing multiple texts
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Whether to truncate input text if it's too long
    #[serde(default = "default_truncate")]
    pub truncate: bool,

    /// Additional provider-specific parameters
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_params: HashMap<String, Value>,

    /// User identifier for tracking/rate limiting (optional)
    pub user: Option<String>,

    /// Encoding format (e.g., "float", "base64")
    pub encoding_format: Option<String>,
}
fn default_normalize() -> bool {
    true
}

fn default_batch_size() -> usize {
    32
}

fn default_truncate() -> bool {
    true
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: None,
            dimensions: None,
            normalize: default_normalize(),
            batch_size: default_batch_size(),
            truncate: default_truncate(),
            additional_params: HashMap::new(),
            user: None,
            encoding_format: None,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new embedding configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model identifier
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the embedding dimensions
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Enable or disable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable truncation
    pub fn with_truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Add an additional parameter
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.additional_params.insert(key.into(), value.into());
        self
    }

    /// Set the user identifier
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the encoding format
    pub fn with_encoding_format(mut self, format: impl Into<String>) -> Self {
        self.encoding_format = Some(format.into());
        self
    }
}

/// Trait for types that can be converted to an EmbeddingConfig
pub trait IntoEmbeddingConfig {
    /// Convert to an EmbeddingConfig
    fn into_embedding_config(self) -> EmbeddingConfig;
}

impl IntoEmbeddingConfig for EmbeddingConfig {
    fn into_embedding_config(self) -> EmbeddingConfig {
        self
    }
}

impl<T: AsRef<str> + Into<String>> IntoEmbeddingConfig for T {
    fn into_embedding_config(self) -> EmbeddingConfig {
        EmbeddingConfig::new().with_model(self.into())
    }
}
