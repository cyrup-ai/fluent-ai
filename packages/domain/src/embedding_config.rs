//! Embedding configuration types and traits
//!
//! This module provides core configuration types for working with embedding models.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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

impl<T: AsRef<str>> IntoEmbeddingConfig for T {
    fn into_embedding_config(self) -> EmbeddingConfig {
        EmbeddingConfig::new().with_model(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_embedding_config_defaults() {
        let config = EmbeddingConfig::default();
        assert!(config.model.is_none());
        assert!(config.dimensions.is_none());
        assert!(config.normalize);
        assert_eq!(config.batch_size, 32);
        assert!(config.truncate);
        assert!(config.additional_params.is_empty());
        assert!(config.user.is_none());
        assert!(config.encoding_format.is_none());
    }

    #[test]
    fn test_embedding_config_builder() {
        let config = EmbeddingConfig::new()
            .with_model("test-model")
            .with_dimensions(768)
            .with_normalize(false)
            .with_batch_size(64)
            .with_truncate(false)
            .with_param("test-param", 42)
            .with_user("test-user")
            .with_encoding_format("float");

        assert_eq!(config.model, Some("test-model".to_string()));
        assert_eq!(config.dimensions, Some(768));
        assert!(!config.normalize);
        assert_eq!(config.batch_size, 64);
        assert!(!config.truncate);
        assert_eq!(config.additional_params.get("test-param"), Some(&json!(42)));
        assert_eq!(config.user, Some("test-user".to_string()));
        assert_eq!(config.encoding_format, Some("float".to_string()));
    }

    #[test]
    fn test_into_embedding_config() {
        let config = "test-model".into_embedding_config();
        assert_eq!(config.model, Some("test-model".to_string()));

        let config = EmbeddingConfig::new()
            .with_model("custom-model")
            .into_embedding_config();
        assert_eq!(config.model, Some("custom-model".to_string()));
    }
}
