//! Embedding Request Models for AI Provider Integration
//!
//! This module provides unified embedding request structures that work across all 8 providers
//! supporting text embeddings while maintaining type safety, zero-allocation patterns, and 
//! efficient batch processing capabilities.
//!
//! # Supported Providers
//!
//! - **OpenAI**: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002
//! - **Cohere**: embed-english-v3.0, embed-multilingual-v3.0, embed-english-light-v3.0
//! - **Azure**: OpenAI models via Azure endpoints with deployment support
//! - **Mistral**: mistral-embed with 1024 dimensions
//! - **Together**: Various open-source embedding models
//! - **Ollama**: Local embedding models with variable dimensions
//! - **Google Gemini**: text-embedding-004 with 768 dimensions
//! - **HuggingFace**: sentence-transformers and custom embedding models
//!
//! # Architecture
//!
//! The embedding system supports both single-text and batch processing with automatic
//! optimization based on input size. Small batches use ArrayVec for zero allocation,
//! while larger batches use Vec for efficiency.
//!
//! # Usage Examples
//!
//! ```rust
//! use fluent_ai_domain::http::requests::embedding::{EmbeddingRequest, EmbeddingInput};
//! use fluent_ai_domain::Provider;
//!
//! // Single text embedding
//! let request = EmbeddingRequest::new("Hello, world!", "text-embedding-3-large")?;
//!
//! // Batch embedding with validation
//! let texts = vec!["Text 1".to_string(), "Text 2".to_string()];
//! let request = EmbeddingRequest::batch(texts, "text-embedding-3-large")?;
//!
//! // Provider-specific request
//! let request = EmbeddingRequest::for_provider(
//!     Provider::Cohere,
//!     EmbeddingInput::Single("Query text".to_string()),
//!     "embed-english-v3.0"
//! )?;
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use arrayvec::{ArrayString, ArrayVec};
use serde::{Deserialize, Serialize};

use crate::http::common::{ValidationError, MAX_IDENTIFIER_LEN};
use crate::Provider;

/// Maximum number of texts in a single embedding request
pub const MAX_BATCH_SIZE: usize = 2048;

/// Maximum number of tokens per text (approximate, varies by model)
pub const MAX_TOKENS_PER_TEXT: usize = 8192;

/// Maximum number of texts for zero-allocation ArrayVec optimization
pub const SMALL_BATCH_SIZE: usize = 32;

/// Maximum length for model names
pub const MAX_MODEL_NAME_LEN: usize = 64;

/// Maximum length for input type and encoding format strings
pub const MAX_FORMAT_LEN: usize = 32;

/// Global statistics for embedding requests
static EMBEDDING_STATS: EmbeddingStats = EmbeddingStats::new();

/// Universal embedding request for text embedding generation
/// 
/// This struct provides a unified interface for embedding requests across all providers
/// while optimizing for both single-text and batch processing scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Input texts to embed
    pub input: EmbeddingInput,
    
    /// Model to use for embedding generation
    pub model: ArrayString<MAX_MODEL_NAME_LEN>,
    
    /// Encoding format (float, base64, etc.) - provider specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<ArrayString<MAX_FORMAT_LEN>>,
    
    /// Number of dimensions for the embeddings (v3 models only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    
    /// User identifier for tracking and billing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<ArrayString<MAX_IDENTIFIER_LEN>>,
    
    /// Provider-specific extensions
    #[serde(flatten)]
    pub extensions: EmbeddingExtensions,
}

/// Input types for embedding generation
/// 
/// Supports single text, batch processing, and token array inputs with
/// automatic optimization based on input size.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text string
    Single(String),
    
    /// Small batch using ArrayVec for zero allocation
    SmallBatch(ArrayVec<String, SMALL_BATCH_SIZE>),
    
    /// Large batch using Vec for efficiency
    LargeBatch(Vec<String>),
    
    /// Token arrays for advanced use cases
    TokenArrays(Vec<Vec<u32>>),
}

/// Provider-specific extensions for embedding requests
/// 
/// Contains optional parameters that are specific to certain providers
/// while maintaining compatibility across all embedding providers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingExtensions {
    /// Cohere-specific input type optimization hint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<CohereInputType>,
    
    /// Cohere truncation mode for oversized texts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<CohereTruncate>,
    
    /// Cohere embedding types to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_types: Option<ArrayVec<CohereEmbeddingType, 4>>,
    
    /// Azure deployment name for Azure OpenAI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment: Option<ArrayString<MAX_IDENTIFIER_LEN>>,
    
    /// API version for Azure and other providers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<ArrayString<16>>,
    
    /// Ollama-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ollama_options: Option<OllamaEmbeddingOptions>,
}

/// Cohere input type for optimization hints
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereInputType {
    /// Text for search queries
    SearchQuery,
    /// Text for search documents
    SearchDocument,
    /// Text for classification tasks
    Classification,
    /// Text for clustering tasks
    Clustering,
}

/// Cohere truncation mode for oversized texts
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CohereTruncate {
    /// No truncation (will error if text is too long)
    None,
    /// Truncate from the start
    Start,
    /// Truncate from the end
    End,
}

/// Cohere embedding types to return
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereEmbeddingType {
    /// Dense float embeddings
    Float,
    /// Integer embeddings (quantized)
    Int8,
    /// Unsigned integer embeddings
    Uint8,
    /// Binary embeddings
    Binary,
    /// Unsigned binary embeddings
    Ubinary,
}

/// Ollama-specific embedding options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingOptions {
    /// Keep model loaded in memory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    
    /// Truncate input to fit context window
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
}

/// Model capabilities and constraints for embedding models
#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    /// Provider that serves this model
    pub provider: Provider,
    
    /// Model name identifier
    pub name: ArrayString<MAX_MODEL_NAME_LEN>,
    
    /// Maximum number of dimensions
    pub max_dimensions: u32,
    
    /// Default number of dimensions
    pub default_dimensions: u32,
    
    /// Maximum input tokens per text
    pub max_input_tokens: u32,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Whether model supports custom dimensions
    pub supports_custom_dimensions: bool,
    
    /// Whether model supports batch processing
    pub supports_batch: bool,
}

/// Embedding request error types
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingRequestError {
    /// Input is empty
    EmptyInput,
    
    /// Batch size exceeds maximum
    BatchTooLarge(usize),
    
    /// Text exceeds token limit
    TextTooLong { index: usize, length: usize, max_length: usize },
    
    /// Model name is invalid or too long
    InvalidModel(String),
    
    /// Dimensions parameter is invalid for this model
    InvalidDimensions { model: String, dimensions: u32, max_dimensions: u32 },
    
    /// Provider doesn't support embeddings
    ProviderNotSupported(Provider),
    
    /// Validation error from domain types
    ValidationError(ValidationError),
    
    /// Model not found in registry
    ModelNotFound(String),
    
    /// Provider-specific parameter error
    ProviderParameterError { provider: Provider, parameter: String, message: String },
}

/// Statistics for embedding request processing
#[derive(Debug)]
pub struct EmbeddingStats {
    /// Total requests processed
    pub total_requests: AtomicU32,
    /// Total texts processed
    pub total_texts: AtomicU32,
    /// Total tokens processed (estimated)
    pub total_tokens: AtomicU32,
    /// Batch requests processed
    pub batch_requests: AtomicU32,
}

impl EmbeddingRequest {
    /// Create a new embedding request for a single text
    #[inline]
    pub fn new(text: impl Into<String>, model: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        let text_string = text.into();
        Self::validate_text(&text_string, 0)?;
        
        let model_name = ArrayString::from(model.as_ref())
            .map_err(|_| EmbeddingRequestError::InvalidModel(model.as_ref().to_string()))?;
        
        EMBEDDING_STATS.record_request(1);
        
        Ok(Self {
            input: EmbeddingInput::Single(text_string),
            model: model_name,
            encoding_format: None,
            dimensions: None,
            user: None,
            extensions: EmbeddingExtensions::default(),
        })
    }
    
    /// Create a new embedding request for batch processing
    #[inline]
    pub fn batch(texts: Vec<String>, model: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        if texts.is_empty() {
            return Err(EmbeddingRequestError::EmptyInput);
        }
        
        if texts.len() > MAX_BATCH_SIZE {
            return Err(EmbeddingRequestError::BatchTooLarge(texts.len()));
        }
        
        // Validate all texts
        for (index, text) in texts.iter().enumerate() {
            Self::validate_text(text, index)?;
        }
        
        let model_name = ArrayString::from(model.as_ref())
            .map_err(|_| EmbeddingRequestError::InvalidModel(model.as_ref().to_string()))?;
        
        // Optimize storage based on batch size
        let input = if texts.len() <= SMALL_BATCH_SIZE {
            let mut small_batch = ArrayVec::new();
            for text in texts {
                small_batch.try_push(text)
                    .map_err(|_| EmbeddingRequestError::BatchTooLarge(small_batch.len()))?;
            }
            EmbeddingInput::SmallBatch(small_batch)
        } else {
            EmbeddingInput::LargeBatch(texts)
        };
        
        EMBEDDING_STATS.record_batch_request(input.count());
        
        Ok(Self {
            input,
            model: model_name,
            encoding_format: None,
            dimensions: None,
            user: None,
            extensions: EmbeddingExtensions::default(),
        })
    }
    
    /// Create a provider-specific embedding request
    #[inline]
    pub fn for_provider(
        provider: Provider,
        input: EmbeddingInput,
        model: impl AsRef<str>,
    ) -> Result<Self, EmbeddingRequestError> {
        // Validate provider supports embeddings
        if !Self::provider_supports_embeddings(provider) {
            return Err(EmbeddingRequestError::ProviderNotSupported(provider));
        }
        
        let model_name = ArrayString::from(model.as_ref())
            .map_err(|_| EmbeddingRequestError::InvalidModel(model.as_ref().to_string()))?;
        
        // Validate input
        input.validate()?;
        
        let extensions = Self::default_extensions_for_provider(provider);
        
        EMBEDDING_STATS.record_request(input.count());
        
        Ok(Self {
            input,
            model: model_name,
            encoding_format: Self::default_encoding_format(provider),
            dimensions: None,
            user: None,
            extensions,
        })
    }
    
    /// Set encoding format
    #[inline]
    pub fn with_encoding_format(mut self, format: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        self.encoding_format = Some(ArrayString::from(format.as_ref())
            .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                provider: Provider::OpenAI, // Generic provider for format error
                parameter: "encoding_format".to_string(),
                message: "Encoding format too long".to_string(),
            })?);
        Ok(self)
    }
    
    /// Set custom dimensions (v3 models only)
    #[inline]
    pub fn with_dimensions(mut self, dimensions: u32) -> Result<Self, EmbeddingRequestError> {
        // Validate dimensions for known models
        if let Some(model_info) = Self::get_model_info(&self.model) {
            if !model_info.supports_custom_dimensions {
                return Err(EmbeddingRequestError::InvalidDimensions {
                    model: self.model.to_string(),
                    dimensions,
                    max_dimensions: model_info.max_dimensions,
                });
            }
            if dimensions > model_info.max_dimensions {
                return Err(EmbeddingRequestError::InvalidDimensions {
                    model: self.model.to_string(),
                    dimensions,
                    max_dimensions: model_info.max_dimensions,
                });
            }
        }
        
        self.dimensions = Some(dimensions);
        Ok(self)
    }
    
    /// Set user identifier
    #[inline]
    pub fn with_user(mut self, user: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        self.user = Some(ArrayString::from(user.as_ref())
            .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                provider: Provider::OpenAI,
                parameter: "user".to_string(),
                message: "User identifier too long".to_string(),
            })?);
        Ok(self)
    }
    
    /// Set Cohere input type
    #[inline]
    pub fn with_cohere_input_type(mut self, input_type: CohereInputType) -> Self {
        self.extensions.input_type = Some(input_type);
        self
    }
    
    /// Set Cohere truncation mode
    #[inline]
    pub fn with_cohere_truncate(mut self, truncate: CohereTruncate) -> Self {
        self.extensions.truncate = Some(truncate);
        self
    }
    
    /// Set Cohere embedding types
    #[inline]
    pub fn with_cohere_embedding_types(mut self, types: Vec<CohereEmbeddingType>) -> Result<Self, EmbeddingRequestError> {
        let mut type_array = ArrayVec::new();
        for embedding_type in types {
            type_array.try_push(embedding_type)
                .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                    provider: Provider::Cohere,
                    parameter: "embedding_types".to_string(),
                    message: "Too many embedding types".to_string(),
                })?;
        }
        self.extensions.embedding_types = Some(type_array);
        Ok(self)
    }
    
    /// Set Azure deployment name
    #[inline]
    pub fn with_azure_deployment(mut self, deployment: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        self.extensions.deployment = Some(ArrayString::from(deployment.as_ref())
            .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                provider: Provider::Azure,
                parameter: "deployment".to_string(),
                message: "Deployment name too long".to_string(),
            })?);
        Ok(self)
    }
    
    /// Set API version
    #[inline]
    pub fn with_api_version(mut self, version: impl AsRef<str>) -> Result<Self, EmbeddingRequestError> {
        self.extensions.api_version = Some(ArrayString::from(version.as_ref())
            .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                provider: Provider::Azure, // Common for versioned APIs
                parameter: "api_version".to_string(),
                message: "API version too long".to_string(),
            })?);
        Ok(self)
    }
    
    /// Set Ollama-specific options
    #[inline]
    pub fn with_ollama_options(mut self, options: OllamaEmbeddingOptions) -> Self {
        self.extensions.ollama_options = Some(options);
        self
    }
    
    /// Validate the entire request
    #[inline]
    pub fn validate(&self) -> Result<(), EmbeddingRequestError> {
        // Validate input
        self.input.validate()?;
        
        // Validate model exists for known models
        if let Some(model_info) = Self::get_model_info(&self.model) {
            // Validate batch size
            if self.input.count() > model_info.max_batch_size {
                return Err(EmbeddingRequestError::BatchTooLarge(self.input.count()));
            }
            
            // Validate dimensions
            if let Some(dimensions) = self.dimensions {
                if !model_info.supports_custom_dimensions {
                    return Err(EmbeddingRequestError::InvalidDimensions {
                        model: self.model.to_string(),
                        dimensions,
                        max_dimensions: model_info.max_dimensions,
                    });
                }
                if dimensions > model_info.max_dimensions {
                    return Err(EmbeddingRequestError::InvalidDimensions {
                        model: self.model.to_string(),
                        dimensions,
                        max_dimensions: model_info.max_dimensions,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the effective dimensions for this request
    #[inline]
    pub fn effective_dimensions(&self) -> Option<u32> {
        self.dimensions.or_else(|| {
            Self::get_model_info(&self.model).map(|info| info.default_dimensions)
        })
    }
    
    /// Estimate total tokens for all texts
    #[inline]
    pub fn estimate_total_tokens(&self) -> u32 {
        self.input.estimate_tokens()
    }
    
    /// Get the number of texts in the request
    #[inline]
    pub fn text_count(&self) -> usize {
        self.input.count()
    }
    
    /// Check if request is optimized for batch processing
    #[inline]
    pub fn is_batch_optimized(&self) -> bool {
        matches!(self.input, EmbeddingInput::SmallBatch(_)) && self.input.count() > 1
    }
    
    /// Convert to provider-specific format
    #[inline]
    pub fn to_provider_json(&self, provider: Provider) -> Result<serde_json::Value, EmbeddingRequestError> {
        match provider {
            Provider::OpenAI | Provider::Azure => self.to_openai_format(),
            Provider::Cohere => self.to_cohere_format(),
            Provider::Mistral => self.to_mistral_format(),
            Provider::Together => self.to_together_format(),
            Provider::Ollama => self.to_ollama_format(),
            Provider::Gemini => self.to_gemini_format(),
            Provider::HuggingFace => self.to_huggingface_format(),
            _ => Err(EmbeddingRequestError::ProviderNotSupported(provider)),
        }
    }
    
    /// Validate individual text input
    #[inline]
    fn validate_text(text: &str, index: usize) -> Result<(), EmbeddingRequestError> {
        if text.is_empty() {
            return Err(EmbeddingRequestError::EmptyInput);
        }
        
        // Rough token estimation: ~4 characters per token
        let estimated_tokens = (text.len() + 3) / 4;
        if estimated_tokens > MAX_TOKENS_PER_TEXT {
            return Err(EmbeddingRequestError::TextTooLong {
                index,
                length: estimated_tokens,
                max_length: MAX_TOKENS_PER_TEXT,
            });
        }
        
        Ok(())
    }
    
    /// Check if provider supports embeddings
    #[inline]
    const fn provider_supports_embeddings(provider: Provider) -> bool {
        matches!(provider, 
            Provider::OpenAI | Provider::Cohere | Provider::Azure | Provider::Mistral |
            Provider::Together | Provider::Ollama | Provider::Gemini | Provider::HuggingFace
        )
    }
    
    /// Get default encoding format for provider
    #[inline]
    fn default_encoding_format(provider: Provider) -> Option<ArrayString<MAX_FORMAT_LEN>> {
        match provider {
            Provider::OpenAI | Provider::Azure => Some(ArrayString::from("float").unwrap_or_default()),
            Provider::Cohere => None, // Cohere doesn't use encoding_format
            _ => None,
        }
    }
    
    /// Get default extensions for provider
    #[inline]
    fn default_extensions_for_provider(provider: Provider) -> EmbeddingExtensions {
        match provider {
            Provider::Cohere => EmbeddingExtensions {
                input_type: Some(CohereInputType::SearchDocument),
                truncate: Some(CohereTruncate::End),
                ..Default::default()
            },
            Provider::Azure => EmbeddingExtensions {
                api_version: Some(ArrayString::from("2024-02-01").unwrap_or_default()),
                ..Default::default()
            },
            _ => EmbeddingExtensions::default(),
        }
    }
    
    /// Get model information for known models
    #[inline]
    fn get_model_info(model: &str) -> Option<EmbeddingModelInfo> {
        match model {
            // OpenAI models
            "text-embedding-3-large" => Some(EmbeddingModelInfo {
                provider: Provider::OpenAI,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 3072,
                default_dimensions: 3072,
                max_input_tokens: 8191,
                max_batch_size: 2048,
                supports_custom_dimensions: true,
                supports_batch: true,
            }),
            "text-embedding-3-small" => Some(EmbeddingModelInfo {
                provider: Provider::OpenAI,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 1536,
                default_dimensions: 1536,
                max_input_tokens: 8191,
                max_batch_size: 2048,
                supports_custom_dimensions: true,
                supports_batch: true,
            }),
            "text-embedding-ada-002" => Some(EmbeddingModelInfo {
                provider: Provider::OpenAI,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 1536,
                default_dimensions: 1536,
                max_input_tokens: 8191,
                max_batch_size: 2048,
                supports_custom_dimensions: false,
                supports_batch: true,
            }),
            // Cohere models
            "embed-english-v3.0" | "embed-multilingual-v3.0" => Some(EmbeddingModelInfo {
                provider: Provider::Cohere,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 1024,
                default_dimensions: 1024,
                max_input_tokens: 512,
                max_batch_size: 96,
                supports_custom_dimensions: false,
                supports_batch: true,
            }),
            // Mistral models
            "mistral-embed" => Some(EmbeddingModelInfo {
                provider: Provider::Mistral,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 1024,
                default_dimensions: 1024,
                max_input_tokens: 8192,
                max_batch_size: 1000,
                supports_custom_dimensions: false,
                supports_batch: true,
            }),
            // Google Gemini models
            "text-embedding-004" => Some(EmbeddingModelInfo {
                provider: Provider::Gemini,
                name: ArrayString::from(model).unwrap_or_default(),
                max_dimensions: 768,
                default_dimensions: 768,
                max_input_tokens: 2048,
                max_batch_size: 100,
                supports_custom_dimensions: false,
                supports_batch: true,
            }),
            _ => None,
        }
    }
    
    /// Convert to OpenAI format
    #[inline]
    fn to_openai_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        let mut json = serde_json::json!({
            "model": self.model,
            "input": self.input.to_openai_input()
        });
        
        if let Some(encoding_format) = &self.encoding_format {
            json["encoding_format"] = serde_json::Value::String(encoding_format.to_string());
        }
        
        if let Some(dimensions) = self.dimensions {
            json["dimensions"] = serde_json::Value::Number(dimensions.into());
        }
        
        if let Some(user) = &self.user {
            json["user"] = serde_json::Value::String(user.to_string());
        }
        
        Ok(json)
    }
    
    /// Convert to Cohere format
    #[inline]
    fn to_cohere_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        let mut json = serde_json::json!({
            "model": self.model,
            "texts": self.input.to_cohere_texts()
        });
        
        if let Some(input_type) = self.extensions.input_type {
            json["input_type"] = serde_json::to_value(input_type)
                .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                    provider: Provider::Cohere,
                    parameter: "input_type".to_string(),
                    message: "Failed to serialize input_type".to_string(),
                })?;
        }
        
        if let Some(truncate) = self.extensions.truncate {
            json["truncate"] = serde_json::to_value(truncate)
                .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                    provider: Provider::Cohere,
                    parameter: "truncate".to_string(),
                    message: "Failed to serialize truncate".to_string(),
                })?;
        }
        
        if let Some(ref embedding_types) = self.extensions.embedding_types {
            json["embedding_types"] = serde_json::to_value(embedding_types)
                .map_err(|_| EmbeddingRequestError::ProviderParameterError {
                    provider: Provider::Cohere,
                    parameter: "embedding_types".to_string(),
                    message: "Failed to serialize embedding_types".to_string(),
                })?;
        }
        
        Ok(json)
    }
    
    /// Convert to Mistral format (OpenAI-compatible)
    #[inline]
    fn to_mistral_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        self.to_openai_format()
    }
    
    /// Convert to Together format (OpenAI-compatible)
    #[inline]
    fn to_together_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        self.to_openai_format()
    }
    
    /// Convert to Ollama format
    #[inline]
    fn to_ollama_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        let mut json = serde_json::json!({
            "model": self.model,
            "prompt": self.input.to_ollama_prompt()
        });
        
        if let Some(ref options) = self.extensions.ollama_options {
            if let Some(ref keep_alive) = options.keep_alive {
                json["keep_alive"] = serde_json::Value::String(keep_alive.clone());
            }
            if let Some(truncate) = options.truncate {
                json["truncate"] = serde_json::Value::Bool(truncate);
            }
        }
        
        Ok(json)
    }
    
    /// Convert to Gemini format
    #[inline]
    fn to_gemini_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        Ok(serde_json::json!({
            "requests": [{
                "model": format!("models/{}", self.model),
                "content": {
                    "parts": [{
                        "text": self.input.to_single_text()
                    }]
                }
            }]
        }))
    }
    
    /// Convert to HuggingFace format
    #[inline]
    fn to_huggingface_format(&self) -> Result<serde_json::Value, EmbeddingRequestError> {
        Ok(serde_json::json!({
            "inputs": self.input.to_huggingface_inputs()
        }))
    }
}