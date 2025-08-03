//! High-level HTTP3 builders with JSONPath selectors for AI model provider APIs
//!
//! This module provides specialized HTTP3 builders that combine the performance and features
//! of the fluent-ai HTTP3 client with JSONPath selectors for targeted data extraction from
//! AI provider API responses.
//!
//! ## Supported Response Types
//!
//! - **OpenAI-compatible**: OpenAI, Mistral, XAI (`{"data": [...], "object": "list"}`)
//! - **Together.ai**: Direct array format (`[...]`)
//! - **HuggingFace**: Wrapped format (`{"models": [...]}`)
//!
//! ## Usage Examples
//!
//! ### OpenAI Provider
//! ```rust
//! use model_info::http3_builders::ModelInfoHttp;
//! 
//! let model_ids = ModelInfoHttp::new()
//!     .bearer_auth("sk-...")
//!     .openai_response()
//!     .model_ids()
//!     .get("https://api.openai.com/v1/models")
//!     .collect_vec()
//!     .await?;
//! ```
//!
//! ### Together.ai Provider
//! ```rust
//! let pricing_data = ModelInfoHttp::new()
//!     .bearer_auth("...")
//!     .together_response()
//!     .pricing_info()
//!     .get("https://api.together.xyz/v1/models")
//!     .stream();
//! ```
//!
//! ### HuggingFace Provider
//! ```rust
//! let popular_models = ModelInfoHttp::new()
//!     .huggingface_response()
//!     .popularity_metrics()
//!     .get("https://huggingface.co/api/models")
//!     .collect_vec()
//!     .await?;
//! ```

use fluent_ai_http3::{Http3Builder, JsonArrayStream, HttpError, HttpResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use futures::Stream;

/// High-level HTTP3 builder with JSONPath selector support for AI model provider APIs
///
/// This builder wraps the fluent-ai HTTP3 client and provides specialized methods
/// for working with different AI provider response formats using JSONPath selectors.
#[derive(Debug, Clone)]
pub struct ModelInfoHttp {
    http3_builder: Http3Builder,
}

impl ModelInfoHttp {
    /// Create a new ModelInfoHttp builder
    ///
    /// Initializes with default HTTP3 configuration optimized for AI provider APIs.
    pub fn new() -> Self {
        Self {
            http3_builder: Http3Builder::new(),
        }
    }

    /// Set Bearer token authentication
    ///
    /// # Arguments
    /// * `token` - The bearer token (e.g., OpenAI API key)
    pub fn bearer_auth(mut self, token: &str) -> Self {
        self.http3_builder = self.http3_builder.bearer_auth(token);
        self
    }

    /// Set API key header authentication
    ///
    /// # Arguments
    /// * `key` - The API key value
    /// * `header_name` - The header name (e.g., "X-API-Key")
    pub fn api_key(mut self, key: &str, header_name: &str) -> Self {
        self.http3_builder = self.http3_builder.header(header_name, key);
        self
    }

    /// Set request timeout
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for the request
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.http3_builder = self.http3_builder.timeout(timeout);
        self
    }

    /// Configure for OpenAI-compatible response format
    ///
    /// Returns a builder configured for responses with format:
    /// ```json
    /// {"data": [...], "object": "list"}
    /// ```
    ///
    /// Used by: OpenAI, Mistral, XAI
    pub fn openai_response(self) -> OpenAiResponseBuilder {
        OpenAiResponseBuilder::new(self.http3_builder)
    }

    /// Configure for Together.ai response format
    ///
    /// Returns a builder configured for direct array responses:
    /// ```json
    /// [...]
    /// ```
    pub fn together_response(self) -> TogetherResponseBuilder {
        TogetherResponseBuilder::new(self.http3_builder)
    }

    /// Configure for HuggingFace response format
    ///
    /// Returns a builder configured for wrapped array responses:
    /// ```json
    /// {"models": [...]}
    /// ```
    pub fn huggingface_response(self) -> HuggingFaceResponseBuilder {
        HuggingFaceResponseBuilder::new(self.http3_builder)
    }
}

impl Default for ModelInfoHttp {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for OpenAI-compatible response format with JSONPath selectors
///
/// Handles responses with the format: `{"data": [...], "object": "list"}`
/// Used by OpenAI, Mistral, and XAI providers.
#[derive(Debug, Clone)]
pub struct OpenAiResponseBuilder {
    http3_builder: Http3Builder,
    jsonpath_selector: Option<String>,
}

impl OpenAiResponseBuilder {
    /// Create a new OpenAI response builder
    pub(crate) fn new(http3_builder: Http3Builder) -> Self {
        Self {
            http3_builder,
            jsonpath_selector: None,
        }
    }

    /// Extract all model objects from the response
    ///
    /// Uses JSONPath selector: `$.data[*]`
    pub fn all_models(mut self) -> Self {
        self.jsonpath_selector = Some("$.data[*]".to_string());
        self
    }

    /// Extract all model IDs from the response
    ///
    /// Uses JSONPath selector: `$.data[*].id`
    pub fn model_ids(mut self) -> Self {
        self.jsonpath_selector = Some("$.data[*].id".to_string());
        self
    }

    /// Extract creation timestamps for all models
    ///
    /// Uses JSONPath selector: `$.data[*].created`
    pub fn created_timestamps(mut self) -> Self {
        self.jsonpath_selector = Some("$.data[*].created".to_string());
        self
    }

    /// Extract ownership information for all models
    ///
    /// Uses JSONPath selector: `$.data[*].owned_by`
    pub fn owned_by(mut self) -> Self {
        self.jsonpath_selector = Some("$.data[*].owned_by".to_string());
        self
    }

    /// Use a custom JSONPath selector
    ///
    /// # Arguments
    /// * `selector` - Custom JSONPath expression (e.g., "$.data[?@.id=='gpt-4']")
    pub fn with_selector(mut self, selector: &str) -> Self {
        self.jsonpath_selector = Some(selector.to_string());
        self
    }

    /// Execute GET request with configured JSONPath selector
    ///
    /// # Arguments
    /// * `url` - The API endpoint URL
    pub fn get(self, url: &str) -> OpenAiResponseExecution {
        OpenAiResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.data[*]".to_string()),
            url.to_string(),
            "GET".to_string(),
        )
    }

    /// Execute POST request with configured JSONPath selector
    ///
    /// # Arguments
    /// * `url` - The API endpoint URL
    pub fn post(self, url: &str) -> OpenAiResponseExecution {
        OpenAiResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.data[*]".to_string()),
            url.to_string(),
            "POST".to_string(),
        )
    }
}

/// Execution context for OpenAI response processing
#[derive(Debug)]
pub struct OpenAiResponseExecution {
    http3_builder: Http3Builder,
    jsonpath_selector: String,
    url: String,
    method: String,
}

impl OpenAiResponseExecution {
    pub(crate) fn new(
        http3_builder: Http3Builder,
        jsonpath_selector: String,
        url: String,
        method: String,
    ) -> Self {
        Self {
            http3_builder,
            jsonpath_selector,
            url,
            method,
        }
    }

    /// Execute request and return a stream of extracted values
    ///
    /// Returns a streaming iterator over the JSONPath-selected values using the HTTP3 high-level builder.
    pub fn stream<T>(self) -> impl Stream<Item = Result<T, ModelInfoError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        use futures::StreamExt;

        let stream = match self.method.as_str() {
            "GET" => self.http3_builder
                .json()
                .array_stream(&self.jsonpath_selector)
                .get(&self.url)
                .stream(),
            "POST" => self.http3_builder
                .json()
                .array_stream(&self.jsonpath_selector)
                .post(&self.url) 
                .stream(),
            _ => return async_stream::stream! {
                yield Err(ModelInfoError::invalid_request(format!("Unsupported HTTP method: {}", self.method)));
            }
        };
        
        // Convert HttpError to ModelInfoError
        stream.map(|result| result.map_err(ModelInfoError::Http))
    }

    /// Execute request and collect all results into a Vec
    ///
    /// # Returns
    /// A Vec containing all extracted values from the JSONPath selector.
    pub async fn collect_vec<T>(self) -> Result<Vec<T>, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_vec method".to_string()))
    }

    /// Execute request and collect the first result
    ///
    /// # Returns
    /// The first value extracted by the JSONPath selector, or an error if none found.
    pub async fn collect_one<T>(self) -> Result<T, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_one method".to_string()))
    }
}

/// Comprehensive error type for ModelInfo HTTP3 builders
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    /// HTTP request or response error
    #[error("HTTP error: {0}")]
    Http(#[from] HttpError),

    /// JSONPath parsing or evaluation error
    #[error("JSONPath error: {0}")]
    JsonPath(String),

    /// JSON deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// Network connectivity error
    #[error("Network error: {message}")]
    Network { message: String },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting error
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Provider API error
    #[error("Provider API error [{status}]: {message}")]
    ProviderApi { status: u16, message: String },

    /// No results found
    #[error("No results found for JSONPath selector: {selector}")]
    NoResults { selector: String },

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotYetImplemented(String),

    /// Generic error
    #[error("ModelInfo error: {0}")]
    Other(String),
}

/// Builder for HuggingFace response format with JSONPath selectors
///
/// Handles wrapped array response format: `{"models": [...]}`
/// Used specifically by HuggingFace provider.
#[derive(Debug, Clone)]
pub struct HuggingFaceResponseBuilder {
    http3_builder: Http3Builder,
    jsonpath_selector: Option<String>,
}

impl HuggingFaceResponseBuilder {
    /// Create a new HuggingFace response builder
    pub(crate) fn new(http3_builder: Http3Builder) -> Self {
        Self {
            http3_builder,
            jsonpath_selector: None,
        }
    }

    /// Extract all model objects from the response
    pub fn all_models(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*]".to_string());
        self
    }

    /// Extract all model IDs from the response
    pub fn model_ids(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].id".to_string());
        self
    }

    /// Extract download statistics for all models
    pub fn download_stats(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].downloads".to_string());
        self
    }

    /// Extract like counts for all models
    pub fn like_counts(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].likes".to_string());
        self
    }

    /// Extract popularity metrics (both downloads and likes) for all models
    pub fn popularity_metrics(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].[downloads,likes]".to_string());
        self
    }

    /// Extract model tags for all models
    pub fn model_tags(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].tags".to_string());
        self
    }

    /// Extract pipeline tags for all models
    pub fn pipeline_tags(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].pipeline_tag".to_string());
        self
    }

    /// Use a custom JSONPath selector
    pub fn with_selector(mut self, selector: &str) -> Self {
        self.jsonpath_selector = Some(selector.to_string());
        self
    }

    /// Execute GET request with configured JSONPath selector
    pub fn get(self, url: &str) -> HuggingFaceResponseExecution {
        HuggingFaceResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.models[*]".to_string()),
            url.to_string(),
            "GET".to_string(),
        )
    }

    /// Execute POST request with configured JSONPath selector
    pub fn post(self, url: &str) -> HuggingFaceResponseExecution {
        HuggingFaceResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.models[*]".to_string()),
            url.to_string(),
            "POST".to_string(),
        )
    }
}

/// Execution context for HuggingFace response processing
#[derive(Debug)]
pub struct HuggingFaceResponseExecution {
    http3_builder: Http3Builder,
    jsonpath_selector: String,
    url: String,
    method: String,
}

impl HuggingFaceResponseExecution {
    pub(crate) fn new(
        http3_builder: Http3Builder,
        jsonpath_selector: String,
        url: String,
        method: String,
    ) -> Self {
        Self {
            http3_builder,
            jsonpath_selector,
            url,
            method,
        }
    }

    /// Execute request and return a stream of extracted values
    pub fn stream<T>(self) -> impl Stream<Item = Result<T, ModelInfoError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        async_stream::stream! {
            yield Err(ModelInfoError::NotYetImplemented("stream method".to_string()));
        }
    }

    /// Execute request and collect all results into a Vec
    pub async fn collect_vec<T>(self) -> Result<Vec<T>, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_vec method".to_string()))
    }

    /// Execute request and collect the first result
    pub async fn collect_one<T>(self) -> Result<T, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_one method".to_string()))
    }
}

/// Comprehensive error type for ModelInfo HTTP3 builders
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    /// HTTP request or response error
    #[error("HTTP error: {0}")]
    Http(#[from] HttpError),

    /// JSONPath parsing or evaluation error
    #[error("JSONPath error: {0}")]
    JsonPath(String),

    /// JSON deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// Network connectivity error
    #[error("Network error: {message}")]
    Network { message: String },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting error
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Provider API error
    #[error("Provider API error [{status}]: {message}")]
    ProviderApi { status: u16, message: String },

    /// No results found
    #[error("No results found for JSONPath selector: {selector}")]
    NoResults { selector: String },

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotYetImplemented(String),

    /// Generic error
    #[error("ModelInfo error: {0}")]
    Other(String),
}

/// Builder for Together.ai response format with JSONPath selectors
///
/// Handles direct array response format: `[...]`
/// Used specifically by Together.ai provider.
#[derive(Debug, Clone)]
pub struct TogetherResponseBuilder {
    http3_builder: Http3Builder,
    jsonpath_selector: Option<String>,
}

impl TogetherResponseBuilder {
    /// Create a new Together.ai response builder
    pub(crate) fn new(http3_builder: Http3Builder) -> Self {
        Self {
            http3_builder,
            jsonpath_selector: None,
        }
    }

    /// Extract all model objects from the response
    ///
    /// Uses JSONPath selector: `$[*]`
    pub fn all_models(mut self) -> Self {
        self.jsonpath_selector = Some("$[*]".to_string());
        self
    }

    /// Extract all model IDs from the response
    ///
    /// Uses JSONPath selector: `$[*].id`
    pub fn model_ids(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].id".to_string());
        self
    }

    /// Extract pricing information for all models
    ///
    /// Uses JSONPath selector: `$[*].pricing`
    pub fn pricing_info(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].pricing".to_string());
        self
    }

    /// Extract input pricing for all models
    ///
    /// Uses JSONPath selector: `$[*].pricing.input`
    pub fn input_pricing(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].pricing.input".to_string());
        self
    }

    /// Extract output pricing for all models
    ///
    /// Uses JSONPath selector: `$[*].pricing.output`
    pub fn output_pricing(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].pricing.output".to_string());
        self
    }

    /// Extract context length information for all models
    ///
    /// Uses JSONPath selector: `$[*].context_length`
    pub fn context_lengths(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].context_length".to_string());
        self
    }

    /// Extract display names for all models
    ///
    /// Uses JSONPath selector: `$[*].display_name`
    pub fn display_names(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].display_name".to_string());
        self
    }

    /// Extract organization information for all models
    ///
    /// Uses JSONPath selector: `$[*].organization`
    pub fn organizations(mut self) -> Self {
        self.jsonpath_selector = Some("$[*].organization".to_string());
        self
    }

    /// Use a custom JSONPath selector
    ///
    /// # Arguments
    /// * `selector` - Custom JSONPath expression (e.g., "$[?@.pricing.input < 0.001]")
    pub fn with_selector(mut self, selector: &str) -> Self {
        self.jsonpath_selector = Some(selector.to_string());
        self
    }

    /// Execute GET request with configured JSONPath selector
    ///
    /// # Arguments
    /// * `url` - The API endpoint URL
    pub fn get(self, url: &str) -> TogetherResponseExecution {
        TogetherResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$[*]".to_string()),
            url.to_string(),
            "GET".to_string(),
        )
    }

    /// Execute POST request with configured JSONPath selector
    ///
    /// # Arguments
    /// * `url` - The API endpoint URL
    pub fn post(self, url: &str) -> TogetherResponseExecution {
        TogetherResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$[*]".to_string()),
            url.to_string(),
            "POST".to_string(),
        )
    }
}

/// Execution context for Together.ai response processing
#[derive(Debug)]
pub struct TogetherResponseExecution {
    http3_builder: Http3Builder,
    jsonpath_selector: String,
    url: String,
    method: String,
}

impl TogetherResponseExecution {
    pub(crate) fn new(
        http3_builder: Http3Builder,
        jsonpath_selector: String,
        url: String,
        method: String,
    ) -> Self {
        Self {
            http3_builder,
            jsonpath_selector,
            url,
            method,
        }
    }

    /// Execute request and return a stream of extracted values
    ///
    /// Returns a streaming iterator over the JSONPath-selected values.
    pub fn stream<T>(self) -> impl Stream<Item = Result<T, ModelInfoError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        async_stream::stream! {
            yield Err(ModelInfoError::NotYetImplemented("stream method".to_string()));
        }
    }

    /// Execute request and collect all results into a Vec
    ///
    /// # Returns
    /// A Vec containing all extracted values from the JSONPath selector.
    pub async fn collect_vec<T>(self) -> Result<Vec<T>, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_vec method".to_string()))
    }

    /// Execute request and collect the first result
    ///
    /// # Returns
    /// The first value extracted by the JSONPath selector, or an error if none found.
    pub async fn collect_one<T>(self) -> Result<T, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_one method".to_string()))
    }
}

/// Comprehensive error type for ModelInfo HTTP3 builders
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    /// HTTP request or response error
    #[error("HTTP error: {0}")]
    Http(#[from] HttpError),

    /// JSONPath parsing or evaluation error
    #[error("JSONPath error: {0}")]
    JsonPath(String),

    /// JSON deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// Network connectivity error
    #[error("Network error: {message}")]
    Network { message: String },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting error
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Provider API error
    #[error("Provider API error [{status}]: {message}")]
    ProviderApi { status: u16, message: String },

    /// No results found
    #[error("No results found for JSONPath selector: {selector}")]
    NoResults { selector: String },

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotYetImplemented(String),

    /// Generic error
    #[error("ModelInfo error: {0}")]
    Other(String),
}

/// Builder for HuggingFace response format with JSONPath selectors
///
/// Handles wrapped array response format: `{"models": [...]}`
/// Used specifically by HuggingFace provider.
#[derive(Debug, Clone)]
pub struct HuggingFaceResponseBuilder {
    http3_builder: Http3Builder,
    jsonpath_selector: Option<String>,
}

impl HuggingFaceResponseBuilder {
    /// Create a new HuggingFace response builder
    pub(crate) fn new(http3_builder: Http3Builder) -> Self {
        Self {
            http3_builder,
            jsonpath_selector: None,
        }
    }

    /// Extract all model objects from the response
    pub fn all_models(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*]".to_string());
        self
    }

    /// Extract all model IDs from the response
    pub fn model_ids(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].id".to_string());
        self
    }

    /// Extract download statistics for all models
    pub fn download_stats(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].downloads".to_string());
        self
    }

    /// Extract like counts for all models
    pub fn like_counts(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].likes".to_string());
        self
    }

    /// Extract popularity metrics (both downloads and likes) for all models
    pub fn popularity_metrics(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].[downloads,likes]".to_string());
        self
    }

    /// Extract model tags for all models
    pub fn model_tags(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].tags".to_string());
        self
    }

    /// Extract pipeline tags for all models
    pub fn pipeline_tags(mut self) -> Self {
        self.jsonpath_selector = Some("$.models[*].pipeline_tag".to_string());
        self
    }

    /// Use a custom JSONPath selector
    pub fn with_selector(mut self, selector: &str) -> Self {
        self.jsonpath_selector = Some(selector.to_string());
        self
    }

    /// Execute GET request with configured JSONPath selector
    pub fn get(self, url: &str) -> HuggingFaceResponseExecution {
        HuggingFaceResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.models[*]".to_string()),
            url.to_string(),
            "GET".to_string(),
        )
    }

    /// Execute POST request with configured JSONPath selector
    pub fn post(self, url: &str) -> HuggingFaceResponseExecution {
        HuggingFaceResponseExecution::new(
            self.http3_builder,
            self.jsonpath_selector.unwrap_or_else(|| "$.models[*]".to_string()),
            url.to_string(),
            "POST".to_string(),
        )
    }
}

/// Execution context for HuggingFace response processing
#[derive(Debug)]
pub struct HuggingFaceResponseExecution {
    http3_builder: Http3Builder,
    jsonpath_selector: String,
    url: String,
    method: String,
}

impl HuggingFaceResponseExecution {
    pub(crate) fn new(
        http3_builder: Http3Builder,
        jsonpath_selector: String,
        url: String,
        method: String,
    ) -> Self {
        Self {
            http3_builder,
            jsonpath_selector,
            url,
            method,
        }
    }

    /// Execute request and return a stream of extracted values
    pub fn stream<T>(self) -> impl Stream<Item = Result<T, ModelInfoError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        async_stream::stream! {
            yield Err(ModelInfoError::NotYetImplemented("stream method".to_string()));
        }
    }

    /// Execute request and collect all results into a Vec
    pub async fn collect_vec<T>(self) -> Result<Vec<T>, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_vec method".to_string()))
    }

    /// Execute request and collect the first result
    pub async fn collect_one<T>(self) -> Result<T, ModelInfoError>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        // Implementation will be added in the streaming execution milestone
        Err(ModelInfoError::NotYetImplemented("collect_one method".to_string()))
    }
}

/// Comprehensive error type for ModelInfo HTTP3 builders
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    /// HTTP request or response error
    #[error("HTTP error: {0}")]
    Http(#[from] HttpError),

    /// JSONPath parsing or evaluation error
    #[error("JSONPath error: {0}")]
    JsonPath(String),

    /// JSON deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// Network connectivity error
    #[error("Network error: {message}")]
    Network { message: String },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting error
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Provider API error
    #[error("Provider API error [{status}]: {message}")]
    ProviderApi { status: u16, message: String },

    /// No results found
    #[error("No results found for JSONPath selector: {selector}")]
    NoResults { selector: String },

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotYetImplemented(String),

    /// Generic error
    #[error("ModelInfo error: {0}")]
    Other(String),
}