// ============================================================================
// File: src/providers/azure_new/client.rs
// ----------------------------------------------------------------------------
// Azure OpenAI client with typestate builder pattern following Anthropic template
// ============================================================================

#![allow(clippy::type_complexity)]

use std::time::Duration;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString, ArrayVec};
use bytes::Bytes;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

// Removed invalid imports - EmbeddingModel and TranscriptionModel are not exported from submodules
use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient},
    clients::azure::completion::{CompletionModel, GPT_4O},
};
use fluent_ai_domain::completion::{self, CompletionRequest, CompletionRequestBuilder};
use fluent_ai_domain::message::Message;
use crate::completion_provider::{CompletionError, CompletionProvider};

/// Azure OpenAI authentication with zero-allocation patterns
#[derive(Clone, Debug)]
pub enum AzureOpenAIAuth {
    ApiKey(ArrayString<128>),
    Token(ArrayString<256>),
}

impl AzureOpenAIAuth {
    /// Create API key authentication with zero allocation validation
    pub fn api_key(key: impl AsRef<str>) -> Result<Self, AzureError> {
        let key_str = key.as_ref();
        if key_str.is_empty() {
            return Err(AzureError::Configuration {
                field: "api_key".to_string(),
                message: "API key cannot be empty".to_string(),
                suggestion: "Provide a valid Azure OpenAI API key".to_string(),
            });
        }

        ArrayString::from(key_str)
            .map(Self::ApiKey)
            .map_err(|_| AzureError::Configuration {
                field: "api_key".to_string(),
                message: format!("API key too long: {} characters (max 128)", key_str.len()),
                suggestion: "Use a valid Azure OpenAI API key".to_string(),
            })
    }

    /// Create token authentication with zero allocation validation
    pub fn token(token: impl AsRef<str>) -> Result<Self, AzureError> {
        let token_str = token.as_ref();
        if token_str.is_empty() {
            return Err(AzureError::Configuration {
                field: "token".to_string(),
                message: "Token cannot be empty".to_string(),
                suggestion: "Provide a valid Azure OpenAI token".to_string(),
            });
        }

        ArrayString::from(token_str)
            .map(Self::Token)
            .map_err(|_| AzureError::Configuration {
                field: "token".to_string(),
                message: format!("Token too long: {} characters (max 256)", token_str.len()),
                suggestion: "Use a valid Azure OpenAI token".to_string(),
            })
    }
}

impl From<String> for AzureOpenAIAuth {
    fn from(token: String) -> Self {
        Self::token(token).unwrap_or_else(|_| {
            // Fallback for compatibility - truncate if too long
            if token.len() <= 256 {
                Self::Token(ArrayString::from(&token).unwrap_or_default())
            } else {
                Self::Token(ArrayString::from(&token[..256]).unwrap_or_default())
            }
        })
    }
}

/// Azure OpenAI error types for comprehensive error handling
#[derive(thiserror::Error, Debug)]
pub enum AzureError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] HttpError),
    #[error("Configuration error in {field}: {message}")]
    Configuration {
        field: String,
        message: String,
        suggestion: String,
    },
    #[error("Authentication failed: {message}")]
    Authentication {
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("Model not supported: {model}")]
    ModelNotSupported {
        model: String,
        supported_models: Vec<String>,
    },
    #[error("Deployment not found: {deployment_id}")]
    DeploymentNotFound {
        deployment_id: String,
        suggestion: String,
    },
}

pub type Result<T> = std::result::Result<T, AzureError>;

/// ------------------------------------------------------------
/// Typestate markers for builder pattern
/// ------------------------------------------------------------
pub struct NeedsPrompt;
pub struct HasPrompt;

/// ------------------------------------------------------------
/// Azure OpenAI Client with zero-allocation HTTP3 architecture
/// ------------------------------------------------------------
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable authentication
    pub(crate) auth: ArcSwap<AzureOpenAIAuth>,
    /// Zero-allocation API version storage
    pub(crate) api_version: ArrayString<32>,
    /// Zero-allocation endpoint storage
    pub(crate) azure_endpoint: ArrayString<256>,
    /// Shared HTTP3 client with connection pooling
    pub(crate) http_client: HttpClient,
    /// Request timeout
    timeout: Duration,
}

impl Client {
    /// Creates a new Azure OpenAI client with HTTP3 and zero-allocation patterns.
    ///
    /// # Arguments
    ///
    /// * `auth` - Azure OpenAI API key or token required for authentication
    /// * `api_version` - API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
    /// * `azure_endpoint` - Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
    pub fn new(
        auth: impl Into<AzureOpenAIAuth>,
        api_version: &str,
        azure_endpoint: &str,
    ) -> Result<Self> {
        let http_client = HttpClient::with_config(HttpConfig::ai_optimized()).map_err(|e| {
            AzureError::Configuration {
                field: "http_client".to_string(),
                message: format!("Failed to create HTTP3 client: {}", e),
                suggestion: "Check network configuration and try again".to_string(),
            }
        })?;

        let auth = auth.into();

        let api_version_array =
            ArrayString::from(api_version).map_err(|_| AzureError::Configuration {
                field: "api_version".to_string(),
                message: format!(
                    "API version too long: {} characters (max 32)",
                    api_version.len()
                ),
                suggestion: "Use a valid Azure API version string".to_string(),
            })?;

        let endpoint_array =
            ArrayString::from(azure_endpoint).map_err(|_| AzureError::Configuration {
                field: "azure_endpoint".to_string(),
                message: format!(
                    "Endpoint URL too long: {} characters (max 256)",
                    azure_endpoint.len()
                ),
                suggestion: "Use a valid Azure OpenAI endpoint URL".to_string(),
            })?;

        Ok(Self {
            auth: ArcSwap::from_pointee(auth),
            api_version: api_version_array,
            azure_endpoint: endpoint_array,
            http_client,
            timeout: Duration::from_secs(30),
        })
    }

    /// Creates a new Azure OpenAI client from an API key.
    pub fn from_api_key(api_key: &str, api_version: &str, azure_endpoint: &str) -> Result<Self> {
        let auth = AzureOpenAIAuth::api_key(api_key)?;
        Self::new(auth, api_version, azure_endpoint)
    }

    /// Creates a new Azure OpenAI client from a token.
    pub fn from_token(token: &str, api_version: &str, azure_endpoint: &str) -> Result<Self> {
        let auth = AzureOpenAIAuth::token(token)?;
        Self::new(auth, api_version, azure_endpoint)
    }

    /// Get the authentication header for requests with zero-allocation patterns
    pub(crate) fn auth_header(&self) -> (&'static str, ArrayString<300>) {
        let auth = self.auth.load();
        match auth.as_ref() {
            AzureOpenAIAuth::ApiKey(api_key) => (
                "api-key",
                ArrayString::from(api_key.as_str()).unwrap_or_default(),
            ),
            AzureOpenAIAuth::Token(token) => {
                let mut bearer_token = ArrayString::<300>::new();
                let _ = bearer_token.try_push_str("Bearer ");
                let _ = bearer_token.try_push_str(token.as_str());
                ("Authorization", bearer_token)
            }
        }
    }

    /// Build optimized headers with zero allocation
    #[inline(always)]
    fn build_headers(&self) -> SmallVec<[(&'static str, ArrayString<300>); 4]> {
        let (auth_header_name, auth_header_value) = self.auth_header();

        smallvec![
            (auth_header_name, auth_header_value),
            (
                "Content-Type",
                ArrayString::from("application/json").unwrap_or_default()
            ),
            (
                "User-Agent",
                ArrayString::from("fluent-ai-http3/1.0").unwrap_or_default()
            ),
        ]
    }

    pub(crate) fn post_embedding(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let headers = self.build_headers();
        let mut request = HttpRequest::post(url, Vec::new())?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        Ok(request.timeout(self.timeout))
    }

    pub(crate) fn post_chat_completion(
        &self,
        deployment_id: &str,
    ) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let headers = self.build_headers();
        let mut request = HttpRequest::post(url, Vec::new())?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        Ok(request.timeout(self.timeout))
    }

    pub(crate) fn post_transcription(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/audio/translations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let headers = self.build_headers();
        let mut request = HttpRequest::post(url, Vec::new())?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        Ok(request.timeout(self.timeout))
    }

    #[cfg(feature = "image")]
    pub(crate) fn post_image_generation(
        &self,
        deployment_id: &str,
    ) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/images/generations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let headers = self.build_headers();
        let mut request = HttpRequest::post(url, Vec::new())?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        Ok(request.timeout(self.timeout))
    }

    #[cfg(feature = "audio")]
    pub(crate) fn post_audio_generation(
        &self,
        deployment_id: &str,
    ) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/audio/speech?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let headers = self.build_headers();
        let mut request = HttpRequest::post(url, Vec::new())?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        Ok(request.timeout(self.timeout))
    }

    /// Test connection to Azure OpenAI service
    pub async fn test_connection(&self) -> Result<()> {
        let url = format!(
            "{}/openai/models?api-version={}",
            self.azure_endpoint, self.api_version
        );

        let headers = self.build_headers();
        let mut request = HttpRequest::get(&url)?;

        for (name, value) in headers.iter() {
            request = request.header(*name, value.as_str());
        }

        let request = request.timeout(Duration::from_secs(10));
        let response = self.http_client.send(request).await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(AzureError::Authentication {
                message: format!("Connection test failed with status: {}", response.status()),
                retry_after: None,
            })
        }
    }
}

/// ------------------------------------------------------------
/// Core builder struct with typestate
/// ------------------------------------------------------------
pub struct AzureCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<completion::Document>,
    tools: Vec<completion::ToolDefinition>,
    max_tokens: Option<u64>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>,
}

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------
impl<'a> AzureCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            max_tokens: None,
            additional_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Convenience helper: all defaults for Azure OpenAI chat.
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> AzureCompletionBuilder<'a, HasPrompt> {
        Self::new(client, GPT_4O)
            .temperature(0.0)
            .max_tokens(4096)
            .prompt(Message::user("")) // dummy; will be replaced in `.chat(..)`
    }
}

// ------------------------------------------------------------
// Fluent setters that keep the typestate
// ------------------------------------------------------------
impl<'a, S> AzureCompletionBuilder<'a, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    #[inline(always)]
    pub fn max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = Some(n);
        self
    }

    #[inline(always)]
    pub fn tool<T: completion::ToolDefinitionInto>(mut self, tool: T) -> Self {
        self.tools.push(tool.into_tool_def());
        self
    }

    #[inline(always)]
    pub fn additional_params(mut self, p: serde_json::Value) -> Self {
        // Merge JSON values in-place
        if let (serde_json::Value::Object(ref mut base), serde_json::Value::Object(other)) = 
            (&mut self.additional_params, p) {
            base.extend(other);
        } else {
            self.additional_params = p;
        }
        self
    }
}

// ------------------------------------------------------------
// Transition from NeedsPrompt → HasPrompt
// ------------------------------------------------------------
impl<'a> AzureCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, p: impl Into<Message>) -> AzureCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(p.into());
        AzureCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            prompt: self.prompt,
            _state: std::marker::PhantomData,
        }
    }
}

// ------------------------------------------------------------
// Final stage — only available once we *have* the prompt
// ------------------------------------------------------------
impl<'a> AzureCompletionBuilder<'a, HasPrompt> {
    /// Build a `CompletionRequest` ready for `.send()` / `.stream()`.
    #[inline(always)]
    fn build(self) -> CompletionRequest {
        let mut req = CompletionRequestBuilder::new(
            CompletionModel::new(self.client.clone(), self.model_name),
            self.prompt.expect("prompt present"),
        )
        .preamble_opt(self.preamble)
        .chat_history(self.chat_history)
        .documents(self.documents)
        .tools(self.tools)
        .temperature_opt(self.temperature)
        .max_tokens_opt(self.max_tokens)
        .additional_params(self.additional_params);

        req.build()
    }

    /// Fire off a **single-shot** completion → `AsyncTask<Result<…>>`
    #[inline(always)]
    pub fn send(
        self,
    ) -> AsyncTask<
        Result<completion::CompletionResponse<completion::CompletionResponseData>, CompletionError>,
    > {
        self.build().send()
    }

    /// Streaming variant → `AsyncTask<Result<StreamingCompletionResponse<…>>>`
    #[inline(always)]
    pub fn stream(
        self,
    ) -> AsyncTask<
        Result<
            crate::streaming::StreamingCompletionResponse<
                super::streaming::StreamingCompletionResponse,
            >,
            CompletionError,
        >,
    > {
        self.build().stream()
    }

    /// Convenience: run chat and get the **first chunk** as `String`.
    #[inline(always)]
    pub fn chat(self, prompt: impl Into<String>) -> AsyncTask<Result<String, PromptError>> {
        let req = self.prompt(prompt.into()).build();
        CompletionModel::new(self.client.clone(), self.model_name).prompt(req)
    }
}

/// ------------------------------------------------------------
/// Convenience type alias for the builder
/// ------------------------------------------------------------
pub type ClientBuilder<'a> = AzureCompletionBuilder<'a, NeedsPrompt>;

/// Create a new Azure OpenAI client from environment variables.
impl Client {
    pub fn from_env() -> Result<Self> {
        let auth = if let Ok(api_key) = std::env::var("AZURE_API_KEY") {
            AzureOpenAIAuth::api_key(api_key)?
        } else if let Ok(token) = std::env::var("AZURE_TOKEN") {
            AzureOpenAIAuth::token(token)?
        } else {
            return Err(AzureError::Configuration {
                field: "authentication".to_string(),
                message: "Neither AZURE_API_KEY nor AZURE_TOKEN environment variable is set"
                    .to_string(),
                suggestion: "Set either AZURE_API_KEY or AZURE_TOKEN environment variable"
                    .to_string(),
            });
        };

        let api_version =
            std::env::var("AZURE_API_VERSION").map_err(|_| AzureError::Configuration {
                field: "AZURE_API_VERSION".to_string(),
                message: "AZURE_API_VERSION environment variable not set".to_string(),
                suggestion:
                    "Set AZURE_API_VERSION to your desired API version (e.g., '2024-10-21')"
                        .to_string(),
            })?;

        let azure_endpoint =
            std::env::var("AZURE_ENDPOINT").map_err(|_| AzureError::Configuration {
                field: "AZURE_ENDPOINT".to_string(),
                message: "AZURE_ENDPOINT environment variable not set".to_string(),
                suggestion: "Set AZURE_ENDPOINT to your Azure OpenAI endpoint URL".to_string(),
            })?;

        Self::new(auth, &api_version, &azure_endpoint)
    }
}

/// Debug implementation that doesn't expose sensitive information
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("auth", &"[REDACTED]")
            .field("api_version", &self.api_version)
            .field("azure_endpoint", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// ------------------------------------------------------------
/// Provider trait implementations
/// ------------------------------------------------------------
impl ProviderClient for Client {
    fn provider_name(&self) -> &'static str {
        "azure"
    }

    fn test_connection(
        &self,
    ) -> DomainAsyncTask<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        DomainAsyncTask::spawn(async move {
            client
                .test_connection()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }
}

impl CompletionClient for Client {
    type Model = CompletionModel;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let gpt4 = azure.completion_model(azure::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type Model = EmbeddingModel;

    /// Create an embedding model with the given name.
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            super::embedding::TEXT_EMBEDDING_3_LARGE => 3072,
            super::embedding::TEXT_EMBEDDING_3_SMALL | super::embedding::TEXT_EMBEDDING_ADA_002 => {
                1536
            }
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with the given name and number of dimensions.
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl TranscriptionClient for Client {
    type Model = TranscriptionModel;

    /// Create a transcription model with the given name.
    fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}
