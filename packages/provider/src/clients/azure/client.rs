// ============================================================================
// File: src/providers/azure_new/client.rs
// ----------------------------------------------------------------------------
// Azure OpenAI client with typestate builder pattern following Anthropic template
// ============================================================================

#![allow(clippy::type_complexity)]

use serde_json::json;
use bytes::Bytes;

use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    domain::message::Message,
    providers::azure::completion::{CompletionModel, GPT_4O},
    runtime::{self as rt, AsyncTask},
    http::{HttpClient, HttpRequest, HttpError},
};

use super::{embedding::EmbeddingModel, transcription::TranscriptionModel};

/// Azure OpenAI authentication
#[derive(Clone, Debug)]
pub enum AzureOpenAIAuth {
    ApiKey(String),
    Token(String),
}

impl From<String> for AzureOpenAIAuth {
    fn from(token: String) -> Self {
        AzureOpenAIAuth::Token(token)
    }
}

/// ------------------------------------------------------------
/// Typestate markers for builder pattern
/// ------------------------------------------------------------
pub struct NeedsPrompt;
pub struct HasPrompt;

/// ------------------------------------------------------------
/// Azure OpenAI Client
/// ------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Client {
    pub(crate) api_version: String,
    pub(crate) azure_endpoint: String,
    pub(crate) http_client: HttpClient,
    pub(crate) auth: AzureOpenAIAuth,
}

impl Client {
    /// Creates a new Azure OpenAI client.
    ///
    /// # Arguments
    ///
    /// * `auth` - Azure OpenAI API key or token required for authentication
    /// * `api_version` - API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
    /// * `azure_endpoint` - Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
    pub fn new(auth: impl Into<AzureOpenAIAuth>, api_version: &str, azure_endpoint: &str) -> Self {
        let http_client = HttpClient::for_provider("azure");
        let auth = auth.into();

        Self {
            api_version: api_version.to_string(),
            azure_endpoint: azure_endpoint.to_string(),
            http_client,
            auth,
        }
    }

    /// Creates a new Azure OpenAI client from an API key.
    pub fn from_api_key(api_key: &str, api_version: &str, azure_endpoint: &str) -> Self {
        Self::new(
            AzureOpenAIAuth::ApiKey(api_key.to_string()),
            api_version,
            azure_endpoint,
        )
    }

    /// Creates a new Azure OpenAI client from a token.
    pub fn from_token(token: &str, api_version: &str, azure_endpoint: &str) -> Self {
        Self::new(
            AzureOpenAIAuth::Token(token.to_string()),
            api_version,
            azure_endpoint,
        )
    }

    /// Get the authentication header for requests
    pub(crate) fn auth_header(&self) -> (&'static str, String) {
        match &self.auth {
            AzureOpenAIAuth::ApiKey(api_key) => ("api-key", api_key.clone()),
            AzureOpenAIAuth::Token(token) => ("Authorization", format!("Bearer {}", token)),
        }
    }

    pub(crate) fn post_embedding(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let (header_name, header_value) = self.auth_header();
        HttpRequest::post(url, Bytes::new())?.header(header_name, header_value)
    }

    pub(crate) fn post_chat_completion(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let (header_name, header_value) = self.auth_header();
        HttpRequest::post(url, Bytes::new())?.header(header_name, header_value)
    }

    pub(crate) fn post_transcription(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/audio/translations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let (header_name, header_value) = self.auth_header();
        HttpRequest::post(url, Bytes::new())?.header(header_name, header_value)
    }

    #[cfg(feature = "image")]
    pub(crate) fn post_image_generation(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/images/generations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let (header_name, header_value) = self.auth_header();
        HttpRequest::post(url, Bytes::new())?.header(header_name, header_value)
    }

    #[cfg(feature = "audio")]
    pub(crate) fn post_audio_generation(&self, deployment_id: &str) -> Result<HttpRequest, HttpError> {
        let url = format!(
            "{}/openai/deployments/{}/audio/speech?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        );
        let (header_name, header_value) = self.auth_header();
        HttpRequest::post(url, Bytes::new())?.header(header_name, header_value)
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
        crate::json_util::merge_inplace(&mut self.additional_params, p);
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
    pub fn from_env() -> Self {
        let auth = if let Ok(api_key) = std::env::var("AZURE_API_KEY") {
            AzureOpenAIAuth::ApiKey(api_key)
        } else if let Ok(token) = std::env::var("AZURE_TOKEN") {
            AzureOpenAIAuth::Token(token)
        } else {
            panic!("Neither AZURE_API_KEY nor AZURE_TOKEN is set");
        };

        let api_version = std::env::var("AZURE_API_VERSION").expect("AZURE_API_VERSION not set");
        let azure_endpoint = std::env::var("AZURE_ENDPOINT").expect("AZURE_ENDPOINT not set");

        Self::new(auth, &api_version, &azure_endpoint)
    }
}

/// ------------------------------------------------------------
/// Provider trait implementations
/// ------------------------------------------------------------
impl ProviderClient for Client {
    fn provider_name(&self) -> &'static str {
        "azure"
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
