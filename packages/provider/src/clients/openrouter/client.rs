// ============================================================================
// File: src/providers/openrouter/client.rs
// ----------------------------------------------------------------------------
// OpenRouter client with typestate-driven builder pattern and HTTP3
// ============================================================================

use std::sync::LazyLock;
use std::time::Duration;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString, ArrayVec};
use atomic_counter::RelaxedCounter;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::completion::{CompletionModel, GPT_4_1};
use crate::{
    client::{CompletionClient, ProviderClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    json_util,
    message::Message,
};
use fluent_ai_domain::{AsyncTask, spawn_async, channel};

// ============================================================================
// OpenRouter API Client with HTTP3 and zero-allocation patterns
// ============================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Global HTTP3 client with AI optimization
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static OPENROUTER_METRICS: LazyLock<OpenRouterMetrics> = LazyLock::new(OpenRouterMetrics::new);

/// OpenRouter performance metrics with atomic counters
#[derive(Debug)]
pub struct OpenRouterMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter,
}

impl OpenRouterMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_requests: RelaxedCounter::new(0),
            successful_requests: RelaxedCounter::new(0),
            failed_requests: RelaxedCounter::new(0),
            concurrent_requests: RelaxedCounter::new(0),
        }
    }
}

/// OpenRouter error types for comprehensive error handling
#[derive(thiserror::Error, Debug)]
pub enum OpenRouterError {
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
}

pub type Result<T> = std::result::Result<T, OpenRouterError>;

/// Zero-allocation OpenRouter client
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    /// Zero-allocation base URL storage
    base_url: ArrayString<256>,
    /// Shared HTTP3 client
    http_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static OpenRouterMetrics,
    /// Request timeout
    timeout: Duration,
}

impl Client {
    /// Create a new OpenRouter client with zero-allocation API key validation
    pub fn new(api_key: &str) -> Result<Self> {
        Self::from_url(api_key, OPENROUTER_API_BASE_URL)
    }

    /// Create a new OpenRouter client with the given API key and base URL
    pub fn from_url(api_key: &str, base_url: &str) -> Result<Self> {
        if api_key.is_empty() {
            return Err(OpenRouterError::Configuration {
                field: "api_key".to_string(),
                message: "API key cannot be empty".to_string(),
                suggestion: "Provide a valid OpenRouter API key".to_string(),
            });
        }

        let api_key_array =
            ArrayString::from(api_key).map_err(|_| OpenRouterError::Configuration {
                field: "api_key".to_string(),
                message: format!("API key too long: {} characters (max 128)", api_key.len()),
                suggestion: "Use a valid OpenRouter API key".to_string(),
            })?;

        let base_url_array =
            ArrayString::from(base_url).map_err(|_| OpenRouterError::Configuration {
                field: "base_url".to_string(),
                message: format!("Base URL too long: {} characters (max 256)", base_url.len()),
                suggestion: "Use a valid OpenRouter API base URL".to_string(),
            })?;

        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            base_url: base_url_array,
            http_client: &HTTP_CLIENT,
            metrics: &OPENROUTER_METRICS,
            timeout: Duration::from_secs(30),
        })
    }

    /// Create from environment (OPENROUTER_API_KEY)
    pub fn from_env() -> Result<Self> {
        let api_key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| OpenRouterError::Configuration {
                field: "OPENROUTER_API_KEY".to_string(),
                message: "OPENROUTER_API_KEY environment variable not set".to_string(),
                suggestion:
                    "Set OPENROUTER_API_KEY environment variable with your OpenRouter API key"
                        .to_string(),
            })?;
        Self::new(&api_key)
    }

    /// Build authenticated request with zero allocations
    pub async fn make_request(
        &self,
        path: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::Response> {
        let url = format!("{}/{}", self.base_url, path);

        // Build headers with zero allocation
        let mut headers: SmallVec<[(&str, ArrayString<180>); 4]> = smallvec![];

        // Build auth header
        let mut auth_header = ArrayString::<180>::new();
        auth_header
            .try_push_str("Bearer ")
            .map_err(|_| OpenRouterError::Configuration {
                field: "auth_header".to_string(),
                message: "Failed to build auth header".to_string(),
                suggestion: "Check API key length".to_string(),
            })?;
        auth_header
            .try_push_str(&self.api_key.load())
            .map_err(|_| OpenRouterError::Configuration {
                field: "auth_header".to_string(),
                message: "Failed to build auth header".to_string(),
                suggestion: "Check API key length".to_string(),
            })?;

        headers.push(("Authorization", auth_header));
        headers.push((
            "Content-Type",
            ArrayString::from("application/json").unwrap(),
        ));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-openrouter/1.0").unwrap(),
        ));

        let request = HttpRequest::post(&url, body)?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(self.timeout);

        // Update metrics atomically
        self.metrics.total_requests.inc();
        self.metrics.concurrent_requests.inc();

        let response = self.http_client.send(request).await;

        self.metrics.concurrent_requests.dec();

        match &response {
            Ok(_) => self.metrics.successful_requests.inc(),
            Err(_) => self.metrics.failed_requests.inc(),
        }

        response.map_err(OpenRouterError::Http)
    }

    /// Test connection to OpenRouter API
    pub async fn test_connection(&self) -> Result<()> {
        let url = format!("{}/models", self.base_url);

        let mut auth_header = ArrayString::<180>::new();
        let _ = auth_header.try_push_str("Bearer ");
        let _ = auth_header.try_push_str(&self.api_key.load());

        let request = HttpRequest::get(&url)?
            .header("Authorization", auth_header.as_str())
            .header("User-Agent", "fluent-ai-openrouter/1.0")
            .timeout(Duration::from_secs(10));

        let response = self.http_client.send(request).await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(OpenRouterError::Authentication {
                message: format!("Connection test failed with status: {}", response.status()),
                retry_after: None,
            })
        }
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> (usize, usize, usize, usize) {
        (
            self.metrics.total_requests.get(),
            self.metrics.successful_requests.get(),
            self.metrics.failed_requests.get(),
            self.metrics.concurrent_requests.get(),
        )
    }
}

// ============================================================================
// Typestate markers
// ============================================================================
pub struct NeedsPrompt;
pub struct HasPrompt;

// ============================================================================
// Core builder (generic over typestate `S`)
// ============================================================================
pub struct OpenRouterCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    stop: Option<Vec<String>>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<completion::Document>,
    tools: Vec<completion::ToolDefinition>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>,
}

// ============================================================================
// Constructors
// ============================================================================
impl<'a> OpenRouterCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            additional_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Convenience helper: sensible defaults for chat
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> OpenRouterCompletionBuilder<'a, HasPrompt> {
        Self::new(client, GPT_4_1)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> OpenRouterCompletionBuilder<'a, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    #[inline(always)]
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    #[inline(always)]
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.stop = Some(sequences);
        self
    }

    #[inline(always)]
    pub fn preamble(mut self, p: impl ToString) -> Self {
        self.preamble = Some(p.to_string());
        self
    }

    #[inline(always)]
    pub fn chat_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = history;
        self
    }

    #[inline(always)]
    pub fn documents(mut self, docs: Vec<completion::Document>) -> Self {
        self.documents = docs;
        self
    }

    #[inline(always)]
    pub fn tools(mut self, tools: Vec<completion::ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    #[inline(always)]
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = json_util::merge(self.additional_params, params);
        self
    }
}

// ============================================================================
// NeedsPrompt -> HasPrompt transition
// ============================================================================
impl<'a> OpenRouterCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> OpenRouterCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        OpenRouterCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop: self.stop,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            additional_params: self.additional_params,
            prompt: self.prompt,
            _state: std::marker::PhantomData::<HasPrompt>,
        }
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> OpenRouterCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, PromptError> {
        let prompt = self.prompt.as_ref().ok_or_else(|| PromptError::MissingPrompt)?;

        let mut builder =
            CompletionRequestBuilder::new(self.model_name.to_string(), prompt.clone())?;

        if let Some(temp) = self.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(ref preamble) = self.preamble {
            builder = builder.preamble(preamble);
        }

        if !self.chat_history.is_empty() {
            builder = builder.chat_history(self.chat_history.clone());
        }

        if !self.documents.is_empty() {
            builder = builder.documents(self.documents.clone());
        }

        if !self.tools.is_empty() {
            builder = builder.tools(self.tools.clone());
        }

        if !self.additional_params.is_null() {
            let mut params = self.additional_params.clone();

            // Add OpenRouter-specific parameters
            if let Some(max_tokens) = self.max_tokens {
                params["max_tokens"] = json!(max_tokens);
            }
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(frequency_penalty) = self.frequency_penalty {
                params["frequency_penalty"] = json!(frequency_penalty);
            }
            if let Some(presence_penalty) = self.presence_penalty {
                params["presence_penalty"] = json!(presence_penalty);
            }
            if let Some(ref stop) = self.stop {
                params["stop"] = json!(stop);
            }

            builder = builder.additional_params(params);
        }

        Ok(builder.build())
    }

    /// Execute the completion request
    pub fn execute(
        self,
    ) -> AsyncTask<
        Result<
            completion::CompletionResponse<super::completion::CompletionResponse>,
            CompletionError,
        >,
    > {
        let (tx, task) = channel();
        let model = CompletionModel::new(self.client.clone(), self.model_name);

        match self.build_request() {
            Ok(request) => {
                spawn_async(async move {
                    let result = model.completion(request).await;
                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(e.into()));
            }
        }

        task
    }

    /// Stream the completion response
    pub fn stream(
        self,
    ) -> AsyncTask<
        Result<
            crate::streaming::StreamingCompletionResponse<
                super::streaming::FinalCompletionResponse,
            >,
            CompletionError,
        >,
    > {
        let (tx, task) = channel();
        let model = CompletionModel::new(self.client.clone(), self.model_name);

        match self.build_request() {
            Ok(request) => {
                spawn_async(async move {
                    let result = model.stream(request).await;
                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(e.into()));
            }
        }

        task
    }
}

// ============================================================================
// Prompt trait implementation
// ============================================================================
impl<'a> Prompt for OpenRouterCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = OpenRouterCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

// ============================================================================
// API types for compatibility
// ============================================================================
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

/// Debug implementation that doesn't expose sensitive information
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for Client {
    type Model = CompletionModel;

    fn completion_model(&self, model: &str) -> Self::Model {
        CompletionModel::new(self.clone(), model)
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for Client {
    fn provider_name(&self) -> &'static str {
        "openrouter"
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
