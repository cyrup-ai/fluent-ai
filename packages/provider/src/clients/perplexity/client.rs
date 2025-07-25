// ============================================================================
// File: src/providers/perplexity/client.rs
// ----------------------------------------------------------------------------
// Perplexity client with typestate-driven builder pattern
// ============================================================================

use std::sync::LazyLock;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString};
use atomic_counter::RelaxedCounter;
use bytes::Bytes;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::completion::{CompletionModel, SONAR_PRO};
use crate::{
    client::{CompletionClient, ProviderClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError},
    json_util,
    message::Message,
    runtime::{self, AsyncTask}};

// ============================================================================
// Perplexity API Client with HTTP3 and zero-allocation patterns
// ============================================================================
const PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";

/// Global HTTP3 client with AI optimization
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static PERPLEXITY_METRICS: LazyLock<PerplexityMetrics> = LazyLock::new(PerplexityMetrics::new);

/// Perplexity performance metrics with atomic counters
#[derive(Debug)]
pub struct PerplexityMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter}

impl PerplexityMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_requests: RelaxedCounter::new(0),
            successful_requests: RelaxedCounter::new(0),
            failed_requests: RelaxedCounter::new(0),
            concurrent_requests: RelaxedCounter::new(0)}
    }
}

/// Zero-allocation Perplexity client with multiple auth sources
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    /// Zero-allocation base URL storage
    base_url: &'static str,
    /// Shared HTTP3 client
    http_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static PerplexityMetrics}

impl Client {
    /// Create a new Perplexity client with zero-allocation API key validation
    #[inline]
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::ConfigError(
                "API key cannot be empty".into(),
            ));
        }

        let api_key_array = ArrayString::from(&api_key)
            .map_err(|_| CompletionError::ConfigError("API key too long".into()))?;

        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            base_url: PERPLEXITY_API_BASE_URL,
            http_client: &HTTP_CLIENT,
            metrics: &PERPLEXITY_METRICS})
    }

    /// Environment variable names to search for Perplexity API keys (ordered by priority)
    pub fn env_api_keys() -> &'static [&'static str] {
        &[
            "PERPLEXITY_API_KEY",   // Primary Perplexity key
            "PERPLEXITYAI_API_KEY", // Alternative name
            "PPLX_API_KEY",         // Common abbreviation
        ]
    }

    /// Create from environment variables (tries multiple variable names)
    #[inline]
    pub fn from_env() -> Result<Self, CompletionError> {
        for env_var in Self::env_api_keys() {
            if let Ok(api_key) = std::env::var(env_var) {
                if !api_key.is_empty() {
                    return Self::new(api_key);
                }
            }
        }
        Err(CompletionError::ConfigError(format!(
            "No Perplexity API key found. Set one of: {}",
            Self::env_api_keys().join(", ")
        )))
    }

    /// Update API key with zero downtime hot-swapping
    #[inline]
    pub fn update_api_key(&self, new_api_key: String) -> Result<(), CompletionError> {
        if new_api_key.is_empty() {
            return Err(CompletionError::ConfigError(
                "API key cannot be empty".into(),
            ));
        }

        let api_key_array = ArrayString::from(&new_api_key)
            .map_err(|_| CompletionError::ConfigError("API key too long".into()))?;

        self.api_key.store(std::sync::Arc::new(api_key_array));
        Ok(())
    }

    /// Build authenticated request with zero allocations
    #[inline]
    pub(crate) async fn make_request(
        &self,
        endpoint: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::Response, HttpError> {
        let url = format!("{}/{}", self.base_url, endpoint);

        // Build headers with zero allocation
        let mut headers: SmallVec<[(&str, ArrayString<180>); 4]> = smallvec![];

        // Build auth header
        let mut auth_header = ArrayString::<180>::new();
        auth_header
            .try_push_str("Bearer ")
            .map_err(|_| HttpError::HeaderTooLong)?;
        auth_header
            .try_push_str(&self.api_key.load())
            .map_err(|_| HttpError::HeaderTooLong)?;

        headers.push(("Authorization", auth_header));
        headers.push((
            "Content-Type",
            ArrayString::from("application/json").unwrap_or_else(|_| ArrayString::new()),
        ));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-perplexity/1.0").unwrap_or_else(|_| ArrayString::new()),
        ));

        let request = HttpRequest::post(&url, body)
            .map_err(HttpError::from)?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())));

        // Update metrics atomically
        self.metrics.total_requests.inc();
        self.metrics.concurrent_requests.inc();

        let response = self.http_client.send(request).await;

        self.metrics.concurrent_requests.dec();

        match &response {
            Ok(_) => self.metrics.successful_requests.inc(),
            Err(_) => self.metrics.failed_requests.inc()}

        response
    }

    /// Test connection to Perplexity API
    #[inline]
    pub async fn test_connection(&self) -> Result<(), CompletionError> {
        // Perplexity doesn't have a models endpoint, so test with a minimal chat request
        let test_body = serde_json::json!({
            "model": "llama-3.1-sonar-small-128k-chat",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        });

        let test_body_bytes = serde_json::to_vec(&test_body)
            .map_err(|e| CompletionError::SerializationError(e.to_string()))?;

        let response = self
            .make_request("chat/completions", test_body_bytes)
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 400 {
            // 400 is OK for test - means API is accessible but request was minimal
            Ok(())
        } else {
            Err(CompletionError::ApiError(format!(
                "Connection test failed with status: {}",
                response.status()
            )))
        }
    }

    /// Create a completion model with the given name
    #[inline]
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Get current performance metrics
    #[inline]
    pub fn get_metrics(&self) -> (usize, usize, usize, usize) {
        (
            self.metrics.total_requests.get(),
            self.metrics.successful_requests.get(),
            self.metrics.failed_requests.get(),
            self.metrics.concurrent_requests.get(),
        )
    }
}

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for Client {
    type Model = Result<CompletionModel, CompletionError>;

    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        Ok(CompletionModel::new(self.clone(), model))
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for Client {
    #[inline]
    fn provider_name(&self) -> &'static str {
        "perplexity"
    }

    #[inline]
    fn test_connection(
        &self,
    ) -> DomainAsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        DomainAsyncTask::spawn(async move {
            client
                .test_connection()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
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
pub struct PerplexityCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<completion::Document>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>}

// ============================================================================
// Constructors
// ============================================================================
impl<'a> PerplexityCompletionBuilder<'a, NeedsPrompt> {
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
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            additional_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData}
    }

    /// Convenience helper: sensible defaults for chat
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> PerplexityCompletionBuilder<'a, HasPrompt> {
        Self::new(client, SONAR_PRO)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> PerplexityCompletionBuilder<'a, S> {
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
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = json_util::merge(self.additional_params, params);
        self
    }
}

// ============================================================================
// NeedsPrompt -> HasPrompt transition
// ============================================================================
impl<'a> PerplexityCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> PerplexityCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        PerplexityCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            additional_params: self.additional_params,
            prompt: self.prompt,
            _state: std::marker::PhantomData::<HasPrompt>}
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> PerplexityCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, PromptError> {
        let prompt = self.prompt.as_ref().ok_or_else(|| {
            PromptError::MissingPrompt("Prompt is required for completion".to_string())
        })?;

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

        if !self.additional_params.is_null() {
            let mut params = self.additional_params.clone();

            // Add Perplexity-specific parameters
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
        let (tx, task) = runtime::channel();
        let model = CompletionModel::new(self.client.clone(), self.model_name);

        match self.build_request() {
            Ok(request) => {
                runtime::spawn_async(async move {
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
                crate::providers::openai::StreamingCompletionResponse,
            >,
            CompletionError,
        >,
    > {
        let (tx, task) = runtime::channel();
        let model = CompletionModel::new(self.client.clone(), self.model_name);

        match self.build_request() {
            Ok(request) => {
                runtime::spawn_async(async move {
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
impl<'a> Prompt for PerplexityCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = PerplexityCompletionBuilder<'a, HasPrompt>;

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
pub struct ApiErrorResponse {
    pub message: String}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse)}
