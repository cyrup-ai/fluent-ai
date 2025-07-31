// ============================================================================
// File: src/providers/together/client.rs
// ----------------------------------------------------------------------------
// Together AI client with typestate-driven builder pattern
// ============================================================================

use std::sync::LazyLock;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString};
use atomic_counter::RelaxedCounter;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_async::channel;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest, HttpResponse};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::{
    completion::{self, CompletionModel, LLAMA_3_2_11B_VISION_INSTRUCT_TURBO},
    embedding::{EmbeddingModel, M2_BERT_80M_8K_RETRIEVAL}};
use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient},
    completion_provider::{CompletionError, CompletionProvider}};
use fluent_ai_domain::{
    completion::{CompletionRequest, CompletionRequestBuilder, CompletionRequestError, types::ToolDefinition},
    context::document::Document,
    chat::message::Message,
    util::json_util};

// ============================================================================
// Together AI API Client with HTTP3 and dual-endpoint optimization
// ============================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

/// Global HTTP3 clients optimized for different endpoints
static CHAT_HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

static EMBED_HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::embedding_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static TOGETHER_METRICS: LazyLock<TogetherMetrics> = LazyLock::new(TogetherMetrics::new);

/// Together AI performance metrics with atomic counters
#[derive(Debug)]
pub struct TogetherMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter,
    pub chat_requests: RelaxedCounter,
    pub embedding_requests: RelaxedCounter}

impl TogetherMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_requests: RelaxedCounter::new(0),
            successful_requests: RelaxedCounter::new(0),
            failed_requests: RelaxedCounter::new(0),
            concurrent_requests: RelaxedCounter::new(0),
            chat_requests: RelaxedCounter::new(0),
            embedding_requests: RelaxedCounter::new(0)}
    }
}

/// Zero-allocation Together AI client with dual endpoints
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    /// Zero-allocation base URL storage
    base_url: &'static str,
    /// Specialized HTTP clients
    chat_client: &'static HttpClient,
    embed_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static TogetherMetrics}

impl Client {
    /// Create a new Together AI client with zero-allocation API key validation
    #[inline]
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::InvalidRequest(
                "API key cannot be empty".into(),
            ));
        }

        let api_key_array = ArrayString::from(&api_key)
            .map_err(|_| CompletionError::InvalidRequest("API key too long".into()))?;

        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            base_url: TOGETHER_AI_BASE_URL,
            chat_client: &CHAT_HTTP_CLIENT,
            embed_client: &EMBED_HTTP_CLIENT,
            metrics: &TOGETHER_METRICS})
    }

    /// Environment variable names to search for Together API keys (ordered by priority)
    pub fn env_api_keys() -> &'static [&'static str] {
        &[
            "TOGETHER_API_KEY",    // Primary Together key
            "TOGETHERAI_API_KEY",  // Alternative name
            "TOGETHER_AI_API_KEY", // Full name variation
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
        Err(CompletionError::InvalidRequest(format!(
            "No Together API key found. Set one of: {}",
            Self::env_api_keys().join(", ")
        )))
    }

    /// Update API key with zero downtime hot-swapping
    #[inline]
    pub fn update_api_key(&self, new_api_key: String) -> Result<(), CompletionError> {
        if new_api_key.is_empty() {
            return Err(CompletionError::InvalidRequest(
                "API key cannot be empty".into(),
            ));
        }

        let api_key_array = ArrayString::from(&new_api_key)
            .map_err(|_| CompletionError::InvalidRequest("API key too long".into()))?;

        self.api_key.store(std::sync::Arc::new(api_key_array));
        Ok(())
    }

    /// Build authenticated chat request with optimized client
    #[inline]
    pub(crate) async fn chat_request(
        &self,
        endpoint: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::HttpResponse, HttpError> {
        let url = format!("{}/{}", self.base_url, endpoint);

        let response = self
            .make_authenticated_request(&self.chat_client, &url, body)
            .await?;
        self.metrics.chat_requests.inc();
        Ok(response)
    }

    /// Build authenticated embedding request with optimized client
    #[inline]
    pub(crate) async fn embedding_request(
        &self,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::HttpResponse, HttpError> {
        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .make_authenticated_request(&self.embed_client, &url, body)
            .await?;
        self.metrics.embedding_requests.inc();
        Ok(response)
    }

    /// Build authenticated request with zero allocations
    #[inline]
    async fn make_authenticated_request(
        &self,
        client: &HttpClient,
        url: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::HttpResponse, HttpError> {
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
            ArrayString::from("fluent-ai-together/1.0").unwrap_or_else(|_| ArrayString::new()),
        ));

        let request = HttpRequest::post(url, body)
            .map_err(HttpError::from)?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())));

        // Update metrics atomically
        self.metrics.total_requests.inc();
        self.metrics.concurrent_requests.inc();

        let response = client.send(request).await;

        self.metrics.concurrent_requests.dec();

        match &response {
            Ok(_) => self.metrics.successful_requests.inc(),
            Err(_) => self.metrics.failed_requests.inc()}

        response
    }

    /// Test connection to Together AI API
    #[inline]
    pub async fn test_connection(&self) -> Result<(), CompletionError> {
        let response = self
            .chat_request("models", vec![])
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if response.status().is_success() {
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

    /// Create an embedding model with the given name
    #[inline]
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            M2_BERT_80M_8K_RETRIEVAL => 8192,
            _ => 0};
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with specific dimensions
    #[inline]
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Get current performance metrics
    #[inline]
    pub fn get_metrics(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.metrics.total_requests.get(),
            self.metrics.successful_requests.get(),
            self.metrics.failed_requests.get(),
            self.metrics.concurrent_requests.get(),
            self.metrics.chat_requests.get(),
            self.metrics.embedding_requests.get(),
        )
    }
}

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for Client {
    type Model = Result<completion::TogetherCompletionModel, CompletionError>;

    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        Ok(completion::TogetherCompletionModel::new(
            self.clone(),
            model,
        ))
    }
}

/// EmbeddingsClient trait implementation for auto-generation
impl EmbeddingsClient for Client {
    type Model = Result<EmbeddingModel, CompletionError>;

    #[inline]
    fn embedding_model(&self, model: &str) -> Self::Model {
        let ndims = match model {
            M2_BERT_80M_8K_RETRIEVAL => 8192,
            _ => 0};
        Ok(EmbeddingModel::new(self.clone(), model, ndims))
    }

    #[inline]
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::Model {
        Ok(EmbeddingModel::new(self.clone(), model, ndims))
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for Client {
    #[inline]
    fn provider_name(&self) -> &'static str {
        "together"
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
pub struct TogetherCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    repetition_penalty: Option<f64>,
    stop: Option<Vec<String>>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<fluent_ai_domain::context::Document>,
    tools: Vec<fluent_ai_domain::completion::ToolDefinition>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>}

// ============================================================================
// Constructors
// ============================================================================
impl<'a> TogetherCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            stop: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            additional_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData}
    }

    /// Convenience helper: sensible defaults for chat
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> TogetherCompletionBuilder<'a, HasPrompt> {
        Self::new(client, LLAMA_3_2_11B_VISION_INSTRUCT_TURBO)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> TogetherCompletionBuilder<'a, S> {
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
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    #[inline(always)]
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty);
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
    pub fn documents(mut self, docs: Vec<fluent_ai_domain::context::Document>) -> Self {
        self.documents = docs;
        self
    }

    #[inline(always)]
    pub fn tools(mut self, tools: Vec<fluent_ai_domain::completion::ToolDefinition>) -> Self {
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
impl<'a> TogetherCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> TogetherCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        TogetherCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            stop: self.stop,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            additional_params: self.additional_params,
            prompt: self.prompt,
            _state: std::marker::PhantomData::<HasPrompt>}
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> TogetherCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, CompletionRequestError> {
        let prompt = self
            .prompt
            .as_ref()
            .ok_or_else(|| CompletionRequestError::InvalidParameter("Prompt is required".to_string()))?;

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

            // Add Together-specific parameters
            if let Some(max_tokens) = self.max_tokens {
                params["max_tokens"] = json!(max_tokens);
            }
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(top_k) = self.top_k {
                params["top_k"] = json!(top_k);
            }
            if let Some(repetition_penalty) = self.repetition_penalty {
                params["repetition_penalty"] = json!(repetition_penalty);
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
            fluent_ai_domain::completion::CompletionResponse<super::completion::CompletionResponse>,
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
            crate::streaming::StreamingCompletionResponse<super::completion::CompletionResponse>,
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
// Prompt trait definition for builder pattern
// ============================================================================
pub trait Prompt {
    type PromptedBuilder;
    
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, CompletionRequestError>;
}

// ============================================================================
// Prompt trait implementation
// ============================================================================
impl<'a> Prompt for TogetherCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = TogetherCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, CompletionRequestError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

// ============================================================================
// API types for compatibility
// ============================================================================
pub mod together_ai_api_types {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: String,
        pub code: String}

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            format!("Code `{}`: {}", self.code, self.error)
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse)}
}
