// ============================================================================
// File: src/providers/ollama/client.rs
// ----------------------------------------------------------------------------
// Ollama client with typestate-driven builder pattern and HTTP3
// ============================================================================

use std::sync::LazyLock;
use std::time::Duration;
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpError;
use fluent_ai_http3::HttpRequest;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString};
use atomic_counter::RelaxedCounter;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::completion::{CompletionModel, EmbeddingModel, MISTRAL_MAGISTRAR_SMALL};
use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError},
    embeddings::{Embed, Embedding, EmbeddingBuilder},
    json_util,
    message::Message,
    runtime::{self, AsyncTask}};

// ============================================================================
// Ollama API Client with HTTP3 and zero-allocation patterns
// ============================================================================
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Global HTTP3 client with AI optimization
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static OLLAMA_METRICS: LazyLock<OllamaMetrics> = LazyLock::new(OllamaMetrics::new);

/// Ollama performance metrics with atomic counters
#[derive(Debug)]
pub struct OllamaMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter}

impl OllamaMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_requests: RelaxedCounter::new(0),
            successful_requests: RelaxedCounter::new(0),
            failed_requests: RelaxedCounter::new(0),
            concurrent_requests: RelaxedCounter::new(0)}
    }
}

/// Ollama error types for comprehensive error handling
#[derive(thiserror::Error, Debug)]
pub enum OllamaError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] HttpError),
    #[error("Configuration error in {field}: {message}")]
    Configuration {
        field: String,
        message: String,
        suggestion: String},
    #[error("Model not found: {model}")]
    ModelNotFound { model: String, suggestion: String },
    #[error("Connection failed: {message}")]
    Connection {
        message: String,
        retry_after: Option<Duration>}}

pub type Result<T> = std::result::Result<T, OllamaError>;

/// Zero-allocation Ollama client
#[derive(Clone)]
pub struct Client {
    /// Zero-allocation base URL storage
    base_url: ArrayString<256>,
    /// Shared HTTP3 client
    http_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static OllamaMetrics,
    /// Request timeout
    timeout: Duration}

impl Default for Client {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| panic!("Failed to create default Ollama client"))
    }
}

impl Client {
    /// Create a new Ollama client with zero-allocation patterns
    pub fn new() -> Result<Self> {
        Self::from_url(OLLAMA_API_BASE_URL)
    }

    /// Get the base URL for this client
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Create a new Ollama client with the given base URL
    pub fn from_url(base_url: &str) -> Result<Self> {
        let base_url_array =
            ArrayString::from(base_url).map_err(|_| OllamaError::Configuration {
                field: "base_url".to_string(),
                message: format!("Base URL too long: {} characters (max 256)", base_url.len()),
                suggestion: "Use a valid Ollama API base URL".to_string()})?;

        Ok(Self {
            base_url: base_url_array,
            http_client: &HTTP_CLIENT,
            metrics: &OLLAMA_METRICS,
            timeout: Duration::from_secs(30)})
    }

    /// Create from environment (Ollama defaults to localhost)
    pub fn from_env() -> Result<Self> {
        // Check for OLLAMA_HOST environment variable
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| OLLAMA_API_BASE_URL.to_string());
        Self::from_url(&base_url)
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
        headers.push((
            "Content-Type",
            ArrayString::from("application/json").unwrap_or_else(|_| ArrayString::new()),
        ));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-ollama/1.0").unwrap_or_else(|_| ArrayString::new()),
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
            Err(_) => self.metrics.failed_requests.inc()}

        response.map_err(OllamaError::Http)
    }

    /// Test connection to Ollama API
    pub async fn test_connection(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);

        let request = HttpRequest::get(&url)?
            .header("User-Agent", "fluent-ai-ollama/1.0")
            .timeout(Duration::from_secs(10));

        let response = self.http_client.send(request).await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(OllamaError::Connection {
                message: format!("Connection test failed with status: {}", response.status()),
                retry_after: None})
        }
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }

    /// Create an embedding model with specific dimensions.
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embeddings builder for the given model.
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingBuilder<EmbeddingModel, D> {
        EmbeddingBuilder::new(self.embedding_model(model))
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
pub struct OllamaCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    repeat_penalty: Option<f64>,
    seed: Option<i32>,
    num_predict: Option<i32>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<completion::Document>,
    tools: Vec<completion::ToolDefinition>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>}

// ============================================================================
// Constructors
// ============================================================================
impl<'a> OllamaCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            seed: None,
            num_predict: None,
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
    pub fn default_for_chat(client: &'a Client) -> OllamaCompletionBuilder<'a, HasPrompt> {
        Self::new(client, MISTRAL_MAGISTRAR_SMALL)
            .temperature(0.8)
            .num_predict(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> OllamaCompletionBuilder<'a, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
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
    pub fn repeat_penalty(mut self, penalty: f64) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn seed(mut self, s: i32) -> Self {
        self.seed = Some(s);
        self
    }

    #[inline(always)]
    pub fn num_predict(mut self, num: i32) -> Self {
        self.num_predict = Some(num);
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
impl<'a> OllamaCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> OllamaCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        OllamaCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
            seed: self.seed,
            num_predict: self.num_predict,
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
impl<'a> OllamaCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, PromptError> {
        let prompt = self
            .prompt
            .as_ref()
            .ok_or_else(|| PromptError::ValidationError("Prompt is required".to_string()))?;

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

            // Add Ollama-specific parameters
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(top_k) = self.top_k {
                params["top_k"] = json!(top_k);
            }
            if let Some(repeat_penalty) = self.repeat_penalty {
                params["repeat_penalty"] = json!(repeat_penalty);
            }
            if let Some(seed) = self.seed {
                params["seed"] = json!(seed);
            }
            if let Some(num_predict) = self.num_predict {
                params["num_predict"] = json!(num_predict);
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
                super::streaming::StreamingCompletionResponse,
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
impl<'a> Prompt for OllamaCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = OllamaCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

/// Debug implementation that doesn't expose sensitive information
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
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

/// EmbeddingsClient trait implementation
impl EmbeddingsClient for Client {
    type Model = EmbeddingModel;

    fn embedding_model(&self, model: &str) -> Self::Model {
        EmbeddingModel::new(self.clone(), model, 0)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::Model {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for Client {
    fn provider_name(&self) -> &'static str {
        "ollama"
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
