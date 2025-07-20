// ============================================================================
// File: src/providers/gemini/client.rs
// ----------------------------------------------------------------------------
// Gemini client with typestate-driven builder pattern
// ============================================================================

use std::sync::LazyLock;
use std::time::Duration;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString, ArrayVec};
use atomic_counter::RelaxedCounter;
use bytes::Bytes;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::completion::{CompletionModel, GEMINI_1_5_PRO};
use super::embedding::EmbeddingModel;
use super::transcription::TranscriptionModel;
use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    embeddings::{Embedding, EmbeddingBuilder},
    json_util,
    message::Message,
    runtime::{self, AsyncTask},
};

// ============================================================================
// Google Gemini API Client with HTTP3 and zero-allocation patterns
// ============================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Global HTTP3 client with AI optimization
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static GEMINI_METRICS: LazyLock<GeminiMetrics> = LazyLock::new(GeminiMetrics::new);

/// Gemini performance metrics with atomic counters
#[derive(Debug)]
pub struct GeminiMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter,
}

impl GeminiMetrics {
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

/// Gemini error types for comprehensive error handling
#[derive(thiserror::Error, Debug)]
pub enum GeminiError {
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
    #[error("API quota exceeded: {message}")]
    QuotaExceeded {
        message: String,
        retry_after: Option<Duration>,
    },
}

pub type Result<T> = std::result::Result<T, GeminiError>;

/// Zero-allocation Gemini client
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    /// Zero-allocation base URL storage
    base_url: ArrayString<256>,
    /// Shared HTTP3 client
    http_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static GeminiMetrics,
    /// Request timeout
    timeout: Duration,
}

impl Client {
    /// Create a new Gemini client with zero-allocation API key validation
    pub fn new(api_key: String) -> Result<Self> {
        Self::from_url(api_key, GEMINI_API_BASE_URL)
    }

    /// Create a new Gemini client with the given API key and base API URL
    pub fn from_url(api_key: String, base_url: &str) -> Result<Self> {
        if api_key.is_empty() {
            return Err(GeminiError::Configuration {
                field: "api_key".to_string(),
                message: "API key cannot be empty".to_string(),
                suggestion: "Provide a valid Gemini API key".to_string(),
            });
        }

        let api_key_array =
            ArrayString::from(&api_key).map_err(|_| GeminiError::Configuration {
                field: "api_key".to_string(),
                message: format!("API key too long: {} characters (max 128)", api_key.len()),
                suggestion: "Use a valid Gemini API key".to_string(),
            })?;

        let base_url_array =
            ArrayString::from(base_url).map_err(|_| GeminiError::Configuration {
                field: "base_url".to_string(),
                message: format!("Base URL too long: {} characters (max 256)", base_url.len()),
                suggestion: "Use a valid Gemini API base URL".to_string(),
            })?;

        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            base_url: base_url_array,
            http_client: &HTTP_CLIENT,
            metrics: &GEMINI_METRICS,
            timeout: Duration::from_secs(30),
        })
    }

    /// Create a new Gemini client from environment variable with validation
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| GeminiError::Configuration {
            field: "GEMINI_API_KEY".to_string(),
            message: "GEMINI_API_KEY environment variable not set".to_string(),
            suggestion: "Set GEMINI_API_KEY environment variable with your Gemini API key"
                .to_string(),
        })?;
        Self::new(api_key)
    }

    /// Build authenticated request with zero allocations
    pub async fn make_request(
        &self,
        path: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::Response> {
        let url = format!("{}/{}?key={}", self.base_url, path, self.api_key.load());

        // Build headers with zero allocation
        let mut headers: SmallVec<[(&str, ArrayString<180>); 4]> = smallvec![];
        headers.push((
            "Content-Type",
            ArrayString::from("application/json").unwrap(),
        ));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-gemini/1.0").unwrap(),
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

        response.map_err(GeminiError::Http)
    }

    /// Build SSE request for streaming
    pub async fn make_sse_request(
        &self,
        path: &str,
        body: Vec<u8>,
    ) -> Result<fluent_ai_http3::Response> {
        let url = format!(
            "{}/{}?alt=sse&key={}",
            self.base_url,
            path,
            self.api_key.load()
        );

        let mut headers: SmallVec<[(&str, ArrayString<180>); 4]> = smallvec![];
        headers.push((
            "Content-Type",
            ArrayString::from("application/json").unwrap(),
        ));
        headers.push(("Accept", ArrayString::from("text/event-stream").unwrap()));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-gemini/1.0").unwrap(),
        ));

        let request = HttpRequest::post(&url, body)?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(self.timeout);

        self.metrics.total_requests.inc();
        self.metrics.concurrent_requests.inc();

        let response = self.http_client.send(request).await;

        self.metrics.concurrent_requests.dec();

        match &response {
            Ok(_) => self.metrics.successful_requests.inc(),
            Err(_) => self.metrics.failed_requests.inc(),
        }

        response.map_err(GeminiError::Http)
    }

    /// Test connection to Gemini API
    pub async fn test_connection(&self) -> Result<()> {
        let url = format!(
            "{}/v1beta/models?key={}",
            self.base_url,
            self.api_key.load()
        );

        let request = HttpRequest::get(&url)?
            .header("User-Agent", "fluent-ai-gemini/1.0")
            .timeout(Duration::from_secs(10));

        let response = self.http_client.send(request).await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(GeminiError::Authentication {
                message: format!("Connection test failed with status: {}", response.status()),
                retry_after: None,
            })
        }
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    /// Create an embedding model with specific dimensions.
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }

    /// Create an embeddings builder for the given model.
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingBuilder<EmbeddingModel, D> {
        EmbeddingBuilder::new(self.embedding_model(model))
    }

    /// Create a transcription model with the given name.
    pub fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
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

/// EmbeddingsClient trait implementation
impl EmbeddingsClient for Client {
    type Model = EmbeddingModel;

    fn embedding_model(&self, model: &str) -> Self::Model {
        EmbeddingModel::new(self.clone(), model, None)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::Model {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }
}

/// TranscriptionClient trait implementation
impl TranscriptionClient for Client {
    type Model = TranscriptionModel;

    fn transcription_model(&self, model: &str) -> Self::Model {
        TranscriptionModel::new(self.clone(), model)
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for Client {
    fn provider_name(&self) -> &'static str {
        "gemini"
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

// ============================================================================
// Typestate markers
// ============================================================================
pub struct NeedsPrompt;
pub struct HasPrompt;

// ============================================================================
// Core builder (generic over typestate `S`)
// ============================================================================
pub struct GeminiCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    max_output_tokens: Option<u32>,
    candidate_count: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    generation_config: Option<serde_json::Value>,
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
impl<'a> GeminiCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            top_p: None,
            top_k: None,
            max_output_tokens: None,
            candidate_count: None,
            stop_sequences: None,
            generation_config: None,
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
    pub fn default_for_chat(client: &'a Client) -> GeminiCompletionBuilder<'a, HasPrompt> {
        Self::new(client, GEMINI_1_5_PRO)
            .temperature(0.8)
            .max_output_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> GeminiCompletionBuilder<'a, S> {
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
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    #[inline(always)]
    pub fn candidate_count(mut self, count: u32) -> Self {
        self.candidate_count = Some(count);
        self
    }

    #[inline(always)]
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    #[inline(always)]
    pub fn generation_config(mut self, config: serde_json::Value) -> Self {
        self.generation_config = Some(config);
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
impl<'a> GeminiCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> GeminiCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        GeminiCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_output_tokens: self.max_output_tokens,
            candidate_count: self.candidate_count,
            stop_sequences: self.stop_sequences,
            generation_config: self.generation_config,
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
impl<'a> GeminiCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, PromptError> {
        let prompt = self.prompt.as_ref().expect("HasPrompt guarantees prompt");

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

            // Add Gemini-specific parameters to generation config
            let mut gen_config = self.generation_config.clone().unwrap_or_else(|| json!({}));

            if let Some(top_p) = self.top_p {
                gen_config["topP"] = json!(top_p);
            }
            if let Some(top_k) = self.top_k {
                gen_config["topK"] = json!(top_k);
            }
            if let Some(max_output_tokens) = self.max_output_tokens {
                gen_config["maxOutputTokens"] = json!(max_output_tokens);
            }
            if let Some(candidate_count) = self.candidate_count {
                gen_config["candidateCount"] = json!(candidate_count);
            }
            if let Some(ref stop_sequences) = self.stop_sequences {
                gen_config["stopSequences"] = json!(stop_sequences);
            }

            if !gen_config.is_null() {
                params["generationConfig"] = gen_config;
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
impl<'a> Prompt for GeminiCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = GeminiCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}
