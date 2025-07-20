// ============================================================================
// File: src/providers/groq/client.rs
// ----------------------------------------------------------------------------
// Groq client with typestate-driven builder pattern following Anthropic template
// ============================================================================

use std::sync::LazyLock;

use arc_swap::ArcSwap;
use arrayvec::{ArrayString, ArrayVec};
use atomic_counter::RelaxedCounter;
use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::json;
use smallvec::{SmallVec, smallvec};

use super::completion::{CompletionModel, LLAMA_3_70B_8192};
use crate::{
    client::{CompletionClient, ProviderClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    json_util,
    message::Message,
    runtime::AsyncTask,
};

// ============================================================================
// Groq API Client with HTTP3 and zero-allocation patterns
// ============================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

/// Global HTTP3 client with AI optimization
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics
static GROQ_METRICS: LazyLock<GroqMetrics> = LazyLock::new(GroqMetrics::new);

/// Groq performance metrics with atomic counters
#[derive(Debug)]
pub struct GroqMetrics {
    pub total_requests: RelaxedCounter,
    pub successful_requests: RelaxedCounter,
    pub failed_requests: RelaxedCounter,
    pub concurrent_requests: RelaxedCounter,
}

impl GroqMetrics {
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

/// Zero-allocation Groq client
#[derive(Clone)]
pub struct Client {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    /// Zero-allocation base URL storage
    base_url: &'static str,
    /// Shared HTTP3 client
    http_client: &'static HttpClient,
    /// Performance metrics
    metrics: &'static GroqMetrics,
}

impl Client {
    /// Create a new Groq client with zero-allocation API key validation
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
            base_url: GROQ_API_BASE_URL,
            http_client: &HTTP_CLIENT,
            metrics: &GROQ_METRICS,
        })
    }

    /// Create a new Groq client from environment variable with validation
    #[inline]
    pub fn from_env() -> Result<Self, CompletionError> {
        let api_key = std::env::var("GROQ_API_KEY").map_err(|_| {
            CompletionError::ConfigError("GROQ_API_KEY environment variable not set".into())
        })?;
        Self::new(api_key)
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
            ArrayString::from("application/json").unwrap(),
        ));
        headers.push((
            "User-Agent",
            ArrayString::from("fluent-ai-groq/1.0").unwrap(),
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
            Err(_) => self.metrics.failed_requests.inc(),
        }

        response
    }

    /// Test connection to Groq API
    #[inline]
    pub async fn test_connection(&self) -> Result<(), CompletionError> {
        let response = self
            .make_request("models", vec![])
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
        "groq"
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
pub struct GroqCompletionBuilder<'a, S> {
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

// ============================================================================
// Constructors
// ============================================================================
impl<'a> GroqCompletionBuilder<'a, NeedsPrompt> {
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

    /// Convenience helper: sensible defaults for chat
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> GroqCompletionBuilder<'a, HasPrompt> {
        Self::new(client, LLAMA_3_70B_8192)
            .temperature(0.7)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> GroqCompletionBuilder<'a, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
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
    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
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
impl<'a> GroqCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> GroqCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        GroqCompletionBuilder {
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
            _state: std::marker::PhantomData::<HasPrompt>,
        }
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> GroqCompletionBuilder<'a, HasPrompt> {
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
            builder = builder.additional_params(self.additional_params.clone());
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
                    tx.send(result);
                });
            }
            Err(e) => {
                tx.send(Err(e.into()));
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
                    tx.send(result);
                });
            }
            Err(e) => {
                tx.send(Err(e.into()));
            }
        }

        task
    }
}

// ============================================================================
// Prompt trait implementation
// ============================================================================
impl<'a> Prompt for GroqCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = GroqCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}
