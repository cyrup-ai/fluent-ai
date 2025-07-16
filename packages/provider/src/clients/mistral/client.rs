// ============================================================================
// File: src/providers/mistral/client.rs
// ----------------------------------------------------------------------------
// Mistral client with typestate-driven builder pattern
// ============================================================================

use serde_json::json;

use crate::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    http::{HttpClient, HttpRequest, HttpError},
    json_util,
    message::Message,
    runtime::{self, AsyncTask},
};

use super::{
    completion::{CompletionModel, MISTRAL_SABA},
    embedding::{EmbeddingModel, MISTRAL_EMBED},
};

// ============================================================================
// Mistral API Client
// ============================================================================
const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) api_key: String,
}

impl Client {
    /// Create a new Mistral client with the given API key.
    pub fn new(api_key: &str) -> Result<Self, HttpError> {
        Self::from_url(api_key, MISTRAL_API_BASE_URL)
    }

    /// Create a new Mistral client with the given API key and base URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Result<Self, HttpError> {
        let http_client = HttpClient::for_provider("mistral")?;
        
        Ok(Self {
            base_url: base_url.to_string(),
            http_client,
            api_key: api_key.to_string(),
        })
    }

    /// Create from environment (MISTRAL_API_KEY)
    pub fn from_env() -> Result<Self, HttpError> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| HttpError::ConfigurationError("MISTRAL_API_KEY not set".to_string()))?;
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str, body: Vec<u8>) -> Result<HttpRequest, HttpError> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        tracing::debug!("POST {}", url);
        
        HttpRequest::post(url, body)?
            .header("Content-Type", "application/json")?
            .header("Authorization", &format!("Bearer {}", self.api_key))
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            MISTRAL_EMBED => 1024,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with specific dimensions.
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
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
pub struct MistralCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    random_seed: Option<i64>,
    safe_prompt: Option<bool>,
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
impl<'a> MistralCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            max_tokens: None,
            top_p: None,
            random_seed: None,
            safe_prompt: None,
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
    pub fn default_for_chat(client: &'a Client) -> MistralCompletionBuilder<'a, HasPrompt> {
        Self::new(client, MISTRAL_SABA)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> MistralCompletionBuilder<'a, S> {
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
    pub fn random_seed(mut self, seed: i64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    #[inline(always)]
    pub fn safe_prompt(mut self, safe: bool) -> Self {
        self.safe_prompt = Some(safe);
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
impl<'a> MistralCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> MistralCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        MistralCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            random_seed: self.random_seed,
            safe_prompt: self.safe_prompt,
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
impl<'a> MistralCompletionBuilder<'a, HasPrompt> {
    /// Build the completion request
    fn build_request(&self) -> Result<CompletionRequest, PromptError> {
        let prompt = self.prompt.as_ref().ok_or_else(|| PromptError::ValidationError("Prompt is required".to_string()))?;

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

            // Add Mistral-specific parameters
            if let Some(max_tokens) = self.max_tokens {
                params["max_tokens"] = json!(max_tokens);
            }
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(random_seed) = self.random_seed {
                params["random_seed"] = json!(random_seed);
            }
            if let Some(safe_prompt) = self.safe_prompt {
                params["safe_prompt"] = json!(safe_prompt);
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
            crate::streaming::StreamingCompletionResponse<super::completion::CompletionResponse>,
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
impl<'a> Prompt for MistralCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = MistralCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

// ============================================================================
// Legacy API types for compatibility
// ============================================================================
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
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

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
