// ============================================================================
// File: src/providers/openai/client.rs
// ----------------------------------------------------------------------------
// OpenAI client with typestate-driven builder pattern
// ============================================================================

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    embeddings::{Embed, EmbeddingsBuilder},
    json_util,
    message::Message,
    runtime::{self, AsyncTask},
};

#[cfg(feature = "audio")]
use super::audio_generation::AudioGenerationModel;
use super::completion::{
    CompletionModel, EmbeddingModel, O4_MINI, TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL,
    TEXT_EMBEDDING_ADA_002,
};
#[cfg(feature = "image")]
use super::image_generation::ImageGenerationModel;
use super::transcription::TranscriptionModel;

// ============================================================================
// OpenAI API Client
// ============================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub(crate) http_client: reqwest::Client,
}

impl Client {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, OPENAI_API_BASE_URL)
    }

    /// Create a new OpenAI client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {api_key}")
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("OpenAI reqwest client should build"),
        }
    }

    /// Create from environment (OpenAI_API_KEY)
    pub fn from_env() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with specific dimensions.
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embeddings builder for the given model.
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create a transcription model with the given name.
    pub fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }

    #[cfg(feature = "image")]
    /// Create an image generation model with the given name.
    pub fn image_generation_model(&self, model: &str) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }

    #[cfg(feature = "audio")]
    /// Create an audio generation model with the given name.
    pub fn audio_generation_model(&self, model: &str) -> AudioGenerationModel {
        AudioGenerationModel::new(self.clone(), model)
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
pub struct OpenAICompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    logit_bias: Option<serde_json::Value>,
    user: Option<String>,
    seed: Option<i32>,
    response_format: Option<serde_json::Value>,
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
impl<'a> OpenAICompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            top_p: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            seed: None,
            response_format: None,
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
    pub fn default_for_chat(client: &'a Client) -> OpenAICompletionBuilder<'a, HasPrompt> {
        Self::new(client, O4_MINI)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> OpenAICompletionBuilder<'a, S> {
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
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    #[inline(always)]
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn logit_bias(mut self, bias: serde_json::Value) -> Self {
        self.logit_bias = Some(bias);
        self
    }

    #[inline(always)]
    pub fn user(mut self, u: impl ToString) -> Self {
        self.user = Some(u.to_string());
        self
    }

    #[inline(always)]
    pub fn seed(mut self, s: i32) -> Self {
        self.seed = Some(s);
        self
    }

    #[inline(always)]
    pub fn response_format(mut self, format: serde_json::Value) -> Self {
        self.response_format = Some(format);
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
impl<'a> OpenAICompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> OpenAICompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        OpenAICompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: self.logit_bias,
            user: self.user,
            seed: self.seed,
            response_format: self.response_format,
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
impl<'a> OpenAICompletionBuilder<'a, HasPrompt> {
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

            // Add OpenAI-specific parameters
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(max_tokens) = self.max_tokens {
                params["max_tokens"] = json!(max_tokens);
            }
            if let Some(presence_penalty) = self.presence_penalty {
                params["presence_penalty"] = json!(presence_penalty);
            }
            if let Some(frequency_penalty) = self.frequency_penalty {
                params["frequency_penalty"] = json!(frequency_penalty);
            }
            if let Some(ref logit_bias) = self.logit_bias {
                params["logit_bias"] = logit_bias.clone();
            }
            if let Some(ref user) = self.user {
                params["user"] = json!(user);
            }
            if let Some(seed) = self.seed {
                params["seed"] = json!(seed);
            }
            if let Some(ref response_format) = self.response_format {
                params["response_format"] = response_format.clone();
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
impl<'a> Prompt for OpenAICompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = OpenAICompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

// ============================================================================
// Common types shared across OpenAI provider
// ============================================================================
#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
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
