// ============================================================================
// File: src/providers/huggingface/client.rs
// ----------------------------------------------------------------------------
// Huggingface client with typestate-driven builder pattern
// ============================================================================

use serde_json::json;
use std::fmt::Display;
use bytes::Bytes;

use crate::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    http::{HttpClient, HttpError},
    json_util,
    message::Message,
    runtime::{self, AsyncTask},
};

use super::completion::{CompletionModel, QWEN2_5_CODER};
#[cfg(feature = "image")]
use super::image_generation::ImageGenerationModel;
use super::transcription::TranscriptionModel;

// ============================================================================
// SubProvider types
// ============================================================================
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SubProvider {
    #[default]
    HFInference,
    Together,
    SambaNova,
    Fireworks,
    Hyperbolic,
    Nebius,
    Novita,
    Custom(String),
}

impl SubProvider {
    /// Get the chat completion endpoint for the SubProvider
    pub fn completion_endpoint(&self, model: &str) -> String {
        match self {
            SubProvider::HFInference => format!("/{model}/v1/chat/completions"),
            _ => "/v1/chat/completions".to_string(),
        }
    }

    /// Get the transcription endpoint for the SubProvider
    pub fn transcription_endpoint(
        &self,
        model: &str,
    ) -> Result<String, crate::transcription::TranscriptionError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{model}")),
            _ => Err(crate::transcription::TranscriptionError::ProviderError(
                format!("transcription endpoint is not supported yet for {self}"),
            )),
        }
    }

    /// Get the image generation endpoint for the SubProvider
    #[cfg(feature = "image")]
    pub fn image_generation_endpoint(
        &self,
        model: &str,
    ) -> Result<String, crate::image_generation::ImageGenerationError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{}", model)),
            _ => Err(
                crate::image_generation::ImageGenerationError::ProviderError(format!(
                    "image generation endpoint is not supported yet for {}",
                    self
                )),
            ),
        }
    }

    pub fn model_identifier(&self, model: &str) -> String {
        match self {
            SubProvider::Fireworks => format!("accounts/fireworks/models/{model}"),
            _ => model.to_string(),
        }
    }
}

impl From<&str> for SubProvider {
    fn from(s: &str) -> Self {
        SubProvider::Custom(s.to_string())
    }
}

impl From<String> for SubProvider {
    fn from(value: String) -> Self {
        SubProvider::Custom(value)
    }
}

impl Display for SubProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let route = match self {
            SubProvider::HFInference => "hf-inference/models".to_string(),
            SubProvider::Together => "together".to_string(),
            SubProvider::SambaNova => "sambanova".to_string(),
            SubProvider::Fireworks => "fireworks-ai".to_string(),
            SubProvider::Hyperbolic => "hyperbolic".to_string(),
            SubProvider::Nebius => "nebius".to_string(),
            SubProvider::Novita => "novita".to_string(),
            SubProvider::Custom(route) => route.clone(),
        };
        write!(f, "{}", route)
    }
}

// ============================================================================
// Huggingface API Client
// ============================================================================
const HUGGINGFACE_API_BASE_URL: &str = "https://router.huggingface.co/";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub sub_provider: SubProvider,
    pub(crate) http_client: HttpClient,
    pub(crate) api_key: String,
}

impl Client {
    /// Create a new Huggingface client with the given API key.
    pub fn new(api_key: &str) -> Result<Self, HttpError> {
        Self::from_url(api_key, HUGGINGFACE_API_BASE_URL, SubProvider::HFInference)
    }

    /// Create a new Huggingface client with the given API key and SubProvider.
    pub fn new_with_sub_provider(api_key: &str, sub_provider: SubProvider) -> Result<Self, HttpError> {
        Self::from_url(api_key, HUGGINGFACE_API_BASE_URL, sub_provider)
    }

    /// Create a new Huggingface client with the given API key, base URL, and SubProvider.
    pub fn from_url(api_key: &str, base_url: &str, sub_provider: SubProvider) -> Result<Self, HttpError> {
        let http_client = HttpClient::for_provider("huggingface")?;
        let url = format!("{base_url}/{sub_provider}");
        
        Ok(Self {
            base_url: url,
            sub_provider,
            http_client,
            api_key: api_key.to_string(),
        })
    }

    /// Create from environment (HF_API_KEY)
    pub fn from_env() -> Result<Self, HttpError> {
        let api_key = std::env::var("HF_API_KEY")
            .map_err(|_| HttpError::ConfigurationError("HF_API_KEY not set".to_string()))?;
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str, body: Vec<u8>) -> Result<crate::http::HttpRequest, HttpError> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        
        crate::http::HttpRequest::post(url, Bytes::from(body))?
            .header("Content-Type", "application/json")?
            .header("Authorization", &format!("Bearer {}", self.api_key))
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
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
}

// ============================================================================
// Typestate markers
// ============================================================================
pub struct NeedsPrompt;
pub struct HasPrompt;

// ============================================================================
// Core builder (generic over typestate `S`)
// ============================================================================
pub struct HuggingfaceCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
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
impl<'a> HuggingfaceCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn new(client: &'a Client, model_name: &'a str) -> Self {
        Self {
            client,
            model_name,
            temperature: None,
            top_p: None,
            max_tokens: None,
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
    pub fn default_for_chat(client: &'a Client) -> HuggingfaceCompletionBuilder<'a, HasPrompt> {
        Self::new(client, QWEN2_5_CODER)
            .temperature(0.8)
            .max_tokens(2048)
            .prompt(Message::user("")) // dummy; will be replaced in actual usage
    }
}

// ============================================================================
// Builder methods available in ALL states
// ============================================================================
impl<'a, S> HuggingfaceCompletionBuilder<'a, S> {
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
impl<'a> HuggingfaceCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, msg: Message) -> HuggingfaceCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(msg);
        HuggingfaceCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
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
impl<'a> HuggingfaceCompletionBuilder<'a, HasPrompt> {
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

            // Add Huggingface-specific parameters
            if let Some(top_p) = self.top_p {
                params["top_p"] = json!(top_p);
            }
            if let Some(max_tokens) = self.max_tokens {
                params["max_tokens"] = json!(max_tokens);
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
impl<'a> Prompt for HuggingfaceCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = HuggingfaceCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}
