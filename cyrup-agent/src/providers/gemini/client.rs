// ============================================================================
// File: src/providers/gemini/client.rs
// ----------------------------------------------------------------------------
// Gemini client with typestate-driven builder pattern
// ============================================================================

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

use super::completion::{CompletionModel, GEMINI_1_5_PRO};
use super::embedding::EmbeddingModel;
use super::transcription::TranscriptionModel;

// ============================================================================
// Google Gemini API Client
// ============================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub api_key: String,
    pub(crate) http_client: reqwest::Client,
}

impl Client {
    /// Create a new Gemini client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, GEMINI_API_BASE_URL)
    }

    /// Create a new Gemini client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::CONTENT_TYPE,
                        "application/json".parse().unwrap(),
                    );
                    headers
                })
                .build()
                .expect("Gemini reqwest client should build"),
        }
    }

    /// Create from environment (GEMINI_API_KEY)
    pub fn from_env() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        Self::new(&api_key)
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}?key={}", self.base_url, path, self.api_key).replace("//", "/");
        tracing::debug!("POST {}/{}?key={}", self.base_url, path, "****");
        self.http_client.post(url)
    }

    pub fn post_sse(&self, path: &str) -> reqwest::RequestBuilder {
        let url =
            format!("{}/{}?alt=sse&key={}", self.base_url, path, self.api_key).replace("//", "/");
        tracing::debug!("POST {}/{}?alt=sse&key={}", self.base_url, path, "****");
        self.http_client.post(url)
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
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create a transcription model with the given name.
    pub fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
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
