// ============================================================================
// File: src/providers/together/client.rs
// ----------------------------------------------------------------------------
// Together AI client with typestate-driven builder pattern
// ============================================================================

use serde_json::json;

use crate::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    json_util,
    message::Message,
    runtime::{self, AsyncTask},
};

use super::{
    completion::{CompletionModel, LLAMA_3_2_11B_VISION_INSTRUCT_TURBO},
    embedding::{EmbeddingModel, M2_BERT_80M_8K_RETRIEVAL},
};

// ============================================================================
// Together AI API Client
// ============================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub(crate) http_client: reqwest::Client,
}

impl Client {
    /// Create a new Together AI client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, TOGETHER_AI_BASE_URL)
    }

    /// Create a new Together AI client with the given API key and base URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::CONTENT_TYPE,
                        "application/json".parse().unwrap(),
                    );
                    headers.insert(
                        "Authorization",
                        format!("Bearer {api_key}")
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("Together AI reqwest client should build"),
        }
    }

    /// Create from environment (TOGETHER_API_KEY)
    pub fn from_env() -> Self {
        let api_key = std::env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY not set");
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        tracing::debug!("POST {}", url);
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            M2_BERT_80M_8K_RETRIEVAL => 8192,
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
    documents: Vec<completion::Document>,
    tools: Vec<completion::ToolDefinition>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>,
}

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
            _state: std::marker::PhantomData,
        }
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
            _state: std::marker::PhantomData::<HasPrompt>,
        }
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> TogetherCompletionBuilder<'a, HasPrompt> {
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
impl<'a> Prompt for TogetherCompletionBuilder<'a, NeedsPrompt> {
    type PromptedBuilder = TogetherCompletionBuilder<'a, HasPrompt>;

    #[inline(always)]
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(self.prompt(Message::user(prompt.to_string())))
    }
}

// ============================================================================
// Legacy API types for compatibility
// ============================================================================
pub mod together_ai_api_types {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: String,
        pub code: String,
    }

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            format!("Code `{}`: {}", self.code, self.error)
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse),
    }
}
