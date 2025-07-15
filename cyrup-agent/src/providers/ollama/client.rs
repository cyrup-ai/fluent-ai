// ============================================================================
// File: src/providers/ollama/client.rs
// ----------------------------------------------------------------------------
// Ollama client with typestate-driven builder pattern
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

use super::completion::{CompletionModel, EmbeddingModel, MISTRAL_MAGISTRAR_SMALL};

// ============================================================================
// Ollama API Client
// ============================================================================
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

#[derive(Clone, Debug)]
pub struct Client {
    pub base_url: String,
    pub(crate) http_client: reqwest::Client,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self::from_url(OLLAMA_API_BASE_URL)
    }

    pub fn from_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_owned(),
            http_client: reqwest::Client::builder()
                .build()
                .expect("Ollama reqwest client should build"),
        }
    }

    /// Create from environment (Ollama defaults to localhost)
    pub fn from_env() -> Self {
        Self::default()
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path);
        self.http_client.post(url)
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
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
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
    _state: std::marker::PhantomData<S>,
}

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
            _state: std::marker::PhantomData,
        }
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
            _state: std::marker::PhantomData::<HasPrompt>,
        }
    }
}

// ============================================================================
// HasPrompt -> execute/stream
// ============================================================================
impl<'a> OllamaCompletionBuilder<'a, HasPrompt> {
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
