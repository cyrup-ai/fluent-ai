// ============================================================================
// File: src/providers/groq/client.rs
// ----------------------------------------------------------------------------
// Groq client with typestate-driven builder pattern following Anthropic template
// ============================================================================

use serde_json::json;

use crate::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    http::{HttpClient, HttpRequest, HttpError},
    json_util,
    message::Message,
    runtime::AsyncTask,
};

use super::completion::{CompletionModel, LLAMA_3_70B_8192};

// ============================================================================
// Groq API Client
// ============================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) api_key: String,
}

impl Client {
    /// Create a new Groq client with the given API key.
    pub fn new(api_key: &str) -> Result<Self, HttpError> {
        Self::from_url(api_key, GROQ_API_BASE_URL)
    }

    /// Create a new Groq client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Result<Self, HttpError> {
        let http_client = HttpClient::for_provider("groq")?;
        
        Ok(Self {
            base_url: base_url.to_string(),
            http_client,
            api_key: api_key.to_string(),
        })
    }

    /// Create a new Groq client from the `GROQ_API_KEY` environment variable.
    pub fn from_env() -> Result<Self, HttpError> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| HttpError::ConfigurationError("GROQ_API_KEY not set".to_string()))?;
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str, body: Vec<u8>) -> Result<HttpRequest, HttpError> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        
        HttpRequest::post(url, body)?
            .header("Content-Type", "application/json")?
            .header("Authorization", &format!("Bearer {}", self.api_key))
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
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
