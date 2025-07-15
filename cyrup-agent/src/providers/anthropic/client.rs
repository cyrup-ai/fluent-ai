// ============================================================================
// File: src/providers/anthropic/client.rs
// ----------------------------------------------------------------------------
// Anthropic API client with typestate builder pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use serde_json::json;

use crate::{
    client::{CompletionClient, ProviderClient},
    completion::{
        self, CompletionError, CompletionRequest, CompletionRequestBuilder, Prompt, PromptError,
    },
    message::Message,
    providers::anthropic::completion::{
        calculate_max_tokens, CacheControl, CompletionModel, ToolChoice, CLAUDE_4_SONNET,
    },
    runtime::{self as rt, AsyncTask},
};

/// Anthropic API authentication
#[derive(Clone, Debug)]
pub struct AnthropicAuth {
    api_key: String,
}

impl From<String> for AnthropicAuth {
    fn from(api_key: String) -> Self {
        AnthropicAuth { api_key }
    }
}

/// ------------------------------------------------------------
/// Typestate markers for builder pattern
/// ------------------------------------------------------------
pub struct NeedsPrompt;
pub struct HasPrompt;

/// ------------------------------------------------------------
/// Anthropic Client
/// ------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Client {
    pub(crate) http_client: reqwest::Client,
}

impl Client {
    /// Creates a new Anthropic client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            api_key.into().parse().expect("API key should parse"),
        );
        headers.insert(
            "anthropic-version",
            "2023-06-01".parse().expect("Version should parse"),
        );

        Self {
            http_client: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .expect("Anthropic reqwest client should build"),
        }
    }

    /// Creates a new Anthropic client from an API key.
    pub fn from_api_key(api_key: &str) -> Self {
        Self::new(api_key.to_string())
    }

    pub(crate) fn post_chat_completion(&self) -> reqwest::RequestBuilder {
        self.http_client
            .post("https://api.anthropic.com/v1/messages")
    }
}

/// ------------------------------------------------------------
/// Core builder struct with typestate
/// ------------------------------------------------------------
pub struct AnthropicCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // mutable fields
    temperature: Option<f64>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<completion::Document>,
    tools: Vec<completion::ToolDefinition>,
    max_tokens: Option<u64>,
    extended_think: bool,
    cache_control: Option<CacheControl>,
    additional_params: serde_json::Value,
    prompt: Option<Message>, // present only when S = HasPrompt
    _state: std::marker::PhantomData<S>,
}

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------
impl<'a> AnthropicCompletionBuilder<'a, NeedsPrompt> {
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
            max_tokens: calculate_max_tokens(model_name),
            extended_think: false,
            cache_control: None,
            additional_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Convenience helper: all defaults for Anthropic chat.
    #[inline(always)]
    pub fn default_for_chat(client: &'a Client) -> AnthropicCompletionBuilder<'a, HasPrompt> {
        Self::new(client, CLAUDE_4_SONNET)
            .temperature(0.0)
            .extended_thinking(true)
            .prompt(Message::user("")) // dummy; will be replaced in `.chat(..)`
    }
}

// ------------------------------------------------------------
// Fluent setters that keep the typestate
// ------------------------------------------------------------
impl<'a, S> AnthropicCompletionBuilder<'a, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    #[inline(always)]
    pub fn extended_thinking(mut self, on: bool) -> Self {
        self.extended_think = on;
        self
    }

    #[inline(always)]
    pub fn cache_control(mut self, c: CacheControl) -> Self {
        self.cache_control = Some(c);
        self
    }

    #[inline(always)]
    pub fn max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = Some(n);
        self
    }

    #[inline(always)]
    pub fn tool<T: completion::ToolDefinitionInto>(mut self, tool: T) -> Self {
        self.tools.push(tool.into_tool_def());
        self
    }

    #[inline(always)]
    pub fn additional_params(mut self, p: serde_json::Value) -> Self {
        crate::json_util::merge_inplace(&mut self.additional_params, p);
        self
    }
}

// ------------------------------------------------------------
// Transition from NeedsPrompt → HasPrompt
// ------------------------------------------------------------
impl<'a> AnthropicCompletionBuilder<'a, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(mut self, p: impl Into<Message>) -> AnthropicCompletionBuilder<'a, HasPrompt> {
        self.prompt = Some(p.into());
        AnthropicCompletionBuilder {
            client: self.client,
            model_name: self.model_name,
            temperature: self.temperature,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            max_tokens: self.max_tokens,
            extended_think: self.extended_think,
            cache_control: self.cache_control,
            additional_params: self.additional_params,
            prompt: self.prompt,
            _state: std::marker::PhantomData,
        }
    }
}

// ------------------------------------------------------------
// Final stage — only available once we *have* the prompt
// ------------------------------------------------------------
impl<'a> AnthropicCompletionBuilder<'a, HasPrompt> {
    /// Build a `CompletionRequest` ready for `.send()` / `.stream()`.
    #[inline(always)]
    fn build(self) -> CompletionRequest {
        let mut req = CompletionRequestBuilder::new(
            CompletionModel::new(self.client.clone(), self.model_name),
            self.prompt.expect("prompt present"),
        )
        .preamble_opt(self.preamble)
        .chat_history(self.chat_history)
        .documents(self.documents)
        .tools(self.tools)
        .temperature_opt(self.temperature)
        .max_tokens_opt(self.max_tokens)
        .additional_params(self.additional_params);

        // Anthropic-specific extras
        if self.extended_think {
            req = req.additional_params(json!({ "extended_thinking": true }));
        }
        if let Some(ctrl) = self.cache_control {
            req = req.additional_params(json!({ "cache_control": ctrl }));
        }

        req.build()
    }

    /// Fire off a **single-shot** completion → `AsyncTask<Result<…>>`
    #[inline(always)]
    pub fn send(
        self,
    ) -> AsyncTask<
        Result<completion::CompletionResponse<completion::CompletionResponseData>, CompletionError>,
    > {
        self.build().send()
    }

    /// Streaming variant → `AsyncTask<Result<StreamingCompletionResponse<…>>>`
    #[inline(always)]
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
        self.build().stream()
    }

    /// Convenience: run chat and get the **first chunk** as `String`.
    #[inline(always)]
    pub fn chat(self, prompt: impl Into<String>) -> AsyncTask<Result<String, PromptError>> {
        let req = self.prompt(prompt.into()).build();
        CompletionModel::new(self.client.clone(), self.model_name).prompt(req)
    }
}

/// ------------------------------------------------------------
/// Convenience type alias for the builder
/// ------------------------------------------------------------
pub type ClientBuilder<'a> = AnthropicCompletionBuilder<'a, NeedsPrompt>;

/// ------------------------------------------------------------
/// Provider trait implementations
/// ------------------------------------------------------------
impl ProviderClient for Client {
    /// Create a new Anthropic client from environment variables.
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        Self::new(api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}
