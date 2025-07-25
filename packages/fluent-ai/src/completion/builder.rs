// ==========================================================================
// File: src/completion/builder.rs
// --------------------------------------------------------------------------
// Generic, zero-alloc completion builder + thin provider-extension traits.
// • Public surface is synchronous (build-time); only `.send()` / `.stream()`
//   return `AsyncTask` / `AsyncStream`.
// • No async fn, no BoxFuture, no async_trait.
// • Provider authors add their own `BuilderExt` impls in their crate/module.
// ==========================================================================

#![allow(clippy::type_complexity)]

use fluent_ai_provider::Model;
use serde_json::json;

use crate::{
    completion::message::Message,
    completion::{
        CompletionError, CompletionRequest, CompletionRequestBuilder, PromptError, ToolDefinition},
    runtime::{self, AsyncTask},
    streaming};

/// ------------------------------------------------------------
/// Typestate markers
/// ------------------------------------------------------------
pub struct NeedsPrompt;
pub struct HasPrompt;

/// ------------------------------------------------------------
/// Generic builder – provider-agnostic
/// ------------------------------------------------------------
pub struct CompletionBuilder<M, S> {
    model: M,
    temperature: Option<f64>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<crate::completion::Document>,
    tools: Vec<ToolDefinition>,
    max_tokens: Option<u64>,
    // universal feature flags
    extended_thinking: bool,
    cache_control: Option<serde_json::Value>,
    prompt_enhancements: Vec<&'static str>,
    extra_params: serde_json::Value,
    prompt: Option<Message>,
    _state: std::marker::PhantomData<S>}

// ---------------------------------------------------------------------
// 1. Bare-bones constructor + default_builder with sensible switches
// ---------------------------------------------------------------------
impl<M: Model> CompletionBuilder<M, NeedsPrompt> {
    #[inline(always)]
    pub fn new(model: M) -> Self {
        Self {
            model,
            temperature: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            max_tokens: None,
            extended_thinking: false,
            cache_control: None,
            prompt_enhancements: Vec::new(),
            extra_params: json!({}),
            prompt: None,
            _state: std::marker::PhantomData}
    }
}

pub fn default_builder<M: Model>(model: M) -> CompletionBuilder<M, NeedsPrompt> {
    CompletionBuilder::new(model)
        .temperature(0.0)
        .extended_thinking(true)
        .prompt_enhancement("prompt-tools-generate")
        .cache_control(json!({ "max_age_secs": 86_400 }))
}

// ---------------------------------------------------------------------
// 2. Fluent setters (provider-agnostic)
// ---------------------------------------------------------------------
impl<M, S> CompletionBuilder<M, S> {
    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }
    #[inline(always)]
    pub fn preamble(mut self, s: impl Into<String>) -> Self {
        self.preamble = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn chat_history(mut self, h: Vec<Message>) -> Self {
        self.chat_history = h;
        self
    }
    #[inline(always)]
    pub fn document(mut self, d: crate::completion::Document) -> Self {
        self.documents.push(d);
        self
    }
    #[inline(always)]
    pub fn tool(mut self, t: ToolDefinition) -> Self {
        self.tools.push(t);
        self
    }
    #[inline(always)]
    pub fn max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = Some(n);
        self
    }
    #[inline(always)]
    pub fn extended_thinking(mut self, on: bool) -> Self {
        self.extended_thinking = on;
        self
    }
    #[inline(always)]
    pub fn cache_control(mut self, v: serde_json::Value) -> Self {
        self.cache_control = Some(v);
        self
    }
    #[inline(always)]
    pub fn prompt_enhancement(mut self, tag: &'static str) -> Self {
        if !self.prompt_enhancements.contains(&tag) {
            self.prompt_enhancements.push(tag);
        }
        self
    }
    #[inline(always)]
    pub fn extra_params(mut self, p: serde_json::Value) -> Self {
        crate::json_util::merge_inplace(&mut self.extra_params, p);
        self
    }
}

// ---------------------------------------------------------------------
// 3. NeedsPrompt → HasPrompt transition
// ---------------------------------------------------------------------
impl<M: Model> CompletionBuilder<M, NeedsPrompt> {
    #[inline(always)]
    pub fn prompt(self, p: impl Into<Message>) -> CompletionBuilder<M, HasPrompt> {
        CompletionBuilder {
            prompt: Some(p.into()),
            model: self.model,
            temperature: self.temperature,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            max_tokens: self.max_tokens,
            extended_thinking: self.extended_thinking,
            cache_control: self.cache_control,
            prompt_enhancements: self.prompt_enhancements,
            extra_params: self.extra_params,
            _state: std::marker::PhantomData}
    }
}

// ---------------------------------------------------------------------
// 4. Final stage (HasPrompt) – produce request / fire async
// ---------------------------------------------------------------------
impl<M: Model> CompletionBuilder<M, HasPrompt> {
    fn into_request(self) -> CompletionRequest {
        let mut b =
            CompletionRequestBuilder::new(self.model.clone(), self.prompt.unwrap_or_default())
                .preamble_opt(self.preamble)
                .chat_history(self.chat_history)
                .documents(self.documents)
                .tools(self.tools)
                .temperature_opt(self.temperature)
                .max_tokens_opt(self.max_tokens)
                .additional_params(self.extra_params);

        if self.extended_thinking {
            b = b.additional_params(json!({ "extended_thinking": true }));
        }
        if let Some(ctrl) = self.cache_control {
            b = b.additional_params(json!({ "cache_control": ctrl }));
        }
        if !self.prompt_enhancements.is_empty() {
            b = b.additional_params(json!({ "prompt_enhancements": self.prompt_enhancements }));
        }

        b.build()
    }

    #[inline(always)]
    pub fn send(self) -> AsyncTask<Result<crate::completion::CompletionResponse, CompletionError>> {
        self.into_request().send()
    }

    #[inline(always)]
    pub fn stream(
        self,
    ) -> AsyncTask<
        Result<streaming::StreamingCompletionResponse<M::StreamingResponse>, CompletionError>,
    > {
        self.into_request().stream()
    }

    #[inline(always)]
    pub fn chat(self) -> AsyncTask<Result<String, PromptError>> {
        self.send()
            .map(|task| task.map(|ok| ok.map(|r| r.choice.first_text().unwrap_or_default())))
    }
}

// ---------------------------------------------------------------------
// 5. Provider-extension *sparse traits*
// ---------------------------------------------------------------------

/// Blanket trait – implemented automatically for every builder.
/// Provider crates add `use` + `impl AnthropicExt for CompletionBuilder<AnthropicModel,_> { … }`.
pub trait BuilderExt: Sized {
    fn provider_param(self, key: &'static str, value: serde_json::Value) -> Self;
}

impl<M, S> BuilderExt for CompletionBuilder<M, S> {
    #[inline(always)]
    fn provider_param(mut self, key: &'static str, value: serde_json::Value) -> Self {
        crate::json_util::merge_inplace(&mut self.extra_params, json!({ key: value }));
        self
    }
}

// ---------------- Anthropic sample extension (in provider crate) ------

// The actual impl belongs inside `providers::anthropic` so downstream users
// only get the methods when the Anthropic feature is enabled.
//
// impl<S> AnthropicBuilderExt for CompletionBuilder<AnthropicCompletionModel, S> { … }

/// Dummy trait shown for clarity – not exported here.
#[allow(dead_code)]
trait AnthropicBuilderExt {
    fn beta(self, tag: &'static str) -> Self;
}
#[allow(dead_code)]
impl<S> AnthropicBuilderExt for CompletionBuilder<fluent_ai_provider::Models, S> {
    #[inline(always)]
    fn beta(self, tag: &'static str) -> Self {
        self.provider_param("anthropic_beta", json!(tag))
    }
}

// ==========================================================================
// End of file
// ==========================================================================
