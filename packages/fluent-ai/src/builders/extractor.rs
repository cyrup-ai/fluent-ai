//! Extractor builder implementations
//!
//! All extractor construction logic and builder patterns.

use std::fmt;
use std::marker::PhantomData;

use fluent_ai_domain::Models;
use fluent_ai_domain::agent::Agent;
use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::extractor::{Extractor, ExtractorImpl};
use fluent_ai_domain::{AsyncTask, spawn_async};
use serde::de::DeserializeOwned;

/// Builder for creating Extractor instances
pub struct ExtractorBuilder<
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel,
> {
    model: M,
    system_prompt: Option<String>,
    _marker: PhantomData<T>,
}

/// Builder with error handler for polymorphic error handling
pub struct ExtractorBuilderWithHandler<
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel,
> {
    #[allow(dead_code)]
    // TODO: Use for completion model integration and structured data extraction
    model: M,
    #[allow(dead_code)] // TODO: Use for extraction guidance and schema specification
    system_prompt: Option<String>,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during extraction operations
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for extraction result processing and validation
    result_handler: Option<Box<dyn FnOnce(T) -> T + Send + 'static>>,
    #[allow(dead_code)] // TODO: Use for streaming extraction chunk processing
    chunk_handler: Option<Box<dyn FnMut(T) -> T + Send + 'static>>,
    #[allow(dead_code)] // TODO: Use for type-level extraction target specification
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> ExtractorImpl<T> {
    // Semantic entry point
    pub fn extract_with<M: CompletionModel>(model: M) -> ExtractorBuilder<T, M> {
        ExtractorBuilder {
            model,
            system_prompt: None,
            _marker: PhantomData,
        }
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static, M: CompletionModel>
    ExtractorBuilder<T, M>
{
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.system_prompt = Some(instructions.into());
        self
    }

    // Error handling - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> ExtractorBuilderWithHandler<T, M>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        ExtractorBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None,
            _marker: PhantomData,
        }
    }

    pub fn on_result<F>(self, handler: F) -> ExtractorBuilderWithHandler<T, M>
    where
        F: FnOnce(T) -> T + Send + 'static,
    {
        ExtractorBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: Box::new(|e| eprintln!("Extractor error: {}", e)),
            result_handler: Some(Box::new(handler)),
            chunk_handler: None,
            _marker: PhantomData,
        }
    }

    pub fn on_chunk<F>(self, handler: F) -> ExtractorBuilderWithHandler<T, M>
    where
        F: FnMut(T) -> T + Send + 'static,
    {
        ExtractorBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: Box::new(|e| eprintln!("Extractor chunk error: {}", e)),
            result_handler: None,
            chunk_handler: Some(Box::new(handler)),
            _marker: PhantomData,
        }
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static, M: CompletionModel + 'static>
    ExtractorBuilderWithHandler<T, M>
{
    // Terminal method - returns impl Extractor
    pub fn build(self) -> impl Extractor<T> {
        // TODO: Convert model to agent properly
        let agent = Agent::new(Models::Gpt35Turbo, "");

        let mut extractor = ExtractorImpl::new(agent);
        if let Some(prompt) = self.system_prompt {
            extractor = extractor.with_system_prompt(prompt);
        }
        extractor
    }

    // Terminal method - async build
    pub fn build_async(self) -> AsyncTask<impl Extractor<T>>
    where
        ExtractorImpl<T>: fluent_ai_domain::async_task::NotResult,
    {
        spawn_async(async move { self.build() })
    }

    // Terminal method - extract from text immediately
    pub fn extract_from_text(self, text: impl Into<String>) -> AsyncTask<T>
    where
        T: fluent_ai_domain::async_task::NotResult,
    {
        let extractor = self.build();
        let text = text.into();
        extractor.extract_from(&text)
    }
}
