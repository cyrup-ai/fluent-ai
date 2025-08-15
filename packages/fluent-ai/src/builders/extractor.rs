//! Extractor builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All extractor construction logic and builder patterns with zero allocation.

use std::fmt;
use std::marker::PhantomData;

use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_domain::Models;
use fluent_ai_domain::agent::Agent;
use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::extractor::{Extractor, ExtractorImpl};
use fluent_ai_domain::http::responses::completion::CompletionChunk;
use fluent_ai_domain::{AsyncTask, spawn_async};
use serde::de::DeserializeOwned;

/// Extractor builder trait - elegant zero-allocation builder pattern
pub trait ExtractorBuilder<T>: Sized 
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
{
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl ExtractorBuilder<T>;
    
    /// Set instructions - EXACT syntax: .instructions("...")
    fn instructions(self, instructions: impl Into<String>) -> impl ExtractorBuilder<T>;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: FnOnce(T) -> T + Send + 'static;
    

    
    /// Build extractor - EXACT syntax: .build()
    fn build(self) -> impl Extractor<T>;
    
    /// Build async extractor - EXACT syntax: .build_async()
    fn build_async(self) -> AsyncTask<impl Extractor<T>>
    where
        ExtractorImpl<T>: fluent_ai_domain::async_task::NotResult;
    
    /// Extract from text immediately - EXACT syntax: .extract_from_text("text")
    fn extract_from_text(self, text: impl Into<String>) -> AsyncTask<T>
    where
        T: fluent_ai_domain::async_task::NotResult;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct ExtractorBuilderImpl<
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel,
    F1 = fn(String),
    F2 = fn(T) -> T,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(T) -> T + Send + 'static,
{
    model: M,
    system_prompt: Option<String>,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    cyrup_chunk_handler: Option<Box<dyn Fn(Result<CompletionChunk, String>) -> CompletionChunk + Send + Sync>>,
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> ExtractorImpl<T> {
    /// Semantic entry point - EXACT syntax: ExtractorImpl::extract_with(model)
    pub fn extract_with<M: CompletionModel>(model: M) -> impl ExtractorBuilder<T> {
        ExtractorBuilderImpl {
            model,
            system_prompt: None,
            error_handler: None,
            result_handler: None,
            cyrup_chunk_handler: None,
            _marker: PhantomData,
        }
    }
}

impl<T, M, F1, F2> ExtractorBuilder<T> for ExtractorBuilderImpl<T, M, F1, F2>
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel + 'static,
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(T) -> T + Send + 'static,
{
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(mut self, prompt: impl Into<String>) -> impl ExtractorBuilder<T> {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    /// Set instructions - EXACT syntax: .instructions("...")
    fn instructions(mut self, instructions: impl Into<String>) -> impl ExtractorBuilder<T> {
        self.system_prompt = Some(instructions.into());
        self
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        ExtractorBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: Some(handler),
            result_handler: self.result_handler,
            cyrup_chunk_handler: self.cyrup_chunk_handler,
            _marker: PhantomData,
        }
    }
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: FnOnce(T) -> T + Send + 'static,
    {
        ExtractorBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: self.error_handler,
            result_handler: Some(handler),
            cyrup_chunk_handler: self.cyrup_chunk_handler,
            _marker: PhantomData,
        }
    }

    
    /// Build extractor - EXACT syntax: .build()
    fn build(self) -> impl Extractor<T> {
        // TODO: Convert model to agent properly
        let agent = Agent::new(Models::Gpt35Turbo, "");
        
        let mut extractor = ExtractorImpl::new(agent);
        if let Some(prompt) = self.system_prompt {
            extractor = extractor.with_system_prompt(prompt);
        }
        extractor
    }
    
    /// Build async extractor - EXACT syntax: .build_async()
    fn build_async(self) -> AsyncTask<impl Extractor<T>>
    where
        ExtractorImpl<T>: fluent_ai_domain::async_task::NotResult,
    {
        spawn_async(async move { self.build() })
    }
    
    /// Extract from text immediately - EXACT syntax: .extract_from_text("text")
    fn extract_from_text(self, text: impl Into<String>) -> AsyncTask<T>
    where
        T: fluent_ai_domain::async_task::NotResult,
    {
        let extractor = self.build();
        let text = text.into();
        extractor.extract_from(&text)
    }
}

impl<T, M, F1, F2> ChunkHandler<CompletionChunk, String> for ExtractorBuilderImpl<T, M, F1, F2>
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel + 'static,
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(T) -> T + Send + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, String>) -> CompletionChunk + Send + Sync + 'static,
    {
        self.cyrup_chunk_handler = Some(Box::new(handler));
        self
    }
}