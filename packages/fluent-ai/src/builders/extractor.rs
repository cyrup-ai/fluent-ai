//! Extractor builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All extractor construction logic and builder patterns with zero allocation.

use std::fmt;
use std::marker::PhantomData;

use fluent_ai_domain::Models;
use fluent_ai_domain::agent::Agent;
use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::extractor::{Extractor, ExtractorImpl};
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
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: FnMut(T) -> T + Send + 'static;
    
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
    F3 = fn(T) -> T,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(T) -> T + Send + 'static,
    F3: FnMut(T) -> T + Send + 'static,
{
    model: M,
    system_prompt: Option<String>,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    chunk_handler: Option<F3>,
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
            chunk_handler: None,
            _marker: PhantomData,
        }
    }
}

impl<T, M, F1, F2, F3> ExtractorBuilder<T> for ExtractorBuilderImpl<T, M, F1, F2, F3>
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel + 'static,
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(T) -> T + Send + 'static,
    F3: FnMut(T) -> T + Send + 'static,
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
            chunk_handler: self.chunk_handler,
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
            chunk_handler: self.chunk_handler,
            _marker: PhantomData,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl ExtractorBuilder<T>
    where
        F: FnMut(T) -> T + Send + 'static,
    {
        ExtractorBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            error_handler: self.error_handler,
            result_handler: self.result_handler,
            chunk_handler: Some(handler),
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