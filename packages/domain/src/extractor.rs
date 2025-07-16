use crate::{AsyncTask, spawn_async};
use crate::agent::Agent;
use crate::completion::CompletionModel;
use crate::Models;
use std::fmt;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

/// Trait defining the core extraction interface
pub trait Extractor<T>: Send + Sync + fmt::Debug + Clone
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
{
    /// Get the agent used for extraction
    fn agent(&self) -> &Agent;
    
    /// Get the system prompt for extraction
    fn system_prompt(&self) -> Option<&str>;
    
    /// Extract structured data from text
    fn extract_from(&self, text: &str) -> AsyncTask<T>
    where
        T: crate::async_task::NotResult;
    
    /// Create new extractor with agent
    fn new(agent: Agent) -> Self;
    
    /// Set system prompt for extraction guidance
    fn with_system_prompt(self, prompt: impl Into<String>) -> Self;
}

/// Implementation of the Extractor trait
#[derive(Debug, Clone)]
pub struct ExtractorImpl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> {
    agent: Agent,
    system_prompt: Option<String>,
    _marker: PhantomData<T>,
}

// ExtractorImpl implements NotResult since it contains no Result types

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> Extractor<T> for ExtractorImpl<T> {
    fn agent(&self) -> &Agent {
        &self.agent
    }
    
    fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    
    fn extract_from(&self, text: &str) -> AsyncTask<T> 
    where
        T: crate::async_task::NotResult,
    {
        let system_prompt = self.system_prompt.clone().unwrap_or_else(|| {
            format!(
                "Extract structured data in JSON format matching the schema for type {}",
                std::any::type_name::<T>()
            )
        });
        
        // Use prompt() method and collect the stream
        use crate::prompt::Prompt;
        let _prompt = Prompt::new(format!("{}

{}", system_prompt, text));
        let _agent = self.agent.clone();
        
        spawn_async(async move {
            // TODO: Implement actual model completion
            // For now, return a placeholder that will fail at compile time if T is Result
            unimplemented!("Extraction implementation pending model integration")
        })
    }
    
    fn new(agent: Agent) -> Self {
        Self {
            agent,
            system_prompt: None,
            _marker: PhantomData,
        }
    }
    
    fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// Builder for creating Extractor instances
pub struct ExtractorBuilder<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static, M: CompletionModel> {
    model: M,
    system_prompt: Option<String>,
    _marker: PhantomData<T>,
}

/// Builder with error handler for polymorphic error handling
pub struct ExtractorBuilderWithHandler<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static, M: CompletionModel> {
    #[allow(dead_code)] // TODO: Use for completion model integration and structured data extraction
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

impl<
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
    M: CompletionModel + 'static,
> ExtractorBuilderWithHandler<T, M>
{
    // Terminal method - returns impl Extractor
    pub fn build(self) -> impl Extractor<T> {
        // TODO: Convert model to agent properly
        let agent = Agent::with_model(Models::Gpt35Turbo)
            .on_error(|_| {})
            .agent();
            
        let mut extractor = ExtractorImpl::new(agent);
        if let Some(prompt) = self.system_prompt {
            extractor = extractor.with_system_prompt(prompt);
        }
        extractor
    }
    
    // Terminal method - async build
    pub fn build_async(self) -> AsyncTask<impl Extractor<T>>
    where
        ExtractorImpl<T>: crate::async_task::NotResult,
    {
        spawn_async(async move {
            self.build()
        })
    }
    
    // Terminal method - extract from text immediately
    pub fn extract_from_text(self, text: impl Into<String>) -> AsyncTask<T> 
    where
        T: crate::async_task::NotResult,
    {
        let extractor = self.build();
        let text = text.into();
        extractor.extract_from(&text)
    }
}

// Type alias for convenience - constraints defined at use site
pub type DefaultExtractor<T> = ExtractorImpl<T>;