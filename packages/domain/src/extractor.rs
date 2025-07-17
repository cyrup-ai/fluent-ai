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

// Builder implementations moved to fluent_ai/src/builders/extractor.rs

// Type alias for convenience - constraints defined at use site
pub type DefaultExtractor<T> = ExtractorImpl<T>;