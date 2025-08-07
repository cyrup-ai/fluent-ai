//! Builders are behavioral/construction logic, separate from core domain models

use fluent_ai_async::AsyncStream;
use crate::domain::chat::CandleChatLoop;
use crate::domain::completion::{
    traits::CandleCompletionModel as DomainCompletionModel,
    types::CandleCompletionParams, 
    CandleCompletionChunk,
};
use crate::domain::prompt::CandlePrompt;
use std::num::NonZeroU64;

// Candle domain types - self-contained
/// Trait for AI completion providers (e.g., OpenAI, Anthropic, local models)  
pub trait CandleCompletionProvider: Send + Sync + 'static {}

/// Trait for context sources (files, directories, web pages, etc.)
pub trait CandleContext: Send + Sync + 'static {}

/// Trait for tools that agents can use (CLI tools, APIs, functions, etc.)
pub trait CandleTool: Send + Sync + 'static {}

/// Trait for memory systems (vector stores, knowledge bases, etc.)
pub trait CandleMemory: Send + Sync + 'static {}

/// Default empty implementations for optional components
pub struct NoProvider;
pub struct NoContext;
pub struct NoTool;
pub struct NoMemory;

impl CandleCompletionProvider for NoProvider {}
impl CandleContext for NoContext {}
impl CandleTool for NoTool {}
impl CandleMemory for NoMemory {}

/// Message roles in conversation
#[derive(Debug, Clone)]
pub enum CandleMessageRole {
    User,
    System,
    Assistant,
}

/// Message chunk for streaming
#[derive(Debug, Clone)]
pub enum CandleMessageChunk {
    Text(String),
    Complete { text: String, finish_reason: Option<String>, usage: Option<String> },
}

impl CandleMessageChunk {
    /// Get the text content of this chunk
    pub fn text(&self) -> &str {
        match self {
            CandleMessageChunk::Text(text) => text,
            CandleMessageChunk::Complete { text, .. } => text,
        }
    }
    
    /// Check if this chunk indicates completion
    pub fn done(&self) -> bool {
        matches!(self, CandleMessageChunk::Complete { .. })
    }
}

impl std::fmt::Display for CandleMessageChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandleMessageChunk::Text(text) => write!(f, "{}", text),
            CandleMessageChunk::Complete { text, .. } => write!(f, "{}", text),
        }
    }
}

// CandleChatLoop is now imported from domain::chat



/// Agent conversation
pub struct CandleAgentConversation {
    messages: Vec<CandleMessage>,
    current_user_input: String,
}

impl CandleAgentConversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            current_user_input: String::new(),
        }
    }
    
    pub fn with_user_input(user_input: impl Into<String>) -> Self {
        let input = user_input.into();
        let mut conversation = Self::new();
        conversation.current_user_input = input.clone();
        conversation.messages.push(CandleMessage { 
            content: input,
            role: CandleMessageRole::User,
        });
        conversation
    }
    
    pub fn latest_user_message(&self) -> &str {
        if self.current_user_input.is_empty() {
            "Hello" // Fallback for compatibility
        } else {
            &self.current_user_input
        }
    }
    
    pub fn last(&self) -> CandleMessage {
        self.messages.last().cloned().unwrap_or_else(|| CandleMessage { 
            content: "Hello".to_string(),
            role: CandleMessageRole::User,
        })
    }
    
    pub fn add_message(&mut self, content: impl Into<String>, role: CandleMessageRole) {
        self.messages.push(CandleMessage {
            content: content.into(),
            role,
        });
    }
}

/// Message in conversation
#[derive(Debug, Clone)]
pub struct CandleMessage {
    content: String,
    role: CandleMessageRole,
}

impl CandleMessage {
    pub fn message(&self) -> &str {
        &self.content
    }
    
    pub fn content(&self) -> &str {
        &self.content
    }
    
    pub fn role(&self) -> &CandleMessageRole {
        &self.role
    }
}

/// Agent role agent
pub struct CandleAgentRoleAgent;

/// Agent role builder trait - elegant zero-allocation builder pattern (PUBLIC API)
pub trait CandleAgentRoleBuilder: Sized + Send {
    /// Create a new agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set the completion provider - EXACT syntax: .completion_provider(CandleKimiK2Provider::new())
    #[must_use]
    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: DomainCompletionModel + Clone + Send + 'static;
    
    /// Set model - EXACT syntax: .model(CandleModels::KIMI_K2)
    #[must_use]
    fn model<M>(self, model: M) -> impl CandleAgentRoleBuilder
    where
        M: DomainCompletionModel;
    
    /// Set temperature - EXACT syntax: .temperature(1.0)
    #[must_use]
    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    #[must_use]
    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    #[must_use]
    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set context - EXACT syntax: .context(CandleContext::<File>::of("/path"))
    #[must_use]
    fn context<C>(self, context: C) -> impl CandleAgentRoleBuilder
    where
        C: CandleContext;
    
    /// Set tools - EXACT syntax: .tools(CandleTool::<Perplexity>::new())
    #[must_use]
    fn tools<T>(self, tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| chunk)
    #[must_use]
    fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static;
        
    /// Convert to agent - EXACT syntax: .into_agent()
    #[must_use]
    fn into_agent(self) -> impl CandleAgentBuilder + Send;
}

/// Agent builder trait (PUBLIC API)
pub trait CandleAgentBuilder: Sized + Send + Sync {
    /// Set conversation history
    #[must_use]
    fn conversation_history<H>(self, history: H) -> impl CandleAgentBuilder;
    
    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;
}

/// First builder - no provider yet
#[derive(Debug, Clone)]
struct CandleAgentRoleBuilderImpl {
    name: String,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
}



impl CandleAgentRoleBuilderImpl {
    /// Create a new agent role builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            temperature: None,
            max_tokens: None,
            system_prompt: None,
        }
    }
}

// Implementation for no-provider builder  
impl CandleAgentRoleBuilder for CandleAgentRoleBuilderImpl {
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }

    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: DomainCompletionModel + Clone + Send + 'static,
    {
        CandleAgentBuilderImpl {
            name: self.name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt,
            provider,
        }
    }

    /// Set model - EXACT syntax: .model(CandleModels::KIMI_K2)
    fn model<M>(self, _model: M) -> impl CandleAgentRoleBuilder
    where
        M: DomainCompletionModel,
    {
        self
    }

    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(mut self, temp: f64) -> impl CandleAgentRoleBuilder {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(mut self, max: u64) -> impl CandleAgentRoleBuilder {
        self.max_tokens = Some(max);
        self
    }

    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(mut self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set context - EXACT syntax: .context(CandleContext::<File>::of("/path"))
    fn context<C>(self, _context: C) -> impl CandleAgentRoleBuilder
    where
        C: CandleContext,
    {
        self
    }

    /// Set tools - EXACT syntax: .tools(CandleTool::<Perplexity>::new())
    fn tools<T>(self, _tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool,
    {
        self
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| chunk)
    fn on_chunk<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        self
    }


    /// Convert to agent - EXACT syntax: .into_agent()
    fn into_agent(self) -> impl CandleAgentBuilder + Send {
        // This shouldn't be called for no-provider builder, but return a placeholder
        NoProviderAgent { _inner: self }
    }
}

/// Debug information for agent configuration
#[derive(Debug, Clone)]
pub struct AgentDebugInfo {
    pub name: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub has_system_prompt: bool,
}

/// Placeholder agent for no-provider case
#[derive(Debug, Clone)]
pub struct NoProviderAgent {
    _inner: CandleAgentRoleBuilderImpl,
}

impl CandleAgentBuilder for NoProviderAgent {
    fn conversation_history<H>(self, _history: H) -> impl CandleAgentBuilder {
        self
    }
    
    fn chat<F>(self, _handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            let error_chunk = CandleMessageChunk::Complete {
                text: "Error: No completion provider configured. Use .completion_provider() before .into_agent()".to_string(),
                finish_reason: Some("error".to_string()),
                usage: None,
            };
            let _ = sender.send(error_chunk);
        })
    }
}

/// Agent builder implementation
#[derive(Debug, Clone)]
pub struct CandleAgentBuilderImpl<P> {
    name: String,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    provider: P,
}

// Implementation for with-provider builder (allows all methods)
impl<P> CandleAgentRoleBuilder for CandleAgentBuilderImpl<P>
where
    P: DomainCompletionModel + Clone + Send + 'static,
{
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }

    fn completion_provider<P2>(self, provider: P2) -> impl CandleAgentRoleBuilder
    where
        P2: DomainCompletionModel + Clone + Send + 'static,
    {
        CandleAgentBuilderImpl {
            name: self.name,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt,
            provider,
        }
    }

    fn model<M>(self, _model: M) -> impl CandleAgentRoleBuilder
    where
        M: DomainCompletionModel,
    {
        self
    }

    fn temperature(mut self, temp: f64) -> impl CandleAgentRoleBuilder {
        self.temperature = Some(temp);
        self
    }

    fn max_tokens(mut self, max: u64) -> impl CandleAgentRoleBuilder {
        self.max_tokens = Some(max);
        self
    }

    fn system_prompt(mut self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }

    fn context<C>(self, _context: C) -> impl CandleAgentRoleBuilder
    where
        C: CandleContext,
    {
        self
    }

    fn tools<T>(self, _tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool,
    {
        self
    }

    fn on_chunk<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        self
    }

    fn into_agent(self) -> impl CandleAgentBuilder + Send {
        self
    }
}

impl<P> CandleAgentBuilder for CandleAgentBuilderImpl<P> 
where 
    P: DomainCompletionModel + Clone + Send + 'static,
{
    /// Set conversation history
    fn conversation_history<H>(self, _history: H) -> impl CandleAgentBuilder {
        self
    }

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        let provider = self.provider;
        let temperature = self.temperature.unwrap_or(0.7);
        let max_tokens = self.max_tokens.unwrap_or(1000);
        let system_prompt = self.system_prompt.clone();

        AsyncStream::with_channel(move |sender| {
            // Create initial empty conversation for handler to inspect
            let initial_conversation = CandleAgentConversation::new();
            
            // Execute handler to get CandleChatLoop result  
            let chat_loop_result = handler(&initial_conversation);
            
            // Process CandleChatLoop result
            match chat_loop_result {
                CandleChatLoop::Break => {
                    // User wants to exit - send final chunk
                    let final_chunk = CandleMessageChunk::Complete {
                        text: String::new(),
                        finish_reason: Some("break".to_string()),
                        usage: None,
                    };
                    let _ = sender.send(final_chunk);
                },
                CandleChatLoop::UserPrompt(user_message) | CandleChatLoop::Reprompt(user_message) => {
                    // Create conversation with real user input for this inference
                    let _conversation_with_input = CandleAgentConversation::with_user_input(&user_message);
                    
                    // Create prompt with system prompt if provided
                    let full_prompt = if let Some(sys_prompt) = system_prompt {
                        format!("{}\n\nUser: {}", sys_prompt, user_message)
                    } else {
                        format!("User: {}", user_message)  
                    };
                    
                    // Create CandlePrompt and CandleCompletionParams
                    let prompt = CandlePrompt::new(full_prompt);
                    let params = CandleCompletionParams {
                        temperature,
                        max_tokens: NonZeroU64::new(max_tokens),
                        ..Default::default()
                    };
                    
                    // Call REAL provider inference
                    let completion_stream = provider.prompt(prompt, &params);
                    
                    // Convert CandleCompletionChunk to CandleMessageChunk and forward
                    completion_stream.for_each(|completion_chunk| {
                        let message_chunk = match completion_chunk {
                            CandleCompletionChunk::Text(text) => {
                                CandleMessageChunk::Text(text)
                            },
                            CandleCompletionChunk::Complete { text, finish_reason, usage } => {
                                CandleMessageChunk::Complete {
                                    text,
                                    finish_reason: finish_reason.map(|f| format!("{:?}", f)),
                                    usage: usage.map(|u| format!("{:?}", u)),
                                }
                            },
                            CandleCompletionChunk::ToolCallStart { id, name } => {
                                CandleMessageChunk::Text(format!("[Tool Call Start: {} - {}]", name, id))
                            },
                            CandleCompletionChunk::ToolCall { id, name, partial_input } => {
                                CandleMessageChunk::Text(format!("[Tool Call: {} - {} with args: {}]", name, id, partial_input))
                            },
                            CandleCompletionChunk::ToolCallComplete { id, name, input } => {
                                CandleMessageChunk::Text(format!("[Tool Call Complete: {} - {} result: {}]", name, id, input))
                            },
                            CandleCompletionChunk::Error(error) => {
                                CandleMessageChunk::Complete {
                                    text: String::new(),
                                    finish_reason: Some("error".to_string()),
                                    usage: Some(format!("Error: {}", error)),
                                }
                            },
                        };
                        
                        if sender.send(message_chunk).is_err() {
                            return; // Client disconnected
                        }
                    });
                }
            }
        })
    }
    

}

/// CandleFluentAi entry point for creating agent roles
pub struct CandleFluentAi;

impl CandleFluentAi {
    /// Create a new agent role builder - main entry point
    pub fn agent_role(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }
}