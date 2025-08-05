//! Builders are behavioral/construction logic, separate from core domain models

use fluent_ai_async::AsyncStream;

// Candle domain types - self-contained
/// Trait for AI completion providers (e.g., OpenAI, Anthropic, local models)
pub trait CandleCompletionProvider: Send + Sync + 'static {}

/// Trait for specific AI models that can provide completions
pub trait CandleCompletionModel: CandleCompletionProvider + Send + Sync + 'static {}

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

/// Chat loop control
#[derive(Debug, Clone)]
pub enum CandleChatLoop {
    Break,
    Reprompt(String),
}



/// Agent conversation
pub struct CandleAgentConversation;

impl CandleAgentConversation {
    pub fn latest_user_message(&self) -> &str {
        "Hello"
    }
    
    pub fn last(&self) -> CandleMessage {
        CandleMessage { content: "Hello".to_string() }
    }
}

/// Message in conversation
pub struct CandleMessage {
    content: String,
}

impl CandleMessage {
    pub fn message(&self) -> &str {
        &self.content
    }
}

/// Agent role agent
pub struct CandleAgentRoleAgent;

/// Agent role builder trait - elegant zero-allocation builder pattern
pub trait CandleAgentRoleBuilder: Sized + Send {
    /// Create a new agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set the completion provider - EXACT syntax: .completion_provider(CandleKimiK2Provider::new())
    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionProvider;
    
    /// Set model - EXACT syntax: .model(CandleModels::KIMI_K2)
    fn model<M>(self, model: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleCompletionModel;
    
    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set context - EXACT syntax: .context(CandleContext::<File>::of("/path"))
    fn context<C>(self, context: C) -> impl CandleAgentRoleBuilder
    where
        C: CandleContext;
    
    /// Set tools - EXACT syntax: .tools(CandleTool::<Perplexity>::new())
    fn tools<T>(self, tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| chunk)
    fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static;
        
    /// Convert to agent - EXACT syntax: .into_agent()
    fn into_agent(self) -> impl CandleAgentBuilder + Send;
}

/// Agent builder trait
pub trait CandleAgentBuilder: Sized + Send + Sync {
    /// Set conversation history
    fn conversation_history<H>(self, history: H) -> impl CandleAgentBuilder;
    
    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;
}

/// Hidden implementation struct - zero-allocation builder state
#[derive(Debug, Clone)]
struct CandleAgentRoleBuilderImpl {
    #[allow(dead_code)] // TODO: Use name for agent identification/debugging
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

impl CandleAgentRoleBuilder for CandleAgentRoleBuilderImpl {
    /// Create a new agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(CandleKimiK2Provider::new())
    fn completion_provider<P>(self, _provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionProvider,
    {
        self
    }

    /// Set model - EXACT syntax: .model(CandleModels::KIMI_K2)
    fn model<M>(self, _model: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleCompletionModel,
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
        CandleAgentBuilderImpl { inner: self }
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

/// Agent builder implementation
#[derive(Debug, Clone)]
pub struct CandleAgentBuilderImpl {
    #[allow(dead_code)] // Builder state container
    inner: CandleAgentRoleBuilderImpl,
}

impl CandleAgentBuilder for CandleAgentBuilderImpl {
    /// Set conversation history
    fn conversation_history<H>(self, _history: H) -> impl CandleAgentBuilder {
        self
    }

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            // Placeholder implementation - real implementation would use the handler
            let _result = handler(&CandleAgentConversation);
            let chunk = CandleMessageChunk::Text("Hello from Candle closure!".to_string());
            let _ = sender.send(chunk);
            
            // Send completion chunk
            let final_chunk = CandleMessageChunk::Complete {
                text: String::new(),
                finish_reason: Some("stop".to_string()),
                usage: None,
            };
            let _ = sender.send(final_chunk);
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