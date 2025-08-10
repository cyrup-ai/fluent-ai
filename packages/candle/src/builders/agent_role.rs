//! Builders are behavioral/construction logic, separate from core domain models

use std::num::NonZeroU64;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_async::AsyncStream;

use crate::domain::chat::message::{CandleMessageChunk, CandleMessageRole};
use crate::domain::chat::CandleChatLoop;
use crate::domain::completion::{
    traits::CandleCompletionModel as DomainCompletionModel, types::CandleCompletionParams,
    CandleCompletionChunk,
};
use crate::domain::context::provider::{
    CandleContext, CandleDirectory, CandleFile, CandleFiles, CandleGithub,
};
use crate::domain::prompt::CandlePrompt;
use crate::domain::tool::traits::CandleTool;

// Candle domain types - self-contained
/// Trait for AI completion providers (e.g., OpenAI, Anthropic, local models)  
pub trait CandleCompletionProvider: Send + Sync + 'static {}

/// Default empty implementations for optional components
pub struct NoProvider;

impl CandleCompletionProvider for NoProvider {}

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
        self.messages
            .last()
            .cloned()
            .unwrap_or_else(|| CandleMessage {
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

impl CandleAgentRoleAgent {
    /// Chat method for use in on_conversation_turn closure
    pub fn chat(&self, chat_loop: CandleChatLoop) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(|sender| match chat_loop {
            CandleChatLoop::Break => {
                let final_chunk = CandleMessageChunk::Complete(
                    String::new(),
                    Some("break".to_string()),
                    None,
                );
                let _ = sender.send(final_chunk);
            }
            CandleChatLoop::UserPrompt(message) | CandleChatLoop::Reprompt(message) => {
                let _ = sender.send(CandleMessageChunk::Text(message));
            }
        })
    }
}

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

    /// Set additional params - EXACT syntax: .additional_params([("key", "value")])
    #[must_use]
    fn additional_params<P>(self, params: P) -> impl CandleAgentRoleBuilder;

    /// Set memory - EXACT syntax: .memory(CandleLibrary::named("name"))
    #[must_use]
    fn memory<M>(self, memory: M) -> impl CandleAgentRoleBuilder;

    /// Set metadata - EXACT syntax: .metadata([("key", "value")])
    #[must_use]
    fn metadata<Meta>(self, metadata: Meta) -> impl CandleAgentRoleBuilder;

    /// Set contexts - EXACT syntax: .context(CandleContext::<CandleFile>::of("/path"), CandleContext::<CandleFiles>::glob("*.rs"), ...)
    #[must_use]
    fn context(
        self,
        context1: CandleContext<CandleFile>,
        context2: CandleContext<CandleFiles>,
        context3: CandleContext<CandleDirectory>,
        context4: CandleContext<CandleGithub>,
    ) -> impl CandleAgentRoleBuilder;

    /// Set tools - EXACT syntax: .tools(CandleTool::<Perplexity>::new())
    #[must_use]
    fn tools<T>(self, tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool;

    /// Set MCP server - EXACT syntax: .mcp_server::<Stdio>().bin("/path").init("command")
    #[must_use]
    fn mcp_server<T>(self) -> impl CandleMcpServerBuilder
    where
        T: 'static;

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| chunk)
    #[must_use]
    fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static;

    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    #[must_use]
    fn on_tool_result<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&[String]) + Send + Sync + 'static;

    /// Set conversation turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    #[must_use]
    fn on_conversation_turn<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) -> AsyncStream<CandleMessageChunk>
            + Send
            + Sync
            + 'static;

    /// Convert to agent - EXACT syntax: .into_agent()
    #[must_use]
    fn into_agent(self) -> impl CandleAgentBuilder + Send;

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;
}

/// MCP server builder for fluent chaining
pub trait CandleMcpServerBuilder: Sized + Send {
    /// Set binary path - EXACT syntax: .bin("/path/to/binary")
    #[must_use]
    fn bin(self, path: impl Into<String>) -> impl CandleMcpServerBuilder;

    /// Initialize with command - EXACT syntax: .init("command args")
    #[must_use]
    fn init(self, command: impl Into<String>) -> impl CandleAgentRoleBuilder;
}

/// Agent builder trait (PUBLIC API)
pub trait CandleAgentBuilder: Sized + Send + Sync {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(CandleMessageRole::User => "content", CandleMessageRole::System => "content", ...)
    #[must_use]
    fn conversation_history(self, history: impl ConversationHistoryArgs)
        -> impl CandleAgentBuilder;

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;

    /// Chat with direct input - EXACT syntax: .chat_direct(ChatLoop)
    fn chat_direct(self, input: CandleChatLoop) -> AsyncStream<CandleMessageChunk>;

    /// Chat with message - EXACT syntax: .chat_with_message("message")
    fn chat_with_message(self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk>;

    /// Chat with message - EXACT syntax: .chat_with_message("message")
    fn chat_with_message(self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk>;
}

/// MCP server builder implementation
#[derive(Debug, Clone)]
pub struct CandleMcpServerBuilderImpl<T> {
    parent_builder: T,
    binary_path: Option<String>,
}

impl<T> CandleMcpServerBuilder for CandleMcpServerBuilderImpl<T>
where
    T: CandleAgentRoleBuilder + Clone,
{
    fn bin(mut self, path: impl Into<String>) -> impl CandleMcpServerBuilder {
        self.binary_path = Some(path.into());
        self
    }

    fn init(self, _command: impl Into<String>) -> impl CandleAgentRoleBuilder {
        // For now, just return the parent builder - in real implementation would configure MCP
        self.parent_builder
    }
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

    /// Set additional params - EXACT syntax: .additional_params([("key", "value")])
    fn additional_params<P>(self, _params: P) -> impl CandleAgentRoleBuilder {
        self
    }

    /// Set memory - EXACT syntax: .memory(CandleLibrary::named("name"))
    fn memory<M>(self, _memory: M) -> impl CandleAgentRoleBuilder {
        self
    }

    /// Set metadata - EXACT syntax: .metadata([("key", "value")])
    fn metadata<Meta>(self, _metadata: Meta) -> impl CandleAgentRoleBuilder {
        self
    }

    /// Set contexts - EXACT syntax: .context(CandleContext::<CandleFile>::of("/path"), ...)
    fn context(
        self,
        _context1: CandleContext<CandleFile>,
        _context2: CandleContext<CandleFiles>,
        _context3: CandleContext<CandleDirectory>,
        _context4: CandleContext<CandleGithub>,
    ) -> impl CandleAgentRoleBuilder {
        self
    }

    /// Set tools - EXACT syntax: .tools(CandleTool::<Perplexity>::new())
    fn tools<T>(self, _tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool,
    {
        self
    }

    /// Set MCP server - EXACT syntax: .mcp_server::<Stdio>().bin("/path").init("command")
    fn mcp_server<T>(self) -> impl CandleMcpServerBuilder
    where
        T: 'static,
    {
        CandleMcpServerBuilderImpl {
            parent_builder: self,
            binary_path: None,
        }
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| chunk)
    fn on_chunk<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        self
    }

    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    fn on_tool_result<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&[String]) + Send + Sync + 'static,
    {
        self
    }

    /// Set conversation turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    fn on_conversation_turn<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) -> AsyncStream<CandleMessageChunk>
            + Send
            + Sync
            + 'static,
    {
        self
    }

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, _handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        AsyncStream::with_channel(|sender| {
            let _ = sender.send(CandleMessageChunk::Error(
                "No provider configured".to_string(),
            ));
        })
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
    fn conversation_history(
        self,
        _history: impl ConversationHistoryArgs,
    ) -> impl CandleAgentBuilder {
        self
    }

    fn chat<F>(self, _handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            let error_chunk = CandleMessageChunk::Error("No completion provider configured. Use .completion_provider() before .into_agent()".to_string());
            let _ = sender.send(error_chunk);
        })
    }

    fn chat_direct(self, _input: CandleChatLoop) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(move |sender| {
            let error_chunk = CandleMessageChunk::Error("No completion provider configured. Use .completion_provider() before .into_agent()".to_string());
            let _ = sender.send(error_chunk);
        })
    }

    fn chat_with_message(self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(move |sender| {
            let error_chunk = CandleMessageChunk::Error(format!("No completion provider configured for message: {}. Use .completion_provider() before .into_agent()", message.into()));
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

    fn additional_params<P2>(self, _params: P2) -> impl CandleAgentRoleBuilder {
        self
    }

    fn memory<M>(self, _memory: M) -> impl CandleAgentRoleBuilder {
        self
    }

    fn metadata<Meta>(self, _metadata: Meta) -> impl CandleAgentRoleBuilder {
        self
    }

    fn context(
        self,
        _context1: CandleContext<CandleFile>,
        _context2: CandleContext<CandleFiles>,
        _context3: CandleContext<CandleDirectory>,
        _context4: CandleContext<CandleGithub>,
    ) -> impl CandleAgentRoleBuilder {
        self
    }

    fn tools<T>(self, _tools: T) -> impl CandleAgentRoleBuilder
    where
        T: CandleTool,
    {
        self
    }

    fn mcp_server<T>(self) -> impl CandleMcpServerBuilder
    where
        T: 'static,
    {
        CandleMcpServerBuilderImpl {
            parent_builder: self,
            binary_path: None,
        }
    }

    fn on_chunk<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        self
    }

    fn on_tool_result<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&[String]) + Send + Sync + 'static,
    {
        self
    }

    fn on_conversation_turn<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) -> AsyncStream<CandleMessageChunk>
            + Send
            + Sync
            + 'static,
    {
        self
    }

    /// Chat with closure - EXACT syntax: .chat(|conversation| ChatLoop)
    fn chat<F>(self, _handler: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        AsyncStream::with_channel(|sender| {
            let _ = sender.send(CandleMessageChunk::Text("Hello from Candle!".to_string()));
        })
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
    fn conversation_history(
        self,
        _history: impl ConversationHistoryArgs,
    ) -> impl CandleAgentBuilder {
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
                    let final_chunk = CandleMessageChunk::Complete(
                        String::new(),
                        Some("break".to_string()),
                        None,
                    );
                    let _ = sender.send(final_chunk);
                }
                CandleChatLoop::UserPrompt(user_message)
                | CandleChatLoop::Reprompt(user_message) => {
                    // Create conversation with real user input for this inference
                    let _conversation_with_input =
                        CandleAgentConversation::with_user_input(&user_message);

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
                            CandleCompletionChunk::Text(text) => CandleMessageChunk::Text(text),
                            CandleCompletionChunk::Complete {
                                text,
                                finish_reason,
                                usage,
                            } => CandleMessageChunk::Complete(
                                text,
                                finish_reason.map(|f| format!("{:?}", f)),
                                usage.map(|u| format!("{:?}", u)),
                            ),
                            CandleCompletionChunk::ToolCallStart { id, name } => {
                                CandleMessageChunk::ToolCallStart(id, name)
                            }
                            CandleCompletionChunk::ToolCall {
                                id,
                                name,
                                partial_input,
                            } => CandleMessageChunk::ToolCall(
                                id,
                                name,
                                partial_input,
                            ),
                            CandleCompletionChunk::ToolCallComplete { id, name, input } => {
                                CandleMessageChunk::ToolCallComplete(id, name, input)
                            }
                            CandleCompletionChunk::Error(error) => CandleMessageChunk::Error(error),
                        };

                        if sender.send(message_chunk).is_err() {
                            return; // Client disconnected
                        }
                    });
                }
            }
        })
    }

    fn chat_direct(self, input: CandleChatLoop) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(move |sender| match input {
            CandleChatLoop::Break => {
                let final_chunk = CandleMessageChunk::Complete(
                    String::new(),
                    Some("break".to_string()),
                    None,
                );
                let _ = sender.send(final_chunk);
            }
            CandleChatLoop::UserPrompt(message) | CandleChatLoop::Reprompt(message) => {
                let _ = sender.send(CandleMessageChunk::Text(message));
            }
        })
    }

    fn chat_with_message(self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
        let provider = self.provider;
        let temperature = self.temperature.unwrap_or(0.7);
        let max_tokens = self.max_tokens.unwrap_or(1000);
        let system_prompt = self.system_prompt.clone();
        let user_message = message.into();

        AsyncStream::with_channel(move |sender| {
            let full_prompt = if let Some(sys_prompt) = system_prompt {
                format!("{}\n\nUser: {}", sys_prompt, user_message)
            } else {
                format!("User: {}", user_message)
            };

            let prompt = CandlePrompt::new(full_prompt);
            let params = CandleCompletionParams {
                temperature,
                max_tokens: NonZeroU64::new(max_tokens),
                ..Default::default()
            };

            let completion_stream = provider.prompt(prompt, &params);

            completion_stream.for_each(|completion_chunk| {
                let message_chunk = match completion_chunk {
                    CandleCompletionChunk::Text(text) => CandleMessageChunk::Text(text),
                    CandleCompletionChunk::Complete {
                        text,
                        finish_reason,
                        usage,
                    } => CandleMessageChunk::Complete(
                        text,
                        finish_reason.map(|f| format!("{:?}", f)),
                        usage.map(|u| format!("{:?}", u)),
                    ),
                    CandleCompletionChunk::ToolCallStart { id, name } => {
                        CandleMessageChunk::ToolCallStart(id, name)
                    }
                    CandleCompletionChunk::ToolCall {
                        id,
                        name,
                        partial_input,
                    } => CandleMessageChunk::ToolCall(
                        id,
                        name,
                        partial_input,
                    ),
                    CandleCompletionChunk::ToolCallComplete { id, name, input } => {
                        CandleMessageChunk::ToolCallComplete(id, name, input)
                    }
                    CandleCompletionChunk::Error(error) => CandleMessageChunk::Error(error),
                };

                if sender.send(message_chunk).is_err() {
                    return;
                }
            });
        })
    }
}

// ConversationHistoryArgs implementations for => syntax
// Enables: .conversation_history(CandleMessageRole::User => "What time is it in Paris, France", CandleMessageRole::System => "...", CandleMessageRole::Assistant => "...")

/// Trait for conversation history arguments supporting arrow syntax
pub trait ConversationHistoryArgs {
    /// Convert this into conversation history format
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>>;
}

impl ConversationHistoryArgs for (CandleMessageRole, &str) {
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(ZeroOneOrMany::one((self.0, self.1.to_string())))
    }
}

impl ConversationHistoryArgs for (CandleMessageRole, String) {
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(ZeroOneOrMany::one(self))
    }
}

impl<T1, T2> ConversationHistoryArgs for (T1, T2)
where
    T1: ConversationHistoryArgs,
    T2: ConversationHistoryArgs,
{
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        match (self.0.into_history(), self.1.into_history()) {
            (Some(h1), Some(h2)) => {
                let mut combined = h1;
                for item in h2.into_iter() {
                    combined = combined.with_pushed(item);
                }
                Some(combined)
            }
            (Some(h), None) | (None, Some(h)) => Some(h),
            (None, None) => None,
        }
    }
}

impl<T1, T2, T3> ConversationHistoryArgs for (T1, T2, T3)
where
    T1: ConversationHistoryArgs,
    T2: ConversationHistoryArgs,
    T3: ConversationHistoryArgs,
{
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        let (h1, h2, h3) = (
            self.0.into_history(),
            self.1.into_history(),
            self.2.into_history(),
        );
        match (h1, h2, h3) {
            (Some(mut combined), h2_opt, h3_opt) => {
                if let Some(h2) = h2_opt {
                    for item in h2.into_iter() {
                        combined = combined.with_pushed(item);
                    }
                }
                if let Some(h3) = h3_opt {
                    for item in h3.into_iter() {
                        combined = combined.with_pushed(item);
                    }
                }
                Some(combined)
            }
            (None, Some(mut combined), h3_opt) => {
                if let Some(h3) = h3_opt {
                    for item in h3.into_iter() {
                        combined = combined.with_pushed(item);
                    }
                }
                Some(combined)
            }
            (None, None, Some(h3)) => Some(h3),
            (None, None, None) => None,
        }
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
