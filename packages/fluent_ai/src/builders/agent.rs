//! Agent builder implementations
//!
//! All agent construction logic and builder patterns.

use fluent_ai_domain::{Models, Document, Memory, Message, MessageRole, ZeroOneOrMany, ByteSize};
use fluent_ai_domain::{AsyncTask, spawn_async, AsyncStream};
use fluent_ai_domain::mcp_tool::McpToolData;
use fluent_ai_domain::chunk::{ChatMessageChunk, CompletionChunk};
use fluent_ai_domain::completion::CompletionRequest;
use fluent_ai_domain::conversation::Conversation;
use fluent_ai_domain::agent::Agent;
use serde_json::Value;

pub struct AgentBuilder {
    model: Models,
    system_prompt: String,
    context: Option<ZeroOneOrMany<Document>>,
    tools: ZeroOneOrMany<McpToolData>,
    memory: Option<Memory>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<Value>,
}

pub struct AgentBuilderWithHandler {
    model: Models,
    system_prompt: String,
    context: Option<ZeroOneOrMany<Document>>,
    tools: ZeroOneOrMany<McpToolData>,
    memory: Option<Memory>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<Value>,
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    result_handler: Option<Box<dyn FnOnce(Agent) -> Agent + Send + 'static>>,
    chunk_handler: Option<Box<dyn FnMut(Agent) -> Agent + Send + 'static>>,
}

impl Agent {
    pub fn with_model(model: Models) -> AgentBuilder {
        AgentBuilder {
            model,
            system_prompt: String::new(),
            context: None,
            tools: ZeroOneOrMany::None,
            memory: None,
            temperature: None,
            max_tokens: None,
            additional_params: None,
        }
    }
}

impl AgentBuilder {
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    pub fn add_context(mut self, document: Document) -> Self {
        match self.context {
            Some(existing) => {
                self.context = Some(existing.with_pushed(document));
            }
            None => {
                self.context = Some(ZeroOneOrMany::one(document));
            }
        }
        self
    }

    pub fn tool(mut self, tool: McpToolData) -> Self {
        self.tools = self.tools.with_pushed(tool);
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn on_error<F>(self, handler: F) -> AgentBuilderWithHandler
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        AgentBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            context: self.context,
            tools: self.tools,
            memory: self.memory,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None,
        }
    }
}

impl AgentBuilderWithHandler {
    pub fn agent(self) -> Agent {
        Agent {
            model: self.model,
            system_prompt: self.system_prompt,
            context: self.context.unwrap_or_else(|| ZeroOneOrMany::one(Document::from_text("").load())),
            tools: self.tools,
            memory: self.memory,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
        }
    }

    pub fn chat_message(self, message: impl Into<String>) -> impl AsyncStream<Item = ChatMessageChunk> {
        let message = message.into();
        
        // Create channel for streaming chunks
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn task to handle simple chat
        tokio::spawn(async move {
            let user_chunk = ChatMessageChunk::new(message.clone(), MessageRole::User);
            let _ = tx.send(user_chunk);

            let response_chunk = ChatMessageChunk::new(
                format!("Echo: {}", message),
                MessageRole::Assistant,
            );
            let _ = tx.send(response_chunk);
        });

        fluent_ai_domain::async_task::AsyncStream::new(rx)
    }

    pub fn stream_completion(self, prompt: impl Into<String>) -> impl AsyncStream<Item = CompletionChunk> {
        let _agent = self.agent();
        let _prompt = prompt.into();
        
        // Create empty stream for now
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        fluent_ai_domain::async_task::AsyncStream::new(rx)
    }

    pub fn on_response<F>(self, _message: impl Into<String>, handler: F) -> AsyncTask<String>
    where
        F: FnOnce(Result<String, String>) -> String + Send + 'static,
    {
        let _agent = self.agent();
        spawn_async(async move {
            let response = "Placeholder response".to_string();
            handler(Ok(response))
        })
    }
}