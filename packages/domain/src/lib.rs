//! Fluent AI Domain Library
//!
//! This crate provides core domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here.

// Re-export cyrup_sugars for convenience
pub use cyrup_sugars::{OneOrMany, ZeroOneOrMany, ByteSize};

// Use std HashMap instead
pub use std::collections::HashMap;

// Re-export Models from provider - temporarily commented out due to circular dependency
// pub use fluent_ai_provider::{Models, ModelInfoData};

// Temporary models types to break circular dependency
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Models {
    Gpt35Turbo,
    Gpt4,
    Gpt4O,
    Claude3Opus,
    Claude3Sonnet,
    Claude3Haiku,
    // Add more variants as needed
}

impl Models {
    /// Get model info for this model
    pub fn info(&self) -> ModelInfoData {
        match self {
            Models::Gpt35Turbo => ModelInfoData {
                name: "gpt-3.5-turbo".to_string(),
                provider: "OpenAI".to_string(),
                context_length: Some(16385),
                max_tokens: Some(4096),
            },
            Models::Gpt4 => ModelInfoData {
                name: "gpt-4".to_string(),
                provider: "OpenAI".to_string(),
                context_length: Some(8192),
                max_tokens: Some(4096),
            },
            Models::Gpt4O => ModelInfoData {
                name: "gpt-4o".to_string(),
                provider: "OpenAI".to_string(),
                context_length: Some(128000),
                max_tokens: Some(4096),
            },
            Models::Claude3Opus => ModelInfoData {
                name: "claude-3-opus-20240229".to_string(),
                provider: "Anthropic".to_string(),
                context_length: Some(200000),
                max_tokens: Some(4096),
            },
            Models::Claude3Sonnet => ModelInfoData {
                name: "claude-3-5-sonnet-20241022".to_string(),
                provider: "Anthropic".to_string(),
                context_length: Some(200000),
                max_tokens: Some(8192),
            },
            Models::Claude3Haiku => ModelInfoData {
                name: "claude-3-haiku-20240307".to_string(),
                provider: "Anthropic".to_string(),
                context_length: Some(200000),
                max_tokens: Some(4096),
            },
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfoData {
    pub name: String,
    pub provider: String,
    pub context_length: Option<u32>,
    pub max_tokens: Option<u32>,
}

// Define our own async task types
pub type AsyncTask<T> = tokio::task::JoinHandle<T>;

pub fn spawn_async<F, T>(future: F) -> AsyncTask<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(future)
}

// Create async_task module for compatibility
pub mod async_task {
    pub use super::{AsyncTask, spawn_async};
    
    // Use the actual AsyncStream implementation from fluent-ai
    pub use futures::stream::Stream;
    use tokio::sync::mpsc::UnboundedReceiver;
    
    /// Zero-allocation async stream implementation
    pub struct AsyncStream<T> {
        receiver: UnboundedReceiver<T>,
    }
    
    impl<T> AsyncStream<T> {
        /// Create a new AsyncStream from a tokio mpsc receiver
        #[inline(always)]
        pub fn new(receiver: UnboundedReceiver<T>) -> Self {
            Self { receiver }
        }
        
        /// Create an empty stream
        #[inline(always)]
        pub fn empty() -> Self {
            let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
            Self::new(rx)
        }
        
        /// Create a stream from a single item
        #[inline(always)]
        pub fn from_single(item: T) -> Self {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let _ = tx.send(item);
            Self::new(rx)
        }
    }
    
    impl<T> Stream for AsyncStream<T> {
        type Item = T;
        
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            use std::pin::Pin;
            Pin::new(&mut self.receiver).poll_recv(cx)
        }
    }
    
    // Define NotResult trait for compatibility
    pub trait NotResult {}
    impl<T> NotResult for T where T: Send + 'static {}
    
    // Error handlers module
    pub mod error_handlers {
        pub fn default_error_handler<T: std::fmt::Debug>(_error: T) {
            // Default error handler implementation
        }
        
        /// Trait for implementing fallback behavior when operations fail
        pub trait BadTraitImpl {
            fn bad_impl(error: &str) -> Self;
        }
    }
}

// Re-export streaming types
pub use futures::stream::Stream as AsyncStream;

// Domain modules
pub mod agent;
pub mod agent_role;
pub mod audio;
pub mod chunk;
pub mod completion;
pub mod context;
pub mod conversation;
pub mod chat;
pub mod engine;
pub mod document;
pub mod embedding;
pub mod extractor;
pub mod image;
pub mod library;
pub mod loader;
pub mod mcp;
pub mod mcp_tool;
pub mod memory;
pub mod memory_ops;
pub mod memory_workflow;
pub mod message;
pub mod model;
pub mod model_info_provider;
pub mod prompt;
pub mod provider;
pub mod tool;
pub mod tool_v2;
pub mod workflow;

// Re-export all types for convenience
// Handle conflicting types by using specific imports to avoid ambiguity

// Agent module exports
pub use agent::{Agent, AgentBuilder};
pub use agent_role::*;

// Audio module exports - specify ContentFormat to avoid conflict with image
pub use audio::{Audio, AudioBuilder, AudioBuilderWithHandler, AudioMediaType};
pub use audio::ContentFormat as AudioContentFormat;

// Chunk module exports
pub use chunk::*;

// Completion module exports - specify ToolDefinition to avoid conflict with tool
pub use completion::{CompletionModel, CompletionBackend, CompletionRequest, CompletionRequestBuilder, CompletionRequestBuilderWithHandler};
pub use completion::ToolDefinition as CompletionToolDefinition;

// Context module exports
pub use context::*;

// Conversation module exports - specify types to avoid conflict with message
pub use conversation::{ConversationImpl, ConversationBuilder, ConversationBuilderWithHandler};
pub use conversation::Conversation as ConversationTrait;

// Document module exports
pub use document::*;

// Embedding module exports
pub use embedding::*;

// Extractor module exports
pub use extractor::*;

// Image module exports - specify ContentFormat to avoid conflict with audio
pub use image::{Image, ImageBuilder, ImageBuilderWithHandler, ImageMediaType, ImageDetail};
pub use image::ContentFormat as ImageContentFormat;

// Library module exports
pub use library::*;

// Loader module exports
pub use loader::*;

// MCP module exports - specify Tool to avoid conflict with mcp_tool
pub use mcp::{McpError, Transport, StdioTransport, Client, McpClient, McpClientBuilder};
pub use mcp::Tool as McpTool;

// MCP Tool module exports - specify Tool to avoid conflict with mcp
pub use mcp_tool::{McpToolImpl, McpToolBuilder, McpToolBuilderWithHandler};
pub use mcp_tool::Tool as McpToolTrait;

// Memory module exports
pub use memory::*;

// Memory ops module exports
pub use memory_ops::*;

// Memory workflow module exports - specify Prompt to avoid conflict with prompt
pub use memory_workflow::{MemoryEnhancedWorkflow, WorkflowError, AdaptiveWorkflow, conversation_workflow, apply_feedback, rag_workflow};
pub use memory_workflow::Prompt as MemoryWorkflowPrompt;

// Message module exports - specify Conversation to avoid conflict with conversation
pub use message::{MessageError, ToolFunction, MimeType, ToolCall, ToolResultContent, Text, UserContent, AssistantContent, MessageRole, Message, MessageChunk, UserContentExt, AssistantContentExt, Content, ConversationMap, ToolResult, MessageBuilder, UserMessageBuilderTrait, AssistantMessageBuilderTrait, MessageFactory, ContentContainer};
pub use message::Conversation as MessageConversation;

// Model module exports
pub use model::*;

// Model info provider module exports
pub use model_info_provider::*;

// Prompt module exports - specify Prompt to avoid conflict with memory_workflow
pub use prompt::{PromptBuilder};
pub use prompt::Prompt as PromptStruct;

// Provider module exports
pub use provider::*;

// Tool module exports - specify ToolDefinition to avoid conflict with completion
pub use tool::{ToolSet, NamedTool, ExecToText, ToolEmbeddingDyn};
pub use tool::Tool as ToolGeneric;
pub use tool::ToolDefinition as ToolDefinitionEnum;

// Workflow module exports
pub use workflow::*;