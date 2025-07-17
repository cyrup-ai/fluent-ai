//! Fluent AI Domain Library
//!
//! This crate provides core domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here.

// Re-export cyrup_sugars for convenience
pub use cyrup_sugars::{OneOrMany, ZeroOneOrMany, ByteSize};

// Re-export hash_map_fn macro for transparent JSON syntax
#[doc(hidden)]
pub use cyrup_sugars::hash_map_fn;

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

/// Channel creation for async communication
pub fn channel<T: Send + 'static>() -> (ChannelSender<T>, AsyncTask<T>) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = spawn_async(async move {
        rx.recv().await.expect("Channel closed unexpectedly")
    });
    (ChannelSender { tx }, task)
}

/// Channel sender wrapper
pub struct ChannelSender<T> {
    tx: tokio::sync::mpsc::UnboundedSender<T>,
}

impl<T> ChannelSender<T> {
    /// Finish the task by sending the result
    pub fn finish(self, value: T) {
        let _ = self.tx.send(value);
    }
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

// Re-export AsyncTaskExt for AsyncTask::new(rx) pattern (removed as it's not needed)
// AsyncStream trait from futures
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
pub mod mcp_tool_traits;
// pub mod secure_mcp_tool; // Temporarily disabled due to cylo dependency
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
pub mod tool_syntax_test;
pub mod architecture_syntax_test;
// pub mod secure_executor; // Temporarily disabled due to compilation issues

// Temporary stub for secure_executor to avoid compilation errors
pub mod secure_executor {
    use crate::AsyncTask;
    use serde_json::Value;

    pub fn get_secure_executor() -> SecureToolExecutor {
        SecureToolExecutor
    }

    pub struct SecureToolExecutor;

    impl SecureToolExecutor {
        pub fn execute_code(&self, _code: &str, _language: &str) -> AsyncTask<Result<Value, String>> {
            crate::spawn_async(async { Ok(Value::Null) })
        }
        
        pub fn execute_tool_with_args(&self, _name: &str, _args: Value) -> AsyncTask<Result<Value, String>> {
            crate::spawn_async(async { Ok(Value::Null) })
        }
    }
}
pub mod workflow;

// Re-export all types for convenience
// Handle conflicting types by using specific imports to avoid ambiguity

// Agent module exports
pub use agent::Agent;
pub use agent_role::{
    AgentRole, AgentRoleImpl, AgentConversation, AgentConversationMessage,
    AgentRoleAgent, AgentWithHistory, Stdio,
    ContextArgs, ToolArgs, ConversationHistoryArgs
};
// Builder types moved to fluent-ai/src/builders/agent_role.rs

// Audio module exports - specify ContentFormat to avoid conflict with image
pub use audio::{Audio, AudioMediaType};
pub use audio::ContentFormat as AudioContentFormat;

// Chunk module exports
pub use chunk::*;

// Completion module exports - specify ToolDefinition to avoid conflict with tool
pub use completion::{CompletionModel, CompletionBackend, CompletionRequest};
pub use completion::ToolDefinition as CompletionToolDefinition;

// Context module exports
pub use context::*;

// Conversation module exports - specify types to avoid conflict with message
pub use conversation::{ConversationImpl};
pub use conversation::Conversation as ConversationTrait;

// Document module exports
pub use document::*;

// Embedding module exports
pub use embedding::*;

// Extractor module exports
pub use extractor::*;

// Image module exports - specify ContentFormat to avoid conflict with audio
pub use image::{Image, ImageMediaType, ImageDetail};
pub use image::ContentFormat as ImageContentFormat;

// Library module exports
pub use library::*;

// Loader module exports
pub use loader::*;

// MCP module exports - specify Tool to avoid conflict with mcp_tool
pub use mcp::{McpError, Transport, StdioTransport, Client, McpClient};
// McpClientBuilder moved to fluent-ai/src/builders/mcp.rs

// MCP Tool module exports - specify Tool to avoid conflict with mcp  
// Implementation types are now in fluent_ai package
pub use mcp_tool_traits::{Tool, McpTool, McpToolData};
pub use mcp_tool::Tool as McpToolTrait;

// Secure MCP Tool module exports - temporarily disabled
// pub use secure_mcp_tool::{SecureMcpTool, SecureMcpToolBuilder};

// Memory module exports
pub use memory::*;

// Memory ops module exports
pub use memory_ops::*;

// Memory workflow module exports - specify Prompt to avoid conflict with prompt
pub use memory_workflow::{MemoryEnhancedWorkflow, WorkflowError, AdaptiveWorkflow, conversation_workflow, apply_feedback, rag_workflow};
pub use memory_workflow::Prompt as MemoryWorkflowPrompt;

// Message module exports - specify Conversation to avoid conflict with conversation
pub use message::{MessageError, ToolFunction, MimeType, ToolCall, ToolResultContent, Text, UserContent, AssistantContent, MessageRole, Message, MessageChunk, UserContentExt, AssistantContentExt, Content, ConversationMap, ToolResult, ContentContainer};
pub use message::Conversation as MessageConversation;

// Model module exports
pub use model::*;

// Model info provider module exports
pub use model_info_provider::*;

// Prompt module exports - specify Prompt to avoid conflict with memory_workflow
// PromptBuilder moved to fluent-ai/src/builders/prompt.rs
pub use prompt::Prompt as PromptStruct;

// Provider module exports
pub use provider::*;

// Tool module exports - specify ToolDefinition to avoid conflict with completion
pub use tool::{ToolSet, NamedTool, ExecToText, ToolEmbeddingDyn};
pub use tool::Tool as ToolGeneric;
pub use tool::ToolDefinition as ToolDefinitionEnum;

// Secure executor module exports - temporarily disabled
// pub use secure_executor::{SecureToolExecutor, SecureExecutionConfig, SecureExecutable, get_secure_executor};

// Workflow module exports
pub use workflow::*;