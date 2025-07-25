//! Builder implementations for fluent-ai
//!
//! All builder traits and implementations belong in this module.
//! Domain contains only pure data structures and interfaces.

// Core builders
pub mod agent;
pub mod agent_role;
pub mod audio;
pub mod chunk;
pub mod completion;
pub mod conversation;
pub mod document;
pub mod embedding;
pub mod extractor;
pub mod image;
pub mod loader;
pub mod mcp;
pub mod mcp_client;
pub mod mcp_tool;
pub mod memory;
pub mod memory_node;
pub mod memory_workflow;
pub mod message;
pub mod model;
pub mod prompt;
pub mod secure_mcp_tool;
pub mod workflow;

// Re-export all builder traits and types
pub use agent::*;
pub use agent_role::{
    AgentRoleBuilder, AgentRoleBuilderWithChunkHandler, AgentRoleBuilderWithHandler,
    McpServerBuilder, Stdio};
pub use audio::{AudioBuilder, AudioBuilderWithHandler};
pub use chunk::*;
pub use completion::*;
pub use conversation::ConversationBuilder;
pub use document::*;
pub use embedding::*;
pub use extractor::*;
pub use image::{ImageBuilder, ImageBuilderWithHandler};
pub use loader::*;
pub use mcp::McpClientBuilder;
pub use mcp_client::*;
pub use mcp_tool::{McpToolBuilder, McpToolImpl};
pub use memory::{VectorQueryBuilder, VectorStoreIndexExt};
pub use memory_node::*;
pub use memory_workflow::*;
pub use message::*;
pub use model::ModelInfoBuilder;
pub use prompt::PromptBuilder;
pub use secure_mcp_tool::SecureMcpToolBuilder;
pub use workflow::WorkflowBuilder;
