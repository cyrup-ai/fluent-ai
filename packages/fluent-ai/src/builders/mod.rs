// Builders module - separated from domain types for clean architecture
// This module contains all builder patterns for fluent-ai types

// Extracted builders from domain
pub mod agent_role;
pub mod audio;
pub mod conversation;
pub mod image;
pub mod mcp;
pub mod mcp_tool;
pub mod memory;
pub mod model;
pub mod prompt;
pub mod secure_mcp_tool;
pub mod workflow;

// Re-export builders for convenient access
pub use agent_role::{
    AgentRoleBuilder, McpServerBuilder, Stdio,
    AgentRoleBuilderWithHandler, AgentRoleBuilderWithChunkHandler
};
pub use audio::{AudioBuilder, AudioBuilderWithHandler};
pub use conversation::ConversationBuilder;
pub use image::{ImageBuilder, ImageBuilderWithHandler};
pub use mcp::McpClientBuilder;
pub use mcp_tool::{McpToolBuilder, McpToolImpl};
pub use memory::VectorQueryBuilder;
pub use model::ModelInfoBuilder;
pub use prompt::PromptBuilder;
pub use secure_mcp_tool::SecureMcpToolBuilder;
pub use workflow::WorkflowBuilder;