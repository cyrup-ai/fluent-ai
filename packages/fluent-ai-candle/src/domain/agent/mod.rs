//! Agent domain types and role implementations
//!
//! This module consolidates agent data structures and role definitions with automatic memory tool injection.

pub mod chat;
pub mod core;
pub mod role;
pub mod types;

// Re-export commonly used types
pub use core::*;
pub use chat::*;
pub use role::*;
pub use types::*;

// Candle-prefixed aliases for external compatibility  
pub use role::{AgentRole as CandleAgentRole, AgentRoleImpl as CandleAgentRoleImpl};
pub use role::{AgentConversation as CandleAgentConversation, AgentConversationMessage as CandleAgentConversationMessage};
pub use role::McpServerConfig as CandleMcpServer;
pub use types::AgentRoleAgent as CandleAgentRoleAgent;
