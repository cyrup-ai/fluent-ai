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
pub use role::{AgentRole, AgentRoleImpl};
pub use types::{AgentConversation, AgentConversationMessage, AgentRoleAgent};
