//! Common imports for fluent-ai

extern crate alloc;

pub use crate::{ZeroOneOrMany, AsyncStream, AsyncTask};
pub use crate::chat::chat_loop::ChatLoop;
pub use crate::domain::conversation::{Conversation as ConversationTrait, ConversationBuilder, ConversationImpl};
pub use crate::engine::*;
pub use crate::fluent::*;
// McpTool is now exported through domain module
pub use crate::memory::*;
pub use crate::workflow::*;

// Re-export hashbrown for transformation system
pub use hashbrown::HashMap;

// Re-export cyrup_sugars for JSON transformation
pub use cyrup_sugars::*;

// Error type definition
pub type Error = Box<dyn std::error::Error + Send + Sync>;
