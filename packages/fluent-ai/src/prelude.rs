//! Common imports for fluent-ai

extern crate alloc;

pub use crate::ZeroOneOrMany;
pub use crate::async_task::{AsyncStream, AsyncTask};
pub use crate::chat_loop::ChatLoop;
pub use crate::collection_ext::prelude::*;
pub use crate::domain::conversation::{Conversation as ConversationTrait, ConversationBuilder, ConversationImpl};
pub use crate::engine::*;
pub use crate::fluent::*;
// McpTool is now exported through domain module
pub use crate::memory::*;
pub use crate::sugars::ByteSize;
pub use crate::sugars::{FutureExt, StreamExt};
pub use crate::workflow::*;

// Re-export hashbrown for transformation system
pub use hashbrown::HashMap;

// Error type definition
pub type Error = Box<dyn std::error::Error + Send + Sync>;
