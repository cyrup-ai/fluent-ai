//! Common imports for fluent-ai

extern crate alloc;

pub use crate::async_task::{AsyncStream, AsyncTask};
pub use crate::chat_loop::ChatLoop;
pub use crate::conversation::Conversation;
pub use crate::collection_ext::prelude::*;
pub use crate::engine::*;
pub use crate::fluent::*;
pub use crate::mcp_tool::McpTool;
pub use crate::memory::*;
pub use crate::sugars::ByteSize;
pub use crate::sugars::{FutureExt, StreamExt};
pub use crate::workflow::*;
pub use crate::ZeroOneOrMany;

// Error type definition
pub type Error = Box<dyn std::error::Error + Send + Sync>;
