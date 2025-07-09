//! Common imports for fluent-ai

extern crate alloc;

pub use crate::async_task::{AsyncStream, AsyncTask};
pub use crate::collection_ext::prelude::*;
pub use crate::mcp_tool::McpTool;
pub use crate::sugars::ByteSize;
pub use crate::sugars::{FutureExt, StreamExt};
pub use crate::ZeroOneOrMany;

// Error type definition
pub type Error = Box<dyn std::error::Error + Send + Sync>;
