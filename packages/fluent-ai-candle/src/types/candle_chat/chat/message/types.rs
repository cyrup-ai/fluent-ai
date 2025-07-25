//! Message types re-exported from parent message module

pub use crate::types::candle_chat::message::{CandleMessage, CandleMessageRole, MessageChunk, MessageType, SearchChatMessage};

// Define aliases for backwards compatibility
pub use CandleMessage as Message;
pub use CandleMessageRole as MessageRole;