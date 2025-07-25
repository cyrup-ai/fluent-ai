//! Message types and processing for the chat system
//!
//! This module provides the core message types and functionality needed
//! for chat message handling within the chat subsystem.

pub mod message_processing;

// Re-export message types from the central types module
pub use crate::types::{
    CandleMessage, CandleMessageRole, CandleSearchChatMessage as SearchChatMessage
};

// MessageChunk and MessageType need to be defined locally or imported from the actual source

// Define local Message and MessageRole aliases that are compatible with the chat system
pub use CandleMessage as Message;
pub use CandleMessageRole as MessageRole;

/// Message processing functionality specific to the chat system
pub mod processing {
    pub use super::message_processing::*;
}