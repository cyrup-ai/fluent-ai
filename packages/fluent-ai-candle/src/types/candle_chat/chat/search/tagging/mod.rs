//! Conversation tagging system with lock-free operations
//!
//! This module provides comprehensive tagging functionality for conversations
//! with zero-allocation streaming patterns and atomic operations.

pub mod types;
pub mod tagger;
pub mod statistics;

// Re-export main types
pub use types::*;
pub use tagger::ConversationTagger;
pub use statistics::TaggingStatistics;