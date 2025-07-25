//! Command parsing module with logical separation
//!
//! Zero-allocation command parsing with dedicated parsers for different
//! command categories to maintain focused, maintainable code.

pub mod core;
pub mod basic;
pub mod advanced;
pub mod advanced_primary;
pub mod advanced_secondary;

// Re-export main parser
pub use core::CommandParser;

// Legacy compatibility
#[deprecated(note = "Use ImmutableChatCommand instead for zero-allocation streaming")]
pub type ChatCommand = crate::types::candle_chat::commands::types::ImmutableChatCommand;