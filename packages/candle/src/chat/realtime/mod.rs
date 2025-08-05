//! Real-time features for chat system
//!
//! This module provides comprehensive real-time features including:
//! - Typing indicators
//! - Live message streaming
//! - Event-driven architecture
//! - Connection management
//!
//! All components use zero-allocation patterns and lock-free operations
//! for maximum performance.

pub mod events;
pub mod streaming;
pub mod typing;

// Re-export the main types for backward compatibility
pub use events::{
    ConnectionStatus, 
    EventBroadcaster, 
    FlumeEventBroadcaster, 
    RealTimeEvent
};
pub use streaming::MessageStream;
pub use typing::TypingIndicator;

// Re-export RealTimeError and RealTimeResult from the domain module
pub use crate::domain::chat::realtime::RealTimeError;
pub type RealTimeResult<T> = std::result::Result<T, RealTimeError>;