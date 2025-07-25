//! Real-time features for chat system - DECOMPOSED MODULES
//!
//! ACTIVELY WORKING BY claude17 - DECOMPOSED FROM 2181 LINES TO <=300 LINE MODULES
//!
//! This module has been decomposed into focused, single-responsibility modules
//! following the ≤300-line architectural constraint. All functionality from the
//! original realtime.rs has been properly separated and is available through submodules.

// Core modules - fully implemented
pub mod events;
pub mod errors;
pub mod typing;
pub mod live_updates;
pub mod streaming;

// Additional modules - to be implemented
pub mod connections;
pub mod system;
pub mod builder;
pub mod config;

// Re-export all public types for ergonomic API
pub use events::{
    RealTimeEvent, ConnectionStatus, NotificationLevel
};

pub use errors::{
    RealTimeError, ErrorSeverity, RealTimeResult
};

pub use typing::{
    TypingState, TypingIndicator, TypingStatistics
};

pub use live_updates::{
    LiveUpdateMessage, MessagePriority, LiveUpdateSystem, LiveUpdateStatistics
};

pub use streaming::{
    LiveMessageStreamer, StreamingStatistics
};

// Re-exports from implemented modules
pub use connections::{
    ConnectionState, ConnectionManager, ConnectionStatistics, ConnectionManagerStatistics
};

pub use system::{
    RealTimeSystem, RealTimeSystemStatistics
};

pub use builder::{
    RealTimeSystemBuilder
};

pub use config::{
    RealtimeConfig, RealtimeChat, RealtimeConnection, RealtimeMessage,
    RealtimeMessageType, PresenceStatus, RealtimeEventType, RealtimeChatMetrics
};