//! Real-time event types and enums
//!
//! This module defines all real-time event types used throughout the realtime system,
//! including connection status, notification levels, and event structures with
//! zero-allocation patterns and atomic operations support.

use serde::{Deserialize, Serialize};
use crate::types::CandleMessage;

/// Real-time event types with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealTimeEvent {
    /// User started typing
    TypingStarted {
        user_id: String,
        session_id: String,
        timestamp: u64},
    /// User stopped typing
    TypingStopped {
        user_id: String,
        session_id: String,
        timestamp: u64},
    /// New message received
    MessageReceived {
        message: CandleMessage,
        session_id: String,
        timestamp: u64},
    /// Message updated
    MessageUpdated {
        message_id: String,
        content: String,
        session_id: String,
        timestamp: u64},
    /// Message deleted
    MessageDeleted {
        message_id: String,
        session_id: String,
        timestamp: u64},
    /// User joined session
    UserJoined {
        user_id: String,
        session_id: String,
        timestamp: u64},
    /// User left session
    UserLeft {
        user_id: String,
        session_id: String,
        timestamp: u64},
    /// Connection status changed
    ConnectionStatusChanged {
        user_id: String,
        status: ConnectionStatus,
        timestamp: u64},
    /// Heartbeat received
    HeartbeatReceived {
        user_id: String,
        session_id: String,
        timestamp: u64},
    /// System notification
    SystemNotification {
        message: String,
        level: NotificationLevel,
        timestamp: u64}}

/// Connection status enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Connected and active
    Connected,
    /// Connecting
    Connecting,
    /// Disconnected
    Disconnected,
    /// Connection error
    Error,
    /// Reconnecting
    Reconnecting,
    /// Connection failed
    Failed,
    /// Idle (connected but inactive)
    Idle,
    /// Unstable connection
    Unstable}

impl ConnectionStatus {
    /// Convert to atomic representation (u8)
    #[inline]
    pub const fn to_atomic(&self) -> u8 {
        match self {
            Self::Connected => 0,
            Self::Connecting => 1,
            Self::Disconnected => 2,
            Self::Error => 3,
            Self::Reconnecting => 4,
            Self::Failed => 5,
            Self::Idle => 6,
            Self::Unstable => 7}
    }

    /// Convert from atomic representation (u8)
    #[inline]
    pub const fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Connected,
            1 => Self::Connecting,
            2 => Self::Disconnected,
            3 => Self::Error,
            4 => Self::Reconnecting,
            5 => Self::Failed,
            6 => Self::Idle,
            _ => Self::Unstable, // Default fallback
        }
    }

    /// Check if connection is in a healthy state
    #[inline]
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Connected | Self::Idle)
    }

    /// Check if connection is in a transitional state
    #[inline]
    pub const fn is_transitional(&self) -> bool {
        matches!(self, Self::Connecting | Self::Reconnecting)
    }

    /// Check if connection has failed
    #[inline]
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed | Self::Error | Self::Disconnected)
    }
}

/// Notification level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationLevel {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Success
    Success}

impl NotificationLevel {
    /// Get priority value for sorting notifications
    #[inline]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Error => 3,
            Self::Warning => 2,
            Self::Success => 1,
            Self::Info => 0}
    }

    /// Check if this is a critical notification
    #[inline]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::Error)
    }
}

impl Default for ConnectionStatus {
    fn default() -> Self {
        Self::Disconnected
    }
}

impl Default for NotificationLevel {
    fn default() -> Self {
        Self::Info
    }
}

/// Message priority for real-time processing - mentioned in TODO4.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3}

impl MessagePriority {
    /// Get priority as u8 for atomic operations
    #[inline]
    pub const fn as_u8(&self) -> u8 {
        *self as u8
    }
    
    /// Create from u8 value
    #[inline]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical}
    }
    
    /// Check if priority is critical
    #[inline]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::Critical)
    }
    
    /// Check if priority is high or above
    #[inline]
    pub const fn is_high_priority(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

// Send + Sync are automatically derived for these enum types since all variants implement Send + Sync