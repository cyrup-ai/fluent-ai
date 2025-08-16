//! Core types and state management for HTTP upgrades
//!
//! Defines protocol types, connection state, and statistics structures
//! for upgraded HTTP connections with thread-safe state tracking.

use std::sync::Arc;

/// Protocol types supported by HTTP upgrades
#[derive(Debug, Clone, PartialEq)]
pub enum UpgradeProtocol {
    WebSocket,
    Http2ServerPush,
    Custom(String),
}

/// Connection state for thread-safe status tracking
#[derive(Debug)]
pub struct ConnectionState {
    pub is_closed: std::sync::atomic::AtomicBool,
    pub bytes_read: std::sync::atomic::AtomicU64,
    pub bytes_written: std::sync::atomic::AtomicU64,
}

/// Connection statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub is_closed: bool,
    pub protocol: UpgradeProtocol,
}
