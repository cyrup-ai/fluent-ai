//! Connection management for upgraded HTTP connections
//!
//! Handles connection lifecycle, statistics, and graceful shutdown
//! with thread-safe state tracking and resource cleanup.

use std::sync::Arc;

use super::types::{ConnectionState, ConnectionStats, UpgradeProtocol};

/// Close the upgraded connection gracefully
pub fn close_connection(connection_state: &Arc<ConnectionState>) {
    connection_state
        .is_closed
        .store(true, std::sync::atomic::Ordering::Release);
}

/// Get current connection statistics
pub fn get_connection_stats(
    connection_state: &Arc<ConnectionState>,
    protocol: &UpgradeProtocol,
) -> ConnectionStats {
    ConnectionStats {
        bytes_read: connection_state
            .bytes_read
            .load(std::sync::atomic::Ordering::Acquire),
        bytes_written: connection_state
            .bytes_written
            .load(std::sync::atomic::Ordering::Acquire),
        is_closed: connection_state
            .is_closed
            .load(std::sync::atomic::Ordering::Acquire),
        protocol: protocol.clone(),
    }
}

/// Check if the connection is still active
pub fn is_connection_active(connection_state: &Arc<ConnectionState>) -> bool {
    !connection_state
        .is_closed
        .load(std::sync::atomic::Ordering::Acquire)
}
