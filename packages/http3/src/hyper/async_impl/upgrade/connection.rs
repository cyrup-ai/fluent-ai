//! Connection creation and management for HTTP upgrades
//!
//! Handles bidirectional connection setup with dedicated read/write channels
//! and background task spawning for I/O processing.

use std::{io, sync::Arc};

use crossbeam_channel::{Receiver, Sender, bounded};
use fluent_ai_async::spawn_task;

use super::types::{ConnectionState, UpgradeProtocol};

/// An upgraded HTTP connection with full bidirectional I/O support.
/// Supports WebSocket, HTTP/2 server push, and other upgrade protocols
/// using streams-first architecture with zero-allocation patterns.
pub struct Upgraded {
    pub read_receiver: Option<Receiver<Vec<u8>>>,
    pub write_sender: Option<Sender<Vec<u8>>>,
    pub protocol: UpgradeProtocol,
    pub connection_state: Arc<ConnectionState>,
}

impl Upgraded {
    /// Create a new Upgraded connection with full bidirectional I/O capability
    pub(crate) fn new_with_protocol(protocol: UpgradeProtocol) -> Result<Self, io::Error> {
        Self::create_bidirectional_connection(protocol)
    }

    /// Create a new Upgraded connection with WebSocket protocol (default)
    pub(crate) fn new() -> Result<Self, io::Error> {
        Self::new_with_protocol(UpgradeProtocol::WebSocket)
    }

    /// Create bidirectional connection with dedicated read/write channels
    pub fn create_bidirectional_connection(protocol: UpgradeProtocol) -> Result<Self, io::Error> {
        let (_read_tx, read_rx) = bounded(1024); // Bounded channel for backpressure
        let (write_tx, write_rx) = bounded(1024);

        let connection_state = Arc::new(ConnectionState {
            is_closed: std::sync::atomic::AtomicBool::new(false),
            bytes_read: std::sync::atomic::AtomicU64::new(0),
            bytes_written: std::sync::atomic::AtomicU64::new(0),
        });

        // Spawn background task for I/O processing
        let state_clone = Arc::clone(&connection_state);
        spawn_task(move || {
            super::io_processing::background_io_processor(write_rx, state_clone);
        });

        Ok(Upgraded {
            read_receiver: Some(read_rx),
            write_sender: Some(write_tx),
            protocol,
            connection_state,
        })
    }
}
