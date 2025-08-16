//! Write operations for upgraded HTTP connections
//!
//! Handles write data operations and stream forwarding with proper
//! error handling and connection state management.

use std::sync::Arc;

use crossbeam_channel::Sender;
use fluent_ai_async::{AsyncStream, emit, spawn_task};

use super::types::ConnectionState;

/// Write data to the upgraded connection
pub fn write_data(
    write_sender: &Option<Sender<Vec<u8>>>,
    data: Vec<u8>,
    connection_state: &Arc<ConnectionState>,
) -> Result<(), std::io::Error> {
    if connection_state
        .is_closed
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "Connection is closed",
        ));
    }

    match write_sender {
        Some(sender) => {
            sender.send(data).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "Write channel disconnected")
            })?;
            Ok(())
        }
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "Write channel not available",
        )),
    }
}

/// Forward data from an input stream to the write channel
pub fn forward_stream_to_write(
    input_stream: AsyncStream<Vec<u8>>,
    write_sender: Option<Sender<Vec<u8>>>,
    connection_state: Arc<ConnectionState>,
) {
    if let Some(sender) = write_sender {
        spawn_task(move || {
            for data in input_stream {
                if connection_state
                    .is_closed
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    break;
                }

                if sender.send(data).is_err() {
                    // Write channel disconnected
                    break;
                }
            }
        });
    }
}

/// Create a write stream that accepts data for transmission
pub fn create_write_stream(
    write_sender: Option<Sender<Vec<u8>>>,
    connection_state: Arc<ConnectionState>,
) -> AsyncStream<Result<(), std::io::Error>> {
    AsyncStream::with_channel(move |sender| {
        // This would typically be used to create a sink-like interface
        // For now, we emit a success result to indicate the write stream is ready
        if write_sender.is_some()
            && !connection_state
                .is_closed
                .load(std::sync::atomic::Ordering::Acquire)
        {
            emit!(sender, Ok(()));
        } else {
            emit!(
                sender,
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Write stream not available"
                ))
            );
        }
    })
}
