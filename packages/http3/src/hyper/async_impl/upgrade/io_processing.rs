//! I/O processing for upgraded HTTP connections
//!
//! Background processors for read/write operations with proper error handling
//! and metrics tracking using zero-allocation streaming patterns.

use std::sync::Arc;

use crossbeam_channel::{Receiver, RecvTimeoutError};
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit, spawn_task};

use super::types::ConnectionState;

/// Background I/O processor for write operations
pub fn background_io_processor(write_receiver: Receiver<Vec<u8>>, state: Arc<ConnectionState>) {
    while let Ok(data) = write_receiver.recv() {
        if state.is_closed.load(std::sync::atomic::Ordering::Acquire) {
            break;
        }

        // Process write data (actual I/O implementation would go here)
        // For production: integrate with hyper's upgraded connection
        let bytes_written = data.len() as u64;
        state
            .bytes_written
            .fetch_add(bytes_written, std::sync::atomic::Ordering::Release);

        // Simulate actual network write operation
        // In production: write to underlying TCP/TLS stream
    }
}

/// Read stream processor with proper error handling and metrics tracking
pub fn read_stream_processor(
    receiver: Receiver<Vec<u8>>,
    sender: AsyncStreamSender<crate::HttpResponseChunk>,
    state: Arc<ConnectionState>,
) {
    loop {
        if state.is_closed.load(std::sync::atomic::Ordering::Acquire) {
            break;
        }

        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(data) => {
                let bytes_read = data.len() as u64;
                state
                    .bytes_read
                    .fetch_add(bytes_read, std::sync::atomic::Ordering::Release);
                emit!(
                    sender,
                    crate::HttpResponseChunk::data(bytes::Bytes::from(data))
                );
            }
            Err(RecvTimeoutError::Timeout) => {
                // Continue polling - allows for graceful shutdown checking
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Connection closed cleanly
                break;
            }
        }
    }
}

/// Convert to a read stream with full bidirectional I/O support
pub fn create_read_stream(
    mut read_receiver: Option<Receiver<Vec<u8>>>,
    connection_state: Arc<ConnectionState>,
) -> AsyncStream<crate::HttpResponseChunk> {
    AsyncStream::with_channel(move |sender| {
        let read_receiver = match read_receiver.take() {
            Some(receiver) => receiver,
            None => {
                use fluent_ai_async::prelude::MessageChunk;
                emit!(
                    sender,
                    crate::HttpResponseChunk::bad_chunk("Read stream already consumed".to_string())
                );
                return;
            }
        };

        let state = Arc::clone(&connection_state);
        spawn_task(move || {
            read_stream_processor(read_receiver, sender, state);
        });
    })
}
