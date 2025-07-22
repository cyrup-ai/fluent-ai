//! Canonical channel types and helpers for the streams-only architecture.

use crate::stream::AsyncStream;
use tokio::sync::mpsc;

/// The sender part of the channel used to create an `AsyncStream`.
pub type AsyncStreamSender<T> = mpsc::UnboundedSender<T>;

/// Creates a new asynchronous stream channel.
///
/// This is the primary factory for creating streams in the fluent-ai ecosystem.
/// It returns a sender and a receiver, where the receiver is already wrapped
/// in the canonical `AsyncStream` type.
// TODO: Convert async_stream_channel to AsyncStream::with_channel pattern
    let (tx, rx) = mpsc::unbounded_channel();
    (tx, tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}
