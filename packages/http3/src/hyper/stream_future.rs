//! Bridge between AsyncStream and Future for Service trait compatibility

use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk;

/// A synchronous wrapper that collects an AsyncStream to completion
///
/// This enables Service trait compatibility by providing synchronous collection
/// over the internal streams-first architecture - NO Future trait used
pub struct StreamCollector<T: MessageChunk + Default + Send + 'static> {
    stream: AsyncStream<T>,
}

impl<T: MessageChunk + Default + Send + 'static> StreamCollector<T> {
    /// Create a new StreamCollector from an AsyncStream
    pub fn new(stream: AsyncStream<T>) -> Self {
        Self { stream }
    }
    
    /// Collect the stream synchronously using blocking collection
    pub fn collect_sync(self) -> T {
        match self.stream.collect().into_iter().next() {
            Some(result) => result,
            None => T::default(),
        }
    }
}

// StreamCollector implementation complete - no Future trait used
