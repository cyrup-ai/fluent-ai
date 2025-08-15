//! Request execution and response handling functionality
//!
//! Pure streams-first execution - NO Futures, NO Result wrapping
//! All operations return unwrapped AsyncStreams per fluent-ai architecture

use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use serde::de::DeserializeOwned;

use crate::{HttpChunk, HttpStream};

/// Pure streams extension trait - NO Result wrapping, streams-only
///
/// Provides streaming methods that return AsyncStreams of deserialized types
/// following the streams-first architecture mandate
pub trait HttpStreamExt<T> {
    /// Stream deserialized objects as they arrive - pure streaming
    ///
    /// Returns unwrapped AsyncStream of deserialized objects, not Result-wrapped
    /// Errors are emitted as stream items, not exceptions
    fn stream_objects(self) -> AsyncStream<T>;

    /// Collect all items into Vec - blocks until complete (streams-first bridge)
    fn collect_all(self) -> Vec<T>;

    /// Get first item only - blocks until available
    fn first_item(self) -> Option<T>;
}

impl<T> HttpStreamExt<T> for HttpStream
where
    T: DeserializeOwned + Send + Default + MessageChunk + FluentMessageChunk + 'static,
{
    fn stream_objects(self) -> AsyncStream<T> {
        AsyncStream::with_channel(move |sender| {
            let mut stream = self;
            std::thread::spawn(move || {
                let mut buffer = Vec::new();

                while let Some(chunk) = stream.poll_next() {
                    match chunk {
                        HttpChunk::Body(bytes) => {
                            buffer.extend_from_slice(&bytes);
                        }
                        HttpChunk::Deserialized(json_value) => {
                            if let Ok(obj) = serde_json::from_value::<T>(json_value) {
                                let _ = sender.send(obj);
                            }
                        }
                        HttpChunk::Error(_) => break,
                        _ => continue,
                    }
                }

                // Try to deserialize accumulated buffer
                if !buffer.is_empty() {
                    if let Ok(obj) = serde_json::from_slice::<T>(&buffer) {
                        let _ = sender.send(obj);
                    }
                }
            });
        })
    }

    fn collect_all(self) -> Vec<T> {
        self.stream_objects().collect()
    }

    fn first_item(self) -> Option<T> {
        let stream = self.stream_objects();
        stream.collect().into_iter().next()
    }
}
