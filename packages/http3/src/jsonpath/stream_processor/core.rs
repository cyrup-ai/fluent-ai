use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, handle_error};
use serde::de::DeserializeOwned;

use super::super::{JsonArrayStream, JsonPathError};
use super::types::{ErrorRecoveryState, JsonStreamProcessor, ProcessorStats};
use crate::prelude::*;
use crate::error::constructors::generic;


impl<T> JsonStreamProcessor<T>
where
    T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
{
    /// Create new JsonStreamProcessor with JSONPath expression
    #[must_use]
    pub fn new(jsonpath_expr: &str) -> Self {
        Self {
            json_array_stream: JsonArrayStream::new_typed(jsonpath_expr),
            chunk_handlers: Vec::new(),
            stats: ProcessorStats::new(),
            error_recovery: ErrorRecoveryState::new(),
        }
    }

    /// Get current processing statistics
    pub fn stats(&self) -> super::types::ProcessorStatsSnapshot {
        self.stats.snapshot()
    }

    /// Add chunk processing handler for custom transformations
    pub fn with_chunk_handler<F>(mut self, handler: F) -> Self
    where
        F: FnMut(Result<T, JsonPathError>) -> Result<T, JsonPathError> + Send + 'static,
    {
        self.chunk_handlers.push(Box::new(handler));
        self
    }

    /// Process HTTP chunks into deserialized objects stream
    pub fn process_chunks<I>(mut self, chunks: I) -> AsyncStream<T, 1024>
    where
        I: Iterator<Item = HttpChunk> + Send + 'static,
        T: MessageChunk + MessageChunk + Default + Send + 'static,
    {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<T>| {
            for chunk in chunks {
                self.stats.update_last_process_time();

                match chunk {
                    HttpChunk::Body(bytes) => {
                        self.stats.record_chunk_processed(bytes.len());

                        match self.process_body_chunk(&sender, bytes) {
                            Ok(_) => {
                                self.record_success();
                            }
                            Err(e) => {
                                self.stats.record_processing_error();
                                let json_error = JsonPathError::new(
                                    crate::jsonpath::error::ErrorKind::Deserialization,
                                    format!("Body chunk processing failed: {}", e),
                                );
                                let http_error =
                                    generic(format!("JSONPath processing error: {}", json_error));

                                if let Err(recovery_error) =
                                    self.handle_error_with_recovery(http_error)
                                {
                                    handle_error!(
                                        recovery_error,
                                        "Failed to process body chunk with recovery"
                                    );
                                }
                            }
                        }
                    }
                    HttpChunk::Error(e) => {
                        self.stats.record_processing_error();
                        if let Err(recovery_error) = self.handle_error_with_recovery(
                            crate::error::Error::new(crate::error::Kind::Request)
                                .with(std::io::Error::new(std::io::ErrorKind::Other, e)),
                        ) {
                            handle_error!(recovery_error, "HTTP chunk error with recovery");
                        }
                    }
                    _ => {
                        continue;
                    }
                }
            }
        })
    }

    /// Process HTTP response body into streaming objects
    pub fn process_body<B>(mut self, body: B) -> AsyncStream<T, 1024>
    where
        B: http_body::Body<Data = Bytes> + Send + Sync + 'static + Unpin,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
        T: MessageChunk + MessageChunk + Default + Send + 'static,
    {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<T>| {
            // TODO: Implement proper body streaming when stream_body_data is available
            // For now, create a simple stream from the body
            let body_stream = std::iter::once(body);

            for body in body_stream {
                // Convert body to bytes for processing
                let body_bytes = Bytes::new(); // Placeholder for now

                // Record processing
                self.stats.record_chunk_processed(0);

                if let Err(e) = self.process_body_chunk(&sender, body_bytes) {
                    self.stats.record_processing_error();
                    handle_error!(e, "Body chunk processing failed");
                } else {
                    self.record_success();
                }
            }
        })
    }

    /// Record successful operation for circuit breaker
    pub(super) fn record_success(&self) {
        self.error_recovery.record_success();
    }
}
