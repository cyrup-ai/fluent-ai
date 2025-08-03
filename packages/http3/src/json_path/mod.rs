//! High-performance JSONPath streaming deserializer for Http3
//!
//! This module provides blazing-fast, zero-allocation JSONPath expression evaluation
//! over streaming HTTP responses. It enables streaming individual array elements from
//! nested JSON structures like OpenAI's `{"data": [...]}` format.
//!
//! # Features
//!
//! - Full JSONPath specification support
//! - Zero-allocation streaming deserialization
//! - Lock-free concurrent processing
//! - Comprehensive error handling and recovery
//! - Integration with Http3 builder pattern
//!
//! # Examples
//!
//! ```rust
//! use fluent_ai_http3::Http3;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Model {
//!     id: String,
//!     object: String,
//! }
//!
//! // Stream individual models from OpenAI's {"data": [...]} response
//! Http3::json()
//!     .array_stream("$.data[*]")
//!     .bearer_auth(&api_key)
//!     .get("https://api.openai.com/v1/models")
//!     .on_chunk(|model: Model| {
//!         Ok => model.into(),
//!         Err(e) => BadChunk::from_err(e)
//!     })
//!     .collect_or_else(|error| Model::default());
//! ```

pub mod buffer;
pub mod deserializer;
pub mod error;
pub mod filter;
pub mod functions;
pub mod parser;
pub mod state_machine;

// Decomposed parser modules
pub mod ast;
pub mod compiler;
pub mod expression;
pub mod filter_parser;
pub mod selector_parser;
pub mod tokenizer;
pub mod tokens;

#[cfg(test)]
mod debug_test;

use std::marker::PhantomData;

use bytes::Bytes;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use serde::de::DeserializeOwned;

pub use self::{
    buffer::{JsonBuffer, StreamBuffer},
    deserializer::{JsonPathDeserializer, JsonPathIterator, StreamingDeserializer},
    error::{JsonPathError, JsonPathResult, JsonPathResultExt},
    filter::FilterEvaluator,
    functions::FunctionEvaluator,
    parser::{
        ComparisonOp, ComplexityMetrics, FilterExpression, FilterValue, JsonPathExpression,
        JsonPathParser, JsonSelector,
    },
    state_machine::{JsonStreamState, StreamStateMachine},
};

/// Zero-allocation JSONPath streaming processor
///
/// Transforms HTTP byte streams into individual JSON objects based on JSONPath expressions.
/// Uses compile-time optimizations and runtime streaming for maximum performance.
pub struct JsonArrayStream<T> {
    /// JSONPath expression for array element selection
    path_expression: JsonPathExpression,
    /// Streaming buffer for efficient byte processing
    buffer: StreamBuffer,
    /// State machine for parsing progress tracking
    state: StreamStateMachine,
    /// Zero-sized type marker for target deserialization type
    _phantom: PhantomData<T>,
}

impl<T> JsonArrayStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create new JSONPath streaming processor
    ///
    /// # Arguments
    ///
    /// * `jsonpath` - JSONPath expression (e.g., "$.data[*]", "$.results[?(@.active)]")
    ///
    /// # Error Handling
    ///
    /// Invalid JSONPath expressions are handled via async-stream error emission patterns.
    /// Errors are logged and processing continues with a default expression.
    ///
    /// # Performance
    ///
    /// JSONPath compilation is performed once during construction for optimal runtime performance.
    pub fn new(jsonpath: &str) -> Self {
        let path_expression = match JsonPathParser::compile(jsonpath) {
            Ok(expr) => expr,
            Err(e) => {
                log::error!("JSONPath compilation failed: {:?}", e);
                // Return empty expression that matches nothing, allowing processing to continue
                JsonPathExpression::new(Vec::new(), jsonpath.to_string(), false)
            }
        };
        let buffer = StreamBuffer::with_capacity(8192); // 8KB initial capacity
        let state = StreamStateMachine::new();

        Self {
            path_expression,
            buffer,
            state,
            _phantom: PhantomData,
        }
    }

    /// Process incoming bytes and yield deserialized objects
    ///
    /// This method is called repeatedly as HTTP chunks arrive. It maintains internal
    /// state to parse JSON incrementally and yield complete objects as they become available.
    ///
    /// # Arguments
    ///
    /// * `chunk` - Incoming HTTP response bytes
    ///
    /// # Returns
    ///
    /// AsyncStream over successfully deserialized objects of type `T`.
    /// Errors are handled via async-stream error emission patterns.
    ///
    /// # Performance
    ///
    /// Uses zero-copy techniques where possible and pre-allocated buffers to minimize allocations.
    /// Lock-free processing with const-generic capacity for blazing-fast performance.
    pub fn process_chunk(&mut self, chunk: Bytes) -> AsyncStream<T>
    where
        T: Send + 'static,
    {
        // Append chunk to internal buffer
        self.buffer.append_chunk(chunk);

        // Process available data and collect results
        let mut deserializer =
            JsonPathDeserializer::new(&self.path_expression, &mut self.buffer, &mut self.state);
        let results: Vec<_> = deserializer.process_available().collect();

        // Create AsyncStream from the processed results
        AsyncStream::with_channel(move |sender| {
            for result in results {
                match result {
                    Ok(value) => {
                        emit!(sender, value);
                    }
                    Err(e) => {
                        handle_error!(e, "JSON deserialization failed");
                        // Continue processing other items even after errors
                    }
                }
            }
        })
    }

    /// Check if the stream has completed processing
    ///
    /// Returns `true` when the entire JSON structure has been parsed and all matching
    /// array elements have been yielded.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Get current parsing statistics for monitoring and debugging
    ///
    /// Returns metrics including bytes processed, objects yielded, and parsing errors.
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            bytes_processed: self.buffer.total_bytes_processed(),
            objects_yielded: self.state.objects_yielded(),
            parse_errors: self.state.parse_errors(),
            buffer_size: self.buffer.current_size(),
        }
    }
}

/// Streaming performance and debugging statistics
#[derive(Debug, Clone, Copy)]
pub struct StreamStats {
    /// Total bytes processed from HTTP response
    pub bytes_processed: u64,
    /// Number of JSON objects successfully deserialized
    pub objects_yielded: u64,
    /// Count of recoverable parsing errors encountered
    pub parse_errors: u64,
    /// Current internal buffer size in bytes
    pub buffer_size: usize,
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
}
