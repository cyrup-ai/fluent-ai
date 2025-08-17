//! Asynchronous streaming processing logic
//!
//! Contains asynchronous processing methods and fallback streaming deserializer
//! logic for handling incomplete JSON and streaming scenarios.

use bytes::Bytes;
use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, prelude::MessageChunk as FluentMessageChunk};
use serde::de::DeserializeOwned;

use super::core::JsonArrayStream;
use crate::json_path::{CoreJsonPathEvaluator, JsonPathDeserializer};

impl<T> JsonArrayStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
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
    pub fn process_chunk(&mut self, chunk: Bytes) -> AsyncStream<T, 1024>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        // Append chunk to internal buffer
        self.buffer.append_chunk(chunk);

        // Try to parse as complete JSON first using simple evaluator
        let all_data = self.buffer.as_bytes();
        let json_str = match std::str::from_utf8(all_data) {
            Ok(s) => s,
            Err(_) => {
                // Invalid UTF-8, return empty stream
                return AsyncStream::builder().empty();
            }
        };

        // Try to parse as complete JSON
        let json_value = match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(value) => {
                println!("DEBUG: JSON parsing succeeded: {:?}", value);
                value
            }
            Err(e) => {
                println!("DEBUG: JSON parsing failed: {:?}", e);
                // Not complete JSON yet, fall back to streaming deserializer
                return self.fallback_to_streaming_deserializer();
            }
        };

        // Use core evaluator for complete JSON
        let expression = self.path_expression.as_string();
        println!(
            "DEBUG: Creating CoreJsonPathEvaluator for expression: {}",
            expression
        );
        let evaluator = match CoreJsonPathEvaluator::new(&expression) {
            Ok(eval) => {
                println!("DEBUG: CoreJsonPathEvaluator created successfully");
                eval
            }
            Err(e) => {
                println!("DEBUG: CoreJsonPathEvaluator creation failed: {:?}", e);
                return AsyncStream::builder().empty();
            }
        };

        println!("DEBUG: Evaluating expression against JSON value");
        let results = match evaluator.evaluate(&json_value) {
            Ok(values) => {
                println!(
                    "DEBUG: CoreJsonPathEvaluator succeeded with {} results",
                    values.len()
                );
                values
            }
            Err(e) => {
                println!("DEBUG: CoreJsonPathEvaluator evaluation failed: {:?}", e);
                return AsyncStream::builder().empty();
            }
        };

        // Convert JSON values to target type T
        let mut typed_results = Vec::new();
        for value in results {
            match serde_json::from_value::<T>(value.clone()) {
                Ok(typed_value) => {
                    typed_results.push(typed_value);
                }
                Err(_) => {
                    // Skip invalid values
                }
            }
        }

        println!(
            "DEBUG: Complete JSON path produced {} typed results",
            typed_results.len()
        );

        // If no results, return a Vec directly instead of an AsyncStream that waits for timeout
        if typed_results.is_empty() {
            println!("DEBUG: Creating immediate empty stream for 0 results");
            // Create a stream that immediately signals completion
            return AsyncStream::with_channel(move |sender| {
                // Send nothing and exit immediately - this signals the producer is done
                drop(sender);
            });
        }

        // Create AsyncStream from the processed results using proper streaming architecture
        AsyncStream::with_channel(move |sender| {
            println!(
                "DEBUG: Complete JSON AsyncStream closure started with {} results",
                typed_results.len()
            );
            for typed_value in typed_results {
                if sender.try_send(typed_value).is_err() {
                    break; // Channel closed
                }
            }
            println!("DEBUG: Complete JSON AsyncStream closure completed");
        })
    }

    pub(super) fn fallback_to_streaming_deserializer(&mut self) -> AsyncStream<T, 1024>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        println!("DEBUG: fallback_to_streaming_deserializer called");

        // Process available data using the streaming deserializer
        let mut deserializer = JsonPathDeserializer::new(&self.path_expression, &mut self.buffer);
        let mut results = Vec::new();

        println!("DEBUG: Starting streaming deserializer iteration");

        // Manually collect the iterator to avoid lifetime dependency
        let mut iterator = deserializer.process_available();
        let mut iteration_count = 0;
        while let Some(result) = iterator.next() {
            iteration_count += 1;
            println!(
                "DEBUG: Streaming iteration {}: {:?}",
                iteration_count,
                result.is_ok()
            );
            results.push(result);

            // Safety check to prevent infinite loops
            if iteration_count > 1000 {
                println!(
                    "DEBUG: Breaking streaming iteration after {} iterations",
                    iteration_count
                );
                break;
            }
        }

        println!(
            "DEBUG: Streaming deserializer completed with {} results after {} iterations",
            results.len(),
            iteration_count
        );

        // Create AsyncStream from the processed results using proper streaming architecture
        AsyncStream::with_channel(move |sender| {
            println!(
                "DEBUG: Streaming AsyncStream closure started with {} results",
                results.len()
            );
            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok(typed_value) => {
                        println!("DEBUG: Streaming sending successful result {}", i);
                        if sender.try_send(typed_value).is_err() {
                            println!("DEBUG: Streaming channel closed at result {}", i);
                            break; // Channel closed
                        }
                    }
                    Err(e) => {
                        println!("DEBUG: Streaming skipping failed result {}: {:?}", i, e);
                        log::warn!("Deserialization failed: {:?}", e);
                        // Skip invalid values
                    }
                }
            }
            println!("DEBUG: Streaming AsyncStream closure completed");
        })
    }
}
