use bytes::Bytes;
use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use fluent_ai_async::{AsyncStreamSender, emit, handle_error};
use serde::de::DeserializeOwned;

use super::super::JsonPathError;
use super::types::JsonStreamProcessor;
use crate::HttpError;

impl<T> JsonStreamProcessor<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Process individual body chunk through JSONPath deserialization
    pub(super) fn process_body_chunk<S>(
        &mut self,
        sender: &AsyncStreamSender<S>,
        bytes: Bytes,
    ) -> Result<(), JsonPathError>
    where
        S: MessageChunk + FluentMessageChunk + Default + Send + 'static,
        T: Into<S>,
    {
        // Check circuit breaker before processing
        let (should_allow, _circuit_state) = self.error_recovery.should_allow_request();
        if !should_allow {
            return Err(JsonPathError::Processing(
                "Circuit breaker open - failing fast".to_string(),
            ));
        }

        // Apply backoff delay if needed
        let backoff_delay_micros = self.error_recovery.get_backoff_delay_micros();
        if backoff_delay_micros > 0 {
            std::thread::sleep(std::time::Duration::from_micros(backoff_delay_micros));
        }

        // Process chunk through JSONPath stream
        let objects_stream = self.json_array_stream.process_chunk(bytes)?;

        // Apply chunk handlers and emit objects
        for object_result in objects_stream {
            let mut processed_result = object_result;

            // Apply all registered chunk handlers
            for handler in &mut self.chunk_handlers {
                processed_result = handler(processed_result);
            }

            match processed_result {
                Ok(object) => {
                    self.stats.record_object_yield();
                    let converted_object: S = object.into();
                    emit!(sender, converted_object);
                }
                Err(e) => {
                    self.stats.record_parse_error();
                    self.error_recovery.record_failure();
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Handle error with recovery mechanisms
    pub(super) fn handle_error_with_recovery(&mut self, error: HttpError) -> Result<(), HttpError> {
        // Record failure for circuit breaker
        self.error_recovery.record_failure();

        // Check if we should attempt recovery
        let (should_allow, circuit_state) = self.error_recovery.should_allow_request();

        if !should_allow {
            handle_error!(
                format!("Circuit breaker {:?} - error: {}", circuit_state, error),
                "stream processing"
            );
            return Err(error);
        }

        // Apply exponential backoff
        let backoff_delay_micros = self.error_recovery.get_backoff_delay_micros();
        if backoff_delay_micros > 0 {
            std::thread::sleep(std::time::Duration::from_micros(backoff_delay_micros));
        }

        // Log error for monitoring
        handle_error!(
            format!("Processing error with recovery: {}", error),
            "json stream processor"
        );

        Err(error)
    }

    /// Reset error recovery state
    pub fn reset_error_recovery(&mut self) {
        self.error_recovery = super::types::ErrorRecoveryState::new();
    }

    /// Get current error recovery state
    pub fn get_circuit_state(&self) -> super::types::CircuitState {
        self.error_recovery.get_current_state()
    }

    /// Check if processor is healthy for new requests
    pub fn is_healthy(&self) -> bool {
        let (should_allow, _) = self.error_recovery.should_allow_request();
        should_allow
    }
}
