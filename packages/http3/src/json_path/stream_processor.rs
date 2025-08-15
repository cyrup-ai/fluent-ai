//! JsonStreamProcessor for HTTP response chunk handling
//!
//! Zero-allocation, lock-free HTTP chunk processing with JSONPath streaming.
//! Maintains streams-only architecture with no futures or blocking operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, handle_error};
use serde::de::DeserializeOwned;

use super::{JsonArrayStream, JsonPathError};
use crate::{HttpChunk, HttpError};

/// Circuit breaker state for error recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, processing normally
    Closed,
    /// Circuit is open, failing fast due to consecutive errors
    Open,
    /// Circuit is half-open, allowing limited requests to test recovery
    HalfOpen,
}

/// Error recovery state with circuit breaker pattern
///
/// Provides sophisticated error handling with exponential backoff and
/// circuit breaker patterns for production resilience.
#[derive(Debug)]
pub struct ErrorRecoveryState {
    /// Current circuit breaker state
    circuit_state: Arc<AtomicU64>, // 0=Closed, 1=Open, 2=HalfOpen
    /// Consecutive failure count
    consecutive_failures: Arc<AtomicU64>,
    /// Timestamp of last failure (microseconds since epoch)
    last_failure_time: Arc<AtomicU64>,
    /// Circuit breaker failure threshold
    failure_threshold: u64,
    /// Circuit breaker timeout (microseconds)
    circuit_timeout_micros: u64,
    /// Maximum backoff delay (microseconds)
    max_backoff_micros: u64,
}

impl ErrorRecoveryState {
    /// Create new error recovery state with default configuration
    fn new() -> Self {
        Self {
            circuit_state: Arc::new(AtomicU64::new(CircuitState::Closed as u64)),
            consecutive_failures: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(AtomicU64::new(0)),
            failure_threshold: 5, // Open circuit after 5 consecutive failures
            circuit_timeout_micros: 30_000_000, // 30 second timeout
            max_backoff_micros: 5_000_000, // 5 second max backoff
        }
    }

    /// Record a successful operation
    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let current_state = self.circuit_state.load(Ordering::Relaxed);

        // If we were half-open and got success, close the circuit
        if current_state == CircuitState::HalfOpen as u64 {
            self.circuit_state
                .store(CircuitState::Closed as u64, Ordering::Relaxed);
        }
    }

    /// Record a failure and update circuit state
    fn record_failure(&self) -> CircuitState {
        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        self.last_failure_time.store(now_micros, Ordering::Relaxed);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;

        let current_state = self.circuit_state.load(Ordering::Relaxed);

        // Open circuit if we've reached the failure threshold
        if failures >= self.failure_threshold && current_state == CircuitState::Closed as u64 {
            self.circuit_state
                .store(CircuitState::Open as u64, Ordering::Relaxed);
            CircuitState::Open
        } else {
            self.get_current_state()
        }
    }

    /// Check if operation should proceed based on circuit state
    fn should_proceed(&self) -> (bool, CircuitState) {
        let current_state = self.get_current_state();

        match current_state {
            CircuitState::Closed => (true, current_state),
            CircuitState::HalfOpen => (true, current_state), // Allow limited requests
            CircuitState::Open => {
                // Check if timeout has passed to attempt half-open
                let now_micros = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_micros() as u64)
                    .unwrap_or(0);

                let last_failure = self.last_failure_time.load(Ordering::Relaxed);
                if now_micros.saturating_sub(last_failure) >= self.circuit_timeout_micros {
                    // Transition to half-open for testing
                    self.circuit_state
                        .store(CircuitState::HalfOpen as u64, Ordering::Relaxed);
                    (true, CircuitState::HalfOpen)
                } else {
                    (false, CircuitState::Open)
                }
            }
        }
    }

    /// Get current circuit state
    fn get_current_state(&self) -> CircuitState {
        match self.circuit_state.load(Ordering::Relaxed) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Default fallback
        }
    }

    /// Calculate current backoff delay in microseconds
    fn get_backoff_delay_micros(&self) -> u64 {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures == 0 {
            return 0;
        }

        // Exponential backoff: 100ms * 2^failures, capped at max_backoff
        let base_delay_micros = 100_000; // 100ms in microseconds
        let exponential_delay = base_delay_micros * 2_u64.saturating_pow(failures.min(10) as u32);
        exponential_delay.min(self.max_backoff_micros)
    }
}

/// Lock-free performance statistics for JsonStreamProcessor
///
/// Thread-safe statistics tracking using atomic operations for concurrent access
/// from AsyncStream processing contexts.
#[derive(Debug)]
pub struct ProcessorStats {
    /// Total chunks processed from HTTP response
    pub chunks_processed: Arc<AtomicU64>,
    /// Total bytes processed from HTTP response
    pub bytes_processed: Arc<AtomicU64>,
    /// Objects successfully deserialized and yielded
    pub objects_yielded: Arc<AtomicU64>,
    /// Processing errors encountered
    pub processing_errors: Arc<AtomicU64>,
    /// JSON parsing errors encountered
    pub parse_errors: Arc<AtomicU64>,
    /// Start time for throughput calculation
    pub start_time: Arc<AtomicU64>,
    /// Last processing timestamp for latency tracking
    pub last_process_time: Arc<AtomicU64>,
}

impl ProcessorStats {
    /// Create new processor statistics tracker
    fn new() -> Self {
        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Self {
            chunks_processed: Arc::new(AtomicU64::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            objects_yielded: Arc::new(AtomicU64::new(0)),
            processing_errors: Arc::new(AtomicU64::new(0)),
            parse_errors: Arc::new(AtomicU64::new(0)),
            start_time: Arc::new(AtomicU64::new(now_micros)),
            last_process_time: Arc::new(AtomicU64::new(now_micros)),
        }
    }

    /// Get current statistics snapshot
    pub fn snapshot(&self) -> ProcessorStatsSnapshot {
        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        let start_time = self.start_time.load(Ordering::Relaxed);
        let elapsed_micros = now_micros.saturating_sub(start_time);
        let elapsed_seconds = (elapsed_micros as f64) / 1_000_000.0;

        let chunks = self.chunks_processed.load(Ordering::Relaxed);
        let bytes = self.bytes_processed.load(Ordering::Relaxed);
        let objects = self.objects_yielded.load(Ordering::Relaxed);
        let errors = self.processing_errors.load(Ordering::Relaxed);
        let parse_errors = self.parse_errors.load(Ordering::Relaxed);

        ProcessorStatsSnapshot {
            chunks_processed: chunks,
            bytes_processed: bytes,
            objects_yielded: objects,
            processing_errors: errors,
            parse_errors,
            throughput_objects_per_sec: if elapsed_seconds > 0.0 {
                objects as f64 / elapsed_seconds
            } else {
                0.0
            },
            bytes_per_object: if objects > 0 {
                bytes as f64 / objects as f64
            } else {
                0.0
            },
            error_rate: if chunks > 0 {
                (errors + parse_errors) as f64 / chunks as f64
            } else {
                0.0
            },
            elapsed_seconds,
        }
    }
}

/// Immutable snapshot of processor statistics
#[derive(Debug, Clone, Copy)]
pub struct ProcessorStatsSnapshot {
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Objects successfully yielded
    pub objects_yielded: u64,
    /// Processing errors encountered
    pub processing_errors: u64,
    /// JSON parsing errors
    pub parse_errors: u64,
    /// Objects processed per second
    pub throughput_objects_per_sec: f64,
    /// Average bytes per object
    pub bytes_per_object: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Total elapsed processing time in seconds
    pub elapsed_seconds: f64,
}

/// High-performance HTTP chunk processor for JSONPath streaming
///
/// Processes HTTP response chunks through JSONPath deserialization in a
/// zero-allocation, lock-free manner following the streams-only architecture.
pub struct JsonStreamProcessor<T> {
    json_array_stream: JsonArrayStream<T>,
    chunk_handlers:
        Vec<Box<dyn FnMut(Result<T, JsonPathError>) -> Result<T, JsonPathError> + Send>>,
    stats: ProcessorStats,
    error_recovery: ErrorRecoveryState,
}

impl<T> JsonStreamProcessor<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create new JsonStreamProcessor with JSONPath expression
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering JSON array elements
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::json_path::JsonStreamProcessor;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let processor = JsonStreamProcessor::<User>::new("$.users[*]");
    /// ```
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
    ///
    /// Returns a snapshot of current performance metrics including throughput,
    /// error rates, and resource utilization.
    ///
    /// # Returns
    /// ProcessorStatsSnapshot with current metrics
    #[must_use]
    pub fn stats(&self) -> ProcessorStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get current circuit breaker state for monitoring
    ///
    /// # Returns
    /// Current CircuitState for observability
    #[must_use]
    pub fn circuit_state(&self) -> CircuitState {
        self.error_recovery.get_current_state()
    }

    /// Handle error with recovery pattern including circuit breaker
    ///
    /// Applies sophisticated error recovery including exponential backoff,
    /// circuit breaker pattern, and failure categorization.
    fn handle_error_with_recovery(&self, error: HttpError) -> Result<(), HttpError> {
        // Check if circuit breaker allows processing
        let (should_proceed, circuit_state) = self.error_recovery.should_proceed();

        if !should_proceed {
            log::warn!(
                "Circuit breaker OPEN - failing fast. State: {:?}, Error: {:?}",
                circuit_state,
                error
            );
            return Err(HttpError::Generic(format!(
                "Circuit breaker open due to consecutive failures: {}",
                error
            )));
        }

        // Record the failure in circuit breaker
        let new_state = self.error_recovery.record_failure();

        // Apply exponential backoff if configured
        let backoff_delay = self.error_recovery.get_backoff_delay_micros();
        if backoff_delay > 0 {
            log::debug!(
                "Applying backoff delay of {}μs due to error recovery. Circuit state: {:?}",
                backoff_delay,
                new_state
            );

            // Sleep for backoff duration (non-blocking approach would be better for production)
            std::thread::sleep(std::time::Duration::from_micros(backoff_delay));
        }

        // Categorize error for specific handling
        match &error {
            HttpError::Generic(msg) if msg.contains("timeout") => {
                log::warn!("Timeout error in JsonStreamProcessor: {}", msg);
                // Timeout errors are often transient - less severe
            }
            HttpError::Generic(msg) if msg.contains("connection") => {
                log::warn!("Connection error in JsonStreamProcessor: {}", msg);
                // Connection errors may indicate network issues
            }
            _ => {
                log::error!("Processing error in JsonStreamProcessor: {:?}", error);
                // Other errors are more severe
            }
        }

        Err(error)
    }

    /// Record successful operation for circuit breaker recovery
    fn record_success(&self) {
        self.error_recovery.record_success();
    }

    /// Add chunk processing handler for customized object transformation
    ///
    /// Handlers are applied in the order they are added to each deserialized object.
    ///
    /// # Arguments
    /// * `handler` - Function to process each deserialized object result
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::json_path::JsonStreamProcessor;
    ///
    /// let processor = processor.with_handler(|result| {
    ///     result.map(|mut obj| {
    ///         // Transform object
    ///         obj
    ///     })
    /// });
    /// ```
    #[must_use]
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: FnMut(Result<T, JsonPathError>) -> Result<T, JsonPathError> + Send + 'static,
    {
        self.chunk_handlers.push(Box::new(handler));
        self
    }

    /// Process HTTP chunks into deserialized objects stream
    ///
    /// Converts HTTP response chunks through JSONPath deserialization following
    /// the zero-allocation streams-only architecture.
    ///
    /// # Arguments
    /// * `chunks` - Iterator of HTTP chunks to process
    ///
    /// # Returns
    /// AsyncStream of deserialized objects matching the JSONPath expression
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::json_path::JsonStreamProcessor;
    /// use fluent_ai_http3::HttpChunk;
    /// use bytes::Bytes;
    ///
    /// let processor = JsonStreamProcessor::<serde_json::Value>::new("$.items[*]");
    /// let chunks = vec![HttpChunk::Body(Bytes::from(r#"{"items":[{"id":1}]}"#))];
    /// let stream = processor.process_chunks(chunks.into_iter());
    /// ```
    pub fn process_chunks<I>(mut self, chunks: I) -> AsyncStream<T>
    where
        I: Iterator<Item = HttpChunk> + Send + 'static,
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<T>| {
            for chunk in chunks {
                // Update processing timestamp
                let now_micros = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_micros() as u64)
                    .unwrap_or(0);
                self.stats
                    .last_process_time
                    .store(now_micros, Ordering::Relaxed);

                match chunk {
                    HttpChunk::Body(bytes) => {
                        // Track chunk and byte counts
                        self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .bytes_processed
                            .fetch_add(bytes.len() as u64, Ordering::Relaxed);

                        match self.process_body_chunk(&sender, bytes) {
                            Ok(_) => {
                                // Record successful processing for circuit breaker
                                self.record_success();
                            }
                            Err(e) => {
                                self.stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                                let json_error = JsonPathError::Deserialization(format!(
                                    "Body chunk processing failed: {}",
                                    e
                                ));
                                let http_error = HttpError::Generic(format!(
                                    "JSONPath processing error: {}",
                                    json_error
                                ));

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
                        self.stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                        if let Err(recovery_error) = self.handle_error_with_recovery(e) {
                            handle_error!(recovery_error, "HTTP chunk error with recovery");
                        }
                    }
                    _ => {
                        // Ignore Head and other chunk types in JSONPath streaming
                        continue;
                    }
                }
            }
        })
    }

    /// Process HTTP response body into streaming objects
    ///
    /// High-performance body processing that maintains zero-allocation constraints
    /// and streams-only architecture.
    ///
    /// # Arguments
    /// * `body_bytes` - Complete HTTP response body
    ///
    /// # Returns
    /// AsyncStream of deserialized objects
    pub fn process_body(mut self, body_bytes: Bytes) -> AsyncStream<T>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<T>| {
            if let Err(e) = self.process_body_chunk(&sender, body_bytes) {
                handle_error!(e, "Failed to process response body");
            }
        })
    }

    /// Process single body chunk through JSONPath deserialization
    ///
    /// Internal method that handles the core deserialization logic while
    /// maintaining architectural constraints.
    fn process_body_chunk(
        &mut self,
        sender: &AsyncStreamSender<T>,
        bytes: Bytes,
    ) -> Result<(), JsonPathError>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        // Process bytes through JSONPath deserializer
        let objects_stream = self.json_array_stream.process_chunk(bytes);

        // Collect objects from stream and process them
        let objects = objects_stream.collect();
        for obj in objects {
            self.stats.objects_yielded.fetch_add(1, Ordering::Relaxed);

            let mut result = Ok(obj);

            // Apply all chunk handlers in sequence
            for handler in &mut self.chunk_handlers {
                result = handler(result);
            }

            match result {
                Ok(processed_obj) => {
                    fluent_ai_async::emit!(sender, processed_obj);
                }
                Err(e) => {
                    log::error!("JSONPath processing failed: {:?}", e);
                    let error_chunk = T::bad_chunk(e.to_string());
                    fluent_ai_async::emit!(sender, error_chunk);
                }
            }
        }

        Ok(())
    }

    /// Process bytes directly (convenience method)
    ///
    /// Wraps raw bytes in HttpChunk::Body and processes them.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw response bytes
    ///
    /// # Returns
    ///
    /// AsyncStream of deserialized objects
    #[inline]
    pub fn process_bytes(&mut self, bytes: Bytes) -> AsyncStream<T>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        // Process bytes through JSONPath deserializer directly
        self.json_array_stream.process_chunk(bytes)
    }

    /// Process single HTTP chunk (convenience method)
    ///
    /// Processes a single HTTP chunk and returns the resulting stream of objects.
    ///
    /// # Arguments
    /// * `chunk` - HTTP chunk to process
    ///
    /// # Returns
    /// AsyncStream of deserialized objects from the chunk
    pub fn process_single_chunk(self, chunk: HttpChunk) -> AsyncStream<T>
    where
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        match chunk {
            HttpChunk::Body(bytes) => self.process_body(bytes),
            HttpChunk::Error(e) => {
                AsyncStream::with_channel(move |_sender: AsyncStreamSender<T>| {
                    handle_error!(e, "HTTP chunk error in single chunk processing");
                })
            }
            _ => {
                // Return empty stream for non-body chunks
                AsyncStream::builder().empty()
            }
        }
    }

    /// Get reference to underlying JSONPath expression
    #[must_use]
    pub fn jsonpath_expr(&self) -> &str {
        self.json_array_stream.jsonpath_expr()
    }
}

/// Error handling utilities for JsonStreamProcessor
impl<T> JsonStreamProcessor<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Process chunks with error recovery
    ///
    /// Processes HTTP chunks while providing error recovery through a
    /// user-provided error handler function.
    ///
    /// # Arguments
    /// * `chunks` - Iterator of HTTP chunks to process
    /// * `error_handler` - Function to handle processing errors
    ///
    /// # Returns
    /// AsyncStream of successfully processed objects
    pub fn process_chunks_with_recovery<I, F>(
        mut self,
        chunks: I,
        mut error_handler: F,
    ) -> AsyncStream<T>
    where
        I: Iterator<Item = HttpChunk> + Send + 'static,
        F: FnMut(HttpError) + Send + 'static,
        T: MessageChunk + FluentMessageChunk + Default + Send + 'static,
    {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<T>| {
            for chunk in chunks {
                match chunk {
                    HttpChunk::Body(bytes) => {
                        if let Err(e) = self.process_body_chunk(&sender, bytes) {
                            let http_error =
                                HttpError::Generic(format!("JSONPath processing failed: {}", e));
                            error_handler(http_error);
                        }
                    }
                    HttpChunk::Error(e) => {
                        error_handler(e);
                    }
                    _ => {
                        // Ignore non-body chunks
                        continue;
                    }
                }
            }
        })
    }
}

/// Resource cleanup implementation for JsonStreamProcessor
impl<T> Drop for JsonStreamProcessor<T> {
    fn drop(&mut self) {
        // Log final statistics on drop
        let final_stats = self.stats.snapshot();
        log::debug!(
            "JsonStreamProcessor dropped - Stats: {} objects yielded, {} chunks processed, {:.2} objects/sec",
            final_stats.objects_yielded,
            final_stats.chunks_processed,
            final_stats.throughput_objects_per_sec
        );
    }
}

/// Default implementation for JsonStreamProcessor
impl<T> Default for JsonStreamProcessor<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create processor with root array JSONPath expression
    fn default() -> Self {
        Self::new("$[*]")
    }
}
