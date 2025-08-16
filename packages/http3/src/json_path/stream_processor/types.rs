use std::sync::Arc;
use std::sync::atomic::AtomicU64;

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
#[derive(Debug)]
pub struct ErrorRecoveryState {
    /// Current circuit breaker state
    pub(super) circuit_state: Arc<AtomicU64>, // 0=Closed, 1=Open, 2=HalfOpen
    /// Consecutive failure count
    pub(super) consecutive_failures: Arc<AtomicU64>,
    /// Timestamp of last failure (microseconds since epoch)
    pub(super) last_failure_time: Arc<AtomicU64>,
    /// Circuit breaker failure threshold
    pub(super) failure_threshold: u64,
    /// Circuit breaker timeout (microseconds)
    pub(super) circuit_timeout_micros: u64,
    /// Maximum backoff delay (microseconds)
    pub(super) max_backoff_micros: u64,
}

/// Lock-free performance statistics for JsonStreamProcessor
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
pub struct JsonStreamProcessor<T> {
    pub(super) json_array_stream: super::super::JsonArrayStream<T>,
    pub(super) chunk_handlers: Vec<
        Box<
            dyn FnMut(
                    Result<T, super::super::JsonPathError>,
                ) -> Result<T, super::super::JsonPathError>
                + Send,
        >,
    >,
    pub(super) stats: ProcessorStats,
    pub(super) error_recovery: ErrorRecoveryState,
}
