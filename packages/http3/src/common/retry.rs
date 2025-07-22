//! Zero allocation retry logic with exponential backoff and jitter

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use fastrand::Rng;
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use futures_util::{StreamExt, pin_mut};

use crate::{HttpError, HttpResult};

/// Retry policy configuration - all durations in milliseconds for zero allocation
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay in milliseconds before first retry
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds between retries
    pub max_delay_ms: u64,
    /// Backoff multiplier (typically 2.0 for exponential backoff)
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0 to 1.0) to prevent thundering herd
    pub jitter_factor: f64,
    /// Timeout per individual attempt in milliseconds
    pub attempt_timeout_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000, // 1 second
            max_delay_ms: 30000,    // 30 seconds
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            attempt_timeout_ms: 30000, // 30 seconds per attempt
        }
    }
}

impl RetryPolicy {
    /// Create aggressive retry policy for critical operations
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay_ms: 100, // 100ms
            max_delay_ms: 10000,   // 10 seconds
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
            attempt_timeout_ms: 15000, // 15 seconds per attempt
        }
    }

    /// Create conservative retry policy for non-critical operations
    pub fn conservative() -> Self {
        Self {
            max_attempts: 2,
            initial_delay_ms: 2000, // 2 seconds
            max_delay_ms: 60000,    // 60 seconds
            backoff_multiplier: 3.0,
            jitter_factor: 0.05,
            attempt_timeout_ms: 60000, // 60 seconds per attempt
        }
    }

    /// Create no-retry policy (single attempt only)
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
            attempt_timeout_ms: 120000, // 2 minutes for single attempt
        }
    }

    /// Calculate delay for specific attempt with exponential backoff and jitter
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }

        // Calculate exponential backoff delay
        let base_delay =
            self.initial_delay_ms as f64 * self.backoff_multiplier.powi((attempt - 1) as i32);

        // Cap at maximum delay
        let capped_delay = base_delay.min(self.max_delay_ms as f64);

        // Add jitter to prevent thundering herd
        let jitter_range = capped_delay * self.jitter_factor;
        let mut rng = Rng::new();
        let jitter = rng.f64() * jitter_range - (jitter_range / 2.0);

        let final_delay = (capped_delay + jitter).max(0.0);
        Duration::from_millis(final_delay as u64)
    }

    /// Check if error is retryable
    pub fn is_retryable_error(&self, error: &HttpError) -> bool {
        match error {
            HttpError::NetworkError { .. } => true,
            HttpError::Timeout { .. } => true,
            HttpError::HttpStatus { status, .. } => {
                // Retry on server errors (5xx) and some client errors (429)
                *status >= 500 || *status == 429
            }
            HttpError::TlsError { .. } => false, // TLS errors usually not transient
            HttpError::InvalidUrl { .. } => false, // Client errors not retryable
            HttpError::SerializationError { .. } => false,
            HttpError::InvalidResponse { .. } => false,
            HttpError::ClientError { .. } => false,
            HttpError::DeserializationError { .. } => false,
            HttpError::UrlParseError { .. } => false,
            HttpError::DownloadInterrupted { .. } => true,
            HttpError::InvalidContentLength { .. } => false,
            HttpError::ChunkProcessingError { .. } => false,
            _ => false,
        }
    }
}

/// Retry statistics for monitoring and observability
#[derive(Debug, Clone)]
pub struct RetryStats {
    /// Total number of attempts made
    pub total_attempts: u32,
    /// Number of successful retries  
    pub successful_retries: u32,
    /// Total time spent in retries (excluding delays)
    pub total_retry_time: Duration,
    /// Total time spent waiting between retries
    pub total_delay_time: Duration,
    /// Timestamp when retry sequence started
    pub start_time: Instant,
    /// Final result timestamp
    pub end_time: Option<Instant>,
    /// List of errors encountered during retries
    pub retry_errors: Vec<String>,
}

impl Default for RetryStats {
    fn default() -> Self {
        Self {
            total_attempts: 0,
            successful_retries: 0,
            total_retry_time: Duration::ZERO,
            total_delay_time: Duration::ZERO,
            start_time: Instant::now(),
            end_time: None,
            retry_errors: Vec::new(),
        }
    }
}

impl RetryStats {
    /// Mark retry sequence as completed
    pub fn complete(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Get total elapsed time for entire retry sequence
    pub fn total_elapsed(&self) -> Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Check if any retries were successful
    pub fn had_successful_retry(&self) -> bool {
        self.successful_retries > 0
    }
}

/// Retry executor trait for generic retry operations
pub trait RetryExecutor<T>: Send + Sync + 'static {
    /// Execute operation with retry logic
    fn execute_with_retry(self, policy: RetryPolicy) -> AsyncStream<RetryResult<T>>;
}

/// Result of a retry attempt
#[derive(Debug)]
pub enum RetryResult<T> {
    /// Attempt succeeded
    Success { result: T, stats: RetryStats },
    /// Attempt failed, will retry
    Retry {
        error: HttpError,
        attempt: u32,
        delay: Duration,
        stats: RetryStats,
    },
    /// Final failure after all retries exhausted
    FinalFailure { error: HttpError, stats: RetryStats },
}

/// Retry executor for HTTP operations
pub struct HttpRetryExecutor<F, T>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    operation: F,
}

impl<F, T> HttpRetryExecutor<F, T>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    /// Create new retry executor for HTTP operation
    pub fn new(operation: F) -> Self {
        Self { operation }
    }
}

impl<F, T> RetryExecutor<T> for HttpRetryExecutor<F, T>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    fn execute_with_retry(self, policy: RetryPolicy) -> AsyncStream<RetryResult<T>> {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<RetryResult<T>>| {
            let operation = self.operation;
            let _handle = tokio::spawn(async move {
                let mut stats = RetryStats::default();

                for attempt in 0..policy.max_attempts {
                    stats.total_attempts = attempt + 1;
                    let attempt_start = Instant::now();

                    // Execute the operation
                    let operation_stream = (operation)();
                    pin_mut!(operation_stream);

                    match operation_stream.next().await {
                        Some(Ok(result)) => {
                            if attempt > 0 {
                                stats.successful_retries += 1;
                            }
                            stats.total_retry_time += attempt_start.elapsed();
                            stats.complete();

                            let _ = sender.send(RetryResult::Success { result, stats });
                            return;
                        }
                        Some(Err(error)) => {
                            stats.total_retry_time += attempt_start.elapsed();
                            stats.retry_errors.push(error.to_string());

                            // Check if we should retry
                            if attempt + 1 < policy.max_attempts
                                && policy.is_retryable_error(&error)
                            {
                                let delay = policy.calculate_delay(attempt + 1);
                                stats.total_delay_time += delay;

                                let _ = sender.send(RetryResult::Retry {
                                    error: error.clone(),
                                    attempt: attempt + 1,
                                    delay,
                                    stats: stats.clone(),
                                });

                                // Apply delay with async sleep
                                if delay > Duration::ZERO {
                                    tokio::time::sleep(delay).await;
                                }
                            } else {
                                // Final failure
                                stats.complete();
                                let _ = sender.send(RetryResult::FinalFailure { error, stats });
                                return;
                            }
                        }
                        None => {
                            // Stream ended without result - treat as error
                            let error = HttpError::InvalidResponse {
                                message: "Empty response stream".to_string(),
                            };
                            stats.total_retry_time += attempt_start.elapsed();
                            stats.retry_errors.push(error.to_string());

                            if attempt + 1 < policy.max_attempts {
                                let delay = policy.calculate_delay(attempt + 1);
                                stats.total_delay_time += delay;

                                let _ = sender.send(RetryResult::Retry {
                                    error: error.clone(),
                                    attempt: attempt + 1,
                                    delay,
                                    stats: stats.clone(),
                                });

                                if delay > Duration::ZERO {
                                    tokio::time::sleep(delay).await;
                                }
                            } else {
                                stats.complete();
                                let _ = sender.send(RetryResult::FinalFailure { error, stats });
                                return;
                            }
                        }
                    }
                }
            });
        })
    }
}

/// Global retry statistics for monitoring across all operations
pub struct GlobalRetryStats {
    total_operations: AtomicU64,
    total_retries: AtomicU64,
    total_failures: AtomicU64,
    total_successes: AtomicU64,
}

impl GlobalRetryStats {
    /// Create new global retry statistics
    pub const fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            total_retries: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
        }
    }

    /// Record an operation start
    pub fn record_operation(&self) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a retry attempt
    pub fn record_retry(&self) {
        self.total_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record final success
    pub fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record final failure
    pub fn record_failure(&self) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics snapshot
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.total_operations.load(Ordering::Relaxed),
            self.total_retries.load(Ordering::Relaxed),
            self.total_successes.load(Ordering::Relaxed),
            self.total_failures.load(Ordering::Relaxed),
        )
    }

    /// Calculate success rate percentage
    pub fn success_rate(&self) -> f64 {
        let (total_ops, _, successes, _) = self.snapshot();
        if total_ops > 0 {
            (successes as f64 / total_ops as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate average retries per operation
    pub fn avg_retries_per_operation(&self) -> f64 {
        let (total_ops, total_retries, _, _) = self.snapshot();
        if total_ops > 0 {
            total_retries as f64 / total_ops as f64
        } else {
            0.0
        }
    }
}

/// Global instance for tracking retry statistics across the application
pub static GLOBAL_RETRY_STATS: GlobalRetryStats = GlobalRetryStats::new();

/// Helper function to create retry executor for HTTP operations
pub fn with_retry<F, T>(operation: F) -> HttpRetryExecutor<F, T>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    HttpRetryExecutor::new(operation)
}

/// Helper function to execute operation with default retry policy
pub fn execute_with_default_retry<F, T>(operation: F) -> AsyncStream<RetryResult<T>>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    let executor = HttpRetryExecutor::new(operation);
    executor.execute_with_retry(RetryPolicy::default())
}

/// Helper function to execute operation with aggressive retry policy
pub fn execute_with_aggressive_retry<F, T>(operation: F) -> AsyncStream<RetryResult<T>>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    let executor = HttpRetryExecutor::new(operation);
    executor.execute_with_retry(RetryPolicy::aggressive())
}

/// Helper function to execute operation with conservative retry policy
pub fn execute_with_conservative_retry<F, T>(operation: F) -> AsyncStream<RetryResult<T>>
where
    F: Fn() -> AsyncStream<HttpResult<T>> + Send + Sync + 'static,
    T: Send + 'static,
{
    let executor = HttpRetryExecutor::new(operation);
    executor.execute_with_retry(RetryPolicy::conservative())
}
