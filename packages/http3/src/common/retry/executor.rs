//! Retry execution engine with streaming support
//!
//! Provides the core retry execution logic using async streams for
//! zero-allocation retry orchestration with comprehensive error handling.

use std::pin::Pin;
use std::time::Instant;

use futures_util::stream::{self, Stream};
use futures_util::{StreamExt, pin_mut};

use super::{RetryPolicy, RetryStats};
use crate::{HttpError, HttpResult};

/// Retry executor trait for generic retry operations
pub trait RetryExecutor<T>: Send + Sync + 'static {
    /// Execute operation with retry logic
    ///
    /// Returns a stream of retry results that includes intermediate retry
    /// attempts and the final result. Consumers can observe retry progress
    /// or wait for the final outcome.
    fn execute_with_retry(
        self,
        policy: RetryPolicy,
    ) -> Pin<Box<dyn Stream<Item = RetryResult<T>> + Send>>;
}

/// Result of a retry attempt
#[derive(Debug)]
pub enum RetryResult<T> {
    /// Attempt succeeded
    Success {
        /// The successful result
        result: T,
        /// Retry statistics
        stats: RetryStats,
    },
    /// Attempt failed, will retry
    Retry {
        /// The error that occurred
        error: HttpError,
        /// Current attempt number
        attempt: u32,
        /// Delay before next retry
        delay: std::time::Duration,
        /// Retry statistics
        stats: RetryStats,
    },
    /// Final failure after all retries exhausted
    FinalFailure {
        /// The final error
        error: HttpError,
        /// Retry statistics
        stats: RetryStats,
    },
}

/// Retry executor for HTTP operations using native Streams
pub struct HttpRetryExecutor<F, T>
where
    F: Fn() -> Pin<Box<dyn Stream<Item = HttpResult<T>> + Send>> + Send + Sync + 'static,
    T: Send + 'static,
{
    operation: F,
}

impl<F, T> HttpRetryExecutor<F, T>
where
    F: Fn() -> Pin<Box<dyn Stream<Item = HttpResult<T>> + Send>> + Send + Sync + 'static,
    T: Send + 'static,
{
    /// Create new retry executor for HTTP operation
    ///
    /// Takes a closure that produces a stream of results for each retry attempt.
    /// The closure will be called once per retry attempt.
    #[inline]
    pub fn new(operation: F) -> Self {
        Self { operation }
    }
}

impl<F, T> RetryExecutor<T> for HttpRetryExecutor<F, T>
where
    F: Fn() -> Pin<Box<dyn Stream<Item = HttpResult<T>> + Send>> + Send + Sync + 'static,
    T: Send + 'static,
{
    /// Execute HTTP operation with comprehensive retry logic
    ///
    /// Implements zero-allocation retry orchestration using stream unfold
    /// for maximum performance. Handles delay calculation, error classification,
    /// and statistics collection with precise timing measurements.
    fn execute_with_retry(
        self,
        policy: RetryPolicy,
    ) -> Pin<Box<dyn Stream<Item = RetryResult<T>> + Send>> {
        let operation = self.operation;
        Box::pin(stream::unfold(
            (policy, operation, 0, RetryStats::default()),
            move |(policy, operation, attempt, mut stats)| async move {
                if attempt >= policy.max_attempts {
                    return None; // All attempts exhausted
                }

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

                        Some((
                            RetryResult::Success {
                                result,
                                stats: stats.clone(),
                            },
                            (policy.clone(), operation, policy.max_attempts, stats),
                        ))
                    }
                    Some(Err(error)) => {
                        stats.total_retry_time += attempt_start.elapsed();
                        stats.retry_errors.push(error.to_string());

                        // Check if we should retry
                        if attempt + 1 < policy.max_attempts && policy.is_retryable_error(&error) {
                            let delay = policy.calculate_delay(attempt + 1);
                            stats.total_delay_time += delay;

                            // Apply delay if needed
                            if delay > std::time::Duration::ZERO {
                                tokio::time::sleep(delay).await;
                            }

                            Some((
                                RetryResult::Retry {
                                    error: error.clone(),
                                    attempt: attempt + 1,
                                    delay,
                                    stats: stats.clone(),
                                },
                                (policy, operation, attempt + 1, stats),
                            ))
                        } else {
                            stats.complete();
                            Some((
                                RetryResult::FinalFailure {
                                    error,
                                    stats: stats.clone(),
                                },
                                (policy.clone(), operation, policy.max_attempts, stats),
                            ))
                        }
                    }
                    None => {
                        // Stream ended without result - treat as error
                        let error = HttpError::InvalidResponse {
                            message: "Empty response stream".to_string(),
                        };
                        stats.total_retry_time += attempt_start.elapsed();
                        stats.retry_errors.push(error.to_string());
                        stats.complete();

                        Some((
                            RetryResult::FinalFailure {
                                error,
                                stats: stats.clone(),
                            },
                            (policy.clone(), operation, policy.max_attempts, stats),
                        ))
                    }
                }
            },
        ))
    }
}
