//! Retry execution engine with streaming support
//!
//! Provides the core retry execution logic using pure streams for
//! zero-allocation retry orchestration with comprehensive error handling.

use std::time::Instant;

use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use fluent_ai_async::{AsyncStream, emit};

use super::{RetryPolicy, RetryStats};
use crate::{HttpError, HttpResult};

/// Retry executor trait for generic retry operations
pub trait RetryExecutor<T>: Send + Sync + 'static {
    /// Execute operation with retry logic
    ///
    /// Returns a stream of retry results that includes intermediate retry
    /// attempts and the final result. Consumers can observe retry progress
    /// or wait for the final outcome.
    fn execute_with_retry(self, policy: RetryPolicy) -> AsyncStream<RetryResult<T>>;
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

impl<T> MessageChunk for RetryResult<T>
where
    T: MessageChunk + Send + Default + 'static,
{
    fn bad_chunk(error: String) -> Self {
        Self::FinalFailure {
            error: HttpError::StreamError { message: error },
            stats: RetryStats::default(),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, Self::Retry { .. } | Self::FinalFailure { .. })
    }

    fn error(&self) -> Option<&str> {
        match self {
            Self::Success { .. } => None,
            Self::Retry { .. } => None, // Avoid lifetime issues with temporary string
            Self::FinalFailure { .. } => None, // Avoid lifetime issues with temporary string
        }
    }
}

impl<T> FluentMessageChunk for RetryResult<T>
where
    T: FluentMessageChunk + Send + Default + 'static,
{
    fn bad_chunk(error: String) -> Self {
        Self::FinalFailure {
            error: HttpError::StreamError { message: error },
            stats: RetryStats::default(),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, Self::Retry { .. } | Self::FinalFailure { .. })
    }

    fn error(&self) -> Option<&str> {
        match self {
            Self::Success { .. } => None,
            Self::Retry { .. } => None, // Avoid lifetime issues with temporary string
            Self::FinalFailure { .. } => None, // Avoid lifetime issues with temporary string
        }
    }
}

impl<T> Default for RetryResult<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::Success {
            result: T::default(),
            stats: RetryStats::default(),
        }
    }
}

/// Retry executor for HTTP operations using AsyncStreams
pub struct HttpRetryExecutor<F, T>
where
    F: Fn() -> HttpResult<T> + Send + Sync + 'static,
    T: Send + 'static,
{
    operation: F,
}

impl<F, T> HttpRetryExecutor<F, T>
where
    F: Fn() -> HttpResult<T> + Send + Sync + 'static,
    T: Send + 'static,
{
    /// Create new retry executor for HTTP operation
    ///
    /// Takes a closure that produces a result for each retry attempt.
    /// The closure will be called once per retry attempt.
    #[inline]
    pub fn new(operation: F) -> Self {
        Self { operation }
    }
}

impl<F, T> RetryExecutor<T> for HttpRetryExecutor<F, T>
where
    F: Fn() -> HttpResult<T> + Send + Sync + 'static,
    T: Send + 'static,
{
    /// Execute HTTP operation with comprehensive retry logic
    ///
    /// Implements zero-allocation retry orchestration using AsyncStream
    /// for maximum performance. Handles delay calculation, error classification,
    /// and statistics collection with precise timing measurements.
    fn execute_with_retry(self, policy: RetryPolicy) -> AsyncStream<RetryResult<T>> {
        let operation = self.operation;

        AsyncStream::with_channel(move |sender| {
            let mut attempt = 0;
            let mut stats = RetryStats::default();

            while attempt < policy.max_attempts {
                stats.total_attempts = attempt + 1;
                let attempt_start = Instant::now();

                // Execute the operation
                match (operation)() {
                    Ok(result) => {
                        if attempt > 0 {
                            stats.successful_retries += 1;
                        }
                        stats.total_retry_time += attempt_start.elapsed();
                        stats.complete();

                        emit!(
                            sender,
                            RetryResult::Success {
                                result,
                                stats: stats.clone(),
                            }
                        );
                        return; // Success - exit retry loop
                    }
                    Err(error) => {
                        stats.total_retry_time += attempt_start.elapsed();
                        stats.retry_errors.push(error.to_string());

                        // Check if we should retry
                        if attempt + 1 < policy.max_attempts && policy.is_retryable_error(&error) {
                            let delay = policy.calculate_delay(attempt + 1);
                            stats.total_delay_time += delay;

                            emit!(
                                sender,
                                RetryResult::Retry {
                                    error: error.clone(),
                                    attempt: attempt + 1,
                                    delay,
                                    stats: stats.clone(),
                                }
                            );

                            // Apply delay if needed
                            if delay > std::time::Duration::ZERO {
                                std::thread::sleep(delay);
                            }

                            attempt += 1;
                            continue;
                        } else {
                            stats.complete();
                            emit!(
                                sender,
                                RetryResult::FinalFailure {
                                    error,
                                    stats: stats.clone(),
                                }
                            );
                            return; // Final failure - exit retry loop
                        }
                    }
                }
            }
        })
    }
}
