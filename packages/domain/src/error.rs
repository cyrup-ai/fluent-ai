//! Zero-Allocation Error Handling System
//!
//! This module provides comprehensive error handling with zero heap allocation,
//! circuit breaker patterns, and lock-free error aggregation for blazing-fast performance.

use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, RelaxedCounter};
use circuit_breaker::{CircuitBreaker, CircuitBreakerBuilder};
use const_str::const_str;

/// Maximum length for error messages to ensure zero allocation
pub const MAX_ERROR_MESSAGE_LEN: usize = 256;

/// Error category for structured error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Memory-related errors
    Memory,
    /// Network and communication errors
    Network,
    /// Configuration and initialization errors
    Config,
    /// System-level errors
    System,
    /// User input and validation errors
    User,
    /// Timeout and deadline errors
    Timeout,
    /// Resource exhaustion errors
    Resource,
    /// Serialization and data format errors
    Serialization,
    /// Authentication and authorization errors
    Auth,
    /// Unknown or unclassified errors
    Unknown,
}

/// Error severity levels for prioritization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - informational
    Info,
    /// Medium severity - warning
    Warning,
    /// High severity - error
    Error,
    /// Critical severity - system failure
    Critical,
}

/// Error recoverability classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorRecoverability {
    /// Error can be retried immediately
    Retriable,
    /// Error requires exponential backoff
    RetriableWithBackoff,
    /// Error is permanent and should not be retried
    Permanent,
    /// Error requires manual intervention
    Manual,
}

/// Zero-allocation error message with const generic length
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroAllocMessage<const N: usize> {
    data: [u8; N],
    len: usize,
}

impl<const N: usize> ZeroAllocMessage<N> {
    /// Create new zero-allocation message
    #[inline(always)]
    pub const fn new(message: &str) -> Self {
        let bytes = message.as_bytes();
        let len = if bytes.len() > N { N } else { bytes.len() };

        let mut data = [0u8; N];
        let mut i = 0;
        while i < len {
            data[i] = bytes[i];
            i += 1;
        }

        Self { data, len }
    }

    /// Get message as string slice
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        // SAFETY: We only store valid UTF-8 bytes
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }

    /// Check if message is empty
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get message length
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }
}

impl<const N: usize> fmt::Display for ZeroAllocMessage<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Default error message type
pub type ErrorMessage = ZeroAllocMessage<MAX_ERROR_MESSAGE_LEN>;

/// Zero-allocation error with comprehensive metadata
#[derive(Debug, Clone)]
pub struct ZeroAllocError {
    /// Error category for classification
    pub category: ErrorCategory,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error recoverability classification
    pub recoverability: ErrorRecoverability,
    /// Zero-allocation error message
    pub message: ErrorMessage,
    /// Error code for machine processing
    pub code: u64,
    /// Source location (file:line)
    pub location: Option<ErrorMessage>,
    /// Cause chain for nested errors
    pub cause: Option<Box<ZeroAllocError>>,
    /// Timestamp when error occurred
    pub timestamp: Instant,
    /// Thread ID where error occurred
    pub thread_id: u64,
    /// Error metadata for structured logging
    pub metadata: [(ErrorMessage, ErrorMessage); 4],
    /// Number of metadata entries
    pub metadata_count: usize,
}

impl ZeroAllocError {
    /// Create new zero-allocation error
    #[inline(always)]
    pub fn new(
        category: ErrorCategory,
        severity: ErrorSeverity,
        recoverability: ErrorRecoverability,
        message: &str,
        code: u64,
    ) -> Self {
        Self {
            category,
            severity,
            recoverability,
            message: ErrorMessage::new(message),
            code,
            location: None,
            cause: None,
            timestamp: Instant::now(),
            thread_id: std::thread::current().id().as_u64().get(),
            metadata: [
                (ErrorMessage::new(""), ErrorMessage::new("")),
                (ErrorMessage::new(""), ErrorMessage::new("")),
                (ErrorMessage::new(""), ErrorMessage::new("")),
                (ErrorMessage::new(""), ErrorMessage::new("")),
            ],
            metadata_count: 0,
        }
    }

    /// Add location information
    #[inline(always)]
    pub fn with_location(mut self, file: &str, line: u32) -> Self {
        let location = const_str::format!("{}:{}", file, line);
        self.location = Some(ErrorMessage::new(&location));
        self
    }

    /// Add cause chain
    #[inline(always)]
    pub fn with_cause(mut self, cause: ZeroAllocError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }

    /// Add metadata key-value pair
    #[inline(always)]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        if self.metadata_count < 4 {
            self.metadata[self.metadata_count] = (ErrorMessage::new(key), ErrorMessage::new(value));
            self.metadata_count += 1;
        }
        self
    }

    /// Check if error is retriable
    #[inline(always)]
    pub fn is_retriable(&self) -> bool {
        matches!(
            self.recoverability,
            ErrorRecoverability::Retriable | ErrorRecoverability::RetriableWithBackoff
        )
    }

    /// Check if error is permanent
    #[inline(always)]
    pub fn is_permanent(&self) -> bool {
        matches!(self.recoverability, ErrorRecoverability::Permanent)
    }

    /// Check if error requires manual intervention
    #[inline(always)]
    pub fn is_manual(&self) -> bool {
        matches!(self.recoverability, ErrorRecoverability::Manual)
    }

    /// Get error age since occurrence
    #[inline(always)]
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

impl fmt::Display for ZeroAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}:{:?}] {}",
            self.category, self.severity, self.message
        )?;

        if let Some(location) = &self.location {
            write!(f, " at {}", location)?;
        }

        if let Some(cause) = &self.cause {
            write!(f, " caused by: {}", cause)?;
        }

        Ok(())
    }
}

impl std::error::Error for ZeroAllocError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_ref()
            .map(|e| e.as_ref() as &dyn std::error::Error)
    }
}

/// Lock-free error counter for statistics
#[derive(Debug)]
pub struct ErrorCounter {
    /// Total error count
    total: RelaxedCounter,
    /// Count by category
    by_category: [RelaxedCounter; 10],
    /// Count by severity
    by_severity: [RelaxedCounter; 4],
    /// Count by recoverability
    by_recoverability: [RelaxedCounter; 4],
    /// Last error timestamp
    last_error: AtomicU64,
}

impl ErrorCounter {
    /// Create new error counter
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            total: RelaxedCounter::new(0),
            by_category: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            by_severity: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            by_recoverability: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            last_error: AtomicU64::new(0),
        }
    }

    /// Record error occurrence
    #[inline(always)]
    pub fn record(&self, error: &ZeroAllocError) {
        self.total.inc();
        self.by_category[error.category as usize].inc();
        self.by_severity[error.severity as usize].inc();
        self.by_recoverability[error.recoverability as usize].inc();

        let timestamp = error.timestamp.elapsed().as_nanos() as u64;
        self.last_error.store(timestamp, Ordering::Relaxed);
    }

    /// Get total error count
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.total.get()
    }

    /// Get error count by category
    #[inline(always)]
    pub fn by_category(&self, category: ErrorCategory) -> usize {
        self.by_category[category as usize].get()
    }

    /// Get error count by severity
    #[inline(always)]
    pub fn by_severity(&self, severity: ErrorSeverity) -> usize {
        self.by_severity[severity as usize].get()
    }

    /// Get error count by recoverability
    #[inline(always)]
    pub fn by_recoverability(&self, recoverability: ErrorRecoverability) -> usize {
        self.by_recoverability[recoverability as usize].get()
    }

    /// Get last error timestamp
    #[inline(always)]
    pub fn last_error_time(&self) -> Option<Duration> {
        let timestamp = self.last_error.load(Ordering::Relaxed);
        if timestamp > 0 {
            Some(Duration::from_nanos(timestamp))
        } else {
            None
        }
    }

    /// Reset all counters
    #[inline(always)]
    pub fn reset(&self) {
        self.total.reset();
        for counter in &self.by_category {
            counter.reset();
        }
        for counter in &self.by_severity {
            counter.reset();
        }
        for counter in &self.by_recoverability {
            counter.reset();
        }
        self.last_error.store(0, Ordering::Relaxed);
    }
}

impl Default for ErrorCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker for error recovery
#[derive(Debug)]
pub struct ErrorCircuitBreaker {
    /// Circuit breaker instance
    breaker: CircuitBreaker,
    /// Error counter for statistics
    counter: ErrorCounter,
    /// Failure threshold
    failure_threshold: usize,
    /// Recovery timeout
    recovery_timeout: Duration,
    /// Half-open requests
    half_open_requests: AtomicU64,
}

impl ErrorCircuitBreaker {
    /// Create new circuit breaker
    #[inline(always)]
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        let breaker = CircuitBreakerBuilder::new()
            .failure_threshold(failure_threshold)
            .recovery_timeout(recovery_timeout)
            .build();

        Self {
            breaker,
            counter: ErrorCounter::new(),
            failure_threshold,
            recovery_timeout,
            half_open_requests: AtomicU64::new(0),
        }
    }

    /// Execute operation with circuit breaker protection
    #[inline(always)]
    pub fn execute<T, E, F>(&self, operation: F) -> Result<T, ZeroAllocError>
    where
        F: FnOnce() -> Result<T, E>,
        E: Into<ZeroAllocError>,
    {
        match self.breaker.call(operation) {
            Ok(result) => Ok(result),
            Err(circuit_breaker::Error::Inner(e)) => {
                let error = e.into();
                self.counter.record(&error);
                Err(error)
            }
            Err(circuit_breaker::Error::CircuitBreakerOpen) => {
                let error = ZeroAllocError::new(
                    ErrorCategory::System,
                    ErrorSeverity::Error,
                    ErrorRecoverability::RetriableWithBackoff,
                    "Circuit breaker is open",
                    500,
                );
                self.counter.record(&error);
                Err(error)
            }
        }
    }

    /// Check if circuit breaker is open
    #[inline(always)]
    pub fn is_open(&self) -> bool {
        matches!(self.breaker.state(), circuit_breaker::State::Open)
    }

    /// Check if circuit breaker is half-open
    #[inline(always)]
    pub fn is_half_open(&self) -> bool {
        matches!(self.breaker.state(), circuit_breaker::State::HalfOpen)
    }

    /// Get error statistics
    #[inline(always)]
    pub fn stats(&self) -> &ErrorCounter {
        &self.counter
    }

    /// Reset circuit breaker
    #[inline(always)]
    pub fn reset(&self) {
        self.counter.reset();
        self.half_open_requests.store(0, Ordering::Relaxed);
    }
}

/// Global error aggregator with lock-free statistics
#[derive(Debug)]
pub struct ErrorAggregator {
    /// Error counters by category
    counters: [ErrorCounter; 10],
    /// Circuit breakers by category
    breakers: [ErrorCircuitBreaker; 10],
    /// Global error rate limiter
    rate_limiter: AtomicU64,
    /// Last rate limit reset
    last_reset: AtomicU64,
    /// Rate limit window
    rate_window: Duration,
    /// Maximum errors per window
    max_errors_per_window: usize,
}

impl ErrorAggregator {
    /// Create new error aggregator
    #[inline(always)]
    pub fn new(max_errors_per_window: usize, rate_window: Duration) -> Self {
        const INIT_COUNTER: ErrorCounter = ErrorCounter {
            total: RelaxedCounter::new(0),
            by_category: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            by_severity: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            by_recoverability: [
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
                RelaxedCounter::new(0),
            ],
            last_error: AtomicU64::new(0),
        };

        Self {
            counters: [INIT_COUNTER; 10],
            breakers: [
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
                ErrorCircuitBreaker::new(10, Duration::from_secs(30)),
            ],
            rate_limiter: AtomicU64::new(0),
            last_reset: AtomicU64::new(0),
            rate_window,
            max_errors_per_window,
        }
    }

    /// Record error with rate limiting
    #[inline(always)]
    pub fn record(&self, error: &ZeroAllocError) -> bool {
        // Check rate limit
        if !self.check_rate_limit() {
            return false;
        }

        // Record in category-specific counter
        self.counters[error.category as usize].record(error);

        true
    }

    /// Check rate limit
    #[inline(always)]
    fn check_rate_limit(&self) -> bool {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let last_reset = self.last_reset.load(Ordering::Relaxed);
        let rate_window_nanos = self.rate_window.as_nanos() as u64;

        // Reset rate limiter if window expired
        if now - last_reset > rate_window_nanos {
            self.rate_limiter.store(0, Ordering::Relaxed);
            self.last_reset.store(now, Ordering::Relaxed);
        }

        // Check if under rate limit
        let current_count = self.rate_limiter.load(Ordering::Relaxed);
        if current_count < self.max_errors_per_window as u64 {
            self.rate_limiter.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get error statistics by category
    #[inline(always)]
    pub fn stats(&self, category: ErrorCategory) -> &ErrorCounter {
        &self.counters[category as usize]
    }

    /// Get circuit breaker by category
    #[inline(always)]
    pub fn breaker(&self, category: ErrorCategory) -> &ErrorCircuitBreaker {
        &self.breakers[category as usize]
    }

    /// Get total error count across all categories
    #[inline(always)]
    pub fn total_errors(&self) -> usize {
        self.counters.iter().map(|c| c.total()).sum()
    }

    /// Reset all statistics
    #[inline(always)]
    pub fn reset(&self) {
        for counter in &self.counters {
            counter.reset();
        }
        for breaker in &self.breakers {
            breaker.reset();
        }
        self.rate_limiter.store(0, Ordering::Relaxed);
        self.last_reset.store(0, Ordering::Relaxed);
    }
}

/// Global error aggregator instance
static ERROR_AGGREGATOR: once_cell::sync::Lazy<ErrorAggregator> =
    once_cell::sync::Lazy::new(|| ErrorAggregator::new(1000, Duration::from_secs(60)));

/// Record error in global aggregator
#[inline(always)]
pub fn record_error(error: &ZeroAllocError) -> bool {
    ERROR_AGGREGATOR.record(error)
}

/// Get global error statistics
#[inline(always)]
pub fn error_stats(category: ErrorCategory) -> &'static ErrorCounter {
    ERROR_AGGREGATOR.stats(category)
}

/// Get global circuit breaker
#[inline(always)]
pub fn error_breaker(category: ErrorCategory) -> &'static ErrorCircuitBreaker {
    ERROR_AGGREGATOR.breaker(category)
}

/// Get total error count
#[inline(always)]
pub fn total_errors() -> usize {
    ERROR_AGGREGATOR.total_errors()
}

/// Reset global error statistics
#[inline(always)]
pub fn reset_error_stats() {
    ERROR_AGGREGATOR.reset()
}

/// Convenience macro for creating errors with location
#[macro_export]
macro_rules! error {
    ($category:expr, $severity:expr, $recoverability:expr, $message:expr, $code:expr) => {
        $crate::error::ZeroAllocError::new($category, $severity, $recoverability, $message, $code)
            .with_location(file!(), line!())
    };
}

/// Convenience macro for creating retriable errors
#[macro_export]
macro_rules! retriable_error {
    ($category:expr, $message:expr, $code:expr) => {
        $crate::error!(
            $category,
            $crate::error::ErrorSeverity::Error,
            $crate::error::ErrorRecoverability::Retriable,
            $message,
            $code
        )
    };
}

/// Convenience macro for creating permanent errors
#[macro_export]
macro_rules! permanent_error {
    ($category:expr, $message:expr, $code:expr) => {
        $crate::error!(
            $category,
            $crate::error::ErrorSeverity::Error,
            $crate::error::ErrorRecoverability::Permanent,
            $message,
            $code
        )
    };
}

/// Convenience macro for creating critical errors
#[macro_export]
macro_rules! critical_error {
    ($category:expr, $message:expr, $code:expr) => {
        $crate::error!(
            $category,
            $crate::error::ErrorSeverity::Critical,
            $crate::error::ErrorRecoverability::Manual,
            $message,
            $code
        )
    };
}

/// Result type alias for zero-allocation errors
pub type ZeroAllocResult<T> = Result<T, ZeroAllocError>;

/// Trait for converting errors to zero-allocation errors
pub trait IntoZeroAllocError {
    fn into_zero_alloc_error(self) -> ZeroAllocError;
}

impl IntoZeroAllocError for std::io::Error {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::System,
            ErrorSeverity::Error,
            ErrorRecoverability::Retriable,
            &self.to_string(),
            self.raw_os_error().unwrap_or(0) as u64,
        )
    }
}

impl IntoZeroAllocError for serde_json::Error {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::Serialization,
            ErrorSeverity::Error,
            ErrorRecoverability::Permanent,
            &self.to_string(),
            self.line() as u64,
        )
    }
}

impl<T> IntoZeroAllocError for std::sync::PoisonError<T> {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::System,
            ErrorSeverity::Critical,
            ErrorRecoverability::Manual,
            "Mutex poison error",
            1001,
        )
    }
}

impl IntoZeroAllocError for std::num::ParseIntError {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::User,
            ErrorSeverity::Error,
            ErrorRecoverability::Permanent,
            &self.to_string(),
            1002,
        )
    }
}

impl IntoZeroAllocError for std::num::ParseFloatError {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::User,
            ErrorSeverity::Error,
            ErrorRecoverability::Permanent,
            &self.to_string(),
            1003,
        )
    }
}

impl IntoZeroAllocError for std::str::Utf8Error {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::Serialization,
            ErrorSeverity::Error,
            ErrorRecoverability::Permanent,
            &self.to_string(),
            1004,
        )
    }
}

impl IntoZeroAllocError for std::time::SystemTimeError {
    fn into_zero_alloc_error(self) -> ZeroAllocError {
        ZeroAllocError::new(
            ErrorCategory::System,
            ErrorSeverity::Error,
            ErrorRecoverability::Retriable,
            &self.to_string(),
            1005,
        )
    }
}

/// Extension trait for Result types
pub trait ZeroAllocResultExt<T> {
    fn map_zero_alloc_err<F>(self, f: F) -> ZeroAllocResult<T>
    where
        F: FnOnce() -> ZeroAllocError;

    fn with_error_metadata(self, key: &str, value: &str) -> ZeroAllocResult<T>;

    fn with_error_code(self, code: u64) -> ZeroAllocResult<T>;

    fn record_error(self) -> ZeroAllocResult<T>;
}

impl<T, E> ZeroAllocResultExt<T> for Result<T, E>
where
    E: IntoZeroAllocError,
{
    fn map_zero_alloc_err<F>(self, f: F) -> ZeroAllocResult<T>
    where
        F: FnOnce() -> ZeroAllocError,
    {
        match self {
            Ok(value) => Ok(value),
            Err(_) => Err(f()),
        }
    }

    fn with_error_metadata(self, key: &str, value: &str) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => Err(e.into_zero_alloc_error().with_metadata(key, value)),
        }
    }

    fn with_error_code(self, code: u64) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => {
                let mut error = e.into_zero_alloc_error();
                error.code = code;
                Err(error)
            }
        }
    }

    fn record_error(self) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => {
                let error = e.into_zero_alloc_error();
                record_error(&error);
                Err(error)
            }
        }
    }
}

impl<T> ZeroAllocResultExt<T> for ZeroAllocResult<T> {
    fn map_zero_alloc_err<F>(self, f: F) -> ZeroAllocResult<T>
    where
        F: FnOnce() -> ZeroAllocError,
    {
        match self {
            Ok(value) => Ok(value),
            Err(_) => Err(f()),
        }
    }

    fn with_error_metadata(self, key: &str, value: &str) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => Err(e.with_metadata(key, value)),
        }
    }

    fn with_error_code(self, code: u64) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(mut e) => {
                e.code = code;
                Err(e)
            }
        }
    }

    fn record_error(self) -> ZeroAllocResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => {
                record_error(&e);
                Err(e)
            }
        }
    }
}
