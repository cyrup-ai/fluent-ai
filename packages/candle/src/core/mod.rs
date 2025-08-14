//! Core domain types and traits

use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use arc_swap::ArcSwap;
use crossbeam_channel;
use crossbeam_utils::CachePadded;
use fluent_ai_async::AsyncStream;
use once_cell::sync::Lazy;

use crate::domain::memory::MemoryError;
use crate::AsyncTask;

/// Domain initialization error types with semantic error handling
#[derive(Debug, thiserror::Error)]
pub enum DomainInitError {
    /// Memory system initialization error
    #[error("Memory initialization failed: {0}")]
    Memory(#[from] MemoryError),
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    /// System error
    #[error("System error: {0}")]
    System(String),
    /// Connection pool error
    #[error("Connection pool error: {0}")]
    Pool(String),
    /// Timeout error
    #[error("Operation timed out after {:?}", .0)]
    Timeout(Duration),
    /// Circuit breaker is open
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Channel error type for proper error handling
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("Channel send error")]
    SendError,
    #[error("Channel receive error")]
    ReceiveError,
    #[error("Channel closed")]
    Closed,
}

/// Channel sender wrapper using crossbeam for zero-allocation performance
pub struct ChannelSender<T> {
    sender: crossbeam_channel::Sender<std::result::Result<T, ChannelError>>,
}

impl<T: Send + 'static> ChannelSender<T> {
    /// Finish the task by sending the result
    #[inline]
    pub fn finish(self, value: T) {
        let _ = self.sender.send(Ok(value));
    }

    /// Finish the task with an error
    #[inline]
    pub fn finish_with_error(self, error: ChannelError) {
        let _ = self.sender.send(Err(error));
    }
}

/// Create a new channel for async communication using crossbeam for zero allocation
#[inline]
pub fn channel<T: Send + 'static>() -> (
    ChannelSender<T>,
    AsyncTask<std::result::Result<T, ChannelError>>,
) {
    let (tx, rx) = crossbeam_channel::bounded(1);
    (ChannelSender { sender: tx }, AsyncTask::new(rx))
}

/// Global state for circuit breaker pattern
static CIRCUIT_BREAKER: Lazy<ArcSwap<CircuitBreaker>> =
    Lazy::new(|| ArcSwap::from_pointee(CircuitBreaker::new()));

/// Circuit breaker state
struct CircuitBreaker {
    _failure_count: AtomicUsize,
    reset_after: Duration,
    last_failure: CachePadded<parking_lot::Mutex<Option<std::time::Instant>>>,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            _failure_count: AtomicUsize::new(0),
            reset_after: Duration::from_secs(60),
            last_failure: CachePadded::new(parking_lot::Mutex::new(None)),
        }
    }

    fn is_open(&self) -> bool {
        let last_failure = self.last_failure.lock();
        if let Some(time) = *last_failure {
            time.elapsed() < self.reset_after
        } else {
            false
        }
    }

    fn record_failure(&self) {
        let mut last_failure = self.last_failure.lock();
        *last_failure = Some(std::time::Instant::now());
    }
}

/// Execute operation with circuit breaker protection using AsyncStream
pub fn execute_with_circuit_breaker<F, T, E>(
    operation: F,
) -> AsyncStream<std::result::Result<T, DomainInitError>>
where
    F: FnOnce() -> std::result::Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: Into<DomainInitError> + Send + 'static,
{
    AsyncStream::with_channel(move |sender| {
        std::thread::spawn(move || {
            let circuit_breaker = CIRCUIT_BREAKER.load();
            if circuit_breaker.is_open() {
                let _ = sender.send(Err(DomainInitError::CircuitBreakerOpen));
                return;
            }

            match operation() {
                Ok(result) => {
                    let _ = sender.send(Ok(result));
                }
                Err(err) => {
                    circuit_breaker.record_failure();
                    let _ = sender.send(Err(err.into()));
                }
            }
        });
    })
}

// Re-export commonly used types
// REMOVED: pub use futures::stream::Stream; - ALL FUTURES ELIMINATED!
/// Core engine for completion processing
pub mod engine;

/// Advanced constrained generation with sampling strategies
pub mod generation;

/// Unified model configuration system for hundreds of models
pub mod model_config;

/// SIMD adapter functions for bridging fluent_ai_simd with generation types
pub mod simd_adapters;

// Re-export core types
pub use engine::*;
pub use generation::*;
pub use model_config::*;
pub use simd_adapters::{
    should_use_simd, simd_argmax_with_bounds, simd_error_to_fallback_strategy,
    simd_softmax_with_cache, simd_temperature_scale,
};
