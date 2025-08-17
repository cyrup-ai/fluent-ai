use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{CircuitState, ErrorRecoveryState};

impl ErrorRecoveryState {
    /// Create new error recovery state with default configuration
    pub fn new() -> Self {
        Self {
            circuit_state: std::sync::Arc::new(AtomicU64::new(CircuitState::Closed as u64)),
            consecutive_failures: std::sync::Arc::new(AtomicU64::new(0)),
            last_failure_time: std::sync::Arc::new(AtomicU64::new(0)),
            failure_threshold: 5,
            circuit_timeout_micros: 30_000_000, // 30 seconds
            max_backoff_micros: 5_000_000,      // 5 seconds
        }
    }

    /// Record successful operation for circuit breaker
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        match self.get_current_state() {
            CircuitState::HalfOpen => {
                // Successful request in half-open state closes the circuit
                self.circuit_state
                    .store(CircuitState::Closed as u64, Ordering::Relaxed);
            }
            _ => {} // No state change needed for Closed state
        }
    }

    /// Record failure and update circuit breaker state
    pub fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;

        // Update last failure timestamp
        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        self.last_failure_time.store(now_micros, Ordering::Relaxed);

        // Open circuit if failure threshold exceeded
        if failures >= self.failure_threshold {
            self.circuit_state
                .store(CircuitState::Open as u64, Ordering::Relaxed);
        }
    }

    /// Check if request should be allowed through circuit breaker
    pub fn should_allow_request(&self) -> (bool, CircuitState) {
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
    pub fn get_current_state(&self) -> CircuitState {
        match self.circuit_state.load(Ordering::Relaxed) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Default fallback
        }
    }

    /// Calculate current backoff delay in microseconds
    pub fn get_backoff_delay_micros(&self) -> u64 {
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

impl Default for ErrorRecoveryState {
    fn default() -> Self {
        Self::new()
    }
}
