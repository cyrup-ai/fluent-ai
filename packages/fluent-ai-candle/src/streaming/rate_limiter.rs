//! Token rate limiting for flow control
//!
//! Provides atomic, lock-free rate limiting for token generation:
//! - Sliding window rate limiting with atomic counters
//! - Zero-allocation token admission control
//! - Configurable rate limits with instant updates

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Token rate limiter for controlling generation speed
pub struct TokenRateLimiter {
    /// Maximum tokens per second (0 = unlimited)
    max_tokens_per_second: f64,
    /// Token count in current window
    current_window_tokens: AtomicUsize,
    /// Current time window start
    window_start: AtomicU64,
    /// Window duration in nanoseconds
    window_duration_ns: u64,
    /// Enable rate limiting
    enabled: bool,
}

impl TokenRateLimiter {
    /// Create new rate limiter
    #[inline]
    pub fn new(max_tokens_per_second: f64) -> Self {
        Self {
            max_tokens_per_second,
            current_window_tokens: AtomicUsize::new(0),
            window_start: AtomicU64::new(0),
            window_duration_ns: 1_000_000_000, // 1 second
            enabled: max_tokens_per_second > 0.0,
        }
    }

    /// Create unlimited rate limiter (no rate limiting)
    #[inline]
    pub fn unlimited() -> Self {
        Self::new(0.0)
    }

    /// Create rate limiter with custom window duration
    #[inline]
    pub fn with_window_duration(max_tokens_per_second: f64, window_duration_ms: u64) -> Self {
        Self {
            max_tokens_per_second,
            current_window_tokens: AtomicUsize::new(0),
            window_start: AtomicU64::new(0),
            window_duration_ns: window_duration_ms * 1_000_000, // Convert ms to ns
            enabled: max_tokens_per_second > 0.0,
        }
    }

    /// Check if token generation should be allowed
    #[inline]
    pub fn should_allow_token(&self) -> bool {
        if !self.enabled || self.max_tokens_per_second <= 0.0 {
            return true;
        }

        let now = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        let current_window_start = self.window_start.load(Ordering::Relaxed);

        // Check if we need a new time window
        if now.saturating_sub(current_window_start) >= self.window_duration_ns {
            // Start new window - use compare_exchange to handle race conditions
            if self.window_start.compare_exchange(
                current_window_start,
                now,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).is_ok() {
                // Successfully started new window, reset token count
                self.current_window_tokens.store(1, Ordering::Relaxed);
                return true;
            }
            // Another thread started the window, fall through to normal check
        }

        // Check current window token count
        let current_tokens = self.current_window_tokens.load(Ordering::Relaxed);
        let max_tokens_in_window = self.max_tokens_per_second as usize;

        if current_tokens < max_tokens_in_window {
            // Attempt to increment token count atomically
            let new_count = self.current_window_tokens.fetch_add(1, Ordering::Relaxed) + 1;
            new_count <= max_tokens_in_window
        } else {
            false
        }
    }

    /// Check current token count in window
    #[inline]
    pub fn current_token_count(&self) -> usize {
        self.current_window_tokens.load(Ordering::Relaxed)
    }

    /// Get maximum tokens allowed per window
    #[inline]
    pub fn max_tokens_per_window(&self) -> usize {
        if self.enabled {
            self.max_tokens_per_second as usize
        } else {
            usize::MAX
        }
    }

    /// Get current utilization ratio (0.0 to 1.0+)
    #[inline]
    pub fn utilization_ratio(&self) -> f64 {
        if !self.enabled {
            return 0.0;
        }

        let current = self.current_token_count() as f64;
        let max = self.max_tokens_per_second;
        
        if max > 0.0 {
            current / max
        } else {
            0.0
        }
    }

    /// Update rate limit
    #[inline]
    pub fn update_rate_limit(&mut self, max_tokens_per_second: f64) {
        self.max_tokens_per_second = max_tokens_per_second;
        self.enabled = max_tokens_per_second > 0.0;

        // Reset window to apply new rate immediately
        self.window_start.store(0, Ordering::Relaxed);
        self.current_window_tokens.store(0, Ordering::Relaxed);
    }

    /// Update window duration
    #[inline]
    pub fn update_window_duration(&mut self, window_duration_ms: u64) {
        self.window_duration_ns = window_duration_ms * 1_000_000;

        // Reset window to apply new duration immediately  
        self.window_start.store(0, Ordering::Relaxed);
        self.current_window_tokens.store(0, Ordering::Relaxed);
    }

    /// Enable or disable rate limiting
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.max_tokens_per_second > 0.0;
        
        if !self.enabled {
            // Reset window when disabling
            self.window_start.store(0, Ordering::Relaxed);
            self.current_window_tokens.store(0, Ordering::Relaxed);
        }
    }

    /// Check if rate limiting is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current rate limit
    #[inline]
    pub fn rate_limit(&self) -> f64 {
        self.max_tokens_per_second
    }

    /// Get window duration in milliseconds
    #[inline]
    pub fn window_duration_ms(&self) -> u64 {
        self.window_duration_ns / 1_000_000
    }

    /// Reset rate limiter state
    #[inline]
    pub fn reset(&mut self) {
        self.window_start.store(0, Ordering::Relaxed);
        self.current_window_tokens.store(0, Ordering::Relaxed);
    }

    /// Estimate time until next token is allowed (microseconds)
    #[inline]
    pub fn time_until_next_token_us(&self) -> u64 {
        if !self.enabled {
            return 0;
        }

        let current_count = self.current_token_count();
        let max_count = self.max_tokens_per_window();

        if current_count < max_count {
            return 0; // Token immediately available
        }

        // Estimate time until window resets
        let now = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        let window_start = self.window_start.load(Ordering::Relaxed);
        let window_age_ns = now.saturating_sub(window_start);
        let remaining_window_ns = self.window_duration_ns.saturating_sub(window_age_ns);

        remaining_window_ns / 1_000 // Convert to microseconds
    }
}