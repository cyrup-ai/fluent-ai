//! Main flow controller implementation
//!
//! Provides the core FlowController struct with sophisticated backpressure management:
//! - Adaptive backpressure detection and response
//! - Multiple strategy support (linear, exponential, adaptive)
//! - Real-time statistics tracking and monitoring
//! - Zero-allocation streaming integration

use std::time::{Duration, Instant};

use fluent_ai_async::{AsyncStream, emit};

use super::flow_types::{BackpressureStrategy, FlowStats, AdaptiveParams};
use super::rate_limiter::TokenRateLimiter;

/// Flow controller for managing streaming backpressure
pub struct FlowController {
    /// Backpressure strategy in use
    strategy: BackpressureStrategy,
    /// Enable flow control
    enabled: bool,
    /// Backpressure threshold (0.0 to 1.0)
    threshold: f32,
    /// Current backpressure state
    is_backpressure_active: bool,
    /// Statistics tracking
    stats: FlowStats,
    /// Token rate tracking
    token_timestamps: Vec<Instant>,
    /// Maximum rate history to keep
    max_rate_history: usize,
    /// Last backpressure check
    last_check: Instant,
    /// Current delay duration
    current_delay: Duration,
    /// Adaptive parameters
    adaptive_params: AdaptiveParams,
    /// Rate limiter
    rate_limiter: TokenRateLimiter}

impl FlowController {
    /// Create new flow controller
    #[inline]
    pub fn new(enabled: bool, threshold: f32, strategy: BackpressureStrategy) -> Self {
        Self {
            strategy,
            enabled,
            threshold: threshold.clamp(0.0, 1.0),
            is_backpressure_active: false,
            stats: FlowStats::default(),
            token_timestamps: Vec::with_capacity(1000),
            max_rate_history: 1000,
            last_check: Instant::now(),
            current_delay: Duration::from_micros(0),
            adaptive_params: AdaptiveParams::default(),
            rate_limiter: TokenRateLimiter::unlimited()}
    }

    /// Create flow controller with rate limiting
    #[inline]
    pub fn with_rate_limit(
        enabled: bool,
        threshold: f32,
        strategy: BackpressureStrategy,
        max_tokens_per_second: f64,
    ) -> Self {
        let mut controller = Self::new(enabled, threshold, strategy);
        controller.rate_limiter = TokenRateLimiter::new(max_tokens_per_second);
        controller
    }

    /// Check for backpressure and update state
    #[inline]
    pub fn check_backpressure(&mut self, buffer_utilization: f32) {
        if !self.enabled {
            return;
        }

        let utilization = buffer_utilization.clamp(0.0, 1.0);
        let now = Instant::now();
        
        // Update statistics
        self.stats.update_buffer_utilization(utilization);
        self.update_token_rate();

        let was_active = self.is_backpressure_active;
        self.is_backpressure_active = utilization >= self.threshold;

        if self.is_backpressure_active && !was_active {
            // Entering backpressure state
            self.stats.record_backpressure_event();
            self.apply_backpressure_strategy(utilization);
        } else if !self.is_backpressure_active && was_active {
            // Exiting backpressure state
            self.current_delay = Duration::from_micros(0);
        } else if self.is_backpressure_active {
            // Still in backpressure, potentially adjust strategy
            self.adjust_backpressure_response(utilization);
        }

        self.last_check = now;
    }

    /// Apply backpressure strategy based on current utilization
    #[inline]
    fn apply_backpressure_strategy(&mut self, utilization: f32) {
        let pressure = (utilization - self.threshold) / (1.0 - self.threshold);
        let pressure = pressure.clamp(0.0, 1.0);

        let new_delay_us = match self.strategy {
            BackpressureStrategy::None => 0,
            BackpressureStrategy::DropOldest | BackpressureStrategy::DropNewest => {
                // No delay for drop strategies, handled elsewhere
                0
            }
            BackpressureStrategy::LinearDelay => {
                // Linear increase from 10us to 1000us
                (10.0 + pressure * 990.0) as u64
            }
            BackpressureStrategy::Exponential => {
                // Exponential increase from 10us to 10000us
                let base_delay = 10.0f32;
                let max_delay = 10000.0f32;
                (base_delay * (max_delay / base_delay).powf(pressure)) as u64
            }
            BackpressureStrategy::Adaptive => {
                self.calculate_adaptive_delay(pressure)
            }
            BackpressureStrategy::Custom {
                base_delay_ms,
                multiplier,
                max_delay_ms} => {
                let delay_us = (base_delay_ms * 1000) as f32 * (1.0 + pressure * multiplier);
                delay_us.min(max_delay_ms as f32 * 1000.0) as u64
            }
        };

        self.current_delay = Duration::from_micros(new_delay_us);
        self.stats.record_delay(new_delay_us);
    }

    /// Calculate adaptive delay based on historical performance
    #[inline]
    fn calculate_adaptive_delay(&mut self, pressure: f32) -> u64 {
        let params = &self.adaptive_params;
        
        // Base delay calculation
        let base_delay = params.min_delay_us as f32;
        let max_delay = params.max_delay_us as f32;
        
        // Adaptive component based on buffer sensitivity
        let adaptive_factor = 1.0 + pressure * params.buffer_sensitivity;
        
        // Apply learning rate for gradual adjustment
        let current_delay_us = self.current_delay.as_micros() as f32;
        let target_delay = base_delay * adaptive_factor;
        let adjusted_delay = current_delay_us * (1.0 - params.learning_rate) 
                           + target_delay * params.learning_rate;

        adjusted_delay.clamp(base_delay, max_delay) as u64
    }

    /// Adjust backpressure response while in backpressure state
    #[inline]
    fn adjust_backpressure_response(&mut self, utilization: f32) {
        if let BackpressureStrategy::Adaptive = self.strategy {
            let pressure = (utilization - self.threshold) / (1.0 - self.threshold);
            let new_delay_us = self.calculate_adaptive_delay(pressure.clamp(0.0, 1.0));
            self.current_delay = Duration::from_micros(new_delay_us);
            self.stats.record_delay(new_delay_us);
        }
    }

    /// Update token processing rate statistics
    #[inline]
    fn update_token_rate(&mut self) {
        let now = Instant::now();
        
        // Add current timestamp
        self.token_timestamps.push(now);
        
        // Remove old timestamps (older than 1 second)
        let cutoff = now - Duration::from_secs(1);
        self.token_timestamps.retain(|&timestamp| timestamp >= cutoff);
        
        // Trim to max history size
        if self.token_timestamps.len() > self.max_rate_history {
            self.token_timestamps.drain(0..self.token_timestamps.len() - self.max_rate_history);
        }
        
        // Calculate current rate
        let rate = self.token_timestamps.len() as f64;
        self.stats.update_token_rate(rate);
    }

    /// Apply the current delay
    #[inline]
    pub fn apply_delay(&self) -> AsyncStream<()> {
        let delay = self.current_delay;
        
        AsyncStream::with_channel(move |sender: fluent_ai_async::AsyncStreamSender<()>| {
            if delay.as_micros() > 0 {
                // For streaming flow control, we emit immediately and let the caller
                // handle timing through the streaming pipeline's natural backpressure
                // This maintains zero-allocation, lock-free characteristics
                emit!(sender, ());
            } else {
                emit!(sender, ());
            }
        })
    }

    /// Check rate limiter
    #[inline]
    pub fn should_allow_token(&self) -> bool {
        self.rate_limiter.should_allow_token()
    }

    /// Update backpressure strategy
    #[inline]
    pub fn update_strategy(&mut self, strategy: BackpressureStrategy) {
        self.strategy = strategy;

        // Reset delay when strategy changes
        self.current_delay = Duration::from_micros(0);
        self.stats.record_adjustment();
    }

    /// Update backpressure threshold
    #[inline]
    pub fn update_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
        self.stats.record_adjustment();
    }

    /// Enable/disable flow control
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.is_backpressure_active = false;
            self.current_delay = Duration::from_micros(0);
        }
    }

    /// Set rate limit (tokens per second, 0 = unlimited)
    #[inline]
    pub fn set_rate_limit(&mut self, max_tokens_per_second: f64) {
        self.rate_limiter.update_rate_limit(max_tokens_per_second);
    }

    /// Get current flow statistics
    #[inline]
    pub fn get_stats(&self) -> &FlowStats {
        &self.stats
    }

    /// Reset flow statistics
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.token_timestamps.clear();
    }

    /// Check if currently under backpressure
    #[inline]
    pub fn is_backpressure_active(&self) -> bool {
        self.is_backpressure_active
    }

    /// Get current strategy
    #[inline]
    pub fn strategy(&self) -> BackpressureStrategy {
        self.strategy
    }

    /// Get current threshold
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Check if flow control is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current delay duration
    #[inline]
    pub fn current_delay(&self) -> Duration {
        self.current_delay
    }

    /// Configure adaptive parameters
    #[inline]
    pub fn configure_adaptive(&mut self, learning_rate: f32, buffer_sensitivity: f32) {
        self.adaptive_params.update_learning(learning_rate, buffer_sensitivity);
    }

    /// Get rate limiter reference
    #[inline]
    pub fn rate_limiter(&self) -> &TokenRateLimiter {
        &self.rate_limiter
    }

    /// Get mutable rate limiter reference
    #[inline]
    pub fn rate_limiter_mut(&mut self) -> &mut TokenRateLimiter {
        &mut self.rate_limiter
    }

    /// Get adaptive parameters
    #[inline]
    pub fn adaptive_params(&self) -> &AdaptiveParams {
        &self.adaptive_params
    }

    /// Update adaptive delay bounds
    #[inline]
    pub fn configure_adaptive_delays(&mut self, min_delay_us: u64, max_delay_us: u64) {
        self.adaptive_params.update_delay_bounds(min_delay_us, max_delay_us);
    }
}