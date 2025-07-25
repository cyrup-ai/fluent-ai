//! Perplexity State Management Module
//!
//! Zero-allocation perplexity tracking with circular buffer and exponential moving average.
//! Provides lock-free perplexity calculations for Mirostat algorithm stability.

use std::sync::atomic::{AtomicU64, Ordering};
use arrayvec::ArrayVec;

/// Maximum perplexity history for moving average (stack allocated)
const MAX_PERPLEXITY_HISTORY: usize = 32;

/// Zero-allocation perplexity state with circular buffer
#[repr(C, align(64))] // Cache line aligned for performance
pub(crate) struct PerplexityState {
    /// Circular buffer for recent perplexity values (stack allocated)
    pub(crate) history: ArrayVec<f32, MAX_PERPLEXITY_HISTORY>,
    /// Current buffer position (atomic for lock-free access)
    pub(crate) position: AtomicU64,
    /// Accumulated surprise for moving average
    pub(crate) surprise_accumulator: f32,
    /// Sample count for averaging
    pub(crate) sample_count: u32,
    /// Exponential moving average parameter (alpha)
    pub(crate) ema_alpha: f32,
    /// Current EMA value
    pub(crate) current_ema: f32,
}

impl PerplexityState {
    /// Create new perplexity state with EMA parameter
    #[inline(always)]
    pub(crate) fn new(ema_alpha: f32) -> Self {
        Self {
            history: ArrayVec::new(),
            position: AtomicU64::new(0),
            surprise_accumulator: 0.0,
            sample_count: 0,
            ema_alpha: ema_alpha.clamp(0.01, 0.5), // Reasonable bounds for stability
            current_ema: 0.0,
        }
    }

    /// Add new perplexity sample with exponential moving average
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn add_sample(&mut self, perplexity: f32) {
        if !perplexity.is_finite() || perplexity <= 0.0 {
            return; // Skip invalid samples
        }

        // Update exponential moving average
        if self.sample_count == 0 {
            self.current_ema = perplexity;
        } else {
            self.current_ema =
                self.ema_alpha * perplexity + (1.0 - self.ema_alpha) * self.current_ema;
        }

        // Add to circular buffer
        if self.history.is_full() {
            let pos = (self.position.load(Ordering::Relaxed) as usize) % MAX_PERPLEXITY_HISTORY;
            self.history[pos] = perplexity;
            self.position.store(
                self.position.load(Ordering::Relaxed).wrapping_add(1),
                Ordering::Relaxed,
            );
        } else if self.history.try_push(perplexity).is_err() {
            // Buffer overflow protection (should not happen)
            return;
        }

        self.sample_count = self.sample_count.saturating_add(1);
    }

    /// Get current moving average perplexity
    #[inline(always)]
    pub(crate) fn current_perplexity(&self) -> f32 {
        if self.sample_count > 0 {
            self.current_ema
        } else {
            1.0 // Default perplexity
        }
    }

    /// Get perplexity variance for stability assessment
    #[inline(always)]
    pub(crate) fn variance(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mean = self.current_ema;
        let mut sum_squared_diff = 0.0;

        for &value in &self.history {
            let diff = value - mean;
            sum_squared_diff += diff * diff;
        }

        sum_squared_diff / (self.history.len() as f32)
    }

    /// Reset state for new sequence
    #[inline(always)]
    pub(crate) fn reset(&mut self) {
        self.history.clear();
        self.position.store(0, Ordering::Relaxed);
        self.surprise_accumulator = 0.0;
        self.sample_count = 0;
        self.current_ema = 0.0;
    }

    /// Get sample count
    #[inline(always)]
    pub(crate) fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Get history length
    #[inline(always)]
    pub(crate) fn history_length(&self) -> usize {
        self.history.len()
    }

    /// Get EMA alpha parameter
    #[inline(always)]
    pub(crate) fn ema_alpha(&self) -> f32 {
        self.ema_alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_state_creation() {
        let state = PerplexityState::new(0.1);
        assert_eq!(state.current_perplexity(), 1.0);
        assert_eq!(state.sample_count(), 0);
        assert_eq!(state.history_length(), 0);
        assert_eq!(state.ema_alpha(), 0.1);
    }

    #[test]
    fn test_perplexity_variance() {
        let mut state = PerplexityState::new(0.1);
        assert_eq!(state.variance(), 0.0);
        
        #[allow(deprecated)]
        {
            state.add_sample(1.0);
            state.add_sample(2.0);
        }
        
        assert!(state.variance() > 0.0);
    }

    #[test]
    fn test_perplexity_reset() {
        let mut state = PerplexityState::new(0.1);
        
        #[allow(deprecated)]
        {
            state.add_sample(1.5);
        }
        
        assert!(state.sample_count() > 0);
        
        state.reset();
        assert_eq!(state.sample_count(), 0);
        assert_eq!(state.current_perplexity(), 1.0);
    }
}