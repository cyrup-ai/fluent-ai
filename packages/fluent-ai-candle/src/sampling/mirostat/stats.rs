//! Mirostat Statistics Module
//!
//! Processing statistics and performance metrics for Mirostat sampling.
//! Provides zero-allocation statistical tracking and convergence analysis.

use super::config::MirostatConfig;

/// Processing statistics for Mirostat sampler
#[derive(Debug, Clone, Copy)]
pub struct MirostatStats {
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Average processing time in nanoseconds
    pub avg_processing_time_nanos: f64,
    /// Current dynamic tau value
    pub current_tau: f32,
    /// Current perplexity estimate
    pub current_perplexity: f32,
    /// Perplexity variance (stability measure)
    pub perplexity_variance: f32,
    /// Algorithm configuration
    pub config: MirostatConfig}

impl MirostatStats {
    /// Check if perplexity is stable (low variance)
    #[inline(always)]
    pub fn is_stable(&self) -> bool {
        self.perplexity_variance < 1.0 && self.tokens_processed > 10
    }

    /// Get convergence ratio (how close tau is to target)
    #[inline(always)]
    pub fn convergence_ratio(&self) -> f32 {
        let target_tau = self.config.tau();
        if target_tau > 0.0 {
            (self.current_tau / target_tau).min(target_tau / self.current_tau)
        } else {
            0.0
        }
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "{} | τ={:.2} (target {:.2}) | perplexity={:.2}±{:.2} | {:.1}μs/token | {} tokens",
            self.config.variant_name(),
            self.current_tau,
            self.config.tau(),
            self.current_perplexity,
            self.perplexity_variance.sqrt(),
            self.avg_processing_time_nanos / 1000.0,
            self.tokens_processed
        )
    }

    /// Get processing rate in tokens per second
    #[inline(always)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.avg_processing_time_nanos > 0.0 {
            1_000_000_000.0 / self.avg_processing_time_nanos
        } else {
            0.0
        }
    }

    /// Get tau deviation from target as percentage
    #[inline(always)]
    pub fn tau_deviation_percent(&self) -> f32 {
        let target = self.config.tau();
        if target > 0.0 {
            ((self.current_tau - target) / target * 100.0).abs()
        } else {
            0.0
        }
    }

    /// Check if tau has converged to target (within 5%)
    #[inline(always)]
    pub fn has_converged(&self) -> bool {
        self.tau_deviation_percent() < 5.0 && self.tokens_processed > 5
    }

    /// Get quality score (0-1) based on stability and convergence
    #[inline(always)]
    pub fn quality_score(&self) -> f32 {
        let stability_score = if self.is_stable() { 1.0 } else { 0.5 };
        let convergence_score = self.convergence_ratio();
        let processing_score = if self.avg_processing_time_nanos < 10_000.0 { 1.0 } else { 0.8 };
        
        (stability_score + convergence_score + processing_score) / 3.0
    }

    /// Create default stats for testing
    #[cfg(test)]
    pub fn default_for_testing() -> Self {
        Self {
            tokens_processed: 0,
            avg_processing_time_nanos: 0.0,
            current_tau: 5.0,
            current_perplexity: 1.0,
            perplexity_variance: 0.0,
            config: MirostatConfig::default()}
    }
}

impl Default for MirostatStats {
    fn default() -> Self {
        Self {
            tokens_processed: 0,
            avg_processing_time_nanos: 0.0,
            current_tau: 5.0,
            current_perplexity: 1.0,
            perplexity_variance: 0.0,
            config: MirostatConfig::default()}
    }
}
