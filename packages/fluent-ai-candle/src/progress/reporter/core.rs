//! Core ProgressHub reporter implementation

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::progress::config::ProgressHubConfig;
use crate::progress::metrics::{MetricsAggregator, InferenceMetrics};

/// High-performance ProgressHub reporter implementation
pub struct ProgressHubReporter {
    /// Reporter configuration
    config: ProgressHubConfig,
    /// Metrics aggregator for statistics
    metrics: Arc<MetricsAggregator>,
    /// Current session ID
    current_session: Arc<parking_lot::RwLock<Option<String>>>,
    /// Reporter active flag
    active: AtomicBool,
    /// Last update timestamp
    last_update: AtomicU64,
    /// Update counter for rate limiting
    update_counter: AtomicU64,
    /// Inference metrics
    inference_metrics: Arc<parking_lot::RwLock<InferenceMetrics>>,
}

impl ProgressHubReporter {
    /// Create new ProgressHub reporter with default configuration
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_config(ProgressHubConfig::default())
    }

    /// Create new ProgressHub reporter with custom configuration
    pub fn with_config(config: ProgressHubConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        config.validate().map_err(|e| format!("Invalid config: {}", e))?;

        let reporter = Self {
            config,
            metrics: Arc::new(MetricsAggregator::new()),
            current_session: Arc::new(parking_lot::RwLock::new(None)),
            active: AtomicBool::new(true),
            last_update: AtomicU64::new(0),
            update_counter: AtomicU64::new(0),
            inference_metrics: Arc::new(parking_lot::RwLock::new(InferenceMetrics::new())),
        };

        Ok(reporter)
    }

    /// Check if reporter is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Deactivate the reporter
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Get current configuration
    pub fn config(&self) -> &ProgressHubConfig {
        &self.config
    }

    /// Get metrics aggregator
    pub fn metrics(&self) -> &Arc<MetricsAggregator> {
        &self.metrics
    }

    /// Get current inference metrics
    pub fn inference_metrics(&self) -> InferenceMetrics {
        self.inference_metrics.read().clone()
    }

    /// Reset all metrics
    pub fn reset_metrics(&self) {
        self.metrics.clear();
        self.inference_metrics.write().reset();
        self.update_counter.store(0, Ordering::Relaxed);
    }

    /// Check if update should be rate limited
    pub(super) fn should_rate_limit(&self) -> bool {
        if !self.config.enable_realtime {
            return false;
        }

        let now = Instant::now().elapsed().as_millis() as u64;
        let last_update = self.last_update.load(Ordering::Relaxed);
        
        if now.saturating_sub(last_update) < self.config.update_interval_ms {
            return true;
        }

        self.last_update.store(now, Ordering::Relaxed);
        false
    }

    /// Internal method to send progress update
    pub(super) fn send_update(&self, message: &str, progress: Option<f64>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_active() || self.should_rate_limit() {
            return Ok(());
        }

        self.update_counter.fetch_add(1, Ordering::Relaxed);

        // In a real implementation, this would send to ProgressHub
        // For now, we'll just log to stderr for debugging
        if self.config.enable_detailed_metrics {
            if let Some(progress) = progress {
                eprintln!("ProgressHub: {} ({:.1}%)", message, progress * 100.0);
            } else {
                eprintln!("ProgressHub: {}", message);
            }
        }

        Ok(())
    }

    /// Get current session ID
    pub fn current_session_id(&self) -> Option<String> {
        self.current_session.read().clone()
    }

    /// Get number of updates sent
    pub fn update_count(&self) -> u64 {
        self.update_counter.load(Ordering::Relaxed)
    }

    /// Get reference to current session lock
    pub(super) fn current_session(&self) -> &Arc<parking_lot::RwLock<Option<String>>> {
        &self.current_session
    }

    /// Get reference to inference metrics lock
    pub(super) fn inference_metrics_ref(&self) -> &Arc<parking_lot::RwLock<InferenceMetrics>> {
        &self.inference_metrics
    }
}

impl Drop for ProgressHubReporter {
    fn drop(&mut self) {
        // End current session on drop
        if let Some(session_id) = self.current_session_id() {
            let _ = self.metrics.end_session(&session_id);
        }
        
        self.deactivate();
    }
}