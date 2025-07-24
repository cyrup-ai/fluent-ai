//! ProgressHub reporter implementation

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::traits::ProgressReporter;
use super::config::ProgressHubConfig;
use super::metrics::{MetricsAggregator, InferenceMetrics};
use super::stages::{DownloadStage, WeightLoadingStage, QuantizationStage};

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
    fn should_rate_limit(&self) -> bool {
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
    fn send_update(&self, message: &str, progress: Option<f64>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
}

impl ProgressReporter for ProgressHubReporter {
    fn report_progress(&self, message: &str, progress: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let clamped_progress = progress.clamp(0.0, 1.0);
        self.send_update(message, Some(clamped_progress))
    }

    fn report_stage_completion(&self, stage_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let message = format!("Completed: {}", stage_name);
        self.send_update(&message, Some(1.0))
    }

    fn report_generation_metrics(
        &self,
        tokens_per_sec: f64,
        cache_hit_rate: f64,
        latency_nanos: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Update internal metrics
        {
            let mut metrics = self.inference_metrics.write();
            let duration = Duration::from_nanos(latency_nanos);
            let tokens = (tokens_per_sec * duration.as_secs_f64()).round() as u64;
            let cache_hits = (cache_hit_rate * 100.0).round() as u64;
            metrics.update_generation(tokens, duration, cache_hits, 100);
        }

        // Record in aggregator
        let tokens = (tokens_per_sec * (latency_nanos as f64 / 1_000_000_000.0)).round() as u64;
        self.metrics.record_tokens(tokens);

        let message = format!(
            "Generation: {:.1} tok/s, {:.1}% cache hit, {:.2}ms latency",
            tokens_per_sec,
            cache_hit_rate * 100.0,
            latency_nanos as f64 / 1_000_000.0
        );
        self.send_update(&message, None)
    }

    fn report_model_loading(
        &self,
        stage: DownloadStage,
        progress: f64,
        bytes_loaded: u64,
        total_bytes: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let percentage = if total_bytes > 0 {
            (bytes_loaded as f64 / total_bytes as f64) * 100.0
        } else {
            progress * 100.0
        };

        let message = format!(
            "{}: {:.1}% ({} / {} bytes)",
            stage.description(),
            percentage,
            bytes_loaded,
            total_bytes
        );
        
        let (stage_start, stage_end) = stage.progress_range();
        let stage_progress = stage_start + (stage_end - stage_start) * progress;
        
        self.send_update(&message, Some(stage_progress))
    }

    fn report_weight_loading(
        &self,
        stage: WeightLoadingStage,
        layer_index: usize,
        total_layers: usize,
        memory_usage_mb: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Update memory metrics
        {
            let mut metrics = self.inference_metrics.write();
            metrics.update_memory(memory_usage_mb);
        }

        let layer_progress = if total_layers > 0 {
            layer_index as f64 / total_layers as f64
        } else {
            0.0
        };

        let message = format!(
            "{}: Layer {}/{} ({:.1} MB)",
            stage.description(),
            layer_index,
            total_layers,
            memory_usage_mb
        );

        let (stage_start, stage_end) = stage.progress_range();
        let stage_progress = stage_start + (stage_end - stage_start) * layer_progress;

        self.send_update(&message, Some(stage_progress))
    }

    fn report_quantization(
        &self,
        stage: QuantizationStage,
        tensors_processed: usize,
        total_tensors: usize,
        compression_ratio: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tensor_progress = if total_tensors > 0 {
            tensors_processed as f64 / total_tensors as f64
        } else {
            0.0
        };

        let message = format!(
            "{}: {}/{} tensors ({:.1}% compression)",
            stage.description(),
            tensors_processed,
            total_tensors,
            compression_ratio * 100.0
        );

        let (stage_start, stage_end) = stage.progress_range();
        let stage_progress = stage_start + (stage_end - stage_start) * tensor_progress;

        self.send_update(&message, Some(stage_progress))
    }

    fn report_error(&self, error_message: &str, context: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.metrics.record_failure();
        
        let message = format!("Error in {}: {}", context, error_message);
        eprintln!("ProgressHub Error: {}", message);
        
        self.send_update(&message, None)
    }

    fn report_completion(&self, success: bool, total_duration_ms: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let status = if success { "SUCCESS" } else { "FAILED" };
        let message = format!("Operation {} in {}ms", status, total_duration_ms);
        
        if !success {
            self.metrics.record_failure();
        }
        
        self.send_update(&message, Some(1.0))
    }

    fn report_memory_usage(&self, allocated_mb: f64, peak_mb: f64, available_mb: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        {
            let mut metrics = self.inference_metrics.write();
            metrics.update_memory(allocated_mb);
        }

        let message = format!(
            "Memory: {:.1} MB allocated, {:.1} MB peak, {:.1} MB available",
            allocated_mb, peak_mb, available_mb
        );
        
        self.send_update(&message, None)
    }

    fn start_session(&self, session_id: &str, operation_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // End current session if exists
        if let Some(current_id) = self.current_session_id() {
            self.metrics.end_session(&current_id);
        }

        // Start new session
        self.metrics.start_session(session_id.to_string(), operation_name.to_string());
        *self.current_session.write() = Some(session_id.to_string());

        let message = format!("Started session '{}': {}", session_id, operation_name);
        self.send_update(&message, Some(0.0))
    }

    fn end_session(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(session_id) = self.current_session_id() {
            if let Some(session_info) = self.metrics.end_session(&session_id) {
                let message = format!(
                    "Ended session '{}': {} tokens in {:.2}s",
                    session_id,
                    session_info.tokens(),
                    session_info.duration().as_secs_f64()
                );
                self.send_update(&message, Some(1.0))?;
            }
            *self.current_session.write() = None;
        }

        Ok(())
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