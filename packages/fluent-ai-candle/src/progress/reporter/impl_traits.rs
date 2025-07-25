//! ProgressReporter trait implementation for ProgressHubReporter

use std::time::Duration;
use crate::progress::traits::ProgressReporter;
use crate::progress::stages::{DownloadStage, WeightLoadingStage, QuantizationStage};
use super::core::ProgressHubReporter;

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
            let mut metrics = self.inference_metrics_ref().write();
            let duration = Duration::from_nanos(latency_nanos);
            let tokens = (tokens_per_sec * duration.as_secs_f64()).round() as u64;
            let cache_hits = (cache_hit_rate * 100.0).round() as u64;
            metrics.update_generation(tokens, duration, cache_hits, 100);
        }

        // Record in aggregator
        let tokens = (tokens_per_sec * (latency_nanos as f64 / 1_000_000_000.0)).round() as u64;
        self.metrics().record_tokens(tokens);

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
            let mut metrics = self.inference_metrics_ref().write();
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
        self.metrics().record_failure();
        
        let message = format!("Error in {}: {}", context, error_message);
        eprintln!("ProgressHub Error: {}", message);
        
        self.send_update(&message, None)
    }

    fn report_completion(&self, success: bool, total_duration_ms: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let status = if success { "SUCCESS" } else { "FAILED" };
        let message = format!("Operation {} in {}ms", status, total_duration_ms);
        
        if !success {
            self.metrics().record_failure();
        }
        
        self.send_update(&message, Some(1.0))
    }

    fn report_memory_usage(&self, allocated_mb: f64, peak_mb: f64, available_mb: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        {
            let mut metrics = self.inference_metrics_ref().write();
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
            self.metrics().end_session(&current_id);
        }

        // Start new session
        self.metrics().start_session(session_id.to_string(), operation_name.to_string());
        *self.current_session().write() = Some(session_id.to_string());

        let message = format!("Started session '{}': {}", session_id, operation_name);
        self.send_update(&message, Some(0.0))
    }

    fn end_session(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(session_id) = self.current_session_id() {
            if let Some(session_info) = self.metrics().end_session(&session_id) {
                let message = format!(
                    "Ended session '{}': {} tokens in {:.2}s",
                    session_id,
                    session_info.tokens(),
                    session_info.duration().as_secs_f64()
                );
                self.send_update(&message, Some(1.0))?;
            }
            *self.current_session().write() = None;
        }

        Ok(())
    }
}