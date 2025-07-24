//! Ultra-High-Performance Progress Reporting for ML Inference
//!
//! Zero-allocation, lock-free progress reporting system integrated with ProgressHub TUI:
//! - Real-time inference progress tracking with nanosecond precision
//! - Token generation metrics with blazing-fast updates
//! - Cache hit rate monitoring with atomic counters
//! - Concurrent operation aggregation with wait-free algorithms
//! - Model loading progress with detailed stage reporting
//! - Weight quantization tracking with memory-efficient updates
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
//! │  ML Operations  │ -> │ ProgressReporter │ -> │   ProgressHub TUI   │
//! │ (Model Loading) │    │  (Lock-Free)     │    │  (Real-Time)        │
//! └─────────────────┘    └──────────────────┘    └─────────────────────┘
//!                               │
//!                        ┌──────────────────┐
//!                        │ MetricsAggregator│
//!                        │ (Concurrent)     │
//!                        └──────────────────┘
//! ```
//!
//! ## Performance Features
//!
//! - **Zero Allocation**: Stack-based progress data, pre-allocated buffers
//! - **Lock-Free**: Atomic operations, wait-free concurrent updates
//! - **Non-Blocking**: Never blocks inference operations
//! - **Real-Time**: Nanosecond-precision timing and updates
//! - **Memory Efficient**: Compact progress structures, intelligent batching
//! - **Thread-Safe**: Concurrent progress reporting without synchronization
//!
//! ## Usage Examples
//!
//! ### Basic Progress Reporting
//!
//! ```rust
//! use fluent_ai_candle::progress::{ProgressReporter, ProgressHubReporter};
//!
//! // Create high-performance reporter
//! let reporter = ProgressHubReporter::new()?;
//!
//! // Report model loading progress
//! reporter.report_progress("Loading weights", 0.45)?;
//! reporter.report_stage_completion("Weight loading")?;
//!
//! // Report generation metrics
//! reporter.report_generation_metrics(tokens_per_sec, cache_hit_rate, latency_nanos)?;
//! ```
//!
//! ### Advanced Session Management
//!
//! ```rust
//! // Start named session for operation tracking
//! reporter.start_session("llama_inference_001", "LLaMA-7B Inference")?;
//!
//! // Report detailed model loading stages
//! reporter.report_model_loading(
//!     DownloadStage::Downloading,
//!     0.75,
//!     1_610_612_736,  // bytes loaded
//!     2_147_483_648   // total bytes
//! )?;
//!
//! // Report weight loading with memory tracking
//! reporter.report_weight_loading(
//!     WeightLoadingStage::LoadingLayers,
//!     12,     // current layer
//!     24,     // total layers
//!     2048.0  // memory usage MB
//! )?;
//!
//! // End session when complete
//! reporter.end_session()?;
//! ```

pub mod traits;
pub mod stages;
pub mod config;
pub mod metrics;
pub mod reporter;

// Re-export main types for convenience
pub use traits::ProgressReporter;
pub use stages::{DownloadStage, WeightLoadingStage, QuantizationStage, InferenceStage};
pub use config::ProgressHubConfig;
pub use metrics::{InferenceMetrics, MetricsAggregator, SessionInfo, AggregatorStats};
pub use reporter::ProgressHubReporter;

// Type aliases for convenience
pub type Reporter = ProgressHubReporter;
pub type Config = ProgressHubConfig;
pub type Metrics = InferenceMetrics;

/// Create a new ProgressHub reporter with default configuration
pub fn create_reporter() -> Result<ProgressHubReporter, Box<dyn std::error::Error + Send + Sync>> {
    ProgressHubReporter::new()
}

/// Create a new ProgressHub reporter with custom configuration
pub fn create_reporter_with_config(config: ProgressHubConfig) -> Result<ProgressHubReporter, Box<dyn std::error::Error + Send + Sync>> {
    ProgressHubReporter::with_config(config)
}

/// Create a low-latency optimized reporter
pub fn create_low_latency_reporter() -> Result<ProgressHubReporter, Box<dyn std::error::Error + Send + Sync>> {
    ProgressHubReporter::with_config(ProgressHubConfig::low_latency())
}

/// Create a high-throughput optimized reporter
pub fn create_high_throughput_reporter() -> Result<ProgressHubReporter, Box<dyn std::error::Error + Send + Sync>> {
    ProgressHubReporter::with_config(ProgressHubConfig::high_throughput())
}

/// Create a minimal resource usage reporter
pub fn create_minimal_reporter() -> Result<ProgressHubReporter, Box<dyn std::error::Error + Send + Sync>> {
    ProgressHubReporter::with_config(ProgressHubConfig::minimal())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_progress_reporting() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test basic progress reporting
        reporter.report_progress("Test progress", 0.5).expect("Failed to report progress");
        reporter.report_stage_completion("Test stage").expect("Failed to report completion");
        
        assert!(reporter.is_active());
        assert_eq!(reporter.update_count(), 2);
    }

    #[test]
    fn test_session_management() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test session management
        reporter.start_session("test_session", "Test Operation").expect("Failed to start session");
        assert_eq!(reporter.current_session_id(), Some("test_session".to_string()));
        
        reporter.end_session().expect("Failed to end session");
        assert_eq!(reporter.current_session_id(), None);
    }

    #[test]
    fn test_concurrent_reporting() {
        let reporter = Arc::new(create_reporter().expect("Failed to create reporter"));
        let mut handles = vec![];

        // Test concurrent access
        for i in 0..10 {
            let reporter_clone = Arc::clone(&reporter);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let progress = (j as f64) / 100.0;
                    let message = format!("Worker {} progress", i);
                    reporter_clone.report_progress(&message, progress).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify metrics were updated
        let stats = reporter.metrics().get_stats();
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_metrics_aggregation() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test generation metrics
        reporter.report_generation_metrics(42.5, 0.85, 125_000).expect("Failed to report metrics");
        
        let metrics = reporter.inference_metrics();
        assert!(metrics.total_operations > 0);
        assert!(metrics.avg_tokens_per_sec > 0.0);
    }

    #[test]
    fn test_configuration_variants() {
        // Test different configuration presets
        let low_latency = create_low_latency_reporter().expect("Failed to create low latency reporter");
        assert!(low_latency.config().is_performance_optimized());

        let high_throughput = create_high_throughput_reporter().expect("Failed to create high throughput reporter");
        assert_eq!(high_throughput.config().max_concurrent_sessions, 1000);

        let minimal = create_minimal_reporter().expect("Failed to create minimal reporter");
        assert!(minimal.config().is_resource_efficient());
    }

    #[test]
    fn test_error_handling() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test error reporting
        reporter.report_error("Test error", "test_context").expect("Failed to report error");
        
        let stats = reporter.metrics().get_stats();
        assert!(stats.failed_operations > 0);
    }
}