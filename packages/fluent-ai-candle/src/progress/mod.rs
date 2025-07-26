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
