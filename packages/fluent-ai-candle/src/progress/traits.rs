//! Progress reporting traits and interfaces

// Removed unused import: Arc

/// Core progress reporting trait for ML operations
/// 
/// Provides a unified interface for reporting progress across different stages
/// of ML model operations including loading, quantization, and inference.
pub trait ProgressReporter: Send + Sync {
    /// Report general progress with message and percentage
    ///
    /// # Arguments
    /// * `message` - Human-readable progress message
    /// * `progress` - Progress percentage (0.0 to 1.0)
    ///
    /// # Example
    /// ```
    /// reporter.report_progress("Loading model weights", 0.75)?;
    /// ```
    fn report_progress(&self, message: &str, progress: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report completion of a specific stage
    ///
    /// # Arguments
    /// * `stage_name` - Name of the completed stage
    ///
    /// # Example
    /// ```
    /// reporter.report_stage_completion("Weight quantization")?;
    /// ```
    fn report_stage_completion(&self, stage_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report detailed generation metrics
    ///
    /// # Arguments
    /// * `tokens_per_sec` - Token generation rate
    /// * `cache_hit_rate` - KV cache hit rate (0.0 to 1.0)
    /// * `latency_nanos` - Generation latency in nanoseconds
    ///
    /// # Example
    /// ```
    /// reporter.report_generation_metrics(42.5, 0.85, 125_000)?;
    /// ```
    fn report_generation_metrics(
        &self,
        tokens_per_sec: f64,
        cache_hit_rate: f64,
        latency_nanos: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report model loading progress with detailed stage information
    ///
    /// # Arguments
    /// * `stage` - Current loading stage
    /// * `progress` - Stage progress (0.0 to 1.0)
    /// * `bytes_loaded` - Number of bytes loaded so far
    /// * `total_bytes` - Total bytes to load
    ///
    /// # Example
    /// ```
    /// reporter.report_model_loading(
    ///     DownloadStage::Downloading, 
    ///     0.65, 
    ///     1_048_576, 
    ///     1_610_612_736
    /// )?;
    /// ```
    fn report_model_loading(
        &self,
        stage: crate::progress::stages::DownloadStage,
        progress: f64,
        bytes_loaded: u64,
        total_bytes: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report weight loading progress
    ///
    /// # Arguments
    /// * `stage` - Current weight loading stage
    /// * `layer_index` - Index of current layer being loaded
    /// * `total_layers` - Total number of layers
    /// * `memory_usage_mb` - Current memory usage in MB
    ///
    /// # Example
    /// ```
    /// reporter.report_weight_loading(
    ///     WeightLoadingStage::LoadingLayers,
    ///     12,
    ///     24,
    ///     2048.5
    /// )?;
    /// ```
    fn report_weight_loading(
        &self,
        stage: crate::progress::stages::WeightLoadingStage,
        layer_index: usize,
        total_layers: usize,
        memory_usage_mb: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report quantization progress
    ///
    /// # Arguments
    /// * `stage` - Current quantization stage
    /// * `tensors_processed` - Number of tensors processed
    /// * `total_tensors` - Total number of tensors
    /// * `compression_ratio` - Current compression ratio
    ///
    /// # Example
    /// ```
    /// reporter.report_quantization(
    ///     QuantizationStage::Quantizing,
    ///     150,
    ///     200,
    ///     0.25
    /// )?;
    /// ```
    fn report_quantization(
        &self,
        stage: crate::progress::stages::QuantizationStage,
        tensors_processed: usize,
        total_tensors: usize,
        compression_ratio: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report error during operation
    ///
    /// # Arguments
    /// * `error_message` - Human-readable error description
    /// * `context` - Context where the error occurred
    ///
    /// # Example
    /// ```
    /// reporter.report_error("Failed to allocate memory", "weight_loading")?;
    /// ```
    fn report_error(&self, error_message: &str, context: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report completion of entire operation
    ///
    /// # Arguments
    /// * `success` - Whether the operation completed successfully
    /// * `total_duration_ms` - Total operation duration in milliseconds
    ///
    /// # Example
    /// ```
    /// reporter.report_completion(true, 5432)?;
    /// ```
    fn report_completion(&self, success: bool, total_duration_ms: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Report memory usage statistics
    ///
    /// # Arguments
    /// * `allocated_mb` - Currently allocated memory in MB
    /// * `peak_mb` - Peak memory usage in MB
    /// * `available_mb` - Available system memory in MB
    ///
    /// # Example
    /// ```
    /// reporter.report_memory_usage(1024.0, 1536.0, 8192.0)?;
    /// ```
    fn report_memory_usage(&self, allocated_mb: f64, peak_mb: f64, available_mb: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Start a new session with given ID
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    /// * `operation_name` - Name of the operation being started
    ///
    /// # Example
    /// ```
    /// reporter.start_session("model_load_001", "LLaMA-7B Loading")?;
    /// ```
    fn start_session(&self, session_id: &str, operation_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// End the current session
    ///
    /// # Example
    /// ```
    /// reporter.end_session()?;
    /// ```
    fn end_session(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}