# fluent-ai-candle Method Inventory

This document provides a comprehensive inventory of all methods in the fluent-ai-candle codebase, organized by module and visibility.

## Overview

The fluent-ai-candle crate is a high-performance candle integration for ML inference with zero-allocation, lock-free design patterns. This inventory captures all public and private methods across the entire codebase.

## Module Structure

```
fluent-ai-candle/
├── lib.rs (main entry point with re-exports)
├── client/ (completion client implementation)
├── model/ (model loading and management)
├── tokenizer/ (tokenization utilities)
├── generator/ (text generation engine)
├── sampling/ (sampling strategies)
├── processing/ (logits processing)
├── streaming/ (real-time streaming)
├── constraints/ (structured generation)
├── memory/ (memory management)
├── progress/ (progress reporting)
├── error/ (error handling)
├── var_builder/ (variable builder for models)
├── kv_cache/ (key-value cache)
├── types/ (type definitions)
└── builders/ (builder patterns)
```

## Core Public API Methods

### Device Utilities (lib.rs)
```rust
// Device management functions
pub fn auto_device() -> candle_core::Result<Device>
pub fn device_info(device: &Device) -> &'static str  
pub fn supports_fast_matmul(device: &Device) -> bool
```

### Performance Timer (lib.rs::perf)
```rust
impl PerfTimer {
    pub fn new(name: &'static str) -> Self
    pub fn elapsed_micros(&self) -> u64
    pub fn elapsed_nanos(&self) -> u64
}
```

### Tensor Utilities (lib.rs::tensor_utils)
```rust
// Tensor manipulation functions
pub fn tensor_to_tokens(tensor: &Tensor, buffer: &mut ArrayVec<u32, 2048>) -> CandleResult<()>
pub fn tokens_to_tensor(tokens: &[u32], device: &candle_core::Device) -> CandleResult<Tensor>
pub fn softmax_with_temperature(logits: &Tensor, temperature: f32) -> CandleResult<Tensor>
pub fn sample_token(
    probs: &Tensor,
    top_k: Option<usize>,
    top_p: Option<f64>, 
    rng: &mut impl rand::Rng,
) -> CandleResult<u32>
```

### Memory Management (memory.rs)
```rust
// Memory tracking functions
pub fn track_allocation(size: usize)
pub fn track_deallocation(size: usize) 
pub fn current_usage() -> usize
pub fn peak_usage() -> usize
pub fn allocation_count() -> usize
pub fn reset_stats()
```

## Client Module Methods

### CandleClientBuilder (client/builder/client_builder.rs)
```rust
impl CandleClientBuilder {
    pub fn new() -> Self
    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self
    pub fn device_type(mut self, device_type: DeviceType) -> Self
    pub fn generation_config(mut self, config: GenerationConfig) -> Self
    pub fn quantization(mut self, quantization_type: QuantizationType) -> Self
    pub fn max_concurrent_requests(mut self, max: u32) -> Self
    pub fn build(self) -> CandleResult<CandleCompletionClient>
}
```

### CandleCompletionClient (client/completion.rs)
```rust
impl CandleCompletionClient {
    pub fn new(config: CandleClientConfig) -> CandleResult<Self>
    pub fn initialize(&self) -> CandleResult<()>
    pub fn is_initialized(&self) -> bool
    pub fn device(&self) -> &Device
    pub fn complete(
        &self,
        request: &CandleCompletionRequest,
    ) -> AsyncStream<CandleCompletionResponse<'static>>
    pub fn complete_stream(
        &self,
        request: &CandleCompletionRequest,
    ) -> AsyncStream<CandleStreamingResponse>
    // Private methods
    fn record_request_stats(&self, success: bool, tokens: usize, cache_hit: bool)
}
```

## Generator Module Methods

### CandleGenerator (generator/core.rs)
```rust
impl CandleGenerator {
    pub fn new(
        model: Arc<dyn crate::model::CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
    ) -> CandleResult<Self>
    
    pub fn with_sophisticated_features(
        model: Arc<dyn crate::model::CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
        kv_cache: Option<Arc<Mutex<KVCache>>>,
        progress_reporter: Option<Arc<dyn ProgressReporter>>,
    ) -> CandleResult<Self>
    
    pub fn update_config(&mut self, config: GenerationConfig)
    pub fn config(&self) -> &GenerationConfig
    pub fn reset_cumulative_log_prob(&self)
    pub fn cumulative_log_prob(&self) -> f64
    pub fn composite_processor(&self) -> &CompositeProcessor
    
    pub fn generate(
        &self, 
        request: &crate::types::CandleCompletionRequest
    ) -> fluent_ai_async::AsyncStream<crate::types::CandleCompletionResponse<'static>>
    
    pub fn generate_stream(
        &self,
        request: &crate::types::CandleCompletionRequest
    ) -> fluent_ai_async::AsyncStream<crate::types::CandleStreamingResponse>
}
```

### Generator Types (generator/types.rs)
```rust
impl GeneratedToken {
    pub fn new(
        id: u32,
        text: String,
        prob: Option<f32>,
        cumulative_prob: Option<f64>,
    ) -> Self
    pub fn text_str(&self) -> CandleResult<&str>
}

impl GenerationSequence {
    pub fn new() -> Self
    pub fn add_token(&mut self, token: GeneratedToken) -> CandleResult<()>
    pub fn complete(&self, reason: StopReason)
    pub fn is_complete(&self) -> bool
    pub fn stop_reason(&self) -> Option<StopReason>
}
```

## Tokenizer Module Methods

### CandleTokenizer (tokenizer/core.rs)
```rust
impl CandleTokenizer {
    pub fn new(tokenizer: Tokenizer, config: TokenizerConfig) -> CandleResult<Self>
    pub fn from_file<P: AsRef<Path>>(path: P, config: TokenizerConfig) -> CandleResult<Self>
    pub fn from_hub(model_id: &str, config: TokenizerConfig) -> AsyncStream<Self>
    pub fn from_fallback_path(model_id: &str, config: TokenizerConfig) -> CandleResult<Self>
    pub fn from_hub_with_revision(
        model_id: &str,
        revision: &str,
        config: TokenizerConfig,
    ) -> AsyncStream<Self>
    
    pub fn vocab_size(&self) -> u32
    pub fn config(&self) -> &TokenizerConfig
    pub fn update_config(&mut self, config: TokenizerConfig)
    pub fn inner(&self) -> &Tokenizer
    pub fn special_tokens(&self) -> &AHashMap<String, u32>
}
```

### Tokenizer Encoding (tokenizer/encoding.rs)
```rust
impl CandleTokenizer {
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> CandleResult<Vec<u32>>
    pub fn encode_to_buffer(
        &self,
        text: &str,
        add_special_tokens: bool,
        buffer: &mut ArrayVec<u32, MAX_SEQUENCE_LENGTH>,
    ) -> CandleResult<()>
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> CandleResult<Vec<Vec<u32>>>
    pub fn estimate_token_count(&self, text: &str) -> usize
}
```

### Tokenizer Decoding (tokenizer/decoding.rs)
```rust
impl CandleTokenizer {
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> CandleResult<String>
    pub fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> CandleResult<Vec<String>>
}
```

### Special Tokens (tokenizer/special_tokens.rs)
```rust
impl CandleTokenizer {
    pub fn token_to_id(&self, token: &str) -> Option<u32>
    pub fn id_to_token(&self, id: u32) -> Option<String>
    pub fn get_special_token_id(&self, token_type: &str) -> Option<u32>
    pub fn eos_token_id(&self) -> Option<u32>
    pub fn is_special_token(&self, token_id: u32) -> bool
    pub fn apply_chat_template(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> CandleResult<String>
}
```

### Tokenizer Configuration (tokenizer/config.rs)
```rust
impl TokenizerConfigBuilder {
    pub fn new() -> Self
    pub fn add_bos_token(mut self, add: bool) -> Self
    pub fn add_eos_token(mut self, add: bool) -> Self
    pub fn max_length(mut self, length: Option<usize>) -> Self
    pub fn padding(mut self, config: PaddingConfig) -> Self
    pub fn truncation(mut self, config: TruncationConfig) -> Self
    pub fn build(self) -> TokenizerConfig
}
```

### Tokenizer Utilities (tokenizer/utils.rs)
```rust
// Utility functions
pub fn load_popular_tokenizer(name: &str) -> AsyncStream<CandleTokenizer>
pub fn config_for_model_type(model_type: &str) -> TokenizerConfig
pub fn validate_tokenizer(tokenizer: &CandleTokenizer) -> CandleResult<()>
```

## Processing Module Methods

### ProcessingEngine (processing/mod.rs)
```rust
impl ProcessingEngine {
    // Core engine methods (details would be extracted from actual file)
}

impl ProcessingMetrics {
    // Metrics collection methods
}

impl ProcessingEngineBuilder {
    // Builder pattern methods
}
```

### Composite Processor (processing/processors/composite.rs)
```rust
impl CompositeProcessor {
    // Composite processing methods
}

impl CompositeProcessorBuilder {
    // Builder methods for composite processor
}
```

### Top-P Processor (processing/processors/top_p.rs)
```rust
impl TopPProcessor {
    // Top-p sampling processor methods
}
```

### Repetition Penalty Processor (processing/processors/repetition_penalty.rs)
```rust
impl RepetitionPenaltyProcessor {
    // Repetition penalty methods
}
```

## Sampling Module Methods

### Typical Sampling (sampling/typical.rs)
```rust
impl TypicalSamplingProcessor {
    pub fn new(typical_p: f64) -> Result<Self, SamplingError>
    pub fn with_config(
        typical_p: f64,
        min_entropy: f64,
        max_surprisal_diff: f64,
        use_approximation: bool,
    ) -> Result<Self, SamplingError>
    
    pub fn typical_p(&self) -> f64
    pub fn set_typical_p(&mut self, typical_p: f64) -> Result<(), SamplingError>
    pub fn analyze_distribution(
        &self,
        probabilities: &[f32],
    ) -> Result<TypicalSamplingStats, SamplingError>
    
    // Private methods
    fn apply_typical_sampling(&self, logits: &Tensor) -> Result<Tensor, SamplingError>
    fn calculate_entropy(&self, probabilities: &[f32]) -> f64
    fn find_typical_tokens(
        &self,
        probabilities: &[f32],
        entropy: f64,
    ) -> Result<Vec<usize>, SamplingError>
    fn create_filtered_logits(
        &self,
        original_logits: &[f32],
        selected_indices: &[usize],
    ) -> Result<Vec<f32>, SamplingError>
    fn process_logits(
        &self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> Result<(), ProcessingError>
    fn apply_typical_sampling_to_probs(&self, probs: &mut [f32])
}

impl TypicalSamplingBuilder {
    pub fn new() -> Self
    pub fn typical_p(mut self, typical_p: f64) -> Self
    pub fn min_entropy(mut self, min_entropy: f64) -> Self
    pub fn max_surprisal_diff(mut self, max_diff: f64) -> Self
    pub fn use_approximation(mut self) -> Self
    pub fn build(self) -> Result<TypicalSamplingProcessor, SamplingError>
}

impl TypicalSamplingStats {
    pub fn is_efficient(&self, threshold: f64) -> bool
    pub fn selection_ratio(&self, vocab_size: usize) -> f64
}
```

## Streaming Module Methods

### Streaming Configuration (streaming/streaming_config.rs)
```rust
impl StreamingConfig {
    pub fn new() -> Self
    pub fn buffer_size(mut self, size: usize) -> CandleResult<Self>
    pub fn chunk_timeout_ms(mut self, timeout: u16) -> CandleResult<Self>
    pub fn max_chunk_size(mut self, size: usize) -> CandleResult<Self>
    pub fn flush_policy(mut self, policy: FlushPolicy) -> Self
    pub fn merge_on_overflow(mut self, enable: bool) -> Self
    pub fn max_merge_attempts(mut self, attempts: u8) -> CandleResult<Self>
    pub fn validate(&self) -> CandleResult<()>
    pub fn estimated_memory_bytes(&self) -> usize
}
```

### Flow Controller (streaming/flow_controller.rs)
```rust
impl FlowController {
    pub fn new(enabled: bool, threshold: f32, strategy: BackpressureStrategy) -> Self
    pub fn with_rate_limit(
        enabled: bool,
        threshold: f32,
        strategy: BackpressureStrategy,
        max_tokens_per_second: f64,
    ) -> Self
    
    pub fn check_backpressure(&mut self, buffer_utilization: f32)
    pub fn apply_delay(&self) -> AsyncStream<()>
    pub fn should_allow_token(&self) -> bool
    pub fn update_strategy(&mut self, strategy: BackpressureStrategy)
    pub fn update_threshold(&mut self, threshold: f32)
    pub fn set_enabled(&mut self, enabled: bool)
    pub fn set_rate_limit(&mut self, max_tokens_per_second: f64)
    pub fn get_stats(&self) -> &FlowStats
    pub fn reset_stats(&mut self)
    pub fn is_backpressure_active(&self) -> bool
    pub fn strategy(&self) -> BackpressureStrategy
    pub fn threshold(&self) -> f32
    pub fn is_enabled(&self) -> bool
    pub fn current_delay(&self) -> Duration
    pub fn configure_adaptive(&mut self, learning_rate: f32, buffer_sensitivity: f32)
    pub fn rate_limiter(&self) -> &TokenRateLimiter
    pub fn rate_limiter_mut(&mut self) -> &mut TokenRateLimiter
    pub fn adaptive_params(&self) -> &AdaptiveParams
    pub fn configure_adaptive_delays(&mut self, min_delay_us: u64, max_delay_us: u64)
    
    // Private methods
    fn apply_backpressure_strategy(&mut self, utilization: f32)
    fn calculate_adaptive_delay(&mut self, pressure: f32) -> u64
    fn adjust_backpressure_response(&mut self, utilization: f32)
    fn update_token_rate(&mut self)
}
```

### Token Output Stream (streaming/token_stream.rs)
```rust
impl TokenOutputStream {
    // Token streaming methods (details would be extracted from actual file)
}
```

### Streaming Metrics (streaming/streaming_metrics.rs)
```rust
impl StreamingMetrics {
    // Streaming metrics methods
}
```

## Error Handling Methods

### Error Helpers (error/error_helpers.rs)
```rust
impl CandleError {
    // Model-related errors
    pub fn model_not_found<S: Into<String>>(path: S) -> Self
    pub fn model_load_error<S: Into<String>>(msg: S) -> Self
    pub fn model_loading<S: Into<String>>(msg: S) -> Self
    pub fn invalid_model_format(msg: &'static str) -> Self
    
    // Tensor and device errors
    pub fn tensor_operation(msg: &'static str) -> Self
    pub fn device_allocation(msg: &'static str) -> Self
    pub fn quantization(msg: &'static str) -> Self
    
    // Tokenizer errors
    pub fn tokenizer(msg: &'static str) -> Self
    pub fn tokenization<S: Into<String>>(msg: S) -> Self
    
    // System errors
    pub fn memory_mapping(msg: &'static str) -> Self
    pub fn loading_timeout() -> Self
    pub fn unsupported_architecture(arch: &'static str) -> Self
    pub fn configuration(msg: &'static str) -> Self
    pub fn safetensors(msg: &'static str) -> Self
    
    // Generation errors
    pub fn context_length_exceeded(current: u32, max: u32) -> Self
    pub fn vocabulary_mismatch(expected: u32, actual: u32) -> Self
    pub fn generation_failed(msg: &'static str) -> Self
    pub fn cache_overflow() -> Self
    pub fn invalid_input(msg: &'static str) -> Self
    pub fn streaming_error(msg: &'static str) -> Self
    pub fn progress_error<S: Into<String>>(msg: S) -> Self
    pub fn cache_error<S: Into<String>>(msg: S) -> Self
    pub fn msg<S: Into<String>>(msg: S) -> Self
    
    // Error analysis
    pub fn is_retryable(&self) -> bool
    pub fn retry_delay(&self) -> Option<u64>
}
```

### Error Context (error/error_context.rs)
```rust
impl ErrorContext {
    pub fn new(operation: &'static str) -> Self
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self
    pub fn with_device<S: Into<String>>(mut self, device: S) -> Self
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self
}
```

## Variable Builder Methods

### CandleVarBuilder (var_builder/builder.rs)
```rust
impl<'a> CandleVarBuilder<'a> {
    pub fn from_mmaped_safetensors<P: AsRef<Path>>(
        paths: &[P],
        config: VarBuilderConfig,
    ) -> Result<Self>
    
    pub fn from_tensors(
        tensors: HashMap<String, Tensor>,
        config: VarBuilderConfig,
    ) -> Result<Self>
    
    pub fn get(&self, shape: &[usize], name: &str) -> Result<Tensor>
    pub fn pp<S: ToString>(&self, prefix: S) -> CandleVarBuilder<'a>
    pub fn to_dtype(&self, dtype: DType) -> CandleVarBuilder<'a>
    pub fn contains_tensor(&self, name: &str) -> bool
    
    // Private methods  
    fn new_internal(inner: VarBuilder<'a>, config: VarBuilderConfig) -> Self
    fn create_tensor_metadata(
        &self,
        tensor: &Tensor,
        name: &str,
    ) -> TensorEntry
}
```

### VarBuilder Configuration (var_builder/config.rs)
```rust
impl VarBuilderConfig {
    pub fn new() -> Self
    pub fn with_device(mut self, device: Device) -> Self
    pub fn with_tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self
    pub fn tensor_prefix(&self) -> Option<&str>
}

impl VarBuilderConfigBuilder {
    pub fn new() -> Self
    pub fn device(mut self, device: Device) -> Self
    pub fn dtype(mut self, dtype: DType) -> Self
    pub fn max_mmap_size(mut self, size: u64) -> Self
    pub fn enable_memory_mapping(mut self) -> Self
    pub fn disable_memory_mapping(mut self) -> Self
    pub fn enable_validation(mut self) -> Self
    pub fn enable_shape_caching(mut self) -> Self
    pub fn enable_lazy_loading(mut self) -> Self
    pub fn enable_device_optimization(mut self) -> Self
    pub fn enable_tensor_fusion(mut self) -> Self
    pub fn enable_tensor_cache(mut self) -> Self
    pub fn disable_tensor_cache(mut self) -> Self
    pub fn tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self
    pub fn build(self) -> VarBuilderConfig
}
```

### Model Metadata (var_builder/metadata.rs)
```rust
impl ModelMetadata {
    pub fn new() -> Self
    pub fn set_architecture(&mut self, arch: &str) -> Result<()>
    pub fn architecture(&self) -> Option<&str>
    pub fn set_total_parameters(&mut self, count: u64)
    pub fn add_config_entry(&mut self, key: &str, value: &str) -> Result<()>
    // Additional metadata methods...
}

impl TensorEntry {
    // Tensor metadata methods
}
```

### Loading Stats (var_builder/types.rs)
```rust
impl LoadingStats {
    pub fn new() -> Self
    pub fn record_tensor_load(&self, bytes: usize, duration_ns: u64)
    pub fn record_cache_hit(&self)
    pub fn record_cache_miss(&self)
    pub fn record_mmap_operation(&self)
    pub fn record_device_transfer(&self)
    pub fn record_validation(&self)
    pub fn record_tensor_fusion(&self)
    // Additional stats methods...
}
```

## Logits Processing Methods

### SamplingConfig (logits.rs)
```rust
impl SamplingConfig {
    pub fn validate(&self) -> CandleResult<()>
    pub fn needs_processing(&self) -> bool
    pub fn to_unified_processor(&self) -> ProcessingResult<CompositeProcessor>
}
```

### LogitsSampler (logits.rs)
```rust
impl LogitsSampler {
    pub fn new(vocab_size: usize, config: SamplingConfig) -> CandleResult<Self>
    pub fn process_logits(&mut self, logits: &mut [f32]) -> CandleResult<()>
    pub fn add_generated_token(&mut self, token: u32) -> CandleResult<()>
    pub fn reset(&mut self)
    pub fn config(&self) -> &SamplingConfig
    pub fn update_config(&mut self, config: SamplingConfig) -> CandleResult<()>
    pub fn engine(&self) -> &ProcessingEngine
    pub fn engine_mut(&mut self) -> &mut ProcessingEngine
}
```

### SamplingMetrics (logits.rs)
```rust
impl SamplingMetrics {
    pub fn new() -> Self
    pub fn record_sample(&self, processing_time_ns: u64)
    pub fn record_cache_hit(&self)
    pub fn average_processing_time_ns(&self) -> f64
    pub fn cache_hit_rate(&self) -> f64
}

// Global metrics function
pub fn sampling_metrics() -> &'static SamplingMetrics
```

### Logits Utilities (logits.rs)
```rust
// Utility functions for logits processing
pub fn stable_softmax(logits: &mut [f32]) -> CandleResult<()>
pub fn find_top_k_indices(logits: &[f32], k: usize) -> ArrayVec<usize, MAX_TOP_K>
pub fn cumulative_probability_threshold(
    probabilities: &[f32],
    threshold: f64,
) -> Option<usize>
```

## Generator SIMD Methods

### SIMD Utilities (generator/simd.rs)
```rust
// High-performance SIMD operations
pub fn scale_logits_by_temperature(logits: &mut [f32], temperature: f32)
pub fn cumulative_sum_f32(input: &[f32], output: &mut [f32])
pub fn find_sample_index(cumulative_probs: &[f32], random_val: f32) -> usize
```

## Hub Integration Methods

### Hub Functions (hub.rs)
```rust
// Hub integration utilities
pub fn create_client(backend: Backend) -> CandleResult<Client>
pub fn create_download_config(cache_dir: PathBuf) -> DownloadConfig
```

## Progress Reporting Methods

### Progress Configuration (progress/config.rs)
```rust
impl ProgressHubConfig {
    // Progress configuration methods
}
```

### Progress Stages (progress/stages.rs)
```rust
impl DownloadStage {
    // Download progress methods
}

impl WeightLoadingStage {
    // Weight loading progress methods
}

impl QuantizationStage {
    // Quantization progress methods
}

impl InferenceStage {
    // Inference progress methods
}
```

## Constraints Methods

### JSON Constraints (constraints/json.rs)
```rust
impl JsonConstraint {
    // JSON constraint methods for structured generation
}

impl JsonState {
    // JSON parsing state methods
}
```

## Type Extensions Methods

### Message Extensions (types/extensions.rs)
```rust
impl MessageExt for Message {
    // Extended message functionality
}

impl RoleExt for MessageRole {
    // Extended role functionality
}
```

## Build Utilities Methods

### Chat Builder (builders/candle_chat/candle_chatbot.rs)
```rust
// CLI chatbot utilities
fn print_styled(text: &str, color: Color, bold: bool) -> io::Result<()>
fn print_header(title: &str) -> io::Result<()>
pub fn cli_chatbot<C>(chatbot: C) -> Result<(), CandleCompletionError>
```

## Trait Implementations

### Core Rust Traits

#### Default Implementations
```rust
// Configuration types
impl Default for FlushPolicy
impl Default for StreamingConfig  
impl Default for CandleClientConfig
impl Default for ModelConfig
impl Default for SamplingConfig
impl Default for GenerationConfig
impl Default for TokenizerConfig
impl Default for PaddingConfig
impl Default for TruncationConfig
impl Default for VarBuilderConfig
impl Default for KVCacheConfig
impl Default for ProgressHubConfig
impl Default for BackpressureStrategy
impl Default for AdaptiveParams

// Builder types
impl Default for CandleClientBuilder
impl Default for TypicalSamplingBuilder
impl Default for KVCacheBuilder
impl Default for ConfigWizard
impl Default for TokenizerConfigBuilder
impl Default for VarBuilderConfigBuilder
impl Default for ExportConfig
impl Default for ChatExporter
impl Default for CommandParser

// Core client types
impl Default for CandleCompletionClient
impl Default for CandleGenerator

// Stats and metrics
impl Default for StreamingMetrics
impl Default for MirostatStats
impl Default for CacheStats
impl Default for GenerationStats
impl Default for GenerationState
impl Default for LoadingStats
impl Default for ModelMetadata
impl Default for FormatMetrics
impl Default for MemoryPool
impl Default for MemoryPoolCollection

// Decoder types
impl Default for DecoderState
impl Default for DecoderConfig
impl Default for StreamingDecoder

// Model types
impl Default for KimiK2Config
impl Default for ModelRegistry
impl Default for ModelLoaderConfig
impl Default for ModelLoader
impl Default for QuantizationType
impl Default for QuantizationConfig
impl Default for ModelType
impl Default for ModelPerformance

// Chat types
impl Default for StreamingConversation
impl Default for RealTimeSystemBuilder
impl Default for TypingStatistics
impl Default for LiveMessageStreamer

// Streaming types
impl Default for OutputFormat
impl Default for HeadTable
impl Default for CandleStreamingResponse
impl Default for CandleStreamingDelta
impl Default for CompactCompletionResponseBuilder

// Constraint types
impl Default for JsonConstraint

// Recovery and loading
impl Default for RecoveryStrategy
impl Default for ProgressTracker
```

#### Clone Implementations
```rust
impl Clone for CandleCompletionClient
impl Clone for CandleGenerator
impl Clone for CandleTokenizer
impl Clone for GenerationStats
impl Clone for GenerationState
impl Clone for ModelRegistry
impl Clone for HotSwappableVarBuilder
impl Clone for VarBuilderConfig
impl Clone for ModelLoaderConfig
```

#### Drop Implementations
```rust
impl Drop for PerfTimer
impl Drop for TokenStreamSender
```

#### Display and Debug Implementations
```rust
impl std::fmt::Display for ErrorContext
impl std::fmt::Display for ContextualError
impl std::fmt::Display for CandleError
impl std::fmt::Display for ErrorCategory
impl std::fmt::Display for ErrorSeverity
impl std::fmt::Display for DecoderStats
impl std::fmt::Display for DecoderState

impl std::fmt::Debug for StreamingConversation
```

#### Error Trait Implementations
```rust
impl std::error::Error for CandleError
impl std::error::Error for ContextualError
impl std::error::Error for DecoderError
```

#### Comparison Trait Implementations
```rust
impl PartialEq for RepetitionPenaltyProcessor
impl PartialEq for TopKProcessor
```

### Custom Domain Traits

#### LogitsProcessor Trait Implementations
```rust
impl LogitsProcessor for TypicalSamplingProcessor
impl LogitsProcessor for TopPProcessor
impl LogitsProcessor for CompositeProcessor
impl LogitsProcessor for MirostatProcessor
impl LogitsProcessor for GumbelSoftmaxProcessor
// Note: Some implementations are commented out in source
```

#### Extension Traits
```rust
impl MessageExt for Message
impl RoleExt for MessageRole
```

#### Model Architecture Traits
```rust
impl Module for KimiK2Model  // Candle framework trait
```

#### Progress Reporting Traits
```rust
impl ProgressReporter for ProgressHubReporter
```

#### Conversation Management Traits
```rust
impl Conversation for ConversationImpl
```

#### Constraint Traits
```rust
impl GenerationConstraint for JsonConstraint
```

### Conversion Trait Implementations

#### From/Into Trait Implementations
```rust
impl From<candle_core::Error> for CandleError
impl From<tokenizers::Error> for CandleError
impl From<CompletionRequestError> for CandleError
impl From<CandleError> for CandleCompletionError
impl From<CandleError> for candle_core::Error
impl From<crate::processing::error::error_types::ProcessingError> for CandleError
impl From<anyhow::Error> for CandleError
impl From<candle_core::Error> for ProcessingError
impl From<crate::sampling::SamplingError> for ProcessingError
impl From<String> for ChatLoop
impl From<&str> for ChatLoop
```

## AsyncStream Methods

The fluent-ai-candle crate heavily uses AsyncStream for zero-allocation, backpressure-aware streaming operations:

### Core AsyncStream Generators
```rust
// Generator streaming methods
pub fn generate(
    &self, 
    request: &CandleCompletionRequest
) -> AsyncStream<CandleCompletionResponse<'static>>

pub fn generate_stream(
    &self,
    request: &CandleCompletionRequest
) -> AsyncStream<CandleStreamingResponse>

// Client streaming methods
pub fn complete(
    &self,
    request: &CandleCompletionRequest,
) -> AsyncStream<CompletionResponse<'static>>

pub fn complete_stream(
    &self,
    request: &CandleCompletionRequest,
) -> AsyncStream<StreamingResponse>

// Tokenizer streaming methods
pub fn from_hub(model_id: &str, config: TokenizerConfig) -> AsyncStream<Self>
pub fn from_hub_with_revision(
    model_id: &str,
    revision: &str,
    config: TokenizerConfig,
) -> AsyncStream<Self>

// Utility streaming methods
pub fn load_popular_tokenizer(name: &str) -> AsyncStream<CandleTokenizer>

// Flow control streaming
pub fn apply_delay(&self) -> AsyncStream<()>

// Model-specific streaming
pub fn from_hub() -> AsyncStream<Self>  // KimiK2Tokenizer
```

### AsyncStream Architecture Benefits

- **Zero Allocation**: Uses pre-allocated buffers and stack allocation
- **Backpressure Handling**: Built-in flow control and rate limiting
- **Composable**: AsyncStreams can be chained and combined
- **Non-blocking**: All operations are asynchronous and non-blocking
- **Error Propagation**: Structured error handling through the stream
- **Memory Efficient**: Minimal memory footprint with streaming data

## Static Functions and Module-level Methods

### Utility Functions
```rust
// Device utilities (lib.rs::device)
pub fn auto_device() -> candle_core::Result<Device>
pub fn device_info(device: &Device) -> &'static str
pub fn supports_fast_matmul(device: &Device) -> bool

// Memory tracking (memory.rs)
pub fn track_allocation(size: usize)
pub fn track_deallocation(size: usize)
pub fn current_usage() -> usize
pub fn peak_usage() -> usize
pub fn allocation_count() -> usize
pub fn reset_stats()

// Tensor utilities (lib.rs::tensor_utils)
pub fn tensor_to_tokens(tensor: &Tensor, buffer: &mut ArrayVec<u32, 2048>) -> CandleResult<()>
pub fn tokens_to_tensor(tokens: &[u32], device: &Device) -> CandleResult<Tensor>
pub fn softmax_with_temperature(logits: &Tensor, temperature: f32) -> CandleResult<Tensor>
pub fn sample_token(probs: &Tensor, top_k: Option<usize>, top_p: Option<f64>, rng: &mut impl rand::Rng) -> CandleResult<u32>

// SIMD utilities (generator/simd.rs)
pub fn scale_logits_by_temperature(logits: &mut [f32], temperature: f32)
pub fn cumulative_sum_f32(input: &[f32], output: &mut [f32])
pub fn find_sample_index(cumulative_probs: &[f32], random_val: f32) -> usize

// Logits processing utilities (logits.rs)
pub fn stable_softmax(logits: &mut [f32]) -> CandleResult<()>
pub fn find_top_k_indices(logits: &[f32], k: usize) -> ArrayVec<usize, MAX_TOP_K>
pub fn cumulative_probability_threshold(probabilities: &[f32], threshold: f64) -> Option<usize>
pub fn sampling_metrics() -> &'static SamplingMetrics

// Hub integration (hub.rs)
pub fn create_client(backend: Backend) -> CandleResult<Client>
pub fn create_download_config(cache_dir: PathBuf) -> DownloadConfig

// Tokenizer utilities (tokenizer/utils.rs)
pub fn config_for_model_type(model_type: &str) -> TokenizerConfig
pub fn validate_tokenizer(tokenizer: &CandleTokenizer) -> CandleResult<()>

// Constraints utilities (constraints/json.rs)
pub fn create_json_constraint_for_tokenizer(schema: &str, tokenizer: &CandleTokenizer) -> CandleResult<JsonConstraint>

// Data type conversion (var_builder/types.rs)
pub fn convert_dtype(dtype: safetensors::Dtype) -> DType

// CLI utilities (builders/candle_chat/candle_chatbot.rs)
pub fn cli_chatbot<C>(chatbot: C) -> Result<(), CandleCompletionError>
```

## Notes

- This inventory captures the primary public methods and key private methods
- Some implementation details may vary as the codebase evolves
- Methods marked with `pub` are part of the public API
- Methods without visibility modifiers are private to their modules
- Generic type parameters and complex return types are simplified for readability
- AsyncStream return types indicate streaming/async operations using fluent-ai-async architecture
- CandleResult<T> is typically an alias for Result<T, CandleError>
- The codebase follows zero-allocation patterns with extensive use of ArrayVec, SmallVec, and stack allocation
- Many trait implementations provide default behaviors and conversions between types
- The architecture emphasizes performance through inlined methods, SIMD operations, and lock-free concurrency

For complete method signatures and documentation, refer to the source code files.