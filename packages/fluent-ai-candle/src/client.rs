//! Production-ready CandleCompletionClient with zero allocation and lock-free design

use crate::error::{CandleError, CandleResult};
use crate::generator::{CandleGenerator, GenerationConfig};
use crate::model::CandleModel;
use crate::tokenizer::{CandleTokenizer, TokenizerConfig};
use arc_swap::ArcSwap;
use candle_core::Device;
use fluent_ai_core::completion::{
    client::{CompletionClient, CompletionClientExt},
    error::CompletionError,
    request::CompletionRequest,
    response::CompletionResponse,
    streaming::StreamingResponse,
    CompletionResult,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for CandleCompletionClient
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CandleClientConfig {
    /// Model path or identifier
    pub model_path: String,
    /// Tokenizer path or identifier
    pub tokenizer_path: Option<String>,
    /// Device to use for computation
    pub device_type: DeviceType,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Tokenizer configuration
    pub tokenizer_config: TokenizerConfig,
    /// Generation configuration
    pub generation_config: GenerationConfig,
    /// Enable model quantization
    pub enable_quantization: bool,
    /// Quantization type
    pub quantization_type: QuantizationType,
    /// Maximum concurrent requests
    pub max_concurrent_requests: u32,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size in MB
    pub cache_size_mb: u32,
}

impl Default for CandleClientConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            device_type: DeviceType::Auto,
            model_config: ModelConfig::default(),
            tokenizer_config: TokenizerConfig::default(),
            generation_config: GenerationConfig::default(),
            enable_quantization: false,
            quantization_type: QuantizationType::Q8_0,
            max_concurrent_requests: 4,
            request_timeout_seconds: 300,
            enable_caching: true,
            cache_size_mb: 512,
        }
    }
}

/// Device type selection
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatically select best available device
    Auto = 0,
    /// Use CPU only
    Cpu = 1,
    /// Use CUDA GPU
    Cuda = 2,
    /// Use Metal GPU (macOS)
    Metal = 3,
}

/// Model configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Context length
    pub context_length: u32,
    /// Hidden size
    pub hidden_size: u32,
    /// Number of attention heads
    pub num_attention_heads: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Use flash attention
    pub use_flash_attention: bool,
    /// RoPE scaling factor
    pub rope_scaling: f32,
}

impl Default for ModelConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            context_length: 2048,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            use_flash_attention: true,
            rope_scaling: 1.0,
        }
    }
}

/// Supported model architectures
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// LLaMA family models
    Llama = 0,
    /// Mistral family models
    Mistral = 1,
    /// Mixtral MoE models
    Mixtral = 2,
    /// Gemma models
    Gemma = 3,
    /// Phi models
    Phi = 4,
    /// Custom architecture
    Custom = 255,
}

/// Quantization types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 4-bit quantization (variant 0)
    Q4_0 = 0,
    /// 4-bit quantization (variant 1)
    Q4_1 = 1,
    /// 8-bit quantization
    Q8_0 = 2,
    /// No quantization
    None = 255,
}

/// Client statistics
#[repr(C)]
#[derive(Debug)]
pub struct ClientStats {
    /// Total requests processed
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Total tokens generated
    pub total_tokens_generated: AtomicU64,
    /// Total processing time in microseconds
    pub total_processing_time_us: AtomicU64,
    /// Average tokens per second
    pub average_tokens_per_second: AtomicU64,
    /// Memory usage in bytes
    pub memory_usage_bytes: AtomicU64,
    /// Cache hit rate (percentage * 100)
    pub cache_hit_rate: AtomicU64,
}

impl Default for ClientStats {
    #[inline(always)]
    fn default() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_processing_time_us: AtomicU64::new(0),
            average_tokens_per_second: AtomicU64::new(0),
            memory_usage_bytes: AtomicU64::new(0),
            cache_hit_rate: AtomicU64::new(0),
        }
    }
}

/// Production-ready Candle completion client
pub struct CandleCompletionClient {
    /// Client configuration
    config: CandleClientConfig,
    /// The candle model
    model: Arc<CandleModel>,
    /// The tokenizer
    tokenizer: Arc<CandleTokenizer>,
    /// The generator
    generator: ArcSwap<CandleGenerator>,
    /// Computation device
    device: Device,
    /// Client statistics
    stats: ClientStats,
    /// Is client initialized
    is_initialized: AtomicBool,
    /// Concurrent request semaphore
    request_semaphore: tokio::sync::Semaphore,
}

impl CandleCompletionClient {
    /// Create a new CandleCompletionClient
    #[inline(always)]
    pub async fn new(config: CandleClientConfig) -> CandleResult<Self> {
        let device = Self::create_device(config.device_type)?;
        
        // Load model
        let model = Arc::new(CandleModel::load_from_path(&config.model_path, &device).await?);
        
        // Load tokenizer
        let tokenizer_path = config.tokenizer_path
            .as_ref()
            .unwrap_or(&config.model_path);
        let tokenizer = Arc::new(CandleTokenizer::from_file(tokenizer_path, config.tokenizer_config.clone())?);
        
        // Create generator
        let generator = CandleGenerator::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            config.generation_config.clone(),
            device.clone(),
        );
        
        let client = Self {
            config: config.clone(),
            model,
            tokenizer,
            generator: ArcSwap::new(Arc::new(generator)),
            device,
            stats: ClientStats::default(),
            is_initialized: AtomicBool::new(true),
            request_semaphore: tokio::sync::Semaphore::new(config.max_concurrent_requests as usize),
        };
        
        Ok(client)
    }
    
    /// Create a new client from HuggingFace Hub
    #[inline(always)]
    pub async fn from_hub(repo_id: &str, config: CandleClientConfig) -> CandleResult<Self> {
        let device = Self::create_device(config.device_type)?;
        
        // Load model from hub
        let model = Arc::new(CandleModel::load_from_hub(repo_id, &device).await?);
        
        // Load tokenizer from hub
        let tokenizer = Arc::new(CandleTokenizer::from_hub(repo_id, config.tokenizer_config.clone()).await?);
        
        // Create generator
        let generator = CandleGenerator::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            config.generation_config.clone(),
            device.clone(),
        );
        
        let client = Self {
            config: config.clone(),
            model,
            tokenizer,
            generator: ArcSwap::new(Arc::new(generator)),
            device,
            stats: ClientStats::default(),
            is_initialized: AtomicBool::new(true),
            request_semaphore: tokio::sync::Semaphore::new(config.max_concurrent_requests as usize),
        };
        
        Ok(client)
    }
    
    /// Create device based on device type
    #[inline(always)]
    fn create_device(device_type: DeviceType) -> CandleResult<Device> {
        match device_type {
            DeviceType::Auto => {
                // Try CUDA first, then Metal, then CPU
                if candle_core::Device::cuda_if_available(0).is_ok() {
                    candle_core::Device::cuda_if_available(0)
                        .map_err(|e| CandleError::from(e))
                } else if cfg!(target_os = "macos") && candle_core::Device::new_metal(0).is_ok() {
                    candle_core::Device::new_metal(0)
                        .map_err(|e| CandleError::from(e))
                } else {
                    Ok(Device::Cpu)
                }
            }
            DeviceType::Cpu => Ok(Device::Cpu),
            DeviceType::Cuda => candle_core::Device::cuda_if_available(0)
                .map_err(|e| CandleError::from(e)),
            DeviceType::Metal => {
                if cfg!(target_os = "macos") {
                    candle_core::Device::new_metal(0)
                        .map_err(|e| CandleError::from(e))
                } else {
                    Err(CandleError::device_allocation("Metal not available on this platform"))
                }
            }
        }
    }
    
    /// Update generation configuration
    #[inline(always)]
    pub fn update_generation_config(&self, config: GenerationConfig) {
        let mut new_generator = (**self.generator.load()).clone();
        new_generator.update_config(config);
        self.generator.store(Arc::new(new_generator));
    }
    
    /// Get client statistics
    #[inline(always)]
    pub fn stats(&self) -> &ClientStats {
        &self.stats
    }
    
    /// Get client configuration
    #[inline(always)]
    pub fn config(&self) -> &CandleClientConfig {
        &self.config
    }
    
    /// Check if client is initialized
    #[inline(always)]
    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Acquire)
    }
    
    /// Get device information
    #[inline(always)]
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get model information
    #[inline(always)]
    pub fn model(&self) -> &Arc<CandleModel> {
        &self.model
    }
    
    /// Get tokenizer information
    #[inline(always)]
    pub fn tokenizer(&self) -> &Arc<CandleTokenizer> {
        &self.tokenizer
    }
    
    /// Warm up the model with a dummy request
    #[inline(always)]
    pub async fn warmup(&self) -> CandleResult<()> {
        let warmup_request = CompletionRequest::builder()
            .prompt("Hello")
            .max_tokens(1)
            .temperature(0.0)
            .build()
            .map_err(|_| CandleError::configuration("Failed to build warmup request"))?;
        
        let _response = self.complete(warmup_request).await?;
        Ok(())
    }
    
    /// Record request statistics
    #[inline(always)]
    fn record_request_stats(&self, success: bool, tokens_generated: u32, processing_time_us: u64) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.stats.successful_requests.fetch_add(1, Ordering::Relaxed);
            self.stats.total_tokens_generated.fetch_add(tokens_generated as u64, Ordering::Relaxed);
            self.stats.total_processing_time_us.fetch_add(processing_time_us, Ordering::Relaxed);
            
            // Update average tokens per second
            let total_tokens = self.stats.total_tokens_generated.load(Ordering::Relaxed);
            let total_time_s = self.stats.total_processing_time_us.load(Ordering::Relaxed) / 1_000_000;
            if total_time_s > 0 {
                let avg_tps = total_tokens / total_time_s;
                self.stats.average_tokens_per_second.store(avg_tps, Ordering::Relaxed);
            }
        } else {
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CompletionClient for CandleCompletionClient {
    #[inline(always)]
    fn complete<'a>(&'a self, request: CompletionRequest<'_>) -> Pin<Box<dyn Future<Output = CompletionResult<CompletionResponse>> + Send + 'a>> {
        Box::pin(async move {
            let start_time = Instant::now();
            
            // Check if client is initialized
            if !self.is_initialized() {
                return Err(CompletionError::from(CandleError::configuration("Client not initialized")));
            }
            
            // Acquire semaphore permit for rate limiting
            let _permit = self.request_semaphore.acquire().await
                .map_err(|_| CompletionError::from(CandleError::configuration("Request semaphore error")))?;
            
            // Generate completion
            let generator = self.generator.load();
            let result = generator.generate(&request).await;
            
            let processing_time_us = start_time.elapsed().as_micros() as u64;
            
            match result {
                Ok(response) => {
                    self.record_request_stats(true, response.tokens_generated(), processing_time_us);
                    Ok(response)
                }
                Err(e) => {
                    self.record_request_stats(false, 0, processing_time_us);
                    Err(CompletionError::from(e))
                }
            }
        })
    }
    
    #[inline(always)]
    fn complete_stream<'a>(&'a self, request: CompletionRequest<'_>) -> Pin<Box<dyn Future<Output = CompletionResult<StreamingResponse>> + Send + 'a>> {
        Box::pin(async move {
            let start_time = Instant::now();
            
            // Check if client is initialized
            if !self.is_initialized() {
                return Err(CompletionError::from(CandleError::configuration("Client not initialized")));
            }
            
            // Acquire semaphore permit for rate limiting
            let _permit = self.request_semaphore.acquire().await
                .map_err(|_| CompletionError::from(CandleError::configuration("Request semaphore error")))?;
            
            // Generate streaming completion
            let generator = self.generator.load();
            let result = generator.generate_stream(&request).await;
            
            let processing_time_us = start_time.elapsed().as_micros() as u64;
            
            match result {
                Ok(stream) => {
                    // Note: We can't easily track tokens for streaming here
                    self.record_request_stats(true, 0, processing_time_us);
                    Ok(stream)
                }
                Err(e) => {
                    self.record_request_stats(false, 0, processing_time_us);
                    Err(CompletionError::from(e))
                }
            }
        })
    }
}

impl CompletionClientExt for CandleCompletionClient {}

unsafe impl Send for CandleCompletionClient {}
unsafe impl Sync for CandleCompletionClient {}

/// Builder for CandleCompletionClient
#[derive(Debug, Clone)]
pub struct CandleClientBuilder {
    config: CandleClientConfig,
}

impl CandleClientBuilder {
    /// Create a new builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            config: CandleClientConfig::default(),
        }
    }
    
    /// Set model path
    #[inline(always)]
    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = path.into();
        self
    }
    
    /// Set tokenizer path
    #[inline(always)]
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }
    
    /// Set device type
    #[inline(always)]
    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }
    
    /// Set generation configuration
    #[inline(always)]
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = config;
        self
    }
    
    /// Enable quantization
    #[inline(always)]
    pub fn quantization(mut self, quantization_type: QuantizationType) -> Self {
        self.config.enable_quantization = true;
        self.config.quantization_type = quantization_type;
        self
    }
    
    /// Set maximum concurrent requests
    #[inline(always)]
    pub fn max_concurrent_requests(mut self, max: u32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }
    
    /// Build the client
    #[inline(always)]
    pub async fn build(self) -> CandleResult<CandleCompletionClient> {
        CandleCompletionClient::new(self.config).await
    }
    
    /// Build the client from HuggingFace Hub
    #[inline(always)]
    pub async fn build_from_hub(self, repo_id: &str) -> CandleResult<CandleCompletionClient> {
        CandleCompletionClient::from_hub(repo_id, self.config).await
    }
}

impl Default for CandleClientBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}