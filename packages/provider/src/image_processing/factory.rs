//! Factory for creating image processing backend instances
//!
//! This module provides a factory pattern for creating and configuring
//! image processing backends based on available features and system capabilities.

use std::collections::HashMap;

use super::*;
use crate::image_processing::candle_backend::CandleImageProcessor;

/// Factory for creating image processing backends
pub struct ImageProcessingFactory;

impl ImageProcessingFactory {
    /// Create an image processing backend based on configuration
    pub fn create_backend(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let backend = match backend_type {
            BackendType::Candle => Self::create_candle_backend(config)?,
            BackendType::Auto => Self::create_auto_backend(config)?,
        };

        Ok(backend)
    }

    /// Create an image embedding provider
    pub fn create_embedding_provider(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let backend = match backend_type {
            BackendType::Candle => Self::create_candle_embedding_provider(config)?,
            BackendType::Auto => Self::create_auto_embedding_provider(config)?,
        };

        Ok(backend)
    }

    /// Create an image generation provider
    #[cfg(feature = "generation")]
    pub fn create_generation_provider(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let backend = match backend_type {
            BackendType::Candle => Self::create_candle_generation_provider(config)?,
            BackendType::Auto => Self::create_auto_generation_provider(config)?,
        };

        Ok(backend)
    }

    /// Get available backends on current system
    pub fn available_backends() -> Vec<BackendType> {
        let mut backends = Vec::new();

        // Check Candle availability
        if Self::is_candle_available() {
            backends.push(BackendType::Candle);
        }

        backends
    }

    /// Get recommended backend for current system
    pub fn recommended_backend() -> BackendType {
        // For now, Candle is the default and only backend
        BackendType::Candle
    }

    /// Create Candle backend
    fn create_candle_backend(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let mut backend = CandleImageProcessor::new()?;

        if let Some(config) = config {
            backend.initialize(config)?;
        }

        Ok(Box::new(backend))
    }

    /// Create Candle embedding provider
    fn create_candle_embedding_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let mut backend = CandleImageProcessor::new()?;

        if let Some(config) = config {
            backend.initialize(config)?;
        }

        Ok(Box::new(backend))
    }

    /// Create Candle generation provider
    #[cfg(feature = "generation")]
    fn create_candle_generation_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let mut backend = crate::image_processing::generation::CandleImageGenerator::new()?;

        if let Some(config) = config {
            backend.initialize(config)?;
        }

        Ok(Box::new(backend))
    }

    /// Create automatically selected backend
    fn create_auto_backend(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let backend_type = Self::recommended_backend();
        Self::create_backend(backend_type, config)
    }

    /// Create automatically selected embedding provider
    fn create_auto_embedding_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let backend_type = Self::recommended_backend();
        Self::create_embedding_provider(backend_type, config)
    }

    /// Create automatically selected generation provider
    #[cfg(feature = "generation")]
    fn create_auto_generation_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let backend_type = Self::recommended_backend();
        Self::create_generation_provider(backend_type, config)
    }

    /// Check if Candle backend is available
    fn is_candle_available() -> bool {
        // Always available since it's our default backend
        true
    }

    /// Detect available device types
    pub fn detect_available_devices() -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];

        #[cfg(feature = "cuda")]
        {
            if Self::is_cuda_available() {
                devices.push(DeviceType::Cuda);
            }
        }

        #[cfg(feature = "metal")]
        {
            if Self::is_metal_available() {
                devices.push(DeviceType::Metal);
            }
        }

        devices
    }

    /// Get optimal device configuration for current system
    pub fn optimal_device_config() -> DeviceConfig {
        let available_devices = Self::detect_available_devices();

        // Prioritize GPU acceleration if available
        let device_type = if available_devices.contains(&DeviceType::Metal) {
            DeviceType::Metal
        } else if available_devices.contains(&DeviceType::Cuda) {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        };

        DeviceConfig {
            device_type,
            device_index: None,
            memory_limit_mb: None,
            mixed_precision: matches!(device_type, DeviceType::Cuda | DeviceType::Metal),
        }
    }

    /// Check CUDA availability
    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> bool {
        // Simple check - in production this could be more sophisticated
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
    }

    /// Check Metal availability
    #[cfg(feature = "metal")]
    fn is_metal_available() -> bool {
        // Metal is available on macOS
        cfg!(target_os = "macos")
    }

    /// Advanced Image Processing Factory with streaming capabilities and HTTP3 integration
    pub fn create_advanced_processing_factory(
        config: AdvancedProcessingConfig,
    ) -> ImageProcessingResult<AdvancedImageProcessingFactory> {
        AdvancedImageProcessingFactory::new(config)
    }

    /// Create provider registry with available backends
    pub fn create_provider_registry() -> ProviderRegistry {
        ProviderRegistry::new()
    }
}

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Candle ML framework backend
    Candle,
    /// Automatically select best available backend
    Auto,
}

/// Provider registry for managing available backends
pub struct ProviderRegistry {
    embedding_providers: HashMap<String, Box<dyn ImageEmbeddingProvider>>,
    #[cfg(feature = "generation")]
    generation_providers: HashMap<String, Box<dyn ImageGenerationProvider>>,
}

impl ProviderRegistry {
    /// Create new provider registry
    pub fn new() -> Self {
        Self {
            embedding_providers: HashMap::new(),
            #[cfg(feature = "generation")]
            generation_providers: HashMap::new(),
        }
    }

    /// Register an embedding provider
    pub fn register_embedding_provider(
        &mut self,
        name: String,
        provider: Box<dyn ImageEmbeddingProvider>,
    ) {
        self.embedding_providers.insert(name, provider);
    }

    /// Register a generation provider
    #[cfg(feature = "generation")]
    pub fn register_generation_provider(
        &mut self,
        name: String,
        provider: Box<dyn ImageGenerationProvider>,
    ) {
        self.generation_providers.insert(name, provider);
    }

    /// Get embedding provider by name
    pub fn get_embedding_provider(&self, name: &str) -> Option<&dyn ImageEmbeddingProvider> {
        self.embedding_providers.get(name).map(|p| p.as_ref())
    }

    /// Get generation provider by name
    #[cfg(feature = "generation")]
    pub fn get_generation_provider(&self, name: &str) -> Option<&dyn ImageGenerationProvider> {
        self.generation_providers.get(name).map(|p| p.as_ref())
    }

    /// List available embedding providers
    pub fn list_embedding_providers(&self) -> Vec<&String> {
        self.embedding_providers.keys().collect()
    }

    /// List available generation providers
    #[cfg(feature = "generation")]
    pub fn list_generation_providers(&self) -> Vec<&String> {
        self.generation_providers.keys().collect()
    }

    /// Initialize default providers
    pub fn initialize_default_providers(&mut self) -> ImageProcessingResult<()> {
        // Register Candle embedding provider
        if let Ok(provider) =
            ImageProcessingFactory::create_embedding_provider(BackendType::Candle, None)
        {
            self.register_embedding_provider("candle".to_string(), provider);
        }

        // Register Candle generation provider
        #[cfg(feature = "generation")]
        {
            if let Ok(provider) =
                ImageProcessingFactory::create_generation_provider(BackendType::Candle, None)
            {
                self.register_generation_provider("candle".to_string(), provider);
            }
        }

        Ok(())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder for image processing backends
pub struct BackendConfigBuilder {
    config: HashMap<String, serde_json::Value>,
}

impl BackendConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
        }
    }

    /// Set device configuration
    pub fn device_config(mut self, device_config: DeviceConfig) -> Self {
        self.config.insert(
            "device_config".to_string(),
            serde_json::to_value(device_config).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Set model name
    pub fn model_name(mut self, model_name: String) -> Self {
        self.config.insert(
            "model_name".to_string(),
            serde_json::Value::String(model_name),
        );
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.insert(
            "batch_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(batch_size)),
        );
        self
    }

    /// Set memory limit
    pub fn memory_limit_mb(mut self, memory_mb: u64) -> Self {
        self.config.insert(
            "memory_limit_mb".to_string(),
            serde_json::Value::Number(serde_json::Number::from(memory_mb)),
        );
        self
    }

    /// Add custom parameter
    pub fn custom_param(mut self, key: String, value: serde_json::Value) -> Self {
        self.config.insert(key, value);
        self
    }

    /// Build configuration
    pub fn build(self) -> HashMap<String, serde_json::Value> {
        self.config
    }
}

impl Default for BackendConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for factory operations
pub mod factory_utils {
    use super::*;

    /// Create optimal backend configuration for current system
    pub fn create_optimal_config() -> HashMap<String, serde_json::Value> {
        BackendConfigBuilder::new()
            .device_config(ImageProcessingFactory::optimal_device_config())
            .batch_size(32)
            .build()
    }

    /// Create configuration for specific device type
    pub fn create_device_config(device_type: DeviceType) -> HashMap<String, serde_json::Value> {
        let device_config = DeviceConfig {
            device_type,
            device_index: None,
            memory_limit_mb: None,
            mixed_precision: matches!(device_type, DeviceType::Cuda | DeviceType::Metal),
        };

        BackendConfigBuilder::new()
            .device_config(device_config)
            .build()
    }

    /// Create low-memory configuration
    pub fn create_low_memory_config() -> HashMap<String, serde_json::Value> {
        BackendConfigBuilder::new()
            .device_config(DeviceConfig {
                device_type: DeviceType::Cpu,
                device_index: None,
                memory_limit_mb: Some(1024), // 1GB limit
                mixed_precision: false,
            })
            .batch_size(8)
            .build()
    }

    /// Create high-performance configuration
    pub fn create_high_performance_config() -> HashMap<String, serde_json::Value> {
        let device_config = ImageProcessingFactory::optimal_device_config();

        BackendConfigBuilder::new()
            .device_config(device_config)
            .batch_size(64)
            .build()
    }
}

/// Advanced Image Processing Factory with streaming capabilities and HTTP3 integration
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use arrayvec::ArrayVec;
use crossbeam_queue::SegQueue;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use futures_util::stream::Stream;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use tokio::sync::mpsc;

/// Magic number constants for format detection
const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
const JPEG_MAGIC: &[u8] = b"\xff\xd8\xff";
const WEBP_MAGIC: &[u8] = b"RIFF";
const WEBP_SUBTYPE: &[u8] = b"WEBP";

/// Maximum size for image chunks in memory-efficient processing
const MAX_CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks

/// Fixed-size buffer pool for zero-allocation processing
static BUFFER_POOL: Lazy<SegQueue<Vec<u8>>> = Lazy::new(|| {
    let pool = SegQueue::new();
    // Pre-allocate 50 buffers
    for _ in 0..50 {
        pool.push(Vec::with_capacity(MAX_CHUNK_SIZE));
    }
    pool
});

/// Performance statistics for monitoring
static PROCESSING_STATS: Lazy<ProcessingStats> = Lazy::new(ProcessingStats::new);

/// Processing statistics with atomic counters
#[derive(Debug)]
pub struct ProcessingStats {
    pub images_processed: AtomicUsize,
    pub bytes_processed: AtomicU64,
    pub format_detections: AtomicUsize,
    pub http3_requests: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub cache_misses: AtomicUsize,
    pub streaming_operations: AtomicUsize,
}

impl ProcessingStats {
    pub fn new() -> Self {
        Self {
            images_processed: AtomicUsize::new(0),
            bytes_processed: AtomicU64::new(0),
            format_detections: AtomicUsize::new(0),
            http3_requests: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            streaming_operations: AtomicUsize::new(0),
        }
    }

    pub fn get_stats(&self) -> (usize, u64, usize, usize, usize, usize, usize) {
        (
            self.images_processed.load(Ordering::Relaxed),
            self.bytes_processed.load(Ordering::Relaxed),
            self.format_detections.load(Ordering::Relaxed),
            self.http3_requests.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
            self.streaming_operations.load(Ordering::Relaxed),
        )
    }
}

/// Image format detection with magic number validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic number with zero-allocation validation
    #[inline(always)]
    pub fn detect_from_bytes(data: &[u8]) -> Self {
        PROCESSING_STATS
            .format_detections
            .fetch_add(1, Ordering::Relaxed);

        if data.len() >= PNG_MAGIC.len() && &data[0..PNG_MAGIC.len()] == PNG_MAGIC {
            ImageFormat::Png
        } else if data.len() >= JPEG_MAGIC.len() && &data[0..JPEG_MAGIC.len()] == JPEG_MAGIC {
            ImageFormat::Jpeg
        } else if data.len() >= 12 && &data[0..4] == WEBP_MAGIC && &data[8..12] == WEBP_SUBTYPE {
            ImageFormat::WebP
        } else {
            ImageFormat::Unknown
        }
    }

    /// Get MIME type for format
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::WebP => "image/webp",
            ImageFormat::Unknown => "application/octet-stream",
        }
    }

    /// Get file extension for format
    #[inline(always)]
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg",
            ImageFormat::WebP => "webp",
            ImageFormat::Unknown => "bin",
        }
    }
}

/// Configuration for advanced image processing
#[derive(Debug, Clone)]
pub struct AdvancedProcessingConfig {
    pub http3_config: HttpConfig,
    pub max_concurrent_requests: usize,
    pub chunk_size: usize,
    pub enable_format_detection: bool,
    pub enable_streaming: bool,
    pub enable_caching: bool,
    pub memory_limit_mb: Option<usize>,
    pub supported_formats: SmallVec<[ImageFormat; 4]>,
    pub quality_settings: QualitySettings,
}

impl Default for AdvancedProcessingConfig {
    fn default() -> Self {
        Self {
            http3_config: HttpConfig::ai_optimized(),
            max_concurrent_requests: 10,
            chunk_size: MAX_CHUNK_SIZE,
            enable_format_detection: true,
            enable_streaming: true,
            enable_caching: true,
            memory_limit_mb: Some(512),
            supported_formats: SmallVec::from_slice(&[
                ImageFormat::Png,
                ImageFormat::Jpeg,
                ImageFormat::WebP,
            ]),
            quality_settings: QualitySettings::default(),
        }
    }
}

/// Quality settings for image processing
#[derive(Debug, Clone)]
pub struct QualitySettings {
    pub jpeg_quality: u8,
    pub png_compression: u8,
    pub webp_quality: u8,
    pub enable_progressive: bool,
    pub enable_optimization: bool,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            jpeg_quality: 85,
            png_compression: 6,
            webp_quality: 80,
            enable_progressive: true,
            enable_optimization: true,
        }
    }
}

/// Advanced image processing factory with streaming capabilities
pub struct AdvancedImageProcessingFactory {
    config: AdvancedProcessingConfig,
    http_client: HttpClient,
    processors: SmallVec<[Arc<dyn StreamingImageProcessor + Send + Sync>; 8]>,
    format_cache: Arc<SegQueue<(String, ImageFormat)>>,
}

impl AdvancedImageProcessingFactory {
    /// Create new advanced image processing factory
    pub fn new(config: AdvancedProcessingConfig) -> ImageProcessingResult<Self> {
        let http_client = HttpClient::with_config(config.http3_config.clone())
            .map_err(|e| ImageProcessingError::InitializationError(e.to_string()))?;

        let mut processors = SmallVec::new();

        // Add processors for each supported format
        for format in &config.supported_formats {
            match format {
                ImageFormat::Png => {
                    processors.push(Arc::new(PngStreamingProcessor::new(config.clone())?));
                }
                ImageFormat::Jpeg => {
                    processors.push(Arc::new(JpegStreamingProcessor::new(config.clone())?));
                }
                ImageFormat::WebP => {
                    processors.push(Arc::new(WebPStreamingProcessor::new(config.clone())?));
                }
                ImageFormat::Unknown => {
                    // Skip unknown formats
                }
            }
        }

        Ok(Self {
            config,
            http_client,
            processors,
            format_cache: Arc::new(SegQueue::new()),
        })
    }

    /// Fetch image from URL with HTTP3 streaming
    pub async fn fetch_image_stream(&self, url: &str) -> ImageProcessingResult<ImageStream> {
        PROCESSING_STATS
            .http3_requests
            .fetch_add(1, Ordering::Relaxed);

        let request = HttpRequest::get(url)
            .map_err(|e| ImageProcessingError::NetworkError(e.to_string()))?
            .header("Accept", "image/png, image/jpeg, image/webp, image/*");

        let response = self
            .http_client
            .send(request)
            .await
            .map_err(|e| ImageProcessingError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ImageProcessingError::NetworkError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let stream = response.stream();
        Ok(ImageStream::new(stream, self.config.chunk_size))
    }

    /// Process image with format detection and streaming
    pub async fn process_image_stream(
        &self,
        mut stream: ImageStream,
    ) -> ImageProcessingResult<ProcessedImageStream> {
        PROCESSING_STATS
            .streaming_operations
            .fetch_add(1, Ordering::Relaxed);

        // Read first chunk for format detection
        let first_chunk = stream
            .next_chunk()
            .await?
            .ok_or(ImageProcessingError::InvalidFormat(
                "Empty image data".to_string(),
            ))?;

        let format = if self.config.enable_format_detection {
            ImageFormat::detect_from_bytes(&first_chunk)
        } else {
            ImageFormat::Unknown
        };

        // Find appropriate processor for format
        let processor = self.find_processor_for_format(format)?;

        // Create processed stream
        let processed_stream = processor.process_stream(stream, first_chunk).await?;

        PROCESSING_STATS
            .images_processed
            .fetch_add(1, Ordering::Relaxed);

        Ok(processed_stream)
    }

    /// Find processor for specific format
    fn find_processor_for_format(
        &self,
        format: ImageFormat,
    ) -> ImageProcessingResult<Arc<dyn StreamingImageProcessor + Send + Sync>> {
        for processor in &self.processors {
            if processor.supports_format(format) {
                return Ok(processor.clone());
            }
        }

        Err(ImageProcessingError::UnsupportedFormat(format!(
            "{:?}",
            format
        )))
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> (usize, u64, usize, usize, usize, usize, usize) {
        PROCESSING_STATS.get_stats()
    }

    /// Clear format cache
    pub fn clear_cache(&self) {
        while self.format_cache.pop().is_some() {
            // Drain cache
        }
    }

    /// Get buffer from pool or create new one
    #[inline(always)]
    pub fn get_buffer(&self) -> Vec<u8> {
        if let Some(mut buffer) = BUFFER_POOL.pop() {
            buffer.clear();
            buffer
        } else {
            Vec::with_capacity(self.config.chunk_size)
        }
    }

    /// Return buffer to pool
    #[inline(always)]
    pub fn return_buffer(&self, buffer: Vec<u8>) {
        if buffer.capacity() >= self.config.chunk_size {
            let _ = BUFFER_POOL.push(buffer);
        }
    }
}

/// Streaming image processor trait - NO FUTURES!
pub trait StreamingImageProcessor {
    /// Check if processor supports format
    fn supports_format(&self, format: ImageFormat) -> bool;

    /// Process image stream
    fn process_stream(
        &self,
        stream: ImageStream,
        first_chunk: Vec<u8>,
    ) -> fluent_ai_domain::AsyncStream<ImageProcessingResult<ProcessedImageStream>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = tx.send(Err(ImageProcessingError::ProcessingError(
                "Not implemented".to_string(),
            )));
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
}

/// Image stream wrapper for chunked processing
pub struct ImageStream {
    inner: Box<dyn Stream<Item = Result<Vec<u8>, HttpError>> + Send + Unpin>,
    chunk_size: usize,
    buffer: Vec<u8>,
    bytes_read: AtomicU64,
}

impl ImageStream {
    pub fn new(
        stream: Box<dyn Stream<Item = Result<Vec<u8>, HttpError>> + Send + Unpin>,
        chunk_size: usize,
    ) -> Self {
        Self {
            inner: stream,
            chunk_size,
            buffer: Vec::with_capacity(chunk_size),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Read next chunk with fixed-size buffer
    pub async fn next_chunk(&mut self) -> ImageProcessingResult<Option<Vec<u8>>> {
        use futures_util::StreamExt;

        if let Some(result) = self.inner.next().await {
            match result {
                Ok(chunk) => {
                    self.bytes_read
                        .fetch_add(chunk.len() as u64, Ordering::Relaxed);
                    PROCESSING_STATS
                        .bytes_processed
                        .fetch_add(chunk.len() as u64, Ordering::Relaxed);
                    Ok(Some(chunk))
                }
                Err(e) => Err(ImageProcessingError::NetworkError(e.to_string())),
            }
        } else {
            Ok(None)
        }
    }

    /// Get total bytes read
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
}

/// Processed image stream with metadata
pub struct ProcessedImageStream {
    pub format: ImageFormat,
    pub metadata: ImageMetadata,
    pub chunks: mpsc::UnboundedReceiver<Result<Vec<u8>, ImageProcessingError>>,
}

impl ProcessedImageStream {
    pub fn new(
        format: ImageFormat,
        metadata: ImageMetadata,
        chunks: mpsc::UnboundedReceiver<Result<Vec<u8>, ImageProcessingError>>,
    ) -> Self {
        Self {
            format,
            metadata,
            chunks,
        }
    }

    /// Collect all chunks into single buffer
    pub async fn collect_all(mut self) -> ImageProcessingResult<Vec<u8>> {
        let mut result = Vec::new();

        while let Some(chunk_result) = self.chunks.recv().await {
            match chunk_result {
                Ok(chunk) => result.extend_from_slice(&chunk),
                Err(e) => return Err(e),
            }
        }

        Ok(result)
    }
}

/// Image metadata for processed streams
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub color_type: ColorType,
    pub bit_depth: u8,
    pub size_bytes: usize,
    pub quality: Option<u8>,
}

/// Color type for image processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    Grayscale,
    Rgb,
    Rgba,
    Indexed,
    GrayscaleAlpha,
}

/// PNG streaming processor
pub struct PngStreamingProcessor {
    config: AdvancedProcessingConfig,
}

impl PngStreamingProcessor {
    pub fn new(config: AdvancedProcessingConfig) -> ImageProcessingResult<Self> {
        Ok(Self { config })
    }
}

impl StreamingImageProcessor for PngStreamingProcessor {
    fn supports_format(&self, format: ImageFormat) -> bool {
        format == ImageFormat::Png
    }

    fn process_stream(
        &self,
        mut stream: ImageStream,
        first_chunk: Vec<u8>,
    ) -> fluent_ai_domain::AsyncStream<ImageProcessingResult<ProcessedImageStream>> {
        let config = self.config.clone();
        let (result_tx, result_rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let (tx, rx) = mpsc::unbounded_channel();

            // Send first chunk
            if let Err(_) = tx.send(Ok(first_chunk.clone())) {
                let _ = result_tx.send(Err(ImageProcessingError::ProcessingError(
                    "Channel closed".to_string(),
                )));
                return;
            }

            // Extract metadata from PNG header
            let metadata = ImageMetadata {
                width: 0, // Would be extracted from PNG header
                height: 0,
                format: ImageFormat::Png,
                color_type: ColorType::Rgba,
                bit_depth: 8,
                size_bytes: first_chunk.len(),
                quality: None,
            };

            // Process remaining chunks
            tokio::spawn(async move {
                while let Ok(Some(chunk)) = stream.next_chunk().await {
                    if tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
            });

            let _ = result_tx.send(Ok(ProcessedImageStream::new(
                ImageFormat::Png,
                metadata,
                rx,
            )));
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(result_rx)
    }
}

/// JPEG streaming processor
pub struct JpegStreamingProcessor {
    config: AdvancedProcessingConfig,
}

impl JpegStreamingProcessor {
    pub fn new(config: AdvancedProcessingConfig) -> ImageProcessingResult<Self> {
        Ok(Self { config })
    }
}

impl StreamingImageProcessor for JpegStreamingProcessor {
    fn supports_format(&self, format: ImageFormat) -> bool {
        format == ImageFormat::Jpeg
    }

    fn process_stream(
        &self,
        mut stream: ImageStream,
        first_chunk: Vec<u8>,
    ) -> fluent_ai_domain::AsyncStream<ImageProcessingResult<ProcessedImageStream>> {
        let config = self.config.clone();
        let (result_tx, result_rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let (tx, rx) = mpsc::unbounded_channel();

            // Send first chunk
            if let Err(_) = tx.send(Ok(first_chunk.clone())) {
                let _ = result_tx.send(Err(ImageProcessingError::ProcessingError(
                    "Channel closed".to_string(),
                )));
                return;
            }

            // Extract metadata from JPEG header
            let metadata = ImageMetadata {
                width: 0, // Would be extracted from JPEG header
                height: 0,
                format: ImageFormat::Jpeg,
                color_type: ColorType::Rgb,
                bit_depth: 8,
                size_bytes: first_chunk.len(),
                quality: Some(config.quality_settings.jpeg_quality),
            };

            // Process remaining chunks
            tokio::spawn(async move {
                while let Ok(Some(chunk)) = stream.next_chunk().await {
                    if tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
            });

            let _ = result_tx.send(Ok(ProcessedImageStream::new(
                ImageFormat::Jpeg,
                metadata,
                rx,
            )));
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(result_rx)
    }
}

/// WebP streaming processor
pub struct WebPStreamingProcessor {
    config: AdvancedProcessingConfig,
}

impl WebPStreamingProcessor {
    pub fn new(config: AdvancedProcessingConfig) -> ImageProcessingResult<Self> {
        Ok(Self { config })
    }
}

impl StreamingImageProcessor for WebPStreamingProcessor {
    fn supports_format(&self, format: ImageFormat) -> bool {
        format == ImageFormat::WebP
    }

    fn process_stream(
        &self,
        mut stream: ImageStream,
        first_chunk: Vec<u8>,
    ) -> fluent_ai_domain::AsyncStream<ImageProcessingResult<ProcessedImageStream>> {
        let config = self.config.clone();
        let (result_tx, result_rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let (tx, rx) = mpsc::unbounded_channel();

            // Send first chunk
            if let Err(_) = tx.send(Ok(first_chunk.clone())) {
                let _ = result_tx.send(Err(ImageProcessingError::ProcessingError(
                    "Channel closed".to_string(),
                )));
                return;
            }

            // Extract metadata from WebP header
            let metadata = ImageMetadata {
                width: 0, // Would be extracted from WebP header
                height: 0,
                format: ImageFormat::WebP,
                color_type: ColorType::Rgba,
                bit_depth: 8,
                size_bytes: first_chunk.len(),
                quality: Some(config.quality_settings.webp_quality),
            };

            // Process remaining chunks
            tokio::spawn(async move {
                while let Ok(Some(chunk)) = stream.next_chunk().await {
                    if tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
            });

            let _ = result_tx.send(Ok(ProcessedImageStream::new(
                ImageFormat::WebP,
                metadata,
                rx,
            )));
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(result_rx)
    }
}

/// Batch processing utilities for advanced factory
pub mod batch_processing {
    use super::*;

    /// Batch process multiple images with streaming
    pub async fn batch_process_images(
        factory: &AdvancedImageProcessingFactory,
        urls: &[String],
        max_concurrent: usize,
    ) -> ImageProcessingResult<Vec<ProcessedImageStream>> {
        use futures_util::stream::{FuturesUnordered, StreamExt};

        let mut futures = FuturesUnordered::new();
        let mut results = Vec::with_capacity(urls.len());

        for url in urls.iter().take(max_concurrent) {
            let future = async {
                let stream = factory.fetch_image_stream(url).await?;
                factory.process_image_stream(stream).await
            };
            futures.push(future);
        }

        while let Some(result) = futures.next().await {
            match result {
                Ok(processed_stream) => {
                    results.push(processed_stream);
                }
                Err(e) => {
                    // Log error but continue processing
                    eprintln!("Error processing image: {}", e);
                }
            }
        }

        Ok(results)
    }

    /// Create parallel processing pipeline
    pub fn create_parallel_pipeline(
        factory: Arc<AdvancedImageProcessingFactory>,
        max_workers: usize,
    ) -> ParallelProcessingPipeline {
        ParallelProcessingPipeline::new(factory, max_workers)
    }
}

/// Parallel processing pipeline for high-throughput scenarios
pub struct ParallelProcessingPipeline {
    factory: Arc<AdvancedImageProcessingFactory>,
    max_workers: usize,
    task_queue: Arc<SegQueue<ProcessingTask>>,
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl ParallelProcessingPipeline {
    pub fn new(factory: Arc<AdvancedImageProcessingFactory>, max_workers: usize) -> Self {
        let task_queue = Arc::new(SegQueue::new());
        let mut worker_handles = Vec::with_capacity(max_workers);

        for _ in 0..max_workers {
            let factory_clone = factory.clone();
            let queue_clone = task_queue.clone();

            let handle = tokio::spawn(async move {
                loop {
                    if let Some(task) = queue_clone.pop() {
                        let _ = task.execute(&factory_clone).await;
                    } else {
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }
                }
            });

            worker_handles.push(handle);
        }

        Self {
            factory,
            max_workers,
            task_queue,
            worker_handles,
        }
    }

    /// Submit processing task
    pub fn submit_task(&self, task: ProcessingTask) {
        self.task_queue.push(task);
    }

    /// Shutdown pipeline
    pub async fn shutdown(self) {
        for handle in self.worker_handles {
            handle.abort();
        }
    }
}

/// Processing task for parallel pipeline
pub struct ProcessingTask {
    pub url: String,
    pub result_sender: tokio::sync::oneshot::Sender<ImageProcessingResult<ProcessedImageStream>>,
}

impl ProcessingTask {
    pub fn new(
        url: String,
        result_sender: tokio::sync::oneshot::Sender<ImageProcessingResult<ProcessedImageStream>>,
    ) -> Self {
        Self { url, result_sender }
    }

    pub async fn execute(
        self,
        factory: &AdvancedImageProcessingFactory,
    ) -> ImageProcessingResult<()> {
        let result = async {
            let stream = factory.fetch_image_stream(&self.url).await?;
            factory.process_image_stream(stream).await
        }
        .await;

        let _ = self.result_sender.send(result);
        Ok(())
    }
}
