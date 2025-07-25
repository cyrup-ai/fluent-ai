//! Image processing providers with Candle backend
//!
//! This module provides a strategy pattern-based architecture for image processing,
//! supporting both embedding generation and image generation through various backends.
//! The default backend uses Candle ML framework for high-performance computer vision.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod candle_backend;
pub mod factory;

#[cfg(feature = "generation")]
pub mod generation;

// Re-export generation types when generation feature is enabled
#[cfg(feature = "generation")]
pub use generation::{CandleImageGenerator, GenerationConfig, SD3ModelVariant};

/// Errors specific to image processing operations
#[derive(Error, Debug, Clone)]
pub enum ImageProcessingError {
    #[error("Backend initialization failed: {0}")]
    BackendInitializationError(String),

    #[error("Device selection failed: {0}")]
    DeviceSelectionError(String),

    #[error("Image preprocessing failed: {0}")]
    PreprocessingError(String),

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionError(String),

    #[error("Image generation failed: {0}")]
    GenerationError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadingError(String),

    #[error("Unsupported image format: {0}")]
    UnsupportedImageFormat(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Resource allocation failed: {0}")]
    ResourceAllocationError(String),

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String)}

/// Result type for image processing operations
pub type ImageProcessingResult<T> = Result<T, ImageProcessingError>;

/// Supported image formats for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// WebP format
    WebP,
    /// GIF format (first frame)
    Gif,
    /// BMP format
    Bmp,
    /// TIFF format
    Tiff}

/// Image data container with metadata
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw image bytes
    pub data: Vec<u8>,
    /// Detected or specified format
    pub format: ImageFormat,
    /// Image dimensions (width, height)
    pub dimensions: Option<(u32, u32)>,
    /// Additional metadata
    pub metadata: HashMap<String, String>}

/// Image preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreprocessingConfig {
    /// Target image size (width, height)
    pub target_size: Option<(u32, u32)>,
    /// Whether to maintain aspect ratio during resize
    pub maintain_aspect_ratio: bool,
    /// Pixel normalization method
    pub pixel_normalization: PixelNormalization,
    /// Whether to convert to grayscale
    pub convert_to_grayscale: bool,
    /// JPEG quality for compression (1-100)
    pub jpeg_quality: u8}

/// Pixel normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PixelNormalization {
    /// Normalize to [0, 1] range
    ZeroToOne,
    /// Normalize to [-1, 1] range
    MinusOneToOne,
    /// Standard ImageNet normalization
    ImageNet,
    /// No normalization
    None}

impl Default for ImagePreprocessingConfig {
    fn default() -> Self {
        Self {
            target_size: Some((224, 224)),
            maintain_aspect_ratio: true,
            pixel_normalization: PixelNormalization::ImageNet,
            convert_to_grayscale: false,
            jpeg_quality: 85}
    }
}

/// Image embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingConfig {
    /// Image preprocessing settings
    pub preprocessing: ImagePreprocessingConfig,
    /// Batch size for processing multiple images
    pub batch_size: usize,
    /// Model-specific parameters
    pub model_params: HashMap<String, serde_json::Value>}

impl Default for ImageEmbeddingConfig {
    fn default() -> Self {
        Self {
            preprocessing: ImagePreprocessingConfig::default(),
            batch_size: 32,
            model_params: HashMap::new()}
    }
}

/// Image generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationConfig {
    /// Output image dimensions
    pub output_size: (u32, u32),
    /// Number of inference steps
    pub num_inference_steps: u32,
    /// Guidance scale for generation
    pub guidance_scale: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Batch size for generation
    pub batch_size: usize,
    /// Model-specific parameters
    pub model_params: HashMap<String, serde_json::Value>}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            output_size: (512, 512),
            num_inference_steps: 50,
            guidance_scale: 7.5,
            seed: None,
            batch_size: 1,
            model_params: HashMap::new()}
    }
}

/// Result of image embedding operation
#[derive(Debug, Clone)]
pub struct ImageEmbeddingResult {
    /// Generated embedding vector
    pub embedding: Vec<f32>,
    /// Processing metadata
    pub metadata: ImageEmbeddingMetadata}

/// Metadata for image embedding operations
#[derive(Debug, Clone)]
pub struct ImageEmbeddingMetadata {
    /// Original image dimensions
    pub original_dimensions: Option<(u32, u32)>,
    /// Processed image dimensions
    pub processed_dimensions: Option<(u32, u32)>,
    /// Detected image format
    pub format: ImageFormat,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Backend used for processing
    pub backend: String,
    /// Model used for embedding
    pub model: String,
    /// Additional backend-specific metadata
    pub backend_metadata: HashMap<String, serde_json::Value>}

/// Result of image generation operation
#[derive(Debug, Clone)]
pub struct ImageGenerationResult {
    /// Generated image data
    pub image: ImageData,
    /// Generation metadata
    pub metadata: ImageGenerationMetadata}

/// Metadata for image generation operations
#[derive(Debug, Clone)]
pub struct ImageGenerationMetadata {
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Backend used for generation
    pub backend: String,
    /// Model used for generation
    pub model: String,
    /// Prompt used for generation
    pub prompt: String,
    /// Configuration used
    pub config: ImageGenerationConfig,
    /// Additional backend-specific metadata
    pub backend_metadata: HashMap<String, serde_json::Value>}

/// Trait for image processing backends (strategy pattern)
pub trait ImageProcessingBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;

    /// Check if backend is available on current system
    fn is_available(&self) -> bool;

    /// Initialize backend with configuration
    fn initialize(
        &mut self,
        config: &HashMap<String, serde_json::Value>,
    ) -> ImageProcessingResult<()>;

    /// Get supported image formats
    fn supported_formats(&self) -> Vec<ImageFormat>;

    /// Get maximum supported image size
    fn max_image_size(&self) -> Option<(u32, u32)>;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

/// Backend capabilities descriptor
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Supports image embedding generation
    pub embedding: bool,
    /// Supports image generation
    pub generation: bool,
    /// Supports batch processing
    pub batch_processing: bool,
    /// Supports GPU acceleration
    pub gpu_acceleration: bool,
    /// Supported acceleration types
    pub acceleration_types: Vec<AccelerationType>}

/// Acceleration types supported by backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationType {
    /// CUDA GPU acceleration
    Cuda,
    /// Metal GPU acceleration (macOS)
    Metal,
    /// Intel MKL acceleration
    Mkl,
    /// Apple Accelerate framework
    Accelerate,
    /// CPU-only processing
    Cpu}

/// Trait for image embedding providers
pub trait ImageEmbeddingProvider: ImageProcessingBackend {
    /// Generate embedding for a single image
    fn embed_image(
        &self,
        image: &ImageData,
        config: Option<&ImageEmbeddingConfig>,
    ) -> ImageProcessingResult<ImageEmbeddingResult>;

    /// Generate embeddings for multiple images
    fn embed_image_batch(
        &self,
        images: &[ImageData],
        config: Option<&ImageEmbeddingConfig>,
    ) -> ImageProcessingResult<Vec<ImageEmbeddingResult>>;

    /// Get embedding dimensions for this provider
    fn embedding_dimensions(&self) -> usize;

    /// Get model information
    fn model_info(&self) -> ImageModelInfo;
}

/// Trait for image generation providers
#[cfg(feature = "generation")]
pub trait ImageGenerationProvider: ImageProcessingBackend {
    /// Generate image from text prompt
    fn generate_image(
        &self,
        prompt: &str,
        config: Option<&ImageGenerationConfig>,
    ) -> ImageProcessingResult<ImageGenerationResult>;

    /// Generate multiple images from text prompt
    fn generate_image_batch(
        &self,
        prompts: &[String],
        config: Option<&ImageGenerationConfig>,
    ) -> ImageProcessingResult<Vec<ImageGenerationResult>>;

    /// Get supported generation models
    fn supported_models(&self) -> Vec<String>;

    /// Load specific generation model
    fn load_model(&mut self, model_name: &str) -> ImageProcessingResult<()>;
}

/// Image model information descriptor
#[derive(Debug, Clone)]
pub struct ImageModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture
    pub architecture: String,
    /// Model parameters count
    pub parameters: Option<u64>,
    /// Model memory requirements in MB
    pub memory_mb: Option<u64>,
    /// Additional model metadata
    pub metadata: HashMap<String, serde_json::Value>}

/// Device configuration for backends
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Preferred device type
    pub device_type: DeviceType,
    /// Device index (for multi-GPU systems)
    pub device_index: Option<u32>,
    /// Memory limit in MB
    pub memory_limit_mb: Option<u64>,
    /// Enable mixed precision
    pub mixed_precision: bool}

/// Device types for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU computation
    Cpu,
    /// CUDA GPU
    Cuda,
    /// Metal GPU (macOS)
    Metal,
    /// Automatic selection
    Auto}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Auto,
            device_index: None,
            memory_limit_mb: None,
            mixed_precision: false}
    }
}

/// Utility functions for image processing
pub mod utils {
    use super::*;

    /// Detect image format from file signature
    pub fn detect_image_format(data: &[u8]) -> Option<ImageFormat> {
        if data.len() < 8 {
            return None;
        }

        // Check for common image file signatures
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some(ImageFormat::Jpeg)
        } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            Some(ImageFormat::Png)
        } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            Some(ImageFormat::WebP)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            Some(ImageFormat::Gif)
        } else if data.starts_with(b"BM") {
            Some(ImageFormat::Bmp)
        } else if data.starts_with(&[0x49, 0x49, 0x2A, 0x00])
            || data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
        {
            Some(ImageFormat::Tiff)
        } else {
            None
        }
    }

    /// Create ImageData from raw bytes with format detection
    pub fn create_image_data(data: Vec<u8>) -> ImageProcessingResult<ImageData> {
        let format = detect_image_format(&data).ok_or_else(|| {
            ImageProcessingError::UnsupportedImageFormat("Unrecognized image format".to_string())
        })?;

        #[cfg(feature = "image")]
        {
            // Extract actual image dimensions using the image crate
            let dimensions = match image::load_from_memory(&data) {
                Ok(img) => Some((img.width(), img.height())),
                Err(_) => None};

            // Generate metadata
            let mut metadata = HashMap::new();
            metadata.insert("format_detected".to_string(), format!("{:?}", format));
            metadata.insert("size_bytes".to_string(), data.len().to_string());

            if let Some((width, height)) = dimensions {
                metadata.insert("dimensions".to_string(), format!("{}x{}", width, height));
                metadata.insert(
                    "aspect_ratio".to_string(),
                    format!("{:.3}", width as f32 / height as f32),
                );
                metadata.insert("total_pixels".to_string(), (width * height).to_string());
            }

            Ok(ImageData {
                data,
                format,
                dimensions,
                metadata})
        }

        #[cfg(not(feature = "image"))]
        {
            // Without image crate, create basic metadata
            let mut metadata = HashMap::new();
            metadata.insert("format_detected".to_string(), format!("{:?}", format));
            metadata.insert("size_bytes".to_string(), data.len().to_string());

            Ok(ImageData {
                data,
                format,
                dimensions: None,
                metadata})
        }
    }

    /// Load image from file path
    pub async fn load_image_from_path(path: &str) -> ImageProcessingResult<ImageData> {
        let data = tokio::fs::read(path).await.map_err(|e| {
            ImageProcessingError::PreprocessingError(format!("Failed to read image file: {}", e))
        })?;

        create_image_data(data)
    }

    /// Validate image dimensions against constraints
    pub fn validate_image_dimensions(
        dimensions: (u32, u32),
        max_size: Option<(u32, u32)>,
    ) -> ImageProcessingResult<()> {
        if let Some((max_width, max_height)) = max_size {
            if dimensions.0 > max_width || dimensions.1 > max_height {
                return Err(ImageProcessingError::ConfigurationError(format!(
                    "Image dimensions {}x{} exceed maximum {}x{}",
                    dimensions.0, dimensions.1, max_width, max_height
                )));
            }
        }

        if dimensions.0 == 0 || dimensions.1 == 0 {
            return Err(ImageProcessingError::ConfigurationError(
                "Image dimensions cannot be zero".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate optimal batch size based on available memory
    pub fn calculate_optimal_batch_size(
        available_memory_mb: usize,
        avg_image_size: (u32, u32),
        channels: u32,
        safety_factor: f32,
    ) -> usize {
        let available_bytes = available_memory_mb * 1024 * 1024;
        let safe_bytes = (available_bytes as f32 * safety_factor) as usize;

        let pixels_per_image = avg_image_size.0 * avg_image_size.1 * channels;
        let bytes_per_image = pixels_per_image * std::mem::size_of::<f32>() as u32;

        if bytes_per_image == 0 {
            return 1;
        }

        (safe_bytes / bytes_per_image as usize).max(1).min(256)
    }
}
