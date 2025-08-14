//! Configuration management for Stable Diffusion 3 image generation
//!
//! This module provides comprehensive configuration structures for SD3 model variants,
//! generation parameters, and optimization settings following stable-diffusion-3 patterns.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Supported Stable Diffusion 3 model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SD3ModelVariant {
    /// SD3 Medium (2.5B parameters)
    ThreeMedium,
    /// SD3.5 Large (8.1B parameters)
    ThreeFiveLarge,
    /// SD3.5 Large Turbo (distilled, 4-step inference)
    ThreeFiveLargeTurbo,
    /// SD3.5 Medium (2.5B parameters, improved architecture)
    ThreeFiveMedium,
}

impl SD3ModelVariant {
    /// Get model identifier for HuggingFace Hub
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::ThreeMedium => "stabilityai/stable-diffusion-3-medium",
            Self::ThreeFiveLarge => "stabilityai/stable-diffusion-3.5-large",
            Self::ThreeFiveLargeTurbo => "stabilityai/stable-diffusion-3.5-large-turbo",
            Self::ThreeFiveMedium => "stabilityai/stable-diffusion-3.5-medium",
        }
    }

    /// Get default number of inference steps
    pub fn default_inference_steps(&self) -> usize {
        match self {
            Self::ThreeMedium => 28,
            Self::ThreeFiveLarge => 50,
            Self::ThreeFiveLargeTurbo => 4,
            Self::ThreeFiveMedium => 28,
        }
    }

    /// Check if model supports Skip Layer Guidance
    pub fn supports_slg(&self) -> bool {
        matches!(self, Self::ThreeFiveMedium)
    }
}

/// Comprehensive generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Model variant to use
    pub model_variant: SD3ModelVariant,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// CFG scale for guidance
    pub cfg_scale: f64,
    /// Time shift factor (alpha)
    pub time_shift: f64,
    /// Enable flash attention
    pub use_flash_attn: bool,
    /// Enable Skip Layer Guidance (SD3.5 Medium only)
    pub use_slg: bool,
    /// Output image size (width, height)
    pub output_size: (u32, u32),
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Batch size for generation
    pub batch_size: usize,
    /// Additional model-specific parameters
    pub additional_params: HashMap<String, serde_json::Value>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model_variant: SD3ModelVariant::ThreeMedium,
            num_inference_steps: 28,
            cfg_scale: 7.0,
            time_shift: 3.0,
            use_flash_attn: false,
            use_slg: false,
            output_size: (1024, 1024),
            seed: None,
            batch_size: 1,
            additional_params: HashMap::new(),
        }
    }
}

/// Model loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadingConfig {
    /// Model identifier (HuggingFace Hub)
    pub model_id: String,
    /// Model revision/branch
    pub revision: Option<String>,
    /// Use safetensors format
    pub use_safetensors: bool,
    /// Cache directory for models
    pub cache_dir: Option<String>,
    /// Enable model caching
    pub cache_enabled: bool,
    /// Model loading timeout (seconds)
    pub timeout_seconds: u64,
}

impl Default for ModelLoadingConfig {
    fn default() -> Self {
        Self {
            model_id: "stabilityai/stable-diffusion-3-medium".to_string(),
            revision: None,
            use_safetensors: true,
            cache_dir: None,
            cache_enabled: true,
            timeout_seconds: 300,
        }
    }
}

/// Configuration validation errors
#[derive(Error, Debug)]
pub enum ConfigValidationError {
    #[error("Invalid inference steps: {0}. Must be between 1 and 1000")]
    InvalidInferenceSteps(usize),

    #[error("Invalid CFG scale: {0}. Must be between 1.0 and 20.0")]
    InvalidCfgScale(f64),

    #[error("Invalid output size: {0}x{1}. Must be between 256x256 and 2048x2048")]
    InvalidOutputSize(u32, u32),

    #[error("Invalid time shift: {0}. Must be between 0.1 and 10.0")]
    InvalidTimeShift(f64),

    #[error("Skip Layer Guidance not supported for model variant: {0:?}")]
    SlgNotSupported(SD3ModelVariant),
}

/// Configuration validation functions
impl GenerationConfig {
    /// Validate inference steps
    pub fn validate_inference_steps(steps: usize) -> Result<(), ConfigValidationError> {
        if steps < 1 || steps > 1000 {
            return Err(ConfigValidationError::InvalidInferenceSteps(steps));
        }
        Ok(())
    }

    /// Validate CFG scale
    pub fn validate_cfg_scale(scale: f64) -> Result<(), ConfigValidationError> {
        if scale < 1.0 || scale > 20.0 {
            return Err(ConfigValidationError::InvalidCfgScale(scale));
        }
        Ok(())
    }

    /// Validate output size
    pub fn validate_output_size(width: u32, height: u32) -> Result<(), ConfigValidationError> {
        if width < 256 || width > 2048 || height < 256 || height > 2048 {
            return Err(ConfigValidationError::InvalidOutputSize(width, height));
        }
        Ok(())
    }

    /// Validate time shift
    pub fn validate_time_shift(shift: f64) -> Result<(), ConfigValidationError> {
        if shift < 0.1 || shift > 10.0 {
            return Err(ConfigValidationError::InvalidTimeShift(shift));
        }
        Ok(())
    }

    /// Validate complete configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        Self::validate_inference_steps(self.num_inference_steps)?;
        Self::validate_cfg_scale(self.cfg_scale)?;
        Self::validate_output_size(self.output_size.0, self.output_size.1)?;
        Self::validate_time_shift(self.time_shift)?;

        if self.use_slg && !self.model_variant.supports_slg() {
            return Err(ConfigValidationError::SlgNotSupported(self.model_variant));
        }

        Ok(())
    }
}

/// Device configuration optimization utilities
pub struct DeviceOptimization;

impl DeviceOptimization {
    /// Calculate optimal batch size for device and model
    pub fn get_optimal_batch_size(
        model_variant: SD3ModelVariant,
        available_memory_gb: usize,
        image_size: (u32, u32),
    ) -> usize {
        let base_memory_per_image = match model_variant {
            SD3ModelVariant::ThreeMedium => 2,         // 2GB per image
            SD3ModelVariant::ThreeFiveLarge => 4,      // 4GB per image
            SD3ModelVariant::ThreeFiveLargeTurbo => 3, // 3GB per image
            SD3ModelVariant::ThreeFiveMedium => 2,     // 2GB per image
        };

        // Scale by image size
        let size_factor = (image_size.0 * image_size.1) as f64 / (1024.0 * 1024.0);
        let memory_per_image = (base_memory_per_image as f64 * size_factor) as usize;

        // Conservative batch size calculation
        let max_batch = available_memory_gb / memory_per_image.max(1);
        max_batch.max(1).min(8) // Clamp between 1 and 8
    }

    /// Calculate memory requirements for configuration
    pub fn calculate_memory_requirements(config: &GenerationConfig) -> usize {
        let base_model_memory = match config.model_variant {
            SD3ModelVariant::ThreeMedium => 6 * 1024 * 1024 * 1024, // 6GB
            SD3ModelVariant::ThreeFiveLarge => 12 * 1024 * 1024 * 1024, // 12GB
            SD3ModelVariant::ThreeFiveLargeTurbo => 10 * 1024 * 1024 * 1024, // 10GB
            SD3ModelVariant::ThreeFiveMedium => 6 * 1024 * 1024 * 1024, // 6GB
        };

        let inference_memory = (config.batch_size * 2 * 1024 * 1024 * 1024); // 2GB per batch item
        let size_factor = (config.output_size.0 * config.output_size.1) as usize / (1024 * 1024);
        let size_memory = size_factor * 100 * 1024; // 100KB per megapixel

        base_model_memory + inference_memory + size_memory
    }
}
