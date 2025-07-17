//! Stable Diffusion 3 image generation module
//!
//! This module provides comprehensive text-to-image generation capabilities using the Candle ML framework.
//! It implements the complete Stable Diffusion 3 pipeline with support for multiple model variants,
//! advanced features like Skip Layer Guidance, and production-ready error handling.
//!
//! ## Architecture
//!
//! The module is organized into several key components:
//!
//! - **Configuration**: Model variants, generation parameters, and device optimization
//! - **Text Encoding**: Triple CLIP encoder (CLIP-L, CLIP-G, T5-XXL) for text understanding
//! - **Sampling**: Euler sampling with CFG and Skip Layer Guidance for SD3.5 models
//! - **VAE Decoding**: Latent to image conversion with proper scaling and post-processing
//! - **Model Management**: HuggingFace Hub integration with caching and memory optimization
//!
//! ## Supported Models
//!
//! - **SD3 Medium**: 2.5B parameter model for balanced performance
//! - **SD3.5 Large**: 8.1B parameter model for highest quality
//! - **SD3.5 Large Turbo**: Distilled 4-step inference model
//! - **SD3.5 Medium**: Improved 2.5B parameter model with SLG support
//!
//! ## Features
//!
//! - **Multi-device Support**: Automatic device detection (Metal, CUDA, CPU)
//! - **Memory Optimization**: Efficient model loading and caching
//! - **Batch Processing**: Optimized batch generation for multiple prompts
//! - **Skip Layer Guidance**: Advanced guidance technique for SD3.5 models
//! - **Flash Attention**: Optional attention optimization for speed
//! - **Configurable Parameters**: Full control over generation parameters
//!
//! ## Usage
//!
//! ```rust
//! use crate::image_processing::generation::{CandleImageGenerator, config::GenerationConfig};
//!
//! // Create generator with default configuration
//! let mut generator = CandleImageGenerator::new()?;
//!
//! // Generate single image
//! let image = generator.generate_image("A beautiful landscape")?;
//!
//! // Generate batch of images
//! let prompts = vec!["A cat", "A dog", "A bird"];
//! let images = generator.generate_image_batch(&prompts)?;
//! ```
//!
//! ## Configuration
//!
//! ```rust
//! use crate::image_processing::generation::config::{GenerationConfig, SD3ModelVariant};
//!
//! let config = GenerationConfig {
//!     model_variant: SD3ModelVariant::ThreeFiveLarge,
//!     num_inference_steps: 50,
//!     cfg_scale: 7.0,
//!     output_size: (1024, 1024),
//!     use_slg: true,
//!     ..Default::default()
//! };
//!
//! let mut generator = CandleImageGenerator::with_config(config)?;
//! ```

pub mod config;
pub mod text_encoder;
pub mod sampling;
pub mod vae;
pub mod models;

// Re-export main types for convenience
pub use config::{GenerationConfig, SD3ModelVariant, ModelLoadingConfig};
pub use text_encoder::StableDiffusion3TripleClipWithTokenizer;
pub use sampling::{euler_sample, SkipLayerGuidanceConfig};
pub use vae::{SD3VAEDecoder, build_sd3_vae_autoencoder};
pub use models::{ModelManager, CompleteModel};

// Re-export main generator
pub use super::generation::CandleImageGenerator;

// Re-export error types
pub use super::generation::{GenerationError, GenerationResult};

/// Generation module information
pub mod info {
    /// Module version
    pub const VERSION: &str = "1.0.0";
    
    /// Supported model variants
    pub const SUPPORTED_MODELS: &[&str] = &[
        "stabilityai/stable-diffusion-3-medium",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-medium",
    ];
    
    /// Default model variant
    pub const DEFAULT_MODEL: &str = "stabilityai/stable-diffusion-3-medium";
    
    /// Module capabilities
    pub const CAPABILITIES: &[&str] = &[
        "text-to-image-generation",
        "batch-processing",
        "multi-device-support",
        "memory-optimization",
        "skip-layer-guidance",
        "flash-attention",
        "model-caching",
    ];
}

/// Utility functions for generation module
pub mod utils {
    use super::*;
    
    /// Create default generation configuration
    pub fn default_generation_config() -> GenerationConfig {
        GenerationConfig::default()
    }
    
    /// Create configuration for specific model variant
    pub fn config_for_model(variant: SD3ModelVariant) -> GenerationConfig {
        GenerationConfig {
            model_variant: variant,
            num_inference_steps: variant.default_inference_steps(),
            use_slg: variant.supports_slg(),
            ..Default::default()
        }
    }
    
    /// Validate generation parameters
    pub fn validate_generation_params(
        num_inference_steps: usize,
        cfg_scale: f64,
        output_size: (u32, u32),
    ) -> Result<(), String> {
        if num_inference_steps == 0 || num_inference_steps > 1000 {
            return Err(format!("Invalid inference steps: {}", num_inference_steps));
        }
        
        if cfg_scale < 1.0 || cfg_scale > 20.0 {
            return Err(format!("Invalid CFG scale: {}", cfg_scale));
        }
        
        if output_size.0 == 0 || output_size.1 == 0 || output_size.0 > 2048 || output_size.1 > 2048 {
            return Err(format!("Invalid output size: {}x{}", output_size.0, output_size.1));
        }
        
        Ok(())
    }
    
    /// Calculate memory requirements for configuration
    pub fn calculate_memory_requirements(config: &GenerationConfig) -> usize {
        config::DeviceOptimization::calculate_memory_requirements(config)
    }
    
    /// Get optimal batch size for available memory
    pub fn get_optimal_batch_size(
        variant: SD3ModelVariant,
        available_memory_gb: usize,
        image_size: (u32, u32),
    ) -> usize {
        config::DeviceOptimization::get_optimal_batch_size(variant, available_memory_gb, image_size)
    }
}

/// Testing utilities (only available in test builds)
#[cfg(test)]
pub mod test_utils {
    use super::*;
    
    /// Create test configuration with minimal resource usage
    pub fn test_config() -> GenerationConfig {
        GenerationConfig {
            model_variant: SD3ModelVariant::ThreeMedium,
            num_inference_steps: 4,
            cfg_scale: 3.0,
            output_size: (512, 512),
            use_flash_attn: false,
            use_slg: false,
            batch_size: 1,
            ..Default::default()
        }
    }
    
    /// Create mock generator for testing
    pub fn create_test_generator() -> Result<CandleImageGenerator, GenerationError> {
        CandleImageGenerator::with_config(test_config())
    }
}

/// Feature flags and conditional compilation
pub mod features {
    /// Check if flash attention is available
    pub fn has_flash_attention() -> bool {
        cfg!(feature = "flash-attn")
    }
    
    /// Check if CUDA support is available
    pub fn has_cuda() -> bool {
        cfg!(feature = "cuda")
    }
    
    /// Check if Metal support is available
    pub fn has_metal() -> bool {
        cfg!(feature = "metal")
    }
    
    /// Check if image processing crate is available
    pub fn has_image_processing() -> bool {
        cfg!(feature = "image")
    }
    
    /// Get available acceleration types
    pub fn available_acceleration() -> Vec<&'static str> {
        let mut types = vec!["cpu"];
        
        if has_cuda() {
            types.push("cuda");
        }
        
        if has_metal() {
            types.push("metal");
        }
        
        types
    }
}

/// Constants for the generation module
pub mod constants {
    /// Default image size for generation
    pub const DEFAULT_IMAGE_SIZE: (u32, u32) = (1024, 1024);
    
    /// Minimum image size
    pub const MIN_IMAGE_SIZE: (u32, u32) = (256, 256);
    
    /// Maximum image size
    pub const MAX_IMAGE_SIZE: (u32, u32) = (2048, 2048);
    
    /// Default inference steps
    pub const DEFAULT_INFERENCE_STEPS: usize = 28;
    
    /// Default CFG scale
    pub const DEFAULT_CFG_SCALE: f64 = 7.0;
    
    /// Default time shift
    pub const DEFAULT_TIME_SHIFT: f64 = 3.0;
    
    /// Maximum batch size
    pub const MAX_BATCH_SIZE: usize = 16;
    
    /// Model cache timeout in seconds
    pub const MODEL_CACHE_TIMEOUT: u64 = 3600;
    
    /// Default model loading timeout
    pub const MODEL_LOADING_TIMEOUT: u64 = 300;
}