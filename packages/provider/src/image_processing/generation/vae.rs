//! VAE decoder implementation for Stable Diffusion 3
//!
//! This module implements the Variational Autoencoder (VAE) decoder following the exact patterns
//! from stable-diffusion-3/vae.rs for optimal compatibility and performance.

use std::path::PathBuf;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::stable_diffusion::vae::{
    AttentionBlock, AutoEncoderKL, AutoEncoderKLConfig, Decoder, ResnetBlock2D, UpSample,
};
use hf_hub::api::sync::Api;
use thiserror::Error;

/// VAE decoding errors
#[derive(Error, Debug)]
pub enum VAEError {
    #[error("VAE model loading failed: {0}")]
    ModelLoadingError(String),

    #[error("VAE decoding failed: {0}")]
    DecodingError(String),

    #[error("Weight mapping failed: {0}")]
    WeightMappingError(String),

    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),

    #[error("HuggingFace Hub error: {0}")]
    HubError(#[from] hf_hub::api::sync::ApiError),

    #[error("Invalid latent dimensions: {0}")]
    InvalidLatentDimensions(String),
}

/// Result type for VAE operations
pub type VAEResult<T> = Result<T, VAEError>;

/// Build SD3 VAE AutoEncoder following stable-diffusion-3/vae.rs patterns
pub fn build_sd3_vae_autoencoder(
    device: &Device,
    model_id: &str,
    revision: Option<&str>,
) -> VAEResult<AutoEncoderKL> {
    // Standard SD3 VAE configuration
    let config = AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 16,
        norm_num_groups: 32,
        sample_size: 512,
        scaling_factor: 1.5305,
        shift_factor: Some(0.0609),
    };

    // Load VAE weights from HuggingFace Hub
    let api = Api::new().map_err(|e| VAEError::HubError(e))?;

    let repo = if let Some(rev) = revision {
        api.model(model_id.to_string()).revision(rev.to_string())
    } else {
        api.model(model_id.to_string())
    };

    let vae_weights = repo
        .get("vae/diffusion_pytorch_model.safetensors")
        .map_err(|e| VAEError::HubError(e))?;

    // Create VarBuilder with weight mapping
    let vae_weights = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(vae_weights).map_err(|e| {
            VAEError::ModelLoadingError(format!("Failed to load VAE weights: {}", e))
        })?
    };

    let mut var_map = VarMap::new();
    for (name, tensor) in vae_weights.tensors() {
        let renamed = sd3_vae_vb_rename(name);
        var_map.set(&renamed, tensor.to_device(device)?)?;
    }

    let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

    // Build VAE model
    let vae = AutoEncoderKL::new(vb, 3, 3, config)
        .map_err(|e| VAEError::ModelLoadingError(format!("VAE model creation failed: {}", e)))?;

    Ok(vae)
}

/// VAE variable name mapping following stable-diffusion-3/vae.rs
fn sd3_vae_vb_rename(name: &str) -> String {
    // Map safetensors weight names to candle VAE structure
    let name = name.replace("first_stage_model.", "");
    let name = name.replace("encoder.", "encoder.");
    let name = name.replace("decoder.", "decoder.");
    let name = name.replace("post_quant_conv.", "post_quant_conv.");
    let name = name.replace("quant_conv.", "quant_conv.");

    // Handle encoder/decoder block mapping
    if name.contains("down.") || name.contains("up.") {
        let name = name.replace("down.", "down_blocks.");
        let name = name.replace("up.", "up_blocks.");
        let name = name.replace("block.", "resnets.");
        let name = name.replace("attn.", "attentions.0.");
        name
    } else if name.contains("mid.") {
        let name = name.replace("mid.", "mid_block.");
        let name = name.replace("block_1.", "resnets.0.");
        let name = name.replace("block_2.", "resnets.1.");
        let name = name.replace("attn_1.", "attentions.0.");
        name
    } else {
        name
    }
}

/// VAE decoder for latent to image conversion
pub struct SD3VAEDecoder {
    vae: AutoEncoderKL,
    scale_factor: f64,
    shift_factor: f64,
}

impl SD3VAEDecoder {
    /// Create new VAE decoder
    pub fn new(vae: AutoEncoderKL) -> Self {
        Self {
            vae,
            scale_factor: 1.5305, // TAESD3 scale factor
            shift_factor: 0.0609, // SD3 shift factor
        }
    }

    /// Create VAE decoder with custom scaling
    pub fn with_scaling(vae: AutoEncoderKL, scale_factor: f64, shift_factor: f64) -> Self {
        Self {
            vae,
            scale_factor,
            shift_factor,
        }
    }

    /// Decode latents to image tensor
    pub fn decode_latents(&self, latents: &Tensor) -> VAEResult<Tensor> {
        // Validate latent dimensions
        let latent_shape = latents.shape();
        if latent_shape.dims().len() != 4 {
            return Err(VAEError::InvalidLatentDimensions(format!(
                "Expected 4D tensor, got {}D: {:?}",
                latent_shape.dims().len(),
                latent_shape
            )));
        }

        // Apply scaling and shift following SD3 patterns
        let scaled_latents = (latents / self.scale_factor)
            .map_err(|e| VAEError::TensorError(e))?
            .add(&Tensor::new(self.shift_factor, latents.device())?)
            .map_err(|e| VAEError::TensorError(e))?;

        // VAE decode
        let decoded = self
            .vae
            .decode(&scaled_latents)
            .map_err(|e| VAEError::DecodingError(format!("VAE decoding failed: {}", e)))?;

        // Post-process to valid image range
        let processed = post_process_image(&decoded)?;

        Ok(processed)
    }

    /// Decode latents with batch processing
    pub fn decode_latents_batch(&self, latents_batch: &[Tensor]) -> VAEResult<Vec<Tensor>> {
        let mut decoded_images = Vec::with_capacity(latents_batch.len());

        for latents in latents_batch {
            let decoded = self.decode_latents(latents)?;
            decoded_images.push(decoded);
        }

        Ok(decoded_images)
    }
}

/// Post-process decoded image tensor to valid range
fn post_process_image(decoded: &Tensor) -> VAEResult<Tensor> {
    // Convert from [-1, 1] to [0, 1] range
    let processed = (decoded + 1.0)
        .map_err(|e| VAEError::TensorError(e))?
        .mul(&Tensor::new(0.5, decoded.device())?)
        .map_err(|e| VAEError::TensorError(e))?;

    // Clamp to valid range
    let clamped = processed
        .clamp(0.0, 1.0)
        .map_err(|e| VAEError::TensorError(e))?;

    Ok(clamped)
}

/// VAE utilities for latent manipulation
pub mod utils {
    use super::*;

    /// Validate latent tensor dimensions
    pub fn validate_latent_dimensions(latents: &Tensor) -> VAEResult<()> {
        let shape = latents.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(VAEError::InvalidLatentDimensions(format!(
                "Expected 4D tensor (batch, channels, height, width), got {}D",
                dims.len()
            )));
        }

        // Check channel dimension (should be 16 for SD3)
        if dims[1] != 16 {
            return Err(VAEError::InvalidLatentDimensions(format!(
                "Expected 16 channels for SD3 latents, got {}",
                dims[1]
            )));
        }

        Ok(())
    }

    /// Calculate output image dimensions from latent dimensions
    pub fn calculate_output_dimensions(latent_shape: &[usize]) -> (usize, usize) {
        // SD3 VAE has 8x upsampling factor
        let height = latent_shape[2] * 8;
        let width = latent_shape[3] * 8;
        (height, width)
    }

    /// Prepare latents for VAE decoding
    pub fn prepare_latents_for_decoding(latents: &Tensor) -> VAEResult<Tensor> {
        validate_latent_dimensions(latents)?;

        // Ensure correct dtype
        let prepared = latents
            .to_dtype(DType::F32)
            .map_err(|e| VAEError::TensorError(e))?;

        // Ensure contiguous memory layout
        let contiguous = prepared
            .contiguous()
            .map_err(|e| VAEError::TensorError(e))?;

        Ok(contiguous)
    }
}

/// VAE configuration validation
pub mod validation {
    use super::*;

    /// Validate VAE configuration parameters
    pub fn validate_vae_config(config: &AutoEncoderKLConfig) -> VAEResult<()> {
        if config.latent_channels == 0 {
            return Err(VAEError::ModelLoadingError(
                "Latent channels cannot be zero".to_string(),
            ));
        }

        if config.block_out_channels.is_empty() {
            return Err(VAEError::ModelLoadingError(
                "Block out channels cannot be empty".to_string(),
            ));
        }

        if config.scaling_factor <= 0.0 {
            return Err(VAEError::ModelLoadingError(
                "Scaling factor must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate scaling factors
    pub fn validate_scaling_factors(scale_factor: f64, shift_factor: f64) -> VAEResult<()> {
        if scale_factor <= 0.0 {
            return Err(VAEError::ModelLoadingError(
                "Scale factor must be positive".to_string(),
            ));
        }

        if shift_factor.is_nan() || shift_factor.is_infinite() {
            return Err(VAEError::ModelLoadingError(
                "Shift factor must be finite".to_string(),
            ));
        }

        Ok(())
    }
}
