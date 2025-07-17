//! Sampling implementation for Stable Diffusion 3
//! 
//! This module implements MMDiT sampling with Euler method following the exact patterns
//! from stable-diffusion-3/sampling.rs for optimal compatibility and performance.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::{flux, mmdit::model::MMDiT};
use thiserror::Error;

/// Skip Layer Guidance configuration for SD3.5 models
#[derive(Debug, Clone)]
pub struct SkipLayerGuidanceConfig {
    /// Guidance scale for skip layers
    pub scale: f64,
    /// Start fraction of inference steps
    pub start: f64,
    /// End fraction of inference steps
    pub end: f64,
    /// Layer indices to apply guidance to
    pub layers: Vec<usize>,
}

impl Default for SkipLayerGuidanceConfig {
    fn default() -> Self {
        Self {
            scale: 2.5,
            start: 0.01,
            end: 0.2,
            layers: vec![7, 8, 9],
        }
    }
}

/// Sampling errors
#[derive(Error, Debug)]
pub enum SamplingError {
    #[error("MMDiT forward pass failed: {0}")]
    MMDiTError(String),
    
    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),
    
    #[error("Invalid sampling parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Noise generation failed: {0}")]
    NoiseGenerationError(String),
}

/// Result type for sampling operations
pub type SamplingResult<T> = Result<T, SamplingError>;

/// Euler sampling implementation following stable-diffusion-3/sampling.rs
#[allow(clippy::too_many_arguments)]
pub fn euler_sample(
    mmdit: &MMDiT,
    y: &Tensor,
    context: &Tensor,
    num_inference_steps: usize,
    cfg_scale: f64,
    time_shift: f64,
    height: usize,
    width: usize,
    slg_config: Option<SkipLayerGuidanceConfig>,
) -> SamplingResult<Tensor> {
    // Initialize noise tensor using flux sampling patterns
    let mut x = flux::sampling::get_noise(1, height, width, y.device())
        .map_err(|e| SamplingError::NoiseGenerationError(format!("Failed to generate noise: {}", e)))?
        .to_dtype(DType::F16)
        .map_err(|e| SamplingError::TensorError(e))?;
    
    // Calculate sigma schedule with time shift
    let sigmas = (0..=num_inference_steps)
        .map(|x| x as f64 / num_inference_steps as f64)
        .rev()
        .map(|x| time_snr_shift(time_shift, x))
        .collect::<Vec<f64>>();
    
    // Main sampling loop
    for (step, window) in sigmas.windows(2).enumerate() {
        let (s_curr, s_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        
        let timestep = (*s_curr) * 1000.0;
        
        // MMDiT forward pass with CFG
        let noise_pred = mmdit.forward(
            &Tensor::cat(&[&x, &x], 0)
                .map_err(|e| SamplingError::TensorError(e))?,
            &Tensor::full(timestep as f32, (2,), x.device())
                .map_err(|e| SamplingError::TensorError(e))?
                .contiguous()
                .map_err(|e| SamplingError::TensorError(e))?,
            y,
            context,
            None,
        )
        .map_err(|e| SamplingError::MMDiTError(format!("MMDiT forward pass failed: {}", e)))?;
        
        // Apply Classifier-Free Guidance
        let mut guidance = apply_cfg(cfg_scale, &noise_pred)?;
        
        // Apply Skip Layer Guidance if configured
        if let Some(slg_config) = slg_config.as_ref() {
            let step_fraction = step as f64 / num_inference_steps as f64;
            
            if step_fraction >= slg_config.start && step_fraction <= slg_config.end {
                let slg_noise_pred = mmdit.forward(
                    &x,
                    &Tensor::full(timestep as f32, (1,), x.device())
                        .map_err(|e| SamplingError::TensorError(e))?
                        .contiguous()
                        .map_err(|e| SamplingError::TensorError(e))?,
                    &y.i(..1)
                        .map_err(|e| SamplingError::TensorError(e))?,
                    &context.i(..1)
                        .map_err(|e| SamplingError::TensorError(e))?,
                    Some(&slg_config.layers),
                )
                .map_err(|e| SamplingError::MMDiTError(format!("SLG forward pass failed: {}", e)))?;
                
                // Apply skip layer guidance
                let slg_adjustment = (noise_pred.i(..1)
                    .map_err(|e| SamplingError::TensorError(e))?
                    - slg_noise_pred.i(..1)
                        .map_err(|e| SamplingError::TensorError(e))?)
                    .map_err(|e| SamplingError::TensorError(e))?
                    * slg_config.scale;
                
                guidance = (guidance + slg_adjustment)
                    .map_err(|e| SamplingError::TensorError(e))?;
            }
        }
        
        // Euler integration step
        x = (x + (guidance * (*s_prev - *s_curr))
            .map_err(|e| SamplingError::TensorError(e))?)
            .map_err(|e| SamplingError::TensorError(e))?;
    }
    
    Ok(x)
}

/// Apply Classifier-Free Guidance (CFG) following stable-diffusion-3/sampling.rs
fn apply_cfg(cfg_scale: f64, noise_pred: &Tensor) -> SamplingResult<Tensor> {
    // CFG formula: guidance = cfg_scale * cond_pred - (cfg_scale - 1) * uncond_pred
    let cond_pred = noise_pred.narrow(0, 0, 1)
        .map_err(|e| SamplingError::TensorError(e))?;
    
    let uncond_pred = noise_pred.narrow(0, 1, 1)
        .map_err(|e| SamplingError::TensorError(e))?;
    
    let guidance = ((cond_pred * cfg_scale)
        .map_err(|e| SamplingError::TensorError(e))?
        - (uncond_pred * (cfg_scale - 1.0))
            .map_err(|e| SamplingError::TensorError(e))?)
        .map_err(|e| SamplingError::TensorError(e))?;
    
    Ok(guidance)
}

/// Time SNR shift function following stable-diffusion-3/sampling.rs
/// Implementation from ComfyUI: https://github.com/comfyanonymous/ComfyUI/blob/main/comfy/model_sampling.py#L181
fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

/// Noise generation utilities
pub mod noise {
    use super::*;
    
    /// Generate initial noise tensor
    pub fn generate_noise(
        batch_size: usize,
        height: usize,
        width: usize,
        device: &Device,
    ) -> SamplingResult<Tensor> {
        flux::sampling::get_noise(batch_size, height, width, device)
            .map_err(|e| SamplingError::NoiseGenerationError(format!("Failed to generate noise: {}", e)))
    }
    
    /// Generate noise with specific seed
    pub fn generate_seeded_noise(
        batch_size: usize,
        height: usize,
        width: usize,
        device: &Device,
        seed: u64,
    ) -> SamplingResult<Tensor> {
        // Set device seed if supported
        if let Err(e) = device.set_seed(seed) {
            return Err(SamplingError::NoiseGenerationError(format!("Failed to set seed: {}", e)));
        }
        
        generate_noise(batch_size, height, width, device)
    }
}

/// Sampling schedule utilities
pub mod schedule {
    use super::*;
    
    /// Generate linear schedule
    pub fn linear_schedule(num_steps: usize) -> Vec<f64> {
        (0..=num_steps)
            .map(|i| i as f64 / num_steps as f64)
            .collect()
    }
    
    /// Generate cosine schedule
    pub fn cosine_schedule(num_steps: usize) -> Vec<f64> {
        (0..=num_steps)
            .map(|i| {
                let t = i as f64 / num_steps as f64;
                0.5 * (1.0 - (t * std::f64::consts::PI).cos())
            })
            .collect()
    }
    
    /// Apply time shift to schedule
    pub fn apply_time_shift(schedule: &[f64], alpha: f64) -> Vec<f64> {
        schedule.iter()
            .map(|&t| time_snr_shift(alpha, t))
            .collect()
    }
}

/// Sampling configuration validation
pub mod validation {
    use super::*;
    
    /// Validate sampling parameters
    pub fn validate_sampling_params(
        num_inference_steps: usize,
        cfg_scale: f64,
        time_shift: f64,
        height: usize,
        width: usize,
    ) -> SamplingResult<()> {
        if num_inference_steps == 0 || num_inference_steps > 1000 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid inference steps: {}", num_inference_steps)
            ));
        }
        
        if cfg_scale < 1.0 || cfg_scale > 30.0 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid CFG scale: {}", cfg_scale)
            ));
        }
        
        if time_shift <= 0.0 || time_shift > 20.0 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid time shift: {}", time_shift)
            ));
        }
        
        if height == 0 || width == 0 || height > 4096 || width > 4096 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid image dimensions: {}x{}", width, height)
            ));
        }
        
        Ok(())
    }
    
    /// Validate Skip Layer Guidance configuration
    pub fn validate_slg_config(config: &SkipLayerGuidanceConfig) -> SamplingResult<()> {
        if config.scale < 0.0 || config.scale > 10.0 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid SLG scale: {}", config.scale)
            ));
        }
        
        if config.start < 0.0 || config.start > 1.0 || config.end < 0.0 || config.end > 1.0 {
            return Err(SamplingError::InvalidParameters(
                format!("Invalid SLG start/end: {}/{}", config.start, config.end)
            ));
        }
        
        if config.start >= config.end {
            return Err(SamplingError::InvalidParameters(
                "SLG start must be less than end".to_string()
            ));
        }
        
        Ok(())
    }
}