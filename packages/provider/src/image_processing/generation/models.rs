//! Model management for Stable Diffusion 3 image generation
//!
//! This module handles model loading, caching, and management for different SD3 variants
//! with efficient memory usage and HuggingFace Hub integration.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::{
    mmdit::model::{Config as MMDiTConfig, MMDiT},
    stable_diffusion::clip::Config as ClipConfig,
    t5::Config as T5Config};
use hf_hub::api::sync::Api;
use thiserror::Error;

use super::config::{GenerationConfig, ModelLoadingConfig, SD3ModelVariant};

/// Model management errors
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model loading failed: {0}")]
    LoadingError(String),

    #[error("Model not found: {0}")]
    NotFoundError(String),

    #[error("Model cache error: {0}")]
    CacheError(String),

    #[error("Weight mapping error: {0}")]
    WeightMappingError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("HuggingFace Hub error: {0}")]
    HubError(#[from] hf_hub::api::sync::ApiError),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error)}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Cached model components
#[derive(Debug)]
struct CachedModel {
    mmdit: Option<Arc<MMDiT>>,
    text_encoder: Option<Arc<super::text_encoder::StableDiffusion3TripleClipWithTokenizer>>,
    vae: Option<Arc<super::vae::SD3VAEDecoder>>,
    variant: SD3ModelVariant,
    device: Device,
    memory_usage: usize,
    last_accessed: std::time::Instant}

/// Model manager for SD3 variants with caching and memory optimization
pub struct ModelManager {
    device: Device,
    model_cache: Arc<Mutex<HashMap<String, CachedModel>>>,
    max_cache_size: usize,
    cache_dir: Option<PathBuf>,
    current_model: Option<String>}

impl ModelManager {
    /// Create new model manager
    pub fn new(device: Device) -> Self {
        Self {
            device,
            model_cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size: 2, // Maximum 2 models in cache
            cache_dir: None,
            current_model: None}
    }

    /// Create model manager with custom cache settings
    pub fn with_cache_settings(
        device: Device,
        max_cache_size: usize,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            device,
            model_cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size,
            cache_dir,
            current_model: None}
    }

    /// Load MMDiT model from HuggingFace Hub
    pub fn load_mmdit_model(
        &self,
        variant: SD3ModelVariant,
        config: &ModelLoadingConfig,
    ) -> ModelResult<Arc<MMDiT>> {
        let model_id = config.model_id.clone();

        // Check cache first
        if let Some(cached) = self.get_cached_mmdit(&model_id) {
            return Ok(cached);
        }

        // Load model configuration
        let mmdit_config = self.get_mmdit_config(variant)?;

        // Download model weights
        let weights_path = self.download_model_weights(&model_id, config)?;

        // Load weights into VarBuilder
        let var_builder = self.load_weights_as_varbuilder(&weights_path)?;

        // Create MMDiT model
        let mmdit = MMDiT::new(var_builder, &mmdit_config)
            .map_err(|e| ModelError::LoadingError(format!("MMDiT creation failed: {}", e)))?;

        let mmdit_arc = Arc::new(mmdit);

        // Cache the model
        self.cache_mmdit_model(&model_id, mmdit_arc.clone(), variant)?;

        Ok(mmdit_arc)
    }

    /// Load complete model (MMDiT + text encoder + VAE)
    pub fn load_complete_model(
        &self,
        variant: SD3ModelVariant,
        config: &ModelLoadingConfig,
    ) -> ModelResult<CompleteModel> {
        let model_id = config.model_id.clone();

        // Load MMDiT
        let mmdit = self.load_mmdit_model(variant, config)?;

        // Load text encoder
        let text_encoder = self.load_text_encoder(variant, config)?;

        // Load VAE
        let vae = self.load_vae_decoder(variant, config)?;

        Ok(CompleteModel {
            mmdit,
            text_encoder,
            vae,
            variant,
            device: self.device.clone()})
    }

    /// Load text encoder components
    fn load_text_encoder(
        &self,
        variant: SD3ModelVariant,
        config: &ModelLoadingConfig,
    ) -> ModelResult<Arc<super::text_encoder::StableDiffusion3TripleClipWithTokenizer>> {
        let model_id = config.model_id.clone();

        // Download text encoder weights
        let weights_path = self.download_text_encoder_weights(&model_id, config)?;

        // Load weights into VarBuilder
        let var_builder = self.load_weights_as_varbuilder(&weights_path)?;

        // Create text encoder
        let text_encoder =
            super::text_encoder::StableDiffusion3TripleClipWithTokenizer::new(var_builder)
                .map_err(|e| {
                    ModelError::LoadingError(format!("Text encoder creation failed: {}", e))
                })?;

        Ok(Arc::new(text_encoder))
    }

    /// Load VAE decoder
    fn load_vae_decoder(
        &self,
        variant: SD3ModelVariant,
        config: &ModelLoadingConfig,
    ) -> ModelResult<Arc<super::vae::SD3VAEDecoder>> {
        let model_id = config.model_id.clone();

        // Build VAE autoencoder
        let vae_autoencoder = super::vae::build_sd3_vae_autoencoder(
            &self.device,
            &model_id,
            config.revision.as_deref(),
        )
        .map_err(|e| ModelError::LoadingError(format!("VAE loading failed: {}", e)))?;

        // Create VAE decoder
        let vae_decoder = super::vae::SD3VAEDecoder::new(vae_autoencoder);

        Ok(Arc::new(vae_decoder))
    }

    /// Get MMDiT configuration for variant
    pub fn get_mmdit_config(&self, variant: SD3ModelVariant) -> ModelResult<MMDiTConfig> {
        let config = match variant {
            SD3ModelVariant::ThreeMedium => MMDiTConfig {
                patch_size: 2,
                in_channels: 16,
                depth: 24,
                num_patches: 1024,
                hidden_size: 1536,
                num_heads: 24,
                mlp_ratio: 4.0,
                class_dropout_prob: 0.1,
                num_classes: 1000,
                learn_sigma: false,
                adm_in_channels: Some(2816),
                context_embedder_config: super::text_encoder::get_context_embedder_config(),
                pos_embed_scaling_factor: None,
                pos_embed_offset: None,
                pos_embed_max_size: None},
            SD3ModelVariant::ThreeFiveLarge => MMDiTConfig {
                patch_size: 2,
                in_channels: 16,
                depth: 38,
                num_patches: 1024,
                hidden_size: 3072,
                num_heads: 24,
                mlp_ratio: 4.0,
                class_dropout_prob: 0.1,
                num_classes: 1000,
                learn_sigma: false,
                adm_in_channels: Some(2816),
                context_embedder_config: super::text_encoder::get_context_embedder_config(),
                pos_embed_scaling_factor: None,
                pos_embed_offset: None,
                pos_embed_max_size: None},
            SD3ModelVariant::ThreeFiveLargeTurbo => MMDiTConfig {
                patch_size: 2,
                in_channels: 16,
                depth: 38,
                num_patches: 1024,
                hidden_size: 3072,
                num_heads: 24,
                mlp_ratio: 4.0,
                class_dropout_prob: 0.1,
                num_classes: 1000,
                learn_sigma: false,
                adm_in_channels: Some(2816),
                context_embedder_config: super::text_encoder::get_context_embedder_config(),
                pos_embed_scaling_factor: None,
                pos_embed_offset: None,
                pos_embed_max_size: None},
            SD3ModelVariant::ThreeFiveMedium => MMDiTConfig {
                patch_size: 2,
                in_channels: 16,
                depth: 24,
                num_patches: 1024,
                hidden_size: 1536,
                num_heads: 24,
                mlp_ratio: 4.0,
                class_dropout_prob: 0.1,
                num_classes: 1000,
                learn_sigma: false,
                adm_in_channels: Some(2816),
                context_embedder_config: super::text_encoder::get_context_embedder_config(),
                pos_embed_scaling_factor: None,
                pos_embed_offset: None,
                pos_embed_max_size: None}};

        Ok(config)
    }

    /// Download model weights from HuggingFace Hub
    fn download_model_weights(
        &self,
        model_id: &str,
        config: &ModelLoadingConfig,
    ) -> ModelResult<PathBuf> {
        let api = Api::new().map_err(|e| ModelError::HubError(e))?;

        let repo = if let Some(revision) = &config.revision {
            api.model(model_id.to_string()).revision(revision.clone())
        } else {
            api.model(model_id.to_string())
        };

        let weights_filename = if config.use_safetensors {
            "transformer/diffusion_pytorch_model.safetensors"
        } else {
            "transformer/diffusion_pytorch_model.bin"
        };

        let weights_path = repo
            .get(weights_filename)
            .map_err(|e| ModelError::HubError(e))?;

        Ok(weights_path)
    }

    /// Download text encoder weights
    fn download_text_encoder_weights(
        &self,
        model_id: &str,
        config: &ModelLoadingConfig,
    ) -> ModelResult<PathBuf> {
        let api = Api::new().map_err(|e| ModelError::HubError(e))?;

        let repo = if let Some(revision) = &config.revision {
            api.model(model_id.to_string()).revision(revision.clone())
        } else {
            api.model(model_id.to_string())
        };

        let weights_filename = if config.use_safetensors {
            "text_encoder/model.safetensors"
        } else {
            "text_encoder/pytorch_model.bin"
        };

        let weights_path = repo
            .get(weights_filename)
            .map_err(|e| ModelError::HubError(e))?;

        Ok(weights_path)
    }

    /// Load weights as VarBuilder
    fn load_weights_as_varbuilder(&self, weights_path: &PathBuf) -> ModelResult<VarBuilder> {
        let weights = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(weights_path)
                .map_err(|e| ModelError::LoadingError(format!("Failed to load weights: {}", e)))?
        };

        let mut var_map = VarMap::new();
        for (name, tensor) in weights.tensors() {
            let device_tensor = tensor.to_device(&self.device)?;
            var_map.set(name, device_tensor)?;
        }

        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &self.device);
        Ok(var_builder)
    }

    /// Get cached MMDiT model
    fn get_cached_mmdit(&self, model_id: &str) -> Option<Arc<MMDiT>> {
        let cache = self.model_cache.lock().ok()?;
        cache.get(model_id).and_then(|cached| cached.mmdit.clone())
    }

    /// Cache MMDiT model
    fn cache_mmdit_model(
        &self,
        model_id: &str,
        mmdit: Arc<MMDiT>,
        variant: SD3ModelVariant,
    ) -> ModelResult<()> {
        let mut cache = self
            .model_cache
            .lock()
            .map_err(|e| ModelError::CacheError(format!("Cache lock failed: {}", e)))?;

        // Estimate memory usage
        let memory_usage = self.estimate_model_memory(variant);

        // Clean cache if needed
        self.clean_cache_if_needed(&mut cache)?;

        // Create cached model entry
        let cached_model = CachedModel {
            mmdit: Some(mmdit),
            text_encoder: None,
            vae: None,
            variant,
            device: self.device.clone(),
            memory_usage,
            last_accessed: std::time::Instant::now()};

        cache.insert(model_id.to_string(), cached_model);
        Ok(())
    }

    /// Clean cache if needed
    fn clean_cache_if_needed(&self, cache: &mut HashMap<String, CachedModel>) -> ModelResult<()> {
        if cache.len() >= self.max_cache_size {
            // Remove oldest accessed model
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, cached)| cached.last_accessed)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
        Ok(())
    }

    /// Estimate model memory usage
    fn estimate_model_memory(&self, variant: SD3ModelVariant) -> usize {
        super::config::DeviceOptimization::calculate_memory_requirements(&GenerationConfig {
            model_variant: variant,
            ..Default::default()
        })
    }

    /// Clear all cached models
    pub fn clear_cache(&self) -> ModelResult<()> {
        let mut cache = self
            .model_cache
            .lock()
            .map_err(|e| ModelError::CacheError(format!("Cache lock failed: {}", e)))?;

        cache.clear();
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> ModelResult<CacheStats> {
        let cache = self
            .model_cache
            .lock()
            .map_err(|e| ModelError::CacheError(format!("Cache lock failed: {}", e)))?;

        let total_memory = cache.values().map(|c| c.memory_usage).sum();
        let model_count = cache.len();

        Ok(CacheStats {
            model_count,
            total_memory_usage: total_memory,
            max_cache_size: self.max_cache_size})
    }
}

/// Complete model with all components
pub struct CompleteModel {
    pub mmdit: Arc<MMDiT>,
    pub text_encoder: Arc<super::text_encoder::StableDiffusion3TripleClipWithTokenizer>,
    pub vae: Arc<super::vae::SD3VAEDecoder>,
    pub variant: SD3ModelVariant,
    pub device: Device}

impl CompleteModel {
    /// Get model variant
    pub fn variant(&self) -> SD3ModelVariant {
        self.variant
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(&self) -> usize {
        super::config::DeviceOptimization::calculate_memory_requirements(&GenerationConfig {
            model_variant: self.variant,
            ..Default::default()
        })
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub model_count: usize,
    pub total_memory_usage: usize,
    pub max_cache_size: usize}

/// Model loading utilities
pub mod utils {
    use super::*;

    /// Validate model configuration
    pub fn validate_model_config(config: &ModelLoadingConfig) -> ModelResult<()> {
        if config.model_id.is_empty() {
            return Err(ModelError::LoadingError(
                "Model ID cannot be empty".to_string(),
            ));
        }

        if config.timeout_seconds == 0 {
            return Err(ModelError::LoadingError(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get default model configuration for variant
    pub fn get_default_model_config(variant: SD3ModelVariant) -> ModelLoadingConfig {
        ModelLoadingConfig {
            model_id: variant.model_id().to_string(),
            revision: None,
            use_safetensors: true,
            cache_dir: None,
            cache_enabled: true,
            timeout_seconds: 300}
    }

    /// Check if model is available locally
    pub fn is_model_cached(model_id: &str, cache_dir: Option<&PathBuf>) -> bool {
        // Implementation depends on HuggingFace Hub cache structure
        // This is a simplified check
        false
    }
}
