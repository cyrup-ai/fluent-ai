//! Model loading system with VarBuilder patterns and progressive loading
//!
//! This module provides production-quality model loading with:
//! - Sophisticated VarBuilder patterns for memory-efficient loading
//! - Progressive loading with detailed progress tracking
//! - Memory-mapped SafeTensors for zero-copy model access
//! - Device-aware loading with automatic device detection
//! - Quantization during loading with validation
//! - Comprehensive error recovery and configuration validation
//! - Model metadata extraction and compatibility checking

#![deny(
    missing_docs,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unused_import_braces,
    unused_qualifications,
    unused_results,
    unused_variables,
    unused_must_use,
    unreachable_pub,
    clippy::all
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_core::safetensors::MmapedSafetensors;

use crate::error::CandleError;

// Re-export submodules
pub mod compatibility;
pub mod metadata;
pub mod progress;
pub mod quantization;
pub mod recovery;
pub mod var_builder;

// Re-export commonly used types
pub use compatibility::check_system_requirements;
pub use metadata::{ModelMetadata, TensorInfo};
pub use progress::{LoadingStage, ProgressCallback, ProgressTracker, SubProgressTracker};
pub use quantization::{QuantizationConfig, QuantizationType, QuantizedTensor};
pub use recovery::{RecoveryAction, RecoveryContext, RecoveryStrategy};
pub use var_builder::{HotSwappableVarBuilder, VarBuilderConfig, VarBuilderFactory};

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    /// Device to load the model onto
    pub device: Device,
    
    /// Data type for model weights
    pub dtype: DType,
    
    /// Whether to use memory mapping for large files
    pub use_mmap: bool,
    
    /// Whether to keep the original tensor data after quantization
    pub keep_original: bool,
    
    /// Whether to validate the model after loading
    pub validate: bool,
    
    /// Recovery strategy for handling errors
    pub recovery_strategy: RecoveryStrategy,
    
    /// Maximum number of retry attempts for downloads
    pub max_retries: u32,
    
    /// Custom tensor transformations
    pub transforms: HashMap<String, Box<dyn Fn(Tensor) -> CandleResult<Tensor> + Send + Sync>>,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            dtype: DType::F32,
            use_mmap: true,
            keep_original: false,
            validate: true,
            recovery_strategy: RecoveryStrategy::Recover,
            max_retries: 3,
            transforms: HashMap::new(),
        }
    }
}

/// Main model loader
pub struct ModelLoader {
    config: ModelLoaderConfig,
    progress: ProgressTracker,
    recovery: RecoveryContext,
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new(ModelLoaderConfig::default())
    }
}

impl ModelLoader {
    /// Create a new model loader with the given configuration
    pub fn new(config: ModelLoaderConfig) -> Self {
        let progress = ProgressTracker::new();
        let recovery = RecoveryContext::new(config.recovery_strategy);
        
        Self {
            config,
            progress,
            recovery,
        }
    }
    
    /// Set the progress callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(f64, &str) + Send + Sync + 'static,
    {
        self.progress = self.progress.with_callback(callback);
        self
    }
    
    /// Load a model from a file
    pub fn load_from_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> CandleResult<(ModelMetadata, VarBuilder<'static>)> {
        self.progress
            .set_stage(LoadingStage::Initializing)?;
            
        // Create a VarBuilderFactory
        let var_builder_config = VarBuilderConfig {
            device: self.config.device.clone(),
            dtype: self.config.dtype,
            use_mmap: self.config.use_mmap,
            keep_original: self.config.keep_original,
            progress: Some(self.progress.subtracker(LoadingStage::LoadingWeights)),
            transforms: self.config.transforms.clone(),
        };
        
        let factory = VarBuilderFactory::from_safetensors(path, var_builder_config)?;
        
        // Extract metadata
        let metadata = ModelMetadata::from_safetensors(factory.get_safetensors()?)?;
        
        // Check system requirements
        if self.config.validate {
            check_system_requirements(&metadata, &self.config.device)?;
        }
        
        // Create VarBuilder
        let var_builder = factory.into_var_builder();
        
        self.progress.set_stage(LoadingStage::Complete)?;
        
        Ok((metadata, var_builder))
    }
    
    /// Get the progress tracker
    pub fn progress_tracker(&self) -> &ProgressTracker {
        &self.progress
    }
    
    /// Get a mutable reference to the progress tracker
    pub fn progress_tracker_mut(&mut self) -> &mut ProgressTracker {
        &mut self.progress
    }
    
    /// Get the recovery context
    pub fn recovery_context(&self) -> &RecoveryContext {
        &self.recovery
    }
    
    /// Get a mutable reference to the recovery context
    pub fn recovery_context_mut(&mut self) -> &mut RecoveryContext {
        &mut self.recovery
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_model_loader_default() {
        let loader = ModelLoader::default();
        assert_eq!(loader.config.dtype, DType::F32);
        assert!(loader.config.use_mmap);
        assert!(!loader.config.keep_original);
        assert!(loader.config.validate);
    }
    
    #[test]
    fn test_model_loader_with_callback() {
        let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();
        
        let callback = move |_progress: f64, _message: &str| {
            callback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        };
        
        let loader = ModelLoader::default().with_progress_callback(callback);
        
        // Trigger a progress update
        loader
            .progress_tracker()
            .update_progress(0.5)
            .unwrap();
            
        assert!(callback_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}