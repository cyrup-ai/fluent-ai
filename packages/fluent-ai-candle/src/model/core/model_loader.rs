//! Model loading methods and loader creation
//!
//! Provides sophisticated model loading capabilities with progressive loading,
//! zero-allocation patterns, and blazing-fast initialization.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use candle_core::DType;
use crate::error::{CandleError, CandleResult};
use crate::memory;
use crate::model::{
    loading::{ModelLoader, ModelMetadata, ProgressCallback, RecoveryStrategy},
    types::QuantizationType,
};
use super::{CandleModel, model_state::ModelState};

impl CandleModel {
    /// Load model using sophisticated ModelLoader with progressive loading
    #[inline(always)]
    pub async fn load_with_loader<P: AsRef<Path>>(
        &self,
        path: P,
        loader: ModelLoader,
    ) -> CandleResult<ModelMetadata> {
        let (metadata, var_builder) = loader.load_model(path).await?;

        // Create model from var_builder based on detected architecture
        let (model, config) = match metadata.architecture.as_str() {
            "llama" => self.create_llama_model(var_builder, &metadata)?,
            "mistral" => self.create_mistral_model(var_builder, &metadata)?,
            "gemma" => self.create_gemma_model(var_builder, &metadata)?,
            "phi" => self.create_phi_model(var_builder, &metadata)?,
            "qwen" => self.create_qwen_model(var_builder, &metadata)?,
            _ => {
                return Err(CandleError::ModelLoadError(format!(
                    "Unsupported architecture: {}",
                    metadata.architecture
                )))
            }
        };

        // Create new model state with blazing-fast atomic swap
        let new_state = ModelState {
            model,
            config,
            _mmap: None, // MmapedSafetensors is managed by VarBuilder
        };

        // Atomically swap the model state
        self.model_state.store(Arc::new(new_state));

        // Update memory usage tracking with zero-allocation patterns
        self.memory_usage
            .store(metadata.model_size_bytes, Ordering::Relaxed);
        memory::track_allocation(metadata.model_size_bytes as usize);

        self.loading_progress.store(100, Ordering::Relaxed);
        self.is_loaded.store(true, Ordering::Relaxed);

        Ok(metadata)
    }

    /// Create sophisticated model loader with progress tracking
    #[inline(always)]
    pub fn create_loader(&self) -> ModelLoader {
        ModelLoader::new(self.device.clone(), DType::F16)
            .with_device_aware_loading(true)
            .with_validation(true)
            .with_recovery_strategy(RecoveryStrategy::Retry)
    }

    /// Create sophisticated model loader with custom configuration
    #[inline(always)]
    pub fn create_loader_with_config(
        &self,
        dtype: DType,
        quantization: Option<QuantizationType>,
        progress_callback: Option<ProgressCallback>,
    ) -> ModelLoader {
        let mut loader = ModelLoader::new(self.device.clone(), dtype)
            .with_device_aware_loading(true)
            .with_validation(true)
            .with_recovery_strategy(RecoveryStrategy::Retry);

        if let Some(quant) = quantization {
            loader = loader.with_quantization(quant);
        }

        if let Some(callback) = progress_callback {
            loader = loader.with_progress_callback(callback);
        }

        loader
    }
}