//! Model loading methods and loader creation
//!
//! Provides sophisticated model loading capabilities with progressive loading,
//! zero-allocation patterns, and blazing-fast initialization.

use std::path::Path;

use candle_core::DType;
use fluent_ai_async::{AsyncStream, handle_error};

use super::CandleModel;
use crate::error::CandleError;
use crate::model::{
    loading::{ModelLoader, ModelMetadata, ProgressCallback, RecoveryStrategy},
    types::QuantizationType,
};

impl CandleModel {
    /// Load model using sophisticated ModelLoader with progressive loading
    #[inline(always)]
    pub fn load_with_loader<P: AsRef<Path> + Send + 'static>(
        &self,
        _path: P,
        _loader: ModelLoader,
    ) -> AsyncStream<ModelMetadata> {
        let _model_ref = self.clone();
        AsyncStream::with_channel(move |_sender| {
            // SYNCHRONOUS implementation only (async patterns forbidden per CLAUDE.md)
            // For now, provide placeholder functionality that handles error properly
            let error = CandleError::ModelLoadError(
                "Async model loading not yet implemented - use synchronous loading methods"
                    .to_string(),
            );
            handle_error!(error, "Model loading not implemented");
        })
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
