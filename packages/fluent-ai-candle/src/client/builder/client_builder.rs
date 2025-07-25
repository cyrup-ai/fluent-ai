//! Client builder for creating CandleCompletionClient instances
//!
//! This module provides a fluent builder API for configuring and creating clients.

use super::super::config::{CandleClientConfig, DeviceType, QuantizationType};
use super::super::completion::CandleCompletionClient;
use crate::error::CandleResult;
use crate::generator::GenerationConfig;

/// Builder for CandleCompletionClient
pub struct CandleClientBuilder {
    config: CandleClientConfig,
}

impl CandleClientBuilder {
    /// Create a new client builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            config: CandleClientConfig::default(),
        }
    }

    /// Set model path
    #[inline(always)]
    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Set tokenizer path
    #[inline(always)]
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    /// Set device type
    #[inline(always)]
    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }

    /// Set generation configuration
    #[inline(always)]
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = config;
        self
    }

    /// Enable quantization
    #[inline(always)]
    pub fn quantization(mut self, quantization_type: QuantizationType) -> Self {
        self.config.enable_quantization = true;
        self.config.quantization_type = quantization_type;
        self
    }

    /// Set maximum concurrent requests
    #[inline(always)]
    pub fn max_concurrent_requests(mut self, max: u32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    /// Build the client
    #[inline(always)]
    pub fn build(self) -> CandleResult<CandleCompletionClient> {
        CandleCompletionClient::new(self.config)
    }
}

impl Default for CandleClientBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}