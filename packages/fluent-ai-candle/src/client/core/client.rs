//! Core CandleCompletionClient struct definition and basic implementations
//!
//! This module contains the main client struct definition and core functionality
//! extracted from the original monolithic client.rs file.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use candle_core::Device;

use super::super::config::{CandleClientConfig, DeviceType};
use super::super::metrics::CandleMetrics;
use crate::generator::CandleGenerator;
use crate::model::CandleModel;
use crate::tokenizer::CandleTokenizer;

/// Zero-allocation Candle completion client with provider pattern alignment
pub struct CandleCompletionClient {
    /// Client configuration
    pub(super) config: CandleClientConfig,
    /// The candle model
    pub(super) model: Arc<CandleModel>,
    /// The tokenizer
    pub(super) tokenizer: Arc<CandleTokenizer>,
    /// The generator
    pub(super) generator: ArcSwap<CandleGenerator>,
    /// Computation device
    pub(super) device: Arc<Device>,
    /// Performance metrics reference
    pub(super) metrics: &'static CandleMetrics,
    /// Is client initialized
    pub(super) is_initialized: AtomicBool,
    /// Maximum concurrent requests allowed
    pub(super) max_concurrent_requests: usize,
}

impl Clone for CandleCompletionClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
            generator: ArcSwap::new(Arc::clone(&self.generator.load())),
            device: Arc::clone(&self.device),
            metrics: self.metrics,
            is_initialized: AtomicBool::new(self.is_initialized.load(Ordering::Acquire)),
            max_concurrent_requests: self.max_concurrent_requests,
        }
    }
}

impl CandleCompletionClient {
    /// Check if client is initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Acquire)
    }

    /// Create device from device type configuration
    pub(super) fn create_device(device_type: DeviceType) -> Result<Device, crate::error::CandleError> {
        match device_type {
            DeviceType::Auto => {
                // Try CUDA first, then Metal, then CPU
                if candle_core::utils::cuda_is_available() {
                    Device::new_cuda(0).map_err(|e| crate::error::CandleError::device_creation(format!("CUDA device creation failed: {}", e)))
                } else if candle_core::utils::metal_is_available() {
                    Device::new_metal(0).map_err(|e| crate::error::CandleError::device_creation(format!("Metal device creation failed: {}", e)))
                } else {
                    Ok(Device::Cpu)
                }
            }
            DeviceType::Cpu => Ok(Device::Cpu),
            DeviceType::Cuda => {
                Device::new_cuda(0).map_err(|e| crate::error::CandleError::device_creation(format!("CUDA device creation failed: {}", e)))
            }
            DeviceType::Metal => {
                Device::new_metal(0).map_err(|e| crate::error::CandleError::device_creation(format!("Metal device creation failed: {}", e)))
            }
        }
    }

    /// Get reference to the metrics instance
    pub fn metrics(&self) -> &'static CandleMetrics {
        self.metrics
    }

    /// Get reference to the device
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get reference to the model
    pub fn model(&self) -> &Arc<CandleModel> {
        &self.model
    }

    /// Get reference to the tokenizer
    pub fn tokenizer(&self) -> &Arc<CandleTokenizer> {
        &self.tokenizer
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &CandleClientConfig {
        &self.config
    }
}

unsafe impl Send for CandleCompletionClient {}
unsafe impl Sync for CandleCompletionClient {}