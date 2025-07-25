//! Model and tensor metadata structures
//!
//! Contains ModelMetadata and TensorEntry types extracted from the original
//! var_builder.rs file, designed for zero-allocation metadata management.

use arrayvec::{ArrayString, ArrayVec};
use candle_core::DType;

use super::types::{
    TensorName, ConfigKey, ConfigValue, DeviceHint,
    MAX_TENSOR_NAME_LEN, MAX_TENSORS, MAX_CONFIG_ENTRIES};
use crate::error::{CandleError, CandleResult as Result};

/// Ultra-compact model metadata with stack-allocated storage
///
/// Contains essential model information in a cache-friendly format.
/// All data is stack-allocated for maximum performance.
#[repr(C, align(64))] // Cache line aligned
#[derive(Clone)]
pub struct ModelMetadata {
    /// Model architecture name
    architecture: Option<ArrayString<64>>,

    /// Total parameter count
    total_parameters: u64,

    /// Model configuration entries (stack allocated)
    config_entries: ArrayVec<(ConfigKey, ConfigValue), MAX_CONFIG_ENTRIES>,

    /// Tensor metadata entries (stack allocated)
    tensor_entries: ArrayVec<TensorEntry, MAX_TENSORS>,

    /// Model creation timestamp
    created_at_nanos: u64,

    /// Model hash for integrity checking
    model_hash: u64}

impl ModelMetadata {
    /// Create new empty metadata
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            architecture: None,
            total_parameters: 0,
            config_entries: ArrayVec::new(),
            tensor_entries: ArrayVec::new(),
            created_at_nanos: Self::current_time_nanos(),
            model_hash: 0}
    }

    /// Get current high-precision timestamp
    #[inline(always)]
    fn current_time_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    }

    /// Set architecture name
    pub fn set_architecture(&mut self, arch: &str) -> Result<()> {
        if arch.len() > 64 {
            return Err(CandleError::ProcessingError("Architecture name too long"));
        }

        let mut arch_string = ArrayString::new();
        if arch_string.try_push_str(arch).is_ok() {
            self.architecture = Some(arch_string);
            Ok(())
        } else {
            Err(CandleError::ProcessingError("Failed to set architecture"))
        }
    }

    /// Get architecture name
    #[inline(always)]
    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_ref().map(|s| s.as_str())
    }

    /// Set total parameters
    #[inline(always)]
    pub fn set_total_parameters(&mut self, count: u64) {
        self.total_parameters = count;
    }

    /// Get total parameters
    #[inline(always)]
    pub const fn total_parameters(&self) -> u64 {
        self.total_parameters
    }

    /// Add configuration entry
    pub fn add_config_entry(&mut self, key: &str, value: &str) -> Result<()> {
        if self.config_entries.is_full() {
            return Err(CandleError::ProcessingError("Configuration entries full"));
        }

        let mut config_key = ConfigKey::new();
        let mut config_value = ConfigValue::new();

        if config_key.try_push_str(key).is_err() || config_value.try_push_str(value).is_err() {
            return Err(CandleError::ProcessingError("Configuration entry too long"));
        }

        self.config_entries.push((config_key, config_value));
        Ok(())
    }

    /// Get configuration value
    pub fn get_config_value(&self, key: &str) -> Option<&str> {
        self.config_entries
            .iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, v)| v.as_str())
    }

    /// Add tensor entry
    pub fn add_tensor_entry(&mut self, entry: TensorEntry) -> Result<()> {
        if self.tensor_entries.is_full() {
            return Err(CandleError::ProcessingError("Tensor entries full"));
        }

        self.tensor_entries.push(entry);
        Ok(())
    }

    /// Get tensor entry by name
    pub fn get_tensor_entry(&self, name: &str) -> Option<&TensorEntry> {
        self.tensor_entries
            .iter()
            .find(|entry| entry.name.as_str() == name)
    }

    /// Get tensor count
    #[inline(always)]
    pub fn tensor_count(&self) -> usize {
        self.tensor_entries.len()
    }

    /// Get configuration entry count
    #[inline(always)]
    pub fn config_count(&self) -> usize {
        self.config_entries.len()
    }

    /// Get model hash
    #[inline(always)]
    pub const fn model_hash(&self) -> u64 {
        self.model_hash
    }

    /// Set model hash
    #[inline(always)]
    pub fn set_model_hash(&mut self, hash: u64) {
        self.model_hash = hash;
    }

    /// Get creation timestamp
    #[inline(always)]
    pub const fn created_at_nanos(&self) -> u64 {
        self.created_at_nanos
    }

    /// Get model age in nanoseconds
    #[inline(always)]
    pub fn age_nanos(&self) -> u64 {
        Self::current_time_nanos().saturating_sub(self.created_at_nanos)
    }
}

impl Default for ModelMetadata {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Ultra-compact tensor metadata entry
///
/// Contains essential tensor information in a cache-friendly format.
/// All data is stack-allocated for maximum performance.
#[repr(C, align(32))] // Cache sub-line aligned
#[derive(Clone)]
pub struct TensorEntry {
    /// Tensor name (stack allocated)
    name: TensorName,

    /// Tensor shape (stack allocated)
    shape: ArrayVec<usize, 8>, // Most tensors have <= 8 dimensions

    /// Data type
    dtype: DType,

    /// Size in bytes
    size_bytes: u64,

    /// Tensor flags (bit-packed)
    /// Bit 0: is_parameter
    /// Bit 1: is_cached
    /// Bit 2: is_device_optimized
    /// Bit 3: requires_grad
    /// Bits 4-31: Reserved
    flags: u32,

    /// Device placement hint
    device_hint: DeviceHint}

impl TensorEntry {
    /// Create new tensor entry
    pub fn new(name: &str, shape: &[usize], dtype: DType, size_bytes: u64) -> Result<Self> {
        if name.len() > MAX_TENSOR_NAME_LEN {
            return Err(CandleError::ProcessingError("Tensor name too long"));
        }

        if shape.len() > 8 {
            return Err(CandleError::ProcessingError("Too many tensor dimensions"));
        }

        let mut tensor_name = TensorName::new();
        if tensor_name.try_push_str(name).is_err() {
            return Err(CandleError::ProcessingError("Failed to create tensor name"));
        }

        let mut tensor_shape = ArrayVec::new();
        for &dim in shape {
            if tensor_shape.try_push(dim).is_err() {
                return Err(CandleError::ProcessingError(
                    "Failed to add shape dimension",
                ));
            }
        }

        Ok(Self {
            name: tensor_name,
            shape: tensor_shape,
            dtype,
            size_bytes,
            flags: 0,
            device_hint: DeviceHint::Auto})
    }

    /// Get tensor name
    #[inline(always)]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Get tensor shape
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get data type
    #[inline(always)]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get size in bytes
    #[inline(always)]
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Check if tensor is a parameter
    #[inline(always)]
    pub const fn is_parameter(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Set parameter flag
    #[inline(always)]
    pub fn set_parameter(&mut self, is_param: bool) {
        if is_param {
            self.flags |= 1;
        } else {
            self.flags &= !1;
        }
    }

    /// Check if tensor is cached
    #[inline(always)]
    pub const fn is_cached(&self) -> bool {
        (self.flags & 2) != 0
    }

    /// Set cached flag
    #[inline(always)]
    pub fn set_cached(&mut self, cached: bool) {
        if cached {
            self.flags |= 2;
        } else {
            self.flags &= !2;
        }
    }

    /// Get device hint
    #[inline(always)]
    pub fn device_hint(&self) -> DeviceHint {
        self.device_hint.clone()
    }

    /// Set device hint
    #[inline(always)]
    pub fn set_device_hint(&mut self, hint: DeviceHint) {
        self.device_hint = hint;
    }

    /// Calculate tensor element count
    #[inline(always)]
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64
    }
}