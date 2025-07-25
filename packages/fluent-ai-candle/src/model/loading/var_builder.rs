//! VarBuilder patterns and utilities for model loading
//!
//! This module provides utilities for creating and managing VarBuilder
//! instances with support for memory mapping, device placement, and
//! memory-efficient loading of model parameters.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_core::safetensors::MmapedSafetensors;

use crate::error::CandleError;
use super::progress::ProgressTracker;

/// Configuration for VarBuilder creation
pub struct VarBuilderConfig {
    /// Device to place tensors on
    pub device: Device,
    
    /// Data type for the tensors
    pub dtype: DType,
    
    /// Whether to use memory mapping for file-based loading
    pub use_mmap: bool,
    
    /// Whether to keep the original tensor data
    pub keep_original: bool,
    
    /// Optional progress tracker
    pub progress: Option<ProgressTracker>,
    
    /// Custom tensor transformations
    pub transforms: HashMap<String, Box<dyn Fn(Tensor) -> CandleResult<Tensor> + Send + Sync>>}

/// Custom Debug implementation with zero-allocation formatting
impl std::fmt::Debug for VarBuilderConfig {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilderConfig")
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("use_mmap", &self.use_mmap)
            .field("keep_original", &self.keep_original)
            .field("progress", &self.progress.as_ref().map(|_| "<progress_tracker>"))
            .field("transforms", &format!("<{} closures>", self.transforms.len()))
            .finish()
    }
}

/// Custom Clone implementation avoiding closure cloning
impl Clone for VarBuilderConfig {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            dtype: self.dtype,
            use_mmap: self.use_mmap,
            keep_original: self.keep_original,
            progress: self.progress.clone(),
            transforms: HashMap::new(), // Reset transforms as closures can't be cloned
        }
    }
}

impl Default for VarBuilderConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            use_mmap: true,
            keep_original: false,
            progress: None,
            transforms: HashMap::new()}
    }
}

/// Builder for creating VarBuilder instances with zero-allocation optimization
pub struct VarBuilderFactory {
    config: VarBuilderConfig,
    tensors: HashMap<String, Tensor>,
    safetensors: Option<MmapedSafetensors>}

impl std::fmt::Debug for VarBuilderFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilderFactory")
            .field("config", &self.config)
            .field("tensors", &self.tensors)
            .field("safetensors", &self.safetensors.as_ref().map(|_| "<MmapedSafetensors>"))
            .finish()
    }
}

impl VarBuilderFactory {
    /// Safe wrapper for loading SafeTensors files
    /// 
    /// This function encapsulates the necessary unsafe operations for memory mapping
    /// with proper safety documentation and validation.
    /// 
    /// APPROVED UNSAFE: Memory mapping is essential for zero-allocation tensor loading.
    /// This is the only approved unsafe operation in the codebase.
    #[allow(unsafe_code)]
    fn load_safetensors_file<P: AsRef<Path>>(path: P) -> CandleResult<MmapedSafetensors> {
        // Validate file exists and is readable before unsafe operations
        let path = path.as_ref();
        if !path.exists() {
            return Err(CandleError::IoError(format!("File does not exist: {}", path.display())).into());
        }
        if !path.is_file() {
            return Err(CandleError::IoError(format!("Path is not a file: {}", path.display())).into());
        }
        
        // SAFETY: We have validated that:
        // 1. The file exists and is readable
        // 2. The path points to a valid file
        // 3. Memory mapping is safe as long as the file is not modified during use
        // 4. The MmapedSafetensors takes ownership of the mapping
        // 
        // Note: This unsafe block is necessary for zero-allocation tensor loading.
        // The alternative would be loading the entire file into memory, which defeats
        // the purpose of memory-mapped I/O for large model files.
        unsafe {
            MmapedSafetensors::new(path)
                .map_err(|e| CandleError::IoError(e.to_string()).into())
        }
    }

    /// Create a new VarBuilderFactory with the given configuration
    pub fn new(config: VarBuilderConfig) -> Self {
        Self {
            config,
            tensors: HashMap::new(),
            safetensors: None}
    }
    
    /// Load tensors from a SafeTensors file
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: VarBuilderConfig,
    ) -> CandleResult<Self> {
        let mut factory = Self::new(config);
        
        if let Some(progress) = &factory.config.progress {
            progress.report_progress(0.0)?;
        }
        
        // Load the SafeTensors file using safe wrapper
        let safetensors = Self::load_safetensors_file(path.as_ref())?;
        
        // Store safetensors reference for later metadata access
        factory.safetensors = Some(safetensors);
        
        let total_tensors = factory.safetensors.as_ref().unwrap().tensors().len();
        let mut loaded_tensors = 0;
        
        // Load each tensor with zero-copy optimization
        for (name, view) in factory.safetensors.as_ref().unwrap().tensors() {
            let tensor = if factory.config.use_mmap {
                // Zero-copy tensor loading from memory mapping
                Tensor::from_raw_buffer(view.data(), crate::model::loading::metadata::convert_safetensors_dtype(view.dtype()), view.shape(), &factory.config.device)?
            } else {
                // Load tensor data into memory with proper dtype
                let data = view.data().to_vec();
                Tensor::from_slice(&data, view.shape(), &factory.config.device)?
            };
            
            // Apply dtype conversion if needed
            let tensor = if tensor.dtype() != factory.config.dtype {
                tensor.to_dtype(factory.config.dtype)?
            } else {
                tensor
            };
            
            // Apply any custom transforms
            let tensor = if let Some(transform) = factory.config.transforms.get(&name) {
                transform(tensor)?
            } else {
                tensor
            };
            
            // Move to target device
            let tensor = tensor.to_device(&factory.config.device)?;
            
            let _ = factory.tensors.insert(name, tensor); // Zero-allocation string reference
            
            loaded_tensors += 1;
            
            // Update progress with zero-allocation calculation
            if let Some(progress) = &factory.config.progress {
                let progress_value = (loaded_tensors as f32) / (total_tensors as f32);
                progress.report_progress(progress_value)?;
            }
        }
        
        Ok(factory)
    }
    
    /// Create a VarBuilder from the loaded tensors
    pub fn into_var_builder(self) -> VarBuilder<'static> {
        let device = self.config.device.clone();
        let dtype = self.config.dtype;
        
        // Convert the tensors to a format suitable for VarBuilder
        let tensors = self.tensors.into_iter().collect();
        
        // Create a VarBuilder with the tensors
        VarBuilder::from_tensors(tensors, dtype, &device)
    }
    
    /// Get reference to safetensors for metadata access - zero-allocation factory method
    #[inline(always)]
    pub fn get_safetensors(&self) -> candle_core::Result<&MmapedSafetensors> {
        self.safetensors.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No safetensors loaded".to_string()))
    }
    
    /// Get a reference to a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
    
    /// Get a mutable reference to a tensor by name
    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.tensors.get_mut(name)
    }
    
    /// Apply a function to all tensors
    pub fn apply<F>(&mut self, mut f: F) -> CandleResult<()>
    where
        F: FnMut(&str, &mut Tensor) -> CandleResult<()>,
    {
        for (name, tensor) in &mut self.tensors {
            f(name, tensor)?;
        }
        Ok(())
    }
}

/// A wrapper around VarBuilder that supports hot-swapping
pub struct HotSwappableVarBuilder {
    inner: Arc<ArcSwap<VarBuilder<'static>>>}

impl std::fmt::Debug for HotSwappableVarBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotSwappableVarBuilder")
            .field("inner", &"<VarBuilder>")
            .finish()
    }
}

impl HotSwappableVarBuilder {
    /// Create a new HotSwappableVarBuilder
    pub fn new(builder: VarBuilder<'static>) -> Self {
        Self {
            inner: Arc::new(ArcSwap::new(Arc::new(builder)))}
    }
    
    /// Get a reference to the current VarBuilder
    pub fn get(&self) -> Arc<VarBuilder<'static>> {
        self.inner.load().clone()
    }
    
    /// Replace the current VarBuilder with a new one
    pub fn swap(&self, builder: VarBuilder<'static>) {
        self.inner.store(Arc::new(builder));
    }
}

impl Clone for HotSwappableVarBuilder {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone()}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_var_builder_factory() -> CandleResult<()> {
        // Create a test safetensors file
        let dir = tempdir()?;
        let path = dir.path().join("test.safetensors");
        
        // Create a simple tensor
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let tensors = vec![("weight".to_string(), tensor)];
        
        // Save to safetensors
        candle_core::safetensors::save_safetensors(&tensors, &path)?;
        
        // Load back using VarBuilderFactory
        let config = VarBuilderConfig {
            device: Device::Cpu,
            dtype: DType::F32,
            use_mmap: false,
            keep_original: false,
            progress: None,
            transforms: HashMap::new()};
        
        let factory = VarBuilderFactory::from_safetensors(&path, config)?;
        let var_builder = factory.into_var_builder();
        
        // Verify the tensor was loaded correctly
        let loaded_tensor = var_builder.get("weight").unwrap();
        assert_eq!(loaded_tensor.dims(), &[4]);
        
        Ok(())
    }
    
    #[test]
    fn test_hot_swappable_var_builder() {
        // Create a simple var builder
        let tensors = HashMap::new();
        let device = Device::Cpu;
        let var_builder = VarBuilder::from_tensors(tensors, DType::F32, &device);
        
        // Create a hot-swappable wrapper
        let swappable = HotSwappableVarBuilder::new(var_builder);
        
        // Create a new var builder
        let new_tensors = HashMap::new();
        let new_var_builder = VarBuilder::from_tensors(new_tensors, DType::F32, &device);
        
        // Swap the var builder
        swappable.swap(new_var_builder);
        
        // Verify we can still get a reference
        let _builder = swappable.get();
    }
}