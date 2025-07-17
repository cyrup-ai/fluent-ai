//! Factory for creating image processing backend instances
//!
//! This module provides a factory pattern for creating and configuring
//! image processing backends based on available features and system capabilities.

use super::*;
use crate::image_processing::candle_backend::CandleImageProcessor;
use std::collections::HashMap;

/// Factory for creating image processing backends
pub struct ImageProcessingFactory;

impl ImageProcessingFactory {
    /// Create an image processing backend based on configuration
    pub fn create_backend(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let backend = match backend_type {
            BackendType::Candle => {
                Self::create_candle_backend(config)?
            }
            BackendType::Auto => {
                Self::create_auto_backend(config)?
            }
        };
        
        Ok(backend)
    }
    
    /// Create an image embedding provider
    pub fn create_embedding_provider(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let backend = match backend_type {
            BackendType::Candle => {
                Self::create_candle_embedding_provider(config)?
            }
            BackendType::Auto => {
                Self::create_auto_embedding_provider(config)?
            }
        };
        
        Ok(backend)
    }
    
    /// Create an image generation provider
    #[cfg(feature = "generation")]
    pub fn create_generation_provider(
        backend_type: BackendType,
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let backend = match backend_type {
            BackendType::Candle => {
                Self::create_candle_generation_provider(config)?
            }
            BackendType::Auto => {
                Self::create_auto_generation_provider(config)?
            }
        };
        
        Ok(backend)
    }
    
    /// Get available backends on current system
    pub fn available_backends() -> Vec<BackendType> {
        let mut backends = Vec::new();
        
        // Check Candle availability
        if Self::is_candle_available() {
            backends.push(BackendType::Candle);
        }
        
        backends
    }
    
    /// Get recommended backend for current system
    pub fn recommended_backend() -> BackendType {
        // For now, Candle is the default and only backend
        BackendType::Candle
    }
    
    /// Create Candle backend
    fn create_candle_backend(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let mut backend = CandleImageProcessor::new()?;
        
        if let Some(config) = config {
            backend.initialize(config)?;
        }
        
        Ok(Box::new(backend))
    }
    
    /// Create Candle embedding provider
    fn create_candle_embedding_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let mut backend = CandleImageProcessor::new()?;
        
        if let Some(config) = config {
            backend.initialize(config)?;
        }
        
        Ok(Box::new(backend))
    }
    
    /// Create Candle generation provider
    #[cfg(feature = "generation")]
    fn create_candle_generation_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let mut backend = crate::image_processing::generation::CandleImageGenerator::new()?;
        
        if let Some(config) = config {
            backend.initialize(config)?;
        }
        
        Ok(Box::new(backend))
    }
    
    /// Create automatically selected backend
    fn create_auto_backend(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageProcessingBackend>> {
        let backend_type = Self::recommended_backend();
        Self::create_backend(backend_type, config)
    }
    
    /// Create automatically selected embedding provider
    fn create_auto_embedding_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageEmbeddingProvider>> {
        let backend_type = Self::recommended_backend();
        Self::create_embedding_provider(backend_type, config)
    }
    
    /// Create automatically selected generation provider
    #[cfg(feature = "generation")]
    fn create_auto_generation_provider(
        config: Option<&HashMap<String, serde_json::Value>>,
    ) -> ImageProcessingResult<Box<dyn ImageGenerationProvider>> {
        let backend_type = Self::recommended_backend();
        Self::create_generation_provider(backend_type, config)
    }
    
    /// Check if Candle backend is available
    fn is_candle_available() -> bool {
        // Always available since it's our default backend
        true
    }
    
    /// Detect available device types
    pub fn detect_available_devices() -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];
        
        #[cfg(feature = "cuda")]
        {
            if Self::is_cuda_available() {
                devices.push(DeviceType::Cuda);
            }
        }
        
        #[cfg(feature = "metal")]
        {
            if Self::is_metal_available() {
                devices.push(DeviceType::Metal);
            }
        }
        
        devices
    }
    
    /// Get optimal device configuration for current system
    pub fn optimal_device_config() -> DeviceConfig {
        let available_devices = Self::detect_available_devices();
        
        // Prioritize GPU acceleration if available
        let device_type = if available_devices.contains(&DeviceType::Metal) {
            DeviceType::Metal
        } else if available_devices.contains(&DeviceType::Cuda) {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        };
        
        DeviceConfig {
            device_type,
            device_index: None,
            memory_limit_mb: None,
            mixed_precision: matches!(device_type, DeviceType::Cuda | DeviceType::Metal),
        }
    }
    
    /// Check CUDA availability
    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> bool {
        // Simple check - in production this could be more sophisticated
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || 
        std::path::Path::new("/usr/local/cuda").exists()
    }
    
    /// Check Metal availability
    #[cfg(feature = "metal")]
    fn is_metal_available() -> bool {
        // Metal is available on macOS
        cfg!(target_os = "macos")
    }
    
    /// Create provider registry with available backends
    pub fn create_provider_registry() -> ProviderRegistry {
        ProviderRegistry::new()
    }
}

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Candle ML framework backend
    Candle,
    /// Automatically select best available backend
    Auto,
}

/// Provider registry for managing available backends
pub struct ProviderRegistry {
    embedding_providers: HashMap<String, Box<dyn ImageEmbeddingProvider>>,
    #[cfg(feature = "generation")]
    generation_providers: HashMap<String, Box<dyn ImageGenerationProvider>>,
}

impl ProviderRegistry {
    /// Create new provider registry
    pub fn new() -> Self {
        Self {
            embedding_providers: HashMap::new(),
            #[cfg(feature = "generation")]
            generation_providers: HashMap::new(),
        }
    }
    
    /// Register an embedding provider
    pub fn register_embedding_provider(
        &mut self,
        name: String,
        provider: Box<dyn ImageEmbeddingProvider>,
    ) {
        self.embedding_providers.insert(name, provider);
    }
    
    /// Register a generation provider
    #[cfg(feature = "generation")]
    pub fn register_generation_provider(
        &mut self,
        name: String,
        provider: Box<dyn ImageGenerationProvider>,
    ) {
        self.generation_providers.insert(name, provider);
    }
    
    /// Get embedding provider by name
    pub fn get_embedding_provider(&self, name: &str) -> Option<&dyn ImageEmbeddingProvider> {
        self.embedding_providers.get(name).map(|p| p.as_ref())
    }
    
    /// Get generation provider by name
    #[cfg(feature = "generation")]
    pub fn get_generation_provider(&self, name: &str) -> Option<&dyn ImageGenerationProvider> {
        self.generation_providers.get(name).map(|p| p.as_ref())
    }
    
    /// List available embedding providers
    pub fn list_embedding_providers(&self) -> Vec<&String> {
        self.embedding_providers.keys().collect()
    }
    
    /// List available generation providers
    #[cfg(feature = "generation")]
    pub fn list_generation_providers(&self) -> Vec<&String> {
        self.generation_providers.keys().collect()
    }
    
    /// Initialize default providers
    pub fn initialize_default_providers(&mut self) -> ImageProcessingResult<()> {
        // Register Candle embedding provider
        if let Ok(provider) = ImageProcessingFactory::create_embedding_provider(
            BackendType::Candle,
            None,
        ) {
            self.register_embedding_provider("candle".to_string(), provider);
        }
        
        // Register Candle generation provider
        #[cfg(feature = "generation")]
        {
            if let Ok(provider) = ImageProcessingFactory::create_generation_provider(
                BackendType::Candle,
                None,
            ) {
                self.register_generation_provider("candle".to_string(), provider);
            }
        }
        
        Ok(())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder for image processing backends
pub struct BackendConfigBuilder {
    config: HashMap<String, serde_json::Value>,
}

impl BackendConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
        }
    }
    
    /// Set device configuration
    pub fn device_config(mut self, device_config: DeviceConfig) -> Self {
        self.config.insert(
            "device_config".to_string(),
            serde_json::to_value(device_config).unwrap_or(serde_json::Value::Null),
        );
        self
    }
    
    /// Set model name
    pub fn model_name(mut self, model_name: String) -> Self {
        self.config.insert("model_name".to_string(), serde_json::Value::String(model_name));
        self
    }
    
    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.insert(
            "batch_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(batch_size)),
        );
        self
    }
    
    /// Set memory limit
    pub fn memory_limit_mb(mut self, memory_mb: u64) -> Self {
        self.config.insert(
            "memory_limit_mb".to_string(),
            serde_json::Value::Number(serde_json::Number::from(memory_mb)),
        );
        self
    }
    
    /// Add custom parameter
    pub fn custom_param(mut self, key: String, value: serde_json::Value) -> Self {
        self.config.insert(key, value);
        self
    }
    
    /// Build configuration
    pub fn build(self) -> HashMap<String, serde_json::Value> {
        self.config
    }
}

impl Default for BackendConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for factory operations
pub mod factory_utils {
    use super::*;
    
    /// Create optimal backend configuration for current system
    pub fn create_optimal_config() -> HashMap<String, serde_json::Value> {
        BackendConfigBuilder::new()
            .device_config(ImageProcessingFactory::optimal_device_config())
            .batch_size(32)
            .build()
    }
    
    /// Create configuration for specific device type
    pub fn create_device_config(device_type: DeviceType) -> HashMap<String, serde_json::Value> {
        let device_config = DeviceConfig {
            device_type,
            device_index: None,
            memory_limit_mb: None,
            mixed_precision: matches!(device_type, DeviceType::Cuda | DeviceType::Metal),
        };
        
        BackendConfigBuilder::new()
            .device_config(device_config)
            .build()
    }
    
    /// Create low-memory configuration
    pub fn create_low_memory_config() -> HashMap<String, serde_json::Value> {
        BackendConfigBuilder::new()
            .device_config(DeviceConfig {
                device_type: DeviceType::Cpu,
                device_index: None,
                memory_limit_mb: Some(1024), // 1GB limit
                mixed_precision: false,
            })
            .batch_size(8)
            .build()
    }
    
    /// Create high-performance configuration
    pub fn create_high_performance_config() -> HashMap<String, serde_json::Value> {
        let device_config = ImageProcessingFactory::optimal_device_config();
        
        BackendConfigBuilder::new()
            .device_config(device_config)
            .batch_size(64)
            .build()
    }
}