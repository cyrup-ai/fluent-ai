//! High-performance device management with atomic selection and intelligent fallback
//!
//! This module provides comprehensive device detection, selection, and management for
//! Candle ML inference with zero-allocation patterns and lock-free atomic operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arc_swap::{ArcSwap, Guard};
use candle_core::Device;
use crossbeam::utils::CachePadded;
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult, ErrorMetrics, record_global_error};
use super::models::CandleDevice;

/// Device type enumeration for platform-specific optimizations
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu = 0,
    CudaGpu = 1,
    MetalGpu = 2,
    RocmGpu = 3,
    VulkanGpu = 4,
}

impl From<CandleDevice> for DeviceType {
    #[inline(always)]
    fn from(device: CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => DeviceType::Cpu,
            CandleDevice::Cuda(_) => DeviceType::CudaGpu,
            CandleDevice::Metal => DeviceType::MetalGpu,
        }
    }
}

/// Comprehensive device information with performance metrics
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type and identifier
    pub device: CandleDevice,
    /// Device type for fast categorization
    pub device_type: DeviceType,
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Compute capability or performance score
    pub compute_capability: f32,
    /// Whether the device is currently available
    pub is_available: bool,
    /// Device name/description (stack-allocated)
    pub name: arrayvec::ArrayString<64>,
    /// Last successful initialization timestamp
    pub last_init_timestamp: u64,
    /// Number of successful operations
    pub success_count: u64,
    /// Number of failed operations
    pub failure_count: u64,
    /// Average operation time in microseconds
    pub avg_operation_time_us: u32,
}

impl DeviceInfo {
    /// Create new device info with validation
    #[inline]
    pub fn new(
        device: CandleDevice,
        name: &str,
        total_memory_bytes: u64,
        compute_capability: f32,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            device,
            device_type: DeviceType::from(device),
            total_memory_bytes,
            available_memory_bytes: total_memory_bytes,
            compute_capability,
            is_available: false, // Will be determined during initialization
            name: arrayvec::ArrayString::from(name).unwrap_or_default(),
            last_init_timestamp: now,
            success_count: 0,
            failure_count: 0,
            avg_operation_time_us: 0,
        }
    }

    /// Update device statistics after operation
    #[inline]
    pub fn record_operation(&mut self, success: bool, duration: Duration) {
        let duration_us = duration.as_micros() as u32;

        if success {
            self.success_count += 1;
            // Update running average (simplified)
            let total_ops = self.success_count;
            self.avg_operation_time_us = ((self.avg_operation_time_us as u64 * (total_ops - 1))
                + duration_us as u64) as u32
                / total_ops as u32;
        } else {
            self.failure_count += 1;
        }
    }

    /// Get success ratio as percentage
    #[inline]
    pub fn success_ratio(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            100.0
        } else {
            (self.success_count as f32 / total as f32) * 100.0
        }
    }

    /// Calculate device performance score
    #[inline]
    pub fn performance_score(&self) -> f32 {
        let base_score = self.compute_capability;
        let memory_factor =
            (self.available_memory_bytes as f32) / (self.total_memory_bytes as f32).max(1.0);
        let success_factor = self.success_ratio() / 100.0;
        let speed_factor = if self.avg_operation_time_us > 0 {
            1000000.0 / (self.avg_operation_time_us as f32) // Inverse of time for speed
        } else {
            1.0
        };

        base_score * memory_factor * success_factor * speed_factor.min(10.0)
    }
}

/// Device manager state for atomic operations
#[derive(Debug)]
struct DeviceManagerState {
    /// Available devices (discovered during initialization)
    available_devices: SmallVec<DeviceInfo, 8>,
    /// Currently selected device index
    current_device_index: usize,
    /// Whether manager is initialized
    is_initialized: bool,
    /// Last device scan timestamp
    last_scan_timestamp: u64,
    /// Device preference order
    preference_order: SmallVec<DeviceType, 4>,
}

impl Default for DeviceManagerState {
    fn default() -> Self {
        Self {
            available_devices: SmallVec::new(),
            current_device_index: 0,
            is_initialized: false,
            last_scan_timestamp: 0,
            preference_order: {
                let mut order = SmallVec::new();
                // Prefer Metal on macOS, CUDA elsewhere, then CPU
                #[cfg(target_os = "macos")]
                {
                    order.push(DeviceType::MetalGpu);
                    order.push(DeviceType::CudaGpu);
                }
                #[cfg(not(target_os = "macos"))]
                {
                    order.push(DeviceType::CudaGpu);
                    order.push(DeviceType::MetalGpu);
                }
                order.push(DeviceType::Cpu);
                order
            },
        }
    }
}

/// Lock-free device manager with intelligent selection and fallback
pub struct DeviceManager {
    /// Hot-swappable device manager state
    state: ArcSwap<DeviceManagerState>,
    /// Atomic flags for coordination
    initialization_in_progress: AtomicBool,
    /// Device scan interval for automatic updates
    scan_interval_ms: AtomicU64,
    /// Performance metrics
    metrics: DeviceMetrics,
}

/// Device manager performance metrics
#[derive(Debug)]
struct DeviceMetrics {
    /// Device initialization attempts
    init_attempts: CachePadded<AtomicU64>,
    /// Successful device initializations
    init_successes: CachePadded<AtomicU64>,
    /// Device fallback operations
    fallback_count: CachePadded<AtomicU64>,
    /// Device switch operations
    switch_count: CachePadded<AtomicU64>,
    /// Average device scan time in microseconds
    avg_scan_time_us: CachePadded<AtomicU32>,
    /// Last error timestamp
    last_error_timestamp: CachePadded<AtomicU64>,
}

impl DeviceMetrics {
    fn new() -> Self {
        Self {
            init_attempts: CachePadded::new(AtomicU64::new(0)),
            init_successes: CachePadded::new(AtomicU64::new(0)),
            fallback_count: CachePadded::new(AtomicU64::new(0)),
            switch_count: CachePadded::new(AtomicU64::new(0)),
            avg_scan_time_us: CachePadded::new(AtomicU32::new(0)),
            last_error_timestamp: CachePadded::new(AtomicU64::new(0)),
        }
    }
}

impl DeviceManager {
    /// Create new device manager with default configuration
    pub fn new() -> CandleResult<Self> {
        Ok(Self {
            state: ArcSwap::from_pointee(DeviceManagerState::default()),
            initialization_in_progress: AtomicBool::new(false),
            scan_interval_ms: AtomicU64::new(30000), // 30 second default scan interval
            metrics: DeviceMetrics::new(),
        })
    }

    /// Initialize device manager and discover available devices
    pub async fn initialize(&self) -> CandleResult<()> {
        // Prevent concurrent initialization
        if self
            .initialization_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Wait for ongoing initialization
            while self.initialization_in_progress.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            return Ok(());
        }

        let result = self.perform_initialization().await;
        self.initialization_in_progress
            .store(false, Ordering::Release);
        result
    }

    /// Perform actual device discovery and initialization
    async fn perform_initialization(&self) -> CandleResult<()> {
        self.metrics.init_attempts.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();

        let mut discovered_devices = SmallVec::<[DeviceInfo; 8]>::new();
        let mut initialization_errors = SmallVec::<[CandleError; 4]>::new();

        // Discover CPU device (always available)
        let cpu_info = self.discover_cpu_device().await?;
        discovered_devices.push(cpu_info);

        // Discover CUDA devices
        match self.discover_cuda_devices().await {
            Ok(mut cuda_devices) => discovered_devices.append(&mut cuda_devices),
            Err(e) => initialization_errors.push(e),
        }

        // Discover Metal devices (macOS only)
        #[cfg(target_os = "macos")]
        {
            match self.discover_metal_devices().await {
                Ok(mut metal_devices) => discovered_devices.append(&mut metal_devices),
                Err(e) => initialization_errors.push(e),
            }
        }

        // Select best available device
        let best_device_index = self.select_best_device(&discovered_devices)?;

        // Update state atomically
        let scan_time = start_time.elapsed();
        let scan_time_us = scan_time.as_micros() as u32;
        self.metrics
            .avg_scan_time_us
            .store(scan_time_us, Ordering::Relaxed);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let new_state = DeviceManagerState {
            available_devices: discovered_devices,
            current_device_index: best_device_index,
            is_initialized: true,
            last_scan_timestamp: now,
            preference_order: self.state.load().preference_order.clone(),
        };

        self.state.store(Arc::new(new_state));
        self.metrics.init_successes.fetch_add(1, Ordering::Relaxed);

        // Log any non-critical initialization errors
        for error in initialization_errors {
            record_global_error(&error);
        }

        Ok(())
    }

    /// Discover CPU device capabilities
    async fn discover_cpu_device(&self) -> CandleResult<DeviceInfo> {
        let start_time = Instant::now();

        // Test CPU device creation
        let _cpu_device = Device::Cpu;

        // Get system memory information
        let total_memory = self.get_system_memory_bytes();
        let compute_capability = self.benchmark_cpu_performance().await;

        let mut cpu_info =
            DeviceInfo::new(CandleDevice::Cpu, "CPU", total_memory, compute_capability);

        cpu_info.is_available = true;
        cpu_info.record_operation(true, start_time.elapsed());

        Ok(cpu_info)
    }

    /// Discover CUDA devices
    async fn discover_cuda_devices(&self) -> CandleResult<SmallVec<[DeviceInfo; 4]>> {
        let mut cuda_devices = SmallVec::new();

        // Try to detect CUDA devices
        for device_id in 0..8 {
            // Check up to 8 CUDA devices
            match self.test_cuda_device(device_id).await {
                Ok(device_info) => cuda_devices.push(device_info),
                Err(_) => break, // Stop on first unavailable device
            }
        }

        if cuda_devices.is_empty() {
            return Err(CandleError::device(
                CandleDevice::Cuda(0),
                "CUDA support",
                &[CandleDevice::Cpu],
            ));
        }

        Ok(cuda_devices)
    }

    /// Test individual CUDA device
    async fn test_cuda_device(&self, device_id: u32) -> CandleResult<DeviceInfo> {
        let start_time = Instant::now();

        // Try to create CUDA device
        let cuda_device = Device::new_cuda(device_id as usize).map_err(|e| {
            CandleError::device(
                CandleDevice::Cuda(device_id),
                "CUDA device creation",
                &[CandleDevice::Cpu],
            )
        })?;

        // Estimate CUDA device capabilities
        let total_memory = self.get_cuda_memory_bytes(device_id);
        let compute_capability = self.benchmark_cuda_performance(device_id).await;

        let mut cuda_info = DeviceInfo::new(
            CandleDevice::Cuda(device_id),
            &format!("CUDA:{}", device_id),
            total_memory,
            compute_capability,
        );

        cuda_info.is_available = true;
        cuda_info.record_operation(true, start_time.elapsed());

        Ok(cuda_info)
    }

    /// Discover Metal devices (macOS only)
    #[cfg(target_os = "macos")]
    async fn discover_metal_devices(&self) -> CandleResult<SmallVec<[DeviceInfo; 2]>> {
        let mut metal_devices = SmallVec::new();

        // Try to detect Metal devices
        for device_id in 0..2 {
            // Check up to 2 Metal devices (integrated + discrete)
            match self.test_metal_device(device_id).await {
                Ok(device_info) => metal_devices.push(device_info),
                Err(_) => continue, // Try next device
            }
        }

        if metal_devices.is_empty() {
            return Err(CandleError::device(
                CandleDevice::Metal(0),
                "Metal support",
                &[CandleDevice::Cpu],
            ));
        }

        Ok(metal_devices)
    }

    /// Test individual Metal device (macOS only)
    #[cfg(target_os = "macos")]
    async fn test_metal_device(&self, device_id: u32) -> CandleResult<DeviceInfo> {
        let start_time = Instant::now();

        // Try to create Metal device
        let _metal_device = Device::new_metal(device_id as usize).map_err(|e| {
            CandleError::device(
                CandleDevice::Metal(device_id),
                "Metal device creation",
                &[CandleDevice::Cpu],
            )
        })?;

        // Estimate Metal device capabilities
        let total_memory = self.get_metal_memory_bytes(device_id);
        let compute_capability = self.benchmark_metal_performance(device_id).await;

        let mut metal_info = DeviceInfo::new(
            CandleDevice::Metal(device_id),
            &format!("Metal:{}", device_id),
            total_memory,
            compute_capability,
        );

        metal_info.is_available = true;
        metal_info.record_operation(true, start_time.elapsed());

        Ok(metal_info)
    }

    /// Select best available device based on performance and preference
    fn select_best_device(&self, devices: &[DeviceInfo]) -> CandleResult<usize> {
        if devices.is_empty() {
            return Err(CandleError::device(
                CandleDevice::Cpu,
                "any available device",
                &[],
            ));
        }

        let state = self.state.load();
        let mut best_index = 0;
        let mut best_score = 0.0;

        // Score devices based on preference order and performance
        for (index, device) in devices.iter().enumerate() {
            if !device.is_available {
                continue;
            }

            let preference_bonus = match state
                .preference_order
                .iter()
                .position(|&pref| pref == device.device_type)
            {
                Some(pos) => 10.0 / (pos as f32 + 1.0), // Higher bonus for preferred devices
                None => 1.0,
            };

            let score = device.performance_score() * preference_bonus;

            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }

        Ok(best_index)
    }

    /// Get current device
    pub fn current_device(&self) -> CandleResult<CandleDevice> {
        let state = self.state.load();

        if !state.is_initialized {
            return Err(CandleError::device(
                CandleDevice::Cpu,
                "initialized device manager",
                &[],
            ));
        }

        state
            .available_devices
            .get(state.current_device_index)
            .map(|info| info.device)
            .ok_or_else(|| CandleError::device(CandleDevice::Cpu, "valid device index", &[]))
    }

    /// Get fallback device using preference order
    pub fn fallback_device(&self) -> CandleResult<CandleDevice> {
        let state = self.state.load();

        if !state.is_initialized {
            return Err(CandleError::device(
                CandleDevice::Cpu,
                "initialized device manager",
                &[],
            ));
        }

        // Try devices in preference order
        for &preferred_type in &state.preference_order {
            if let Some(device_info) = state
                .available_devices
                .iter()
                .find(|info| info.device_type == preferred_type && info.is_available)
            {
                self.metrics.fallback_count.fetch_add(1, Ordering::Relaxed);
                return Ok(device_info.device);
            }
        }

        // Fallback to first available device
        state
            .available_devices
            .iter()
            .find(|info| info.is_available)
            .map(|info| info.device)
            .ok_or_else(|| CandleError::device(CandleDevice::Cpu, "any available device", &[]))
    }

    /// Switch to specific device
    pub async fn switch_device(&self, target_device: CandleDevice) -> CandleResult<()> {
        let state = self.state.load();

        if !state.is_initialized {
            return Err(CandleError::device(
                target_device,
                "initialized device manager",
                &[],
            ));
        }

        // Find target device in available devices
        let target_index = state
            .available_devices
            .iter()
            .position(|info| info.device == target_device && info.is_available)
            .ok_or_else(|| {
                CandleError::device(
                    target_device,
                    "available device",
                    &state
                        .available_devices
                        .iter()
                        .map(|info| info.device)
                        .collect::<Vec<_>>(),
                )
            })?;

        // Update state if different from current
        if target_index != state.current_device_index {
            let mut new_state = (**state).clone();
            new_state.current_device_index = target_index;
            self.state.store(Arc::new(new_state));
            self.metrics.switch_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get all available devices
    pub fn available_devices(&self) -> SmallVec<[DeviceInfo; 8]> {
        let state = self.state.load();
        state.available_devices.clone()
    }

    /// Check if device manager is initialized
    #[inline(always)]
    pub fn is_initialized(&self) -> bool {
        self.state.load().is_initialized
    }

    /// Get device manager statistics
    pub fn statistics(&self) -> DeviceManagerStatistics {
        DeviceManagerStatistics {
            init_attempts: self.metrics.init_attempts.load(Ordering::Relaxed),
            init_successes: self.metrics.init_successes.load(Ordering::Relaxed),
            fallback_count: self.metrics.fallback_count.load(Ordering::Relaxed),
            switch_count: self.metrics.switch_count.load(Ordering::Relaxed),
            avg_scan_time_us: self.metrics.avg_scan_time_us.load(Ordering::Relaxed),
            last_error_timestamp: self.metrics.last_error_timestamp.load(Ordering::Relaxed),
        }
    }

    // Helper methods for device capability detection

    /// Get system memory in bytes
    fn get_system_memory_bytes(&self) -> u64 {
        // Simplified system memory detection
        // In a real implementation, this would use platform-specific APIs
        8 * 1024 * 1024 * 1024 // 8GB default
    }

    /// Get CUDA device memory in bytes
    fn get_cuda_memory_bytes(&self, _device_id: u32) -> u64 {
        // Simplified CUDA memory detection
        // In a real implementation, this would query CUDA device properties
        8 * 1024 * 1024 * 1024 // 8GB default
    }

    /// Get Metal device memory in bytes (macOS only)
    #[cfg(target_os = "macos")]
    fn get_metal_memory_bytes(&self, _device_id: u32) -> u64 {
        // Simplified Metal memory detection
        // In a real implementation, this would query Metal device properties
        8 * 1024 * 1024 * 1024 // 8GB default
    }

    /// Benchmark CPU performance
    async fn benchmark_cpu_performance(&self) -> f32 {
        // Simplified CPU benchmarking
        // In a real implementation, this would run matrix operations
        1.0 // Base score for CPU
    }

    /// Benchmark CUDA performance
    async fn benchmark_cuda_performance(&self, _device_id: u32) -> f32 {
        // Simplified CUDA benchmarking
        // In a real implementation, this would run GPU kernels
        5.0 // Higher score for CUDA
    }

    /// Benchmark Metal performance (macOS only)
    #[cfg(target_os = "macos")]
    async fn benchmark_metal_performance(&self, _device_id: u32) -> f32 {
        // Simplified Metal benchmarking
        // In a real implementation, this would run Metal compute shaders
        4.0 // High score for Metal
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| panic!("Failed to create default DeviceManager"))
    }
}

/// Device manager statistics for monitoring
#[derive(Debug, Clone)]
pub struct DeviceManagerStatistics {
    pub init_attempts: u64,
    pub init_successes: u64,
    pub fallback_count: u64,
    pub switch_count: u64,
    pub avg_scan_time_us: u32,
    pub last_error_timestamp: u64,
}

impl DeviceManagerStatistics {
    /// Get initialization success rate
    #[inline]
    pub fn success_rate(&self) -> f32 {
        if self.init_attempts == 0 {
            100.0
        } else {
            (self.init_successes as f32 / self.init_attempts as f32) * 100.0
        }
    }
}
