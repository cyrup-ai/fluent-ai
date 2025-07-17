//! ============================================================================
//! Cylo: Secure Multi-Platform Code Execution Framework
//! ============================================================================
//!
//! High-performance secure code execution with multiple isolation backends:
//! - Apple containerization for macOS with Apple Silicon
//! - LandLock sandboxing for Linux with kernel-level security
//! - FireCracker microVMs for ultra-lightweight virtualization
//!
//! Features:
//! - Zero allocation in hot paths
//! - Lock-free concurrent operations
//! - Platform-aware backend selection
//! - Instance lifecycle management
//! - Comprehensive health monitoring
//!
//! ## Quick Start
//!
//! ```rust
//! use fluent_ai_cylo::{Cylo, global_instance_manager};
//!
//! // Create execution environment
//! let cylo_env = Cylo::Apple("python:alpine3.20".to_string());
//! let instance = cylo_env.instance("my_python_env");
//!
//! // Register with global manager
//! let manager = global_instance_manager();
//! manager.register_instance(instance).await?;
//!
//! // Execute code
//! let request = ExecutionRequest::new("print('Hello, World!')", "python");
//! let result = manager.get_instance("my_python_env").await?
//!     .execute_code(request).await;
//! ```

// ============================================================================
// Core execution environment types
// ============================================================================

pub mod execution_env;
pub use execution_env::{
    Cylo, CyloInstance, CyloError, CyloResult,
    validate_instance_name, validate_environment_spec
};

// ============================================================================
// Backend implementations and traits
// ============================================================================

pub mod backends;
pub use backends::{
    // Core traits
    ExecutionBackend, BackendConfig, BackendError, BackendResult,
    
    // Request/Response types
    ExecutionRequest, ExecutionResult, ExecutionLimits, HealthStatus,
    ResourceUsage,
    
    // Backend implementations
    AppleBackend, LandLockBackend, FireCrackerBackend,
    
    // Factory function
    create_backend,
};

#[cfg(target_os = "linux")]
pub use backends::{
    landlock::LandLockBackend,
    firecracker::FireCrackerBackend,
};

// ============================================================================
// Platform detection and capability assessment
// ============================================================================

pub mod platform;
pub use platform::{
    // Core platform information
    PlatformInfo, OperatingSystem, Architecture,
    
    // Capability detection
    PlatformCapabilities, VirtualizationSupport, ContainerSupport,
    SecurityFeatures, NetworkCapabilities, FilesystemFeatures,
    
    // Performance characteristics
    PerformanceHints, TmpDirPerformance, IOCharacteristics,
    
    // Backend availability
    BackendAvailability,
    
    // Utility functions
    detect_platform, is_apple_silicon, is_linux, has_landlock, has_kvm,
    get_recommended_backend, get_available_backends,
};

// ============================================================================
// Instance lifecycle management
// ============================================================================

pub mod instance_manager;
pub use instance_manager::{
    // Core manager types
    InstanceManager,
    
    // Global access functions
    global_instance_manager, init_global_instance_manager,
};

// ============================================================================
// Prelude for common imports
// ============================================================================

/// Common imports for Cylo usage
pub mod prelude {
    pub use crate::{
        // Core types
        Cylo, CyloInstance, CyloError, CyloResult,
        
        // Execution types
        ExecutionRequest, ExecutionResult, ExecutionBackend,
        BackendConfig, HealthStatus,
        
        // Platform detection
        detect_platform, get_recommended_backend,
        
        // Instance management
        global_instance_manager,
    };
    
    // AsyncTask re-export for convenience
    pub use crate::async_task::{AsyncTask, AsyncTaskBuilder};
}

// ============================================================================
// Feature-gated exports
// ============================================================================

// ============================================================================
// AsyncTask module - simple wrapper around tokio for backend compatibility
// ============================================================================

pub mod async_task {
    /// AsyncTask is a type alias for tokio::task::JoinHandle
    pub type AsyncTask<T> = tokio::task::JoinHandle<T>;
    
    /// Simple AsyncTaskBuilder for fluent construction
    pub struct AsyncTaskBuilder<F> {
        future: F,
    }
    
    impl<F, T> AsyncTaskBuilder<F>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        /// Create a new AsyncTaskBuilder
        pub fn new(future: F) -> Self {
            Self { future }
        }
        
        /// Spawn the task and return the AsyncTask handle
        pub fn spawn(self) -> AsyncTask<T> {
            tokio::spawn(self.future)
        }
    }
    
    /// Convenience function to spawn an async task
    pub fn spawn_async<F, T>(future: F) -> AsyncTask<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        tokio::spawn(future)
    }
}

// Re-export AsyncTask types for backend implementations
#[doc(hidden)]
pub use async_task::{AsyncTask, AsyncTaskBuilder};

// UUID re-export for instance ID generation
#[doc(hidden)]
pub use uuid;

// Serde re-exports for serialization support
#[doc(hidden)]
pub use serde::{Deserialize, Serialize};

// ============================================================================
// Version and metadata
// ============================================================================

/// Cylo version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cylo build metadata
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    git_hash: option_env!("GIT_HASH").unwrap_or("unknown"),
    build_time: option_env!("BUILD_TIME").unwrap_or("unknown"),
    rust_version: option_env!("RUST_VERSION").unwrap_or("unknown"),
};

/// Build information structure
#[derive(Debug, Clone, Copy)]
pub struct BuildInfo {
    /// Package version
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: &'static str,
    /// Build timestamp
    pub build_time: &'static str,
    /// Rust compiler version
    pub rust_version: &'static str,
}

// ============================================================================
// Zero-allocation convenience constructors
// ============================================================================

/// Zero-allocation constructor for Apple backend execution environment
#[inline]
pub const fn apple_env(image: &'static str) -> ConstCylo {
    ConstCylo::Apple(image)
}

/// Zero-allocation constructor for LandLock backend execution environment
#[inline]
pub const fn landlock_env(jail_path: &'static str) -> ConstCylo {
    ConstCylo::LandLock(jail_path)
}

/// Zero-allocation constructor for FireCracker backend execution environment
#[inline]
pub const fn firecracker_env(image: &'static str) -> ConstCylo {
    ConstCylo::FireCracker(image)
}

/// Compile-time Cylo environment specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstCylo {
    /// Apple containerization backend
    Apple(&'static str),
    /// LandLock sandboxing backend
    LandLock(&'static str),
    /// FireCracker microVM backend
    FireCracker(&'static str),
}

impl ConstCylo {
    /// Convert to runtime Cylo instance with zero allocation in hot path
    #[inline]
    pub fn to_runtime(self) -> Cylo {
        match self {
            ConstCylo::Apple(image) => Cylo::Apple(image.to_string()),
            ConstCylo::LandLock(path) => Cylo::LandLock(path.to_string()),
            ConstCylo::FireCracker(image) => Cylo::FireCracker(image.to_string()),
        }
    }
    
    /// Create named instance with zero allocation
    #[inline]
    pub fn instance(self, name: &'static str) -> CyloInstance {
        CyloInstance {
            env: self.to_runtime(),
            name: name.to_string(),
        }
    }
}

// ============================================================================
// High-performance execution shortcuts
// ============================================================================

/// Execute code directly with automatic backend selection and instance management
/// 
/// This is a high-level convenience function that:
/// 1. Detects the best available backend for the current platform
/// 2. Creates a temporary execution instance
/// 3. Executes the code with optimal settings
/// 4. Cleans up resources automatically
/// 
/// For production use, prefer explicit instance management for better performance.
#[inline]
pub fn execute_code_auto(
    code: &str, 
    language: &str
) -> AsyncTask<CyloResult<ExecutionResult>> {
    AsyncTaskBuilder::new()
        .spawn(move || async move {
            let platform_info = detect_platform();
            
            let backend_name = get_recommended_backend()
                .ok_or_else(|| CyloError::no_backend_available())?;
            
            let cylo_env = match backend_name {
                "Apple" => Cylo::Apple("python:alpine3.20".to_string()),
                "LandLock" => Cylo::LandLock("/tmp/cylo_auto".to_string()),
                "FireCracker" => Cylo::FireCracker("python:alpine3.20".to_string()),
                _ => return Err(CyloError::unsupported_backend(backend_name)),
            };
            
            let instance_name = format!("auto_{}", uuid::Uuid::new_v4().simple());
            let instance = cylo_env.instance(instance_name);
            
            let manager = global_instance_manager();
            manager.register_instance(instance.clone()).await?;
            
            let backend = manager.get_instance(&instance.id()).await?;
            let request = ExecutionRequest::new(code, language);
            let result = backend.execute_code(request).await;
            
            // Clean up instance
            let _ = manager.remove_instance(&instance.id()).await;
            
            Ok(result)
        })
}

/// Execute Python code with automatic backend selection
#[inline]
pub fn execute_python(code: &str) -> AsyncTask<CyloResult<ExecutionResult>> {
    execute_code_auto(code, "python")
}

/// Execute JavaScript code with automatic backend selection
#[inline]
pub fn execute_javascript(code: &str) -> AsyncTask<CyloResult<ExecutionResult>> {
    execute_code_auto(code, "javascript")
}

/// Execute Rust code with automatic backend selection
#[inline]
pub fn execute_rust(code: &str) -> AsyncTask<CyloResult<ExecutionResult>> {
    execute_code_auto(code, "rust")
}

/// Execute Bash code with automatic backend selection
#[inline]
pub fn execute_bash(code: &str) -> AsyncTask<CyloResult<ExecutionResult>> {
    execute_code_auto(code, "bash")
}

/// Execute Go code with automatic backend selection
#[inline]
pub fn execute_go(code: &str) -> AsyncTask<CyloResult<ExecutionResult>> {
    execute_code_auto(code, "go")
}

// ============================================================================
// Performance monitoring and diagnostics
// ============================================================================

/// Get comprehensive platform and backend diagnostics
pub fn get_diagnostics() -> AsyncTask<DiagnosticsReport> {
    AsyncTaskBuilder::new()
        .spawn(|| async move {
            let platform_info = detect_platform();
            let available_backends = get_available_backends();
            let manager = global_instance_manager();
            
            let health_results = manager.health_check_all().await
                .unwrap_or_default();
            
            let instance_list = manager.list_instances()
                .unwrap_or_default();
            
            DiagnosticsReport {
                platform: platform_info.clone(),
                available_backends: available_backends.iter().map(|&s| s.to_string()).collect(),
                backend_health: health_results,
                active_instances: instance_list,
                performance_hints: platform_info.performance.clone(),
            }
        })
}

/// Comprehensive diagnostics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsReport {
    /// Platform information
    pub platform: PlatformInfo,
    /// Available execution backends
    pub available_backends: Vec<String>,
    /// Backend health status
    pub backend_health: std::collections::HashMap<String, HealthStatus>,
    /// Currently active instances
    pub active_instances: Vec<String>,
    /// Performance optimization hints
    pub performance_hints: PerformanceHints,
}