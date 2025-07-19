//! ============================================================================
//! Secure tool executor using Cylo for multi-platform sandboxed execution
//! ============================================================================
//!
//! High-performance secure execution of tools using Cylo's comprehensive
//! isolation backends including Apple containerization, LandLock sandboxing,
//! and FireCracker microVMs for maximum security and performance.

use crate::{AsyncTask, HashMap};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::time::Duration;

// Import Cylo types for secure execution
use cylo::{
    ExecutionRequest, ExecutionResult, Cylo, CyloInstance, CyloResult,
    ExecutionBackend, BackendConfig, HealthStatus, ResourceLimits, create_backend,
    detect_platform, get_recommended_backend
};

/// Security level for execution isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Basic isolation (fastest, least secure)
    Basic,
    /// Standard isolation (balanced security and performance)
    Standard,
    /// High isolation (strong security with good performance)
    High,
    /// Maximum isolation (strongest security, may impact performance)
    Maximum,
}

/// Secure execution configuration with comprehensive options
#[derive(Debug, Clone)]
pub struct SecureExecutionConfig {
    /// Security isolation level
    pub security_level: SecurityLevel,
    /// Preferred execution backend (None for auto-selection)
    pub preferred_backend: Option<String>,
    /// Maximum execution time
    pub timeout: Duration,
    /// Memory limit in bytes
    pub memory_limit: Option<u64>,
    /// CPU core limit
    pub cpu_limit: Option<u32>,
    /// Enable instance reuse for performance
    pub instance_reuse: bool,
    /// Custom environment variables
    pub env_vars: HashMap<String, String>,
    /// Working directory within execution environment
    pub working_dir: Option<String>,
}

impl Default for SecureExecutionConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Standard,
            preferred_backend: None,
            timeout: Duration::from_secs(30),
            memory_limit: Some(512 * 1024 * 1024), // 512MB
            cpu_limit: Some(1),
            instance_reuse: true,
            env_vars: HashMap::new(),
            working_dir: None,
        }
    }
}

/// High-performance secure tool executor using Cylo backends
pub struct SecureToolExecutor {
    /// Execution configuration
    config: SecureExecutionConfig,
    /// Cylo backend for secure execution
    backend: Box<dyn ExecutionBackend>,
    /// Cached instance for reuse
    cached_instance: Option<CyloInstance>,
}

impl SecureToolExecutor {
    /// Create a new secure executor with default configuration
    /// 
    /// Automatically selects the best available backend for the current platform
    /// with balanced security and performance settings.
    #[inline]
    pub fn new() -> Self {
        Self::with_config(SecureExecutionConfig::default())
    }
    
    /// Create a new secure executor with custom configuration
    /// 
    /// # Arguments
    /// * `config` - Secure execution configuration
    /// 
    /// # Returns
    /// Configured secure executor ready for use
    pub fn with_config(config: SecureExecutionConfig) -> Self {
        // Create Cylo environment based on security level
        let cylo_env = match config.security_level {
            SecurityLevel::Basic | SecurityLevel::Standard => {
                // Use auto-detection for basic/standard security
                if let Some(backend_name) = get_recommended_backend() {
                    match backend_name {
                        "Apple" => Cylo::Apple("python:alpine3.20".to_string()),
                        "LandLock" => Cylo::LandLock("/tmp/cylo_jail".to_string()),
                        "FireCracker" => Cylo::FireCracker("python:alpine3.20".to_string()),
                        _ => Cylo::Apple("python:alpine3.20".to_string()), // Default fallback
                    }
                } else {
                    Cylo::Apple("python:alpine3.20".to_string()) // Default fallback
                }
            }
            SecurityLevel::High => {
                // Prefer LandLock for high security
                Cylo::LandLock("/tmp/cylo_jail_high".to_string())
            }
            SecurityLevel::Maximum => {
                // Use FireCracker for maximum security
                Cylo::FireCracker("python:alpine3.20".to_string())
            }
        };
        
        // Create backend configuration based on security level
        let backend_config = BackendConfig {
            name: "secure_executor".to_string(),
            enabled: true,
            default_timeout: config.timeout,
            default_limits: ResourceLimits {
                max_memory: config.memory_limit,
                max_cpu_time: Some(config.timeout.as_secs()),
                max_processes: config.cpu_limit,
                max_file_size: Some(1024 * 1024 * 100), // 100MB default
            },
            backend_specific: config.env_vars.clone(),
        };
        
        // Create backend using cylo factory
        let backend = create_backend(&cylo_env, backend_config)
            .unwrap_or_else(|_| {
                // Fallback to Apple backend on error
                let fallback_env = Cylo::Apple("python:alpine3.20".to_string());
                let fallback_config = BackendConfig {
                    name: "secure_executor_fallback".to_string(),
                    enabled: true,
                    default_timeout: config.timeout,
                    default_limits: ResourceLimits {
                        max_memory: config.memory_limit,
                        max_cpu_time: Some(config.timeout.as_secs()),
                        max_processes: config.cpu_limit,
                        max_file_size: Some(1024 * 1024 * 100), // 100MB default
                    },
                    backend_specific: config.env_vars.clone(),
                };
                create_backend(&fallback_env, fallback_config)
                    .expect("Failed to create fallback backend")
            });
        
        Self {
            config,
            backend,
            cached_instance: None,
        }
    }
    
    /// Create a performance-optimized secure executor
    /// 
    /// Prioritizes execution speed while maintaining security requirements.
    #[inline]
    pub fn performance_optimized() -> Self {
        let mut config = SecureExecutionConfig::default();
        config.security_level = SecurityLevel::Standard;
        config.instance_reuse = true;
        
        Self::with_config(config)
    }
    
    /// Create a security-focused executor
    /// 
    /// Maximizes isolation and security with optimal performance.
    #[inline]
    pub fn security_focused() -> Self {
        let mut config = SecureExecutionConfig::default();
        config.security_level = SecurityLevel::Maximum;
        config.timeout = Duration::from_secs(60); // Longer timeout for security operations
        
        Self::with_config(config)
    }
    
    /// Get the current execution configuration
    #[inline]
    pub fn config(&self) -> &SecureExecutionConfig {
        &self.config
    }
    
    /// Update execution configuration
    /// 
    /// # Arguments
    /// * `config` - New configuration to apply
    pub fn update_config(&mut self, config: SecureExecutionConfig) {
        // Create new executor with updated config
        let new_executor = Self::with_config(config.clone());
        
        // Update our state
        self.config = config;
        self.backend = new_executor.backend;
        
        // Clear cached instance when config changes
        self.cached_instance = None;
    }
    
    /// Execute code securely with automatic language detection and routing
    /// 
    /// # Arguments
    /// * `code` - Source code to execute
    /// * `language` - Programming language (python, javascript, rust, bash, go)
    /// 
    /// # Returns
    /// AsyncTask that resolves to execution result
    pub fn execute_code(&self, code: &str, language: &str) -> AsyncTask<Result<Value, String>> {
        let request = self.create_execution_request(code, language);
        let backend = &self.backend;
        
        crate::spawn_async(async move {
            // Execute using the backend's execute_code method
            let execution_task = backend.execute_code(request);
            
            match execution_task.await {
                Ok(result) => Ok(Self::format_execution_result(result, language)),
                Err(e) => Err(format!("Secure execution failed: {}", e)),
            }
        })
    }
    
    /// Execute Bash/shell code securely
    #[inline]
    pub fn execute_bash(&self, code: &str) -> AsyncTask<Result<Value, String>> {
        self.execute_code(code, "bash")
    }
    
    /// Execute Python code securely
    #[inline]
    pub fn execute_python(&self, code: &str) -> AsyncTask<Result<Value, String>> {
        self.execute_code(code, "python")
    }
    
    /// Execute JavaScript code securely
    #[inline]
    pub fn execute_javascript(&self, code: &str) -> AsyncTask<Result<Value, String>> {
        self.execute_code(code, "javascript")
    }
    
    /// Execute Rust code securely
    #[inline]
    pub fn execute_rust(&self, code: &str) -> AsyncTask<Result<Value, String>> {
        self.execute_code(code, "rust")
    }
    
    /// Execute Go code securely
    #[inline]
    pub fn execute_go(&self, code: &str) -> AsyncTask<Result<Value, String>> {
        self.execute_code(code, "go")
    }
    
    /// Execute code with specific Cylo instance
    /// 
    /// # Arguments
    /// * `instance` - Cylo instance to use for execution
    /// * `code` - Source code to execute
    /// * `language` - Programming language
    /// 
    /// # Returns
    /// AsyncTask that resolves to execution result
    pub fn execute_with_instance(
        &self,
        _instance: &CyloInstance,
        code: &str,
        language: &str
    ) -> AsyncTask<Result<Value, String>> {
        // For now, delegate to regular execute_code since backend handles instance management
        self.execute_code(code, language)
    }
    
    /// Execute a tool with arguments securely
    /// 
    /// This is the main integration point for Tool trait implementations.
    /// Supports both direct code execution and tool-specific parameters.
    /// 
    /// # Arguments
    /// * `tool_name` - Name of the tool being executed
    /// * `args` - Tool arguments including code and language
    /// 
    /// # Returns
    /// Future that resolves to execution result
    pub fn execute_tool_with_args(
        &self, 
        tool_name: &str, 
        args: Value
    ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        let tool_name = tool_name.to_string();
        let executor = self.clone();
        
        Box::pin(async move {
            // Extract execution parameters from args
            let (code, language) = match Self::extract_execution_params(&args, &tool_name) {
                Ok(params) => params,
                Err(e) => return Err(e),
            };
            
            // Execute the code securely
            executor.execute_code(&code, &language).await
        })
    }
    
    /// Get execution metrics and diagnostics
    /// 
    /// # Returns
    /// AsyncTask that resolves to diagnostic information
    pub fn get_diagnostics(&self) -> AsyncTask<Result<Value, String>> {
        let backend = &self.backend;
        
        crate::spawn_async(async move {
            // Use backend's health_check method
            let health_task = backend.health_check();
            
            match health_task.await {
                Ok(health_status) => {
                    Ok(serde_json::json!({
                        "status": match health_status {
                            HealthStatus::Healthy => "healthy",
                            HealthStatus::Degraded => "degraded",
                            HealthStatus::Unhealthy => "unhealthy",
                        },
                        "backend_type": "cylo",
                        "timestamp": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    }))
                },
                Err(e) => Err(format!("Failed to get diagnostics: {}", e)),
            }
        })
    }
    
    // ========================================================================
    // Internal Implementation Methods
    // ========================================================================
    
    /// Create execution request from parameters
    fn create_execution_request(&self, code: &str, language: &str) -> ExecutionRequest {
        let mut request = ExecutionRequest::new(code, language)
            .with_timeout(self.config.timeout);
        
        // Apply resource limits
        if let Some(memory_limit) = self.config.memory_limit {
            request = request.with_memory_limit(memory_limit);
        }
        
        if let Some(cpu_limit) = self.config.cpu_limit {
            request = request.with_cpu_limit(cpu_limit);
        }
        
        // Apply environment variables
        for (key, value) in &self.config.env_vars {
            request = request.with_env_var(key.clone(), value.clone());
        }
        
        // Apply working directory
        if let Some(workdir) = &self.config.working_dir {
            request = request.with_working_dir(workdir.clone());
        }
        
        request
    }
    
    /// Format execution result as JSON
    fn format_execution_result(result: ExecutionResult, language: &str) -> Value {
        serde_json::json!({
            "status": if result.is_success() { "success" } else { "error" },
            "language": language,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration.as_millis(),
            "resource_usage": {
                "peak_memory": result.resource_usage.peak_memory,
                "cpu_time_ms": result.resource_usage.cpu_time_ms,
                "process_count": result.resource_usage.process_count,
                "disk_bytes_written": result.resource_usage.disk_bytes_written,
                "disk_bytes_read": result.resource_usage.disk_bytes_read,
                "network_bytes_sent": result.resource_usage.network_bytes_sent,
                "network_bytes_received": result.resource_usage.network_bytes_received,
            },
            "metadata": result.metadata,
        })
    }
    
    /// Extract execution parameters from tool arguments
    fn extract_execution_params(args: &Value, tool_name: &str) -> Result<(String, String), String> {
        // Extract code parameter
        let code = match args.get("code") {
            Some(Value::String(code)) => code.clone(),
            Some(value) => value.to_string(),
            None => return Err("Missing required 'code' parameter".to_string()),
        };
        
        // Extract or infer language parameter
        let language = match args.get("language") {
            Some(Value::String(lang)) => lang.clone(),
            _ => Self::infer_language_from_tool_name(tool_name),
        };
        
        Ok((code, language))
    }
    
    /// Infer programming language from tool name
    fn infer_language_from_tool_name(tool_name: &str) -> String {
        let tool_lower = tool_name.to_lowercase();
        
        if tool_lower.contains("python") || tool_lower.contains("py") {
            "python".to_string()
        } else if tool_lower.contains("javascript") || tool_lower.contains("js") || tool_lower.contains("node") {
            "javascript".to_string()
        } else if tool_lower.contains("rust") || tool_lower.contains("rs") {
            "rust".to_string()
        } else if tool_lower.contains("go") || tool_lower.contains("golang") {
            "go".to_string()
        } else if tool_lower.contains("bash") || tool_lower.contains("shell") || tool_lower.contains("sh") {
            "bash".to_string()
        } else {
            "bash".to_string() // Safe default for shell commands
        }
    }
}

impl Clone for SecureToolExecutor {
    fn clone(&self) -> Self {
        // Create new executor with same configuration
        Self::with_config(self.config.clone())
    }
}

impl Default for SecureToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for tools that can be executed securely with Cylo
pub trait SecureExecutable {
    /// Execute the tool securely using the Cylo system
    /// 
    /// # Arguments
    /// * `args` - Tool execution arguments
    /// * `executor` - Secure executor instance
    /// 
    /// # Returns
    /// Future that resolves to execution result
    fn execute_securely(
        &self, 
        args: Value, 
        executor: &SecureToolExecutor
    ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;
}

// ============================================================================
// Global Executor Management
// ============================================================================

/// Global secure executor instance for high-performance shared usage
static GLOBAL_EXECUTOR: OnceLock<SecureToolExecutor> = OnceLock::new();

/// Get the global secure executor instance
/// 
/// Uses optimal default configuration for the current platform.
#[inline]
pub fn get_secure_executor() -> &'static SecureToolExecutor {
    GLOBAL_EXECUTOR.get_or_init(SecureToolExecutor::new)
}

/// Initialize the global secure executor with custom configuration
/// 
/// # Arguments
/// * `config` - Custom execution configuration
/// 
/// # Returns
/// Result indicating success or if already initialized
pub fn init_secure_executor(config: SecureExecutionConfig) -> Result<(), SecureToolExecutor> {
    GLOBAL_EXECUTOR.set(SecureToolExecutor::with_config(config))
}

/// Initialize the global secure executor with a specific instance
/// 
/// # Arguments
/// * `executor` - Configured executor instance
/// 
/// # Returns
/// Result indicating success or if already initialized
pub fn set_secure_executor(executor: SecureToolExecutor) -> Result<(), SecureToolExecutor> {
    GLOBAL_EXECUTOR.set(executor)
}

// ============================================================================
// High-Level Convenience Functions
// ============================================================================

/// Execute code securely with automatic backend selection
/// 
/// Uses the global executor instance for optimal performance.
#[inline]
pub fn execute_code_securely(code: &str, language: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_code(code, language)
}

/// Execute Python code securely with global executor
#[inline]
pub fn execute_python_securely(code: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_python(code)
}

/// Execute JavaScript code securely with global executor
#[inline]
pub fn execute_javascript_securely(code: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_javascript(code)
}

/// Execute Rust code securely with global executor
#[inline]
pub fn execute_rust_securely(code: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_rust(code)
}

/// Execute Bash code securely with global executor
#[inline]
pub fn execute_bash_securely(code: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_bash(code)
}

/// Execute Go code securely with global executor
#[inline]
pub fn execute_go_securely(code: &str) -> AsyncTask<Result<Value, String>> {
    get_secure_executor().execute_go(code)
}