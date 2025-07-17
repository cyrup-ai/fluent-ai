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

// Import Cylo types with feature gate
#[cfg(feature = "cylo")]
use fluent_ai_cylo::{
    CyloExecutor, RoutingStrategy, BackendPreferences, OptimizationConfig,
    ExecutionRequest, ExecutionResult, Cylo, CyloInstance, CyloResult,
    global_executor, create_executor, create_performance_executor, create_security_executor
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
#[derive(Debug)]
pub struct SecureToolExecutor {
    /// Execution configuration
    config: SecureExecutionConfig,
    /// Cylo executor for routing and management
    #[cfg(feature = "cylo")]
    executor: CyloExecutor,
    /// Cached instance for reuse
    #[cfg(feature = "cylo")]
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
        #[cfg(feature = "cylo")]
        {
            let executor = match config.security_level {
                SecurityLevel::Basic | SecurityLevel::Standard => create_executor(),
                SecurityLevel::High => create_security_executor(),
                SecurityLevel::Maximum => {
                    let mut executor = create_security_executor();
                    let mut preferences = BackendPreferences::default();
                    // Prefer FireCracker for maximum security
                    preferences.preferred_order = vec![
                        "FireCracker".to_string(),
                        "LandLock".to_string(),
                        "Apple".to_string(),
                    ];
                    executor.update_preferences(preferences);
                    executor
                }
            };
            
            Self {
                config,
                executor,
                cached_instance: None,
            }
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            Self { config }
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
        
        #[cfg(feature = "cylo")]
        {
            Self {
                config,
                executor: create_performance_executor(),
                cached_instance: None,
            }
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            Self { config }
        }
    }
    
    /// Create a security-focused executor
    /// 
    /// Maximizes isolation and security with optimal performance.
    #[inline]
    pub fn security_focused() -> Self {
        let mut config = SecureExecutionConfig::default();
        config.security_level = SecurityLevel::Maximum;
        config.timeout = Duration::from_secs(60); // Longer timeout for security operations
        
        #[cfg(feature = "cylo")]
        {
            Self {
                config,
                executor: create_security_executor(),
                cached_instance: None,
            }
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            Self { config }
        }
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
        self.config = config;
        
        #[cfg(feature = "cylo")]
        {
            // Update Cylo executor optimization config
            let optimization = OptimizationConfig {
                instance_reuse: self.config.instance_reuse,
                instance_pool_size: 5,
                max_idle_time: Duration::from_secs(300),
                load_balancing: true,
                monitoring_interval: Duration::from_secs(60),
            };
            self.executor.update_config(optimization);
            
            // Clear cached instance when config changes
            self.cached_instance = None;
        }
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
        #[cfg(feature = "cylo")]
        {
            let request = self.create_execution_request(code, language);
            let executor = &self.executor;
            let instance = self.cached_instance.clone();
            
            crate::spawn_async(async move {
                match executor.execute(request, instance.as_ref()).await {
                    Ok(result) => Ok(Self::format_execution_result(result, language)),
                    Err(e) => Err(format!("Secure execution failed: {}", e)),
                }
            })
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            let language = language.to_string();
            crate::spawn_async(async move {
                Err(format!("Cylo feature not enabled. Cannot execute {} code.", language))
            })
        }
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
    #[cfg(feature = "cylo")]
    pub fn execute_with_instance(
        &self,
        instance: &CyloInstance,
        code: &str,
        language: &str
    ) -> AsyncTask<Result<Value, String>> {
        let request = self.create_execution_request(code, language);
        let executor = &self.executor;
        let instance = instance.clone();
        
        crate::spawn_async(async move {
            match executor.execute_with_instance(&instance, request).await {
                Ok(result) => Ok(Self::format_execution_result(result, language)),
                Err(e) => Err(format!("Secure execution with instance failed: {}", e)),
            }
        })
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
    #[cfg(feature = "cylo")]
    pub fn get_diagnostics(&self) -> AsyncTask<Result<Value, String>> {
        let executor = &self.executor;
        
        crate::spawn_async(async move {
            match executor.get_metrics() {
                Ok(metrics) => {
                    Ok(serde_json::json!({
                        "executions_per_backend": metrics.executions_per_backend,
                        "avg_execution_time": metrics.avg_execution_time
                            .iter()
                            .map(|(k, v)| (k.clone(), v.as_millis()))
                            .collect::<HashMap<String, u128>>(),
                        "success_rate": metrics.success_rate,
                        "last_updated": metrics.last_updated
                            .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()),
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
    #[cfg(feature = "cylo")]
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
    #[cfg(feature = "cylo")]
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
        #[cfg(feature = "cylo")]
        {
            Self {
                config: self.config.clone(),
                executor: create_executor(), // Create new executor instance
                cached_instance: self.cached_instance.clone(),
            }
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            Self {
                config: self.config.clone(),
            }
        }
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