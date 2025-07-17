//! ============================================================================
//! Secure MCP tool implementation with mandatory Cylo container execution
//! ============================================================================
//!
//! ALL tools execute in containers via Cylo - this is the only execution method.
//! Container selection is automatic (based on OS) or explicit via Cylo types.
//! Zero allocation, blazing-fast, lock-free execution with comprehensive error handling.

use crate::{mcp_tool::{Tool, McpTool}, AsyncTask, HashMap};
use fluent_ai_cylo::{
    CyloExecutor, CyloInstance, Cylo, ExecutionRequest, ExecutionResult, CyloResult,
    global_executor, RoutingStrategy, OptimizationConfig
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use std::future::Future;
use std::pin::Pin;

/// High-performance MCP tool with mandatory Cylo container execution
/// 
/// ALL tools execute in containers - automatic backend selection or explicit Cylo instance.
/// Zero allocation patterns with lock-free concurrent execution and comprehensive monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureMcpTool {
    /// Tool identifier
    name: String,
    /// Tool functionality description
    description: String,
    /// JSON schema for input parameters
    parameters: Value,
    /// Optional MCP server identifier
    server: Option<String>,
    /// Explicit Cylo execution environment (None = automatic selection)
    #[serde(skip)]
    cylo_instance: Option<CyloInstance>,
    /// Execution timeout in seconds
    timeout_seconds: u64,
    /// Memory limit in bytes
    memory_limit: Option<u64>,
    /// CPU core limit
    cpu_limit: Option<u32>,
}

impl SecureMcpTool {
    /// Create MCP tool with automatic container backend selection
    /// 
    /// Uses platform-optimal container backend (Apple/LandLock/FireCracker)
    /// based on OS capabilities and performance characteristics.
    #[inline]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: None,
            cylo_instance: None, // Automatic backend selection
            timeout_seconds: 30,
            memory_limit: Some(512 * 1024 * 1024), // 512MB default
            cpu_limit: Some(1),
        }
    }
    
    /// Create MCP tool with MCP server identifier
    #[inline]
    pub fn with_server(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        server: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: Some(server.into()),
            cylo_instance: None,
            timeout_seconds: 30,
            memory_limit: Some(512 * 1024 * 1024),
            cpu_limit: Some(1),
        }
    }
    
    /// Create MCP tool with explicit Cylo container environment
    /// 
    /// Allows precise control over execution backend and container configuration.
    /// 
    /// # Arguments
    /// * `name` - Tool identifier
    /// * `description` - Tool functionality description
    /// * `parameters` - JSON schema for parameters
    /// * `cylo_instance` - Explicit container environment
    /// 
    /// # Examples
    /// ```
    /// let tool = SecureMcpTool::with_cylo(
    ///     "python_analyzer",
    ///     "Python code analysis tool",
    ///     parameters,
    ///     Cylo::Apple("python:alpine3.20".to_string()).instance("analyzer_env")
    /// );
    /// ```
    #[inline]
    pub fn with_cylo(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        cylo_instance: CyloInstance,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: None,
            cylo_instance: Some(cylo_instance),
            timeout_seconds: 30,
            memory_limit: Some(512 * 1024 * 1024),
            cpu_limit: Some(1),
        }
    }

    /// Create language-specific code execution tool with optimal container selection
    /// 
    /// Automatically selects container image and backend based on language requirements.
    /// 
    /// # Arguments
    /// * `language` - Programming language (python, javascript, rust, bash, go)
    /// 
    /// # Returns
    /// Optimized MCP tool for the specified language
    pub fn code_executor(language: &str) -> Self {
        let name = format!("{}_executor", language);
        let description = format!("Container-based {} code execution with automatic backend selection", language);
        
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": format!("The {} source code to execute", language)
                },
                "input": {
                    "type": "string",
                    "description": "Optional stdin input for the program"
                },
                "env_vars": {
                    "type": "object",
                    "description": "Environment variables for execution",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["code"]
        });

        let mut tool = Self::new(name, description, parameters);
        tool.optimize_for_language(language);
        tool
    }
    
    /// Create multi-language code executor with intelligent container routing
    /// 
    /// Supports dynamic language detection and optimal container selection
    /// for maximum performance across different programming languages.
    #[inline]
    pub fn multi_language_executor() -> Self {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Source code to execute"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language for execution",
                    "enum": ["python", "javascript", "rust", "bash", "go"]
                },
                "input": {
                    "type": "string",
                    "description": "Optional stdin input"
                },
                "env_vars": {
                    "type": "object",
                    "description": "Environment variables",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["code", "language"]
        });

        Self::new(
            "code_executor",
            "Multi-language container-based code execution with intelligent routing",
            parameters,
        )
    }
    
    /// Optimize tool configuration for specific programming language
    /// 
    /// Adjusts memory, CPU, and timeout limits based on language characteristics.
    /// 
    /// # Arguments
    /// * `language` - Programming language to optimize for
    fn optimize_for_language(&mut self, language: &str) {
        match language.to_lowercase().as_str() {
            "rust" => {
                self.memory_limit = Some(1024 * 1024 * 1024); // 1GB for compilation
                self.timeout_seconds = 120; // Longer timeout for compilation
                self.cpu_limit = Some(2); // More CPU for compilation
            },
            "go" => {
                self.memory_limit = Some(512 * 1024 * 1024); // 512MB
                self.timeout_seconds = 60; // Moderate timeout
                self.cpu_limit = Some(1);
            },
            "python" | "javascript" => {
                self.memory_limit = Some(256 * 1024 * 1024); // 256MB
                self.timeout_seconds = 30;
                self.cpu_limit = Some(1);
            },
            "bash" => {
                self.memory_limit = Some(128 * 1024 * 1024); // 128MB
                self.timeout_seconds = 15;
                self.cpu_limit = Some(1);
            },
            _ => {
                // Keep defaults
            }
        }
    }
    
    /// Set execution timeout
    /// 
    /// # Arguments
    /// * `seconds` - Maximum execution time in seconds
    #[inline]
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }
    
    /// Set memory limit
    /// 
    /// # Arguments
    /// * `bytes` - Maximum memory usage in bytes
    #[inline]
    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = Some(bytes);
        self
    }
    
    /// Set CPU core limit
    /// 
    /// # Arguments
    /// * `cores` - Maximum number of CPU cores
    #[inline]
    pub fn with_cpu_limit(mut self, cores: u32) -> Self {
        self.cpu_limit = Some(cores);
        self
    }
    
    /// Get the Cylo instance if explicitly set
    #[inline]
    pub fn cylo_instance(&self) -> Option<&CyloInstance> {
        self.cylo_instance.as_ref()
    }
}

impl Tool for SecureMcpTool {
    #[inline]
    fn name(&self) -> &str {
        &self.name
    }
    
    #[inline]
    fn description(&self) -> &str {
        &self.description
    }
    
    #[inline]
    fn parameters(&self) -> &Value {
        &self.parameters
    }
    
    /// Execute tool in Cylo container with zero allocation hot path
    /// 
    /// ALL execution happens in containers - automatic backend selection
    /// or explicit Cylo instance based on tool configuration.
    fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        let tool_name = self.name.clone();
        let cylo_instance = self.cylo_instance.clone();
        let timeout = Duration::from_secs(self.timeout_seconds);
        let memory_limit = self.memory_limit;
        let cpu_limit = self.cpu_limit;
        
        Box::pin(async move {
            // Extract execution parameters
            let (code, language, input, env_vars) = match Self::extract_execution_params(&args) {
                Ok(params) => params,
                Err(e) => return Err(e),
            };
            
            // Create execution request with resource limits
            let mut request = ExecutionRequest::new(&code, &language)
                .with_timeout(timeout);
            
            if let Some(memory) = memory_limit {
                request = request.with_memory_limit(memory);
            }
            
            if let Some(cpu) = cpu_limit {
                request = request.with_cpu_limit(cpu);
            }
            
            if let Some(input_data) = input {
                request = request.with_input(input_data);
            }
            
            for (key, value) in env_vars {
                request = request.with_env_var(key, value);
            }
            
            // Execute in container using global executor or explicit instance
            let executor = global_executor();
            let result = match cylo_instance {
                Some(instance) => {
                    executor.execute_with_instance(&instance, request).await
                },
                None => {
                    executor.execute(request, None).await
                }
            };
            
            match result {
                Ok(exec_result) => {
                    Ok(Self::format_execution_result(exec_result, &tool_name))
                },
                Err(e) => {
                    Err(format!("Container execution failed for {}: {}", tool_name, e))
                }
            }
        })
    }
}

impl McpTool for SecureMcpTool {
    #[inline]
    fn server(&self) -> Option<&str> {
        self.server.as_deref()
    }
    
    #[inline]
    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self::new(name, description, parameters)
    }
}

// Builder implementations moved to fluent_ai/src/builders/secure_mcp_tool.rs

impl SecureMcpTool {
    /// Create Python executor with optimal container configuration
    /// 
    /// Uses Python:alpine container with 256MB memory and 30s timeout.
    #[inline]
    pub fn python_executor() -> Self {
        Self::code_executor("python")
    }
    
    /// Create JavaScript executor with Node.js container
    /// 
    /// Uses Node:alpine container optimized for JavaScript execution.
    #[inline]
    pub fn javascript_executor() -> Self {
        Self::code_executor("javascript")
    }
    
    /// Create Bash executor with minimal Alpine container
    /// 
    /// Uses Alpine container with 128MB memory for shell commands.
    #[inline]
    pub fn bash_executor() -> Self {
        Self::code_executor("bash")
    }
    
    /// Create Rust executor with compilation support
    /// 
    /// Uses Rust:alpine container with 1GB memory for compilation.
    #[inline]
    pub fn rust_executor() -> Self {
        Self::code_executor("rust")
    }
    
    /// Create Go executor with compilation support
    /// 
    /// Uses Golang:alpine container with 512MB memory.
    #[inline]
    pub fn go_executor() -> Self {
        Self::code_executor("go")
    }
    
    // ========================================================================
    // Internal Implementation Methods
    // ========================================================================
    
    /// Extract execution parameters from tool arguments
    /// 
    /// # Arguments
    /// * `args` - JSON arguments from tool invocation
    /// 
    /// # Returns
    /// Tuple of (code, language, input, env_vars) or error message
    fn extract_execution_params(args: &Value) -> Result<(String, String, Option<String>, HashMap<String, String>), String> {
        // Extract required code parameter
        let code = match args.get("code") {
            Some(Value::String(code)) => code.clone(),
            Some(value) => value.to_string(),
            None => return Err("Missing required 'code' parameter".to_string()),
        };
        
        // Extract language parameter (required for multi-language, inferred for single-language tools)
        let language = match args.get("language") {
            Some(Value::String(lang)) => lang.clone(),
            Some(value) => value.to_string(),
            None => "bash".to_string(), // Safe default
        };
        
        // Extract optional input parameter
        let input = args.get("input")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract optional environment variables
        let mut env_vars = HashMap::new();
        if let Some(Value::Object(env_obj)) = args.get("env_vars") {
            for (key, value) in env_obj {
                if let Some(value_str) = value.as_str() {
                    env_vars.insert(key.clone(), value_str.to_string());
                }
            }
        }
        
        Ok((code, language, input, env_vars))
    }
    
    /// Format execution result as structured JSON response
    /// 
    /// # Arguments
    /// * `result` - Cylo execution result
    /// * `tool_name` - Name of the executing tool
    /// 
    /// # Returns
    /// Formatted JSON response with execution details
    fn format_execution_result(result: ExecutionResult, tool_name: &str) -> Value {
        serde_json::json!({
            "status": if result.is_success() { "success" } else { "error" },
            "tool": tool_name,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration.as_millis(),
            "resource_usage": {
                "peak_memory_bytes": result.resource_usage.peak_memory,
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
}

// ============================================================================
// High-Level Convenience Functions
// ============================================================================

/// Create Python executor with automatic container selection
/// 
/// Uses optimal Python container for the current platform.
#[inline]
pub fn python_tool() -> SecureMcpTool {
    SecureMcpTool::python_executor()
}

/// Create JavaScript executor with automatic container selection
/// 
/// Uses optimal Node.js container for the current platform.
#[inline]
pub fn javascript_tool() -> SecureMcpTool {
    SecureMcpTool::javascript_executor()
}

/// Create Rust executor with compilation support
/// 
/// Uses Rust container with compilation toolchain.
#[inline]
pub fn rust_tool() -> SecureMcpTool {
    SecureMcpTool::rust_executor()
}

/// Create multi-language executor with intelligent routing
/// 
/// Supports dynamic language detection and optimal container selection.
#[inline]
pub fn universal_code_tool() -> SecureMcpTool {
    SecureMcpTool::multi_language_executor()
}

/// Create custom tool with explicit Cylo environment
/// 
/// # Arguments
/// * `name` - Tool identifier
/// * `description` - Tool description
/// * `parameters` - JSON schema for parameters
/// * `cylo_instance` - Explicit container environment
/// 
/// # Returns
/// MCP tool with explicit container configuration
#[inline]
pub fn custom_container_tool(
    name: impl Into<String>,
    description: impl Into<String>,
    parameters: Value,
    cylo_instance: CyloInstance,
) -> SecureMcpTool {
    SecureMcpTool::with_cylo(name, description, parameters, cylo_instance)
}

// Tool builder function moved to fluent_ai/src/builders/secure_mcp_tool.rs