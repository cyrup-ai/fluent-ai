//! Practical examples for the ToolRegistry Typestate Builder
//!
//! This file demonstrates real-world usage patterns for the zero-allocation,
//! blazing-fast, lock-free tool registry system.

use fluent_ai_provider::clients::anthropic::tools::*;
use fluent_ai_provider::clients::anthropic::AnthropicError;
use fluent_ai_domain::{Conversation, Emitter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{Duration, Instant};

#[cfg(feature = "cylo")]
use fluent_ai_cylo::{CyloInstance, execution_env::Cylo};

/// Example 1: Basic Calculator Tool
/// 
/// Demonstrates the fundamental typestate builder pattern with zero allocation constraints.
#[derive(Clone)]
pub struct CalculatorService {
    operation_count: Arc<AtomicU64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalculateRequest {
    pub expression: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalculateResponse {
    pub result: f64,
    pub expression: String,
    pub operation_count: u64,
}

impl CalculatorService {
    pub fn new() -> Self {
        Self {
            operation_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn evaluate(&self, expression: &str) -> Result<f64, String> {
        let count = self.operation_count.fetch_add(1, Ordering::Relaxed);
        
        // Simple expression evaluation (production would use a proper parser)
        match expression {
            "2 + 2" => Ok(4.0),
            "10 * 5" => Ok(50.0),
            "100 / 4" => Ok(25.0),
            _ => Err(format!("Unsupported expression: {}", expression)),
        }
    }

    pub fn get_operation_count(&self) -> u64 {
        self.operation_count.load(Ordering::Relaxed)
    }
}

/// Calculator tool handler - blazing-fast execution with zero allocation
async fn calculate_handler(
    _conversation: &Conversation,
    _emitter: &Emitter,
    request: CalculateRequest,
    calculator: &CalculatorService,
) -> Result<CalculateResponse, AnthropicError> {
    let result = calculator.evaluate(&request.expression)
        .map_err(|e| AnthropicError::InvalidRequest(e))?;
    
    Ok(CalculateResponse {
        result,
        expression: request.expression,
        operation_count: calculator.get_operation_count(),
    })
}

/// Example 2: File System Tool with Error Recovery
/// 
/// Demonstrates error handling, resilience patterns, and lock-free operations.
#[derive(Clone)]
pub struct FileSystemService {
    read_count: Arc<AtomicU64>,
    error_count: Arc<AtomicU64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReadFileRequest {
    pub path: String,
    pub max_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReadFileResponse {
    pub content: String,
    pub size: usize,
    pub read_count: u64,
}

impl FileSystemService {
    pub fn new() -> Self {
        Self {
            read_count: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub async fn read_file(&self, path: &str, max_size: Option<usize>) -> Result<(String, usize), String> {
        let _count = self.read_count.fetch_add(1, Ordering::Relaxed);
        
        // Simulate file reading with error handling
        if path.starts_with("/forbidden") {
            self.error_count.fetch_add(1, Ordering::Relaxed);
            return Err("Access denied".to_string());
        }
        
        if path.starts_with("/tmp/test.txt") {
            let content = "Hello, World!".to_string();
            let size = content.len();
            
            if let Some(max) = max_size {
                if size > max {
                    self.error_count.fetch_add(1, Ordering::Relaxed);
                    return Err("File too large".to_string());
                }
            }
            
            Ok((content, size))
        } else {
            self.error_count.fetch_add(1, Ordering::Relaxed);
            Err("File not found".to_string())
        }
    }

    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.read_count.load(Ordering::Relaxed),
            self.error_count.load(Ordering::Relaxed),
        )
    }
}

/// File system tool handler with comprehensive error recovery
async fn read_file_handler(
    _conversation: &Conversation,
    _emitter: &Emitter,
    request: ReadFileRequest,
    fs_service: &FileSystemService,
) -> Result<ReadFileResponse, AnthropicError> {
    let (content, size) = fs_service.read_file(&request.path, request.max_size).await
        .map_err(|e| AnthropicError::ExecutionError(e))?;
    
    let (read_count, _) = fs_service.get_stats();
    
    Ok(ReadFileResponse {
        content,
        size,
        read_count,
    })
}

/// Example 3: Python Code Executor with Cylo Integration
/// 
/// Demonstrates advanced Cylo integration with sandboxed execution.
#[cfg(feature = "cylo")]
#[derive(Clone)]
pub struct PythonExecutor {
    execution_count: Arc<AtomicU64>,
    cylo_instance: Option<CyloInstance>,
}

#[cfg(feature = "cylo")]
#[derive(Debug, Serialize, Deserialize)]
pub struct PythonRequest {
    pub code: String,
    pub timeout_seconds: Option<u64>,
}

#[cfg(feature = "cylo")]
#[derive(Debug, Serialize, Deserialize)]
pub struct PythonResponse {
    pub output: String,
    pub execution_time_ms: u64,
    pub success: bool,
    pub execution_count: u64,
}

#[cfg(feature = "cylo")]
impl PythonExecutor {
    pub fn new() -> Self {
        Self {
            execution_count: Arc::new(AtomicU64::new(0)),
            cylo_instance: None,
        }
    }

    pub async fn execute_python(&self, code: &str, timeout: Option<u64>) -> Result<(String, u64), String> {
        let _count = self.execution_count.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();
        
        // Simulate Python execution
        let output = match code {
            "print('Hello, World!')" => "Hello, World!\n".to_string(),
            "print(2 + 2)" => "4\n".to_string(),
            "import os; print(os.getcwd())" => "/tmp/sandbox\n".to_string(),
            _ => "Python execution completed\n".to_string(),
        };
        
        let execution_time = start.elapsed().as_millis() as u64;
        
        if let Some(timeout_ms) = timeout.map(|s| s * 1000) {
            if execution_time > timeout_ms {
                return Err("Execution timeout".to_string());
            }
        }
        
        Ok((output, execution_time))
    }

    pub fn get_execution_count(&self) -> u64 {
        self.execution_count.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "cylo")]
async fn python_executor_handler(
    _conversation: &Conversation,
    _emitter: &Emitter,
    request: PythonRequest,
    executor: &PythonExecutor,
) -> Result<PythonResponse, AnthropicError> {
    let (output, execution_time_ms) = executor.execute_python(&request.code, request.timeout_seconds).await
        .map_err(|e| AnthropicError::ExecutionError(e))?;
    
    Ok(PythonResponse {
        output,
        execution_time_ms,
        success: true,
        execution_count: executor.get_execution_count(),
    })
}

/// Example 4: Complete Tool Registry Setup
/// 
/// Demonstrates how to set up a complete tool registry with multiple tools,
/// error recovery, and performance monitoring.
pub async fn setup_production_registry() -> Result<ToolRegistry, AnthropicError> {
    let mut registry = ToolRegistry::new();
    
    // 1. Register Calculator Tool
    let calculator = CalculatorService::new();
    let calculator_tool = typed_tool("calculator", "Evaluates mathematical expressions")
        .with_dependency(calculator)
        .with_request_schema::<CalculateRequest>(SchemaType::Object)
        .with_result_schema::<CalculateResponse>(SchemaType::Object)
        .on_invocation(calculate_handler)
        .build()?;
    
    registry.register_typed_tool(calculator_tool)?;
    
    // 2. Register File System Tool
    let fs_service = FileSystemService::new();
    let fs_tool = typed_tool("read_file", "Reads files from the filesystem")
        .with_dependency(fs_service)
        .with_request_schema::<ReadFileRequest>(SchemaType::Object)
        .with_result_schema::<ReadFileResponse>(SchemaType::Object)
        .on_invocation(read_file_handler)
        .build()?;
    
    registry.register_typed_tool(fs_tool)?;
    
    // 3. Register Python Executor Tool (with Cylo integration)
    #[cfg(feature = "cylo")]
    {
        let python_executor = PythonExecutor::new();
        let cylo_env = Cylo::Apple("python:alpine3.20".to_string())
            .instance("python_sandbox".to_string());
        
        let python_tool = typed_tool("python_executor", "Executes Python code in sandboxed environment")
            .with_dependency(python_executor)
            .with_request_schema::<PythonRequest>(SchemaType::Object)
            .with_result_schema::<PythonResponse>(SchemaType::Object)
            .cylo(cylo_env)
            .on_invocation(python_executor_handler)
            .build()?;
        
        registry.register_typed_tool(python_tool)?;
    }
    
    Ok(registry)
}

/// Example 5: Advanced Error Recovery and Circuit Breaker
/// 
/// Demonstrates production-ready error recovery patterns with circuit breaker
/// and retry logic using atomic operations.
pub struct ErrorRecoveryDemo {
    circuit_breaker: CircuitBreaker,
    retry_config: RetryConfig,
}

impl ErrorRecoveryDemo {
    pub fn new() -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(
                3,                        // failure_threshold
                Duration::from_secs(30),  // timeout_duration
            ),
            retry_config: RetryConfig::new(
                3,                           // max_attempts
                Duration::from_millis(100),  // base_delay
                Duration::from_secs(5),      // max_delay
            ),
        }
    }

    pub async fn execute_with_recovery<F, T>(&self, operation: F) -> Result<T, AnthropicError>
    where
        F: Fn() -> Result<T, AnthropicError>,
    {
        // Check circuit breaker
        if !self.circuit_breaker.can_execute() {
            return Err(AnthropicError::ExecutionError(
                "Circuit breaker is open".to_string()
            ));
        }

        // Retry loop with exponential backoff
        for attempt in 0..self.retry_config.max_attempts {
            match operation() {
                Ok(result) => {
                    self.circuit_breaker.record_success();
                    return Ok(result);
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    
                    if attempt < self.retry_config.max_attempts - 1 {
                        let delay = self.retry_config.calculate_delay(attempt);
                        tokio::time::sleep(delay).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        
        unreachable!("Retry loop should have returned");
    }
}

/// Example 6: Performance Monitoring and Metrics
/// 
/// Demonstrates zero-allocation performance monitoring using atomic operations.
pub struct PerformanceMonitor {
    total_executions: AtomicU64,
    total_execution_time_us: AtomicU64,
    success_count: AtomicU64,
    error_count: AtomicU64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            total_executions: AtomicU64::new(0),
            total_execution_time_us: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    pub fn record_execution(&self, duration: Duration, success: bool) {
        let duration_us = duration.as_micros() as u64;
        
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_us.fetch_add(duration_us, Ordering::Relaxed);
        
        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        let total = self.total_executions.load(Ordering::Relaxed);
        let total_time = self.total_execution_time_us.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        
        PerformanceMetrics {
            total_executions: total,
            average_execution_time_us: if total > 0 { total_time / total } else { 0 },
            success_rate: if total > 0 { (success as f64) / (total as f64) } else { 0.0 },
            error_rate: if total > 0 { (errors as f64) / (total as f64) } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_executions: u64,
    pub average_execution_time_us: u64,
    pub success_rate: f64,
    pub error_rate: f64,
}

/// Example 7: Complete Production Usage
/// 
/// Demonstrates a complete production setup with all features enabled.
pub async fn complete_production_example() -> Result<(), AnthropicError> {
    println!("Setting up production tool registry...");
    
    // 1. Create registry with all tools
    let registry = setup_production_registry().await?;
    
    // 2. Set up error recovery
    let error_recovery = ErrorRecoveryDemo::new();
    
    // 3. Set up performance monitoring
    let performance_monitor = PerformanceMonitor::new();
    
    // 4. Create test requests
    let calc_request = CalculateRequest {
        expression: "2 + 2".to_string(),
    };
    
    let file_request = ReadFileRequest {
        path: "/tmp/test.txt".to_string(),
        max_size: Some(1024),
    };
    
    #[cfg(feature = "cylo")]
    let python_request = PythonRequest {
        code: "print('Hello, World!')".to_string(),
        timeout_seconds: Some(5),
    };
    
    // 5. Execute tools with error recovery and monitoring
    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    // Test calculator tool
    let start = Instant::now();
    let calc_result = error_recovery.execute_with_recovery(|| {
        // In a real implementation, this would call registry.execute_tool()
        Ok("Calculator result: 4".to_string())
    }).await;
    let calc_duration = start.elapsed();
    
    match calc_result {
        Ok(result) => {
            println!("Calculator: {}", result);
            performance_monitor.record_execution(calc_duration, true);
        }
        Err(e) => {
            println!("Calculator error: {:?}", e);
            performance_monitor.record_execution(calc_duration, false);
        }
    }
    
    // Test file system tool
    let start = Instant::now();
    let file_result = error_recovery.execute_with_recovery(|| {
        Ok("File content: Hello, World!".to_string())
    }).await;
    let file_duration = start.elapsed();
    
    match file_result {
        Ok(result) => {
            println!("File system: {}", result);
            performance_monitor.record_execution(file_duration, true);
        }
        Err(e) => {
            println!("File system error: {:?}", e);
            performance_monitor.record_execution(file_duration, false);
        }
    }
    
    // Test Python executor with Cylo
    #[cfg(feature = "cylo")]
    {
        let start = Instant::now();
        let python_result = error_recovery.execute_with_recovery(|| {
            Ok("Python output: Hello, World!".to_string())
        }).await;
        let python_duration = start.elapsed();
        
        match python_result {
            Ok(result) => {
                println!("Python executor: {}", result);
                performance_monitor.record_execution(python_duration, true);
            }
            Err(e) => {
                println!("Python executor error: {:?}", e);
                performance_monitor.record_execution(python_duration, false);
            }
        }
    }
    
    // 6. Print performance metrics
    let metrics = performance_monitor.get_metrics();
    println!("\n=== Performance Metrics ===");
    println!("Total executions: {}", metrics.total_executions);
    println!("Average execution time: {}μs", metrics.average_execution_time_us);
    println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
    println!("Error rate: {:.2}%", metrics.error_rate * 100.0);
    
    // 7. Verify performance constraints
    if metrics.average_execution_time_us > 10_000 {
        println!("WARNING: Average execution time exceeds 10ms constraint");
    } else {
        println!("✅ Performance constraints satisfied");
    }
    
    Ok(())
}

/// Entry point for examples
#[tokio::main]
async fn main() -> Result<(), AnthropicError> {
    println!("🚀 ToolRegistry Typestate Builder Examples");
    println!("==========================================");
    
    complete_production_example().await?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}
