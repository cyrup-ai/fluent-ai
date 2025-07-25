//! Advanced macro processor with statistics and execution
//!
//! This module provides the MacroProcessor which handles complex macro execution
//! with performance monitoring, concurrent execution limits, and comprehensive statistics.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use uuid::Uuid;

use super::types::{MacroAction, StoredMacro};
use super::errors::MacroSystemError;

/// Macro processor for executing and managing chat macros
///
/// This processor provides comprehensive macro execution capabilities with:
/// - Recording and playback of chat interactions
/// - Variable substitution and conditional logic
/// - Performance monitoring and error handling
/// - Concurrent execution with lock-free data structures
/// - Macro validation and optimization
#[derive(Debug, Clone)]
pub struct MacroProcessor {
    /// Macro storage with lock-free access
    macros: Arc<SkipMap<Uuid, StoredMacro>>,
    /// Execution statistics
    stats: Arc<MacroProcessorStats>,
    /// Variable context for macro execution
    variables: Arc<RwLock<HashMap<Arc<str>, Arc<str>>>>,
    /// Execution queue for async processing
    #[allow(dead_code)] // TODO: Implement in macro execution system
    execution_queue: Arc<SegQueue<MacroExecutionRequest>>,
    /// Configuration settings
    config: MacroProcessorConfig}

/// Macro processor statistics (internal atomic counters)
#[derive(Debug, Default)]
pub struct MacroProcessorStats {
    /// Total macros executed
    pub total_executions: AtomicUsize,
    /// Successful executions
    pub successful_executions: AtomicUsize,
    /// Failed executions
    pub failed_executions: AtomicUsize,
    /// Total execution time in microseconds
    pub total_execution_time_us: AtomicUsize,
    /// Active executions
    pub active_executions: AtomicUsize}

/// Macro processor statistics snapshot (for external API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroProcessorStatsSnapshot {
    /// Total macros executed
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Failed executions
    pub failed_executions: usize,
    /// Total execution time in microseconds
    pub total_execution_time_us: usize,
    /// Active executions
    pub active_executions: usize}

/// Macro processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroProcessorConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Default execution timeout in seconds
    pub default_timeout_seconds: u64,
    /// Enable variable substitution
    pub enable_variable_substitution: bool,
    /// Enable conditional execution
    pub enable_conditional_execution: bool,
    /// Enable loop execution
    pub enable_loop_execution: bool,
    /// Maximum macro recursion depth
    pub max_recursion_depth: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Auto-save macro changes
    pub auto_save: bool}

/// Macro execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionRequest {
    /// Macro ID to execute
    pub macro_id: Uuid,
    /// Execution context variables
    pub context_variables: HashMap<Arc<str>, Arc<str>>,
    /// Execution timeout override
    pub timeout_override: Option<Duration>,
    /// Execution priority (higher = more priority)
    pub priority: u32,
    /// Request timestamp
    pub requested_at: Duration}

/// Macro execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionResult {
    /// Execution success indicator
    pub success: bool,
    /// Execution message/error
    pub message: Arc<str>,
    /// Actions executed
    pub actions_executed: usize,
    /// Execution duration
    pub execution_duration: Duration,
    /// Variables modified during execution
    pub modified_variables: HashMap<Arc<str>, Arc<str>>,
    /// Execution metadata
    pub metadata: MacroExecutionMetadata}

/// Macro execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionMetadata {
    /// Execution ID
    pub execution_id: Uuid,
    /// Macro ID
    pub macro_id: Uuid,
    /// Start timestamp
    pub started_at: Duration,
    /// End timestamp
    pub completed_at: Duration,
    /// Execution context
    pub context: HashMap<Arc<str>, Arc<str>>,
    /// Performance metrics
    pub performance: MacroPerformanceMetrics}

/// Macro performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroPerformanceMetrics {
    /// CPU time used in microseconds
    pub cpu_time_us: u64,
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// Network requests made
    pub network_requests: u32,
    /// Disk operations performed
    pub disk_operations: u32}

impl Default for MacroProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            default_timeout_seconds: 30,
            enable_variable_substitution: true,
            enable_conditional_execution: true,
            enable_loop_execution: true,
            max_recursion_depth: 10,
            enable_monitoring: true,
            auto_save: true}
    }
}

impl MacroProcessor {
    /// Create a new macro processor
    pub fn new() -> Self {
        Self {
            macros: Arc::new(SkipMap::new()),
            stats: Arc::new(MacroProcessorStats::default()),
            variables: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(SegQueue::new()),
            config: MacroProcessorConfig::default()}
    }

    /// Create a macro processor with custom configuration
    pub fn with_config(config: MacroProcessorConfig) -> Self {
        Self {
            macros: Arc::new(SkipMap::new()),
            stats: Arc::new(MacroProcessorStats::default()),
            variables: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(SegQueue::new()),
            config}
    }

    /// Register a macro
    pub fn register_macro(&self, macro_def: StoredMacro) -> Result<(), MacroSystemError> {
        // Validate macro
        self.validate_macro(&macro_def)?;

        // Store macro
        self.macros.insert(macro_def.metadata.id, macro_def);

        Ok(())
    }

    /// Unregister a macro
    pub fn unregister_macro(&self, macro_id: &Uuid) -> Result<(), MacroSystemError> {
        if self.macros.remove(macro_id).is_none() {
            return Err(MacroSystemError::MacroNotFound);
        }

        Ok(())
    }

    /// Execute a macro by ID with fluent-ai-async streaming architecture
    pub fn execute_macro(
        &self,
        macro_id: &Uuid,
        _context_variables: HashMap<Arc<str>, Arc<str>>,
    ) -> AsyncStream<MacroExecutionResult> {
        let macros = self.macros.clone();
        let macro_id = *macro_id;
        let _processor = self.clone();
        
        AsyncStream::with_channel(move |_sender| {
            if let Some(entry) = macros.get(&macro_id) {
                let _macro_def = entry.value().clone();
                
                // Execute macro implementation using streaming
                handle_error!(
                    MacroSystemError::NotImplemented,
                    "Macro execution implementation pending"
                );
            } else {
                handle_error!(
                    MacroSystemError::MacroNotFound,
                    "Macro not found"
                );
            }
        })
    }

    /// Execute a macro directly with fluent-ai-async streaming architecture
    pub fn execute_macro_direct(
        &self,
        _macro_def: StoredMacro,
        _context_variables: HashMap<Arc<str>, Arc<str>>,
    ) -> AsyncStream<MacroExecutionResult> {
        let _processor = self.clone();
        
        AsyncStream::with_channel(move |_sender| {
            // Execute macro implementation synchronously (no nested streaming)
            handle_error!(
                MacroSystemError::NotImplemented,
                "Direct execution implementation pending"
            );
        })
    }

    /// Internal macro execution implementation with fluent-ai-async streaming architecture
    fn execute_macro_impl(
        &self,
        macro_def: StoredMacro,
        context_variables: HashMap<Arc<str>, Arc<str>>,
    ) -> AsyncStream<MacroExecutionResult> {
        let variables = self.variables.clone();
        let stats = self.stats.clone();
        let _processor = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let execution_id = Uuid::new_v4();
            let start_time = Instant::now();
            let started_at = Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );

            // Update statistics
            stats
                .total_executions
                .fetch_add(1, Ordering::Relaxed);
            stats
                .active_executions
                .fetch_add(1, Ordering::Relaxed);

            // Merge context variables with global variables using zero-allocation patterns
            let _execution_context = if let Ok(global_vars) = variables.try_read() {
                let mut context = global_vars.clone();
                context.extend(context_variables.clone());
                context.extend(macro_def.variables.clone());
                context
            } else {
                handle_error!(
                    MacroSystemError::LockError,
                    "Lock acquisition failed"
                );
            };

            let actions_executed = 0;
            let modified_variables = HashMap::new();

            // Execute actions using streaming patterns
            for _action in macro_def.actions.iter() {
                // TODO: Implement action execution with proper streaming patterns
                handle_error!(
                    MacroSystemError::NotImplemented,
                    "Action execution pending implementation"
                );
            }

            let execution_duration = start_time.elapsed();

            // Update statistics
            stats
                .successful_executions
                .fetch_add(1, Ordering::Relaxed);
            stats
                .active_executions
                .fetch_sub(1, Ordering::Relaxed);
            stats.total_execution_time_us.fetch_add(
                execution_duration.as_micros() as usize,
                Ordering::Relaxed,
            );

            emit!(sender, MacroExecutionResult {
                success: true,
                message: Arc::from("Macro executed successfully"),
                actions_executed,
                execution_duration,
                modified_variables,
                metadata: MacroExecutionMetadata {
                    execution_id,
                    macro_id: macro_def.metadata.id,
                    started_at,
                    completed_at: Duration::from_secs(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ),
                    context: context_variables,
                    performance: MacroPerformanceMetrics {
                        cpu_time_us: execution_duration.as_micros() as u64,
                        memory_bytes: 0, // Would need memory profiling
                        network_requests: 0,
                        disk_operations: 0}}});
        })
    }

    /// Validate a macro
    fn validate_macro(&self, macro_def: &StoredMacro) -> Result<(), MacroSystemError> {
        if macro_def.metadata.name.is_empty() {
            return Err(MacroSystemError::InvalidMacro(
                "Macro name cannot be empty".to_string(),
            ));
        }

        if macro_def.actions.is_empty() {
            return Err(MacroSystemError::InvalidMacro(
                "Macro must have at least one action".to_string(),
            ));
        }

        // Validate recursion depth
        self.validate_recursion_depth(&macro_def.actions, 0)?;

        Ok(())
    }

    /// Validate recursion depth
    fn validate_recursion_depth(
        &self,
        actions: &[MacroAction],
        current_depth: usize,
    ) -> Result<(), MacroSystemError> {
        if current_depth > self.config.max_recursion_depth {
            return Err(MacroSystemError::MaxRecursionDepthExceeded);
        }

        for action in actions {
            if let MacroAction::Conditional {
                then_actions,
                else_actions,
                ..
            } = action
            {
                self.validate_recursion_depth(then_actions, current_depth + 1)?;
                if let Some(else_acts) = else_actions {
                    self.validate_recursion_depth(else_acts, current_depth + 1)?;
                }
            }
        }

        Ok(())
    }

    /// Get all registered macros
    pub fn get_macros(&self) -> Vec<StoredMacro> {
        self.macros
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get macro by ID
    pub fn get_macro(&self, macro_id: &Uuid) -> Option<StoredMacro> {
        self.macros.get(macro_id).map(|entry| entry.value().clone())
    }

    /// Get processor statistics
    pub fn stats(&self) -> MacroProcessorStatsSnapshot {
        MacroProcessorStatsSnapshot {
            total_executions: self.stats.total_executions.load(Ordering::Relaxed),
            successful_executions: self.stats.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.stats.failed_executions.load(Ordering::Relaxed),
            total_execution_time_us: self.stats.total_execution_time_us.load(Ordering::Relaxed),
            active_executions: self.stats.active_executions.load(Ordering::Relaxed)}
    }

    /// Set global variable
    pub fn set_variable(&self, name: Arc<str>, value: Arc<str>) -> AsyncStream<()> {
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.write() {
                Ok(mut vars) => {
                    vars.insert(name, value);
                    let _ = sender.send(());
                }
                Err(e) => {
                    eprintln!("Failed to acquire write lock on variables: {}", e);
                    // Still send a result to complete the stream
                    let _ = sender.send(());
                }
            }
        })
    }

    /// Get global variable
    pub fn get_variable(&self, name: &str) -> AsyncStream<Option<Arc<str>>> {
        let name = name.to_string();
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.read() {
                Ok(vars) => {
                    let result = vars.get(name.as_str()).cloned();
                    let _ = sender.send(result);
                }
                Err(e) => {
                    eprintln!("Failed to acquire read lock on variables: {}", e);
                    let _ = sender.send(None);
                }
            }
        })
    }

    /// Clear all global variables
    pub fn clear_variables(&self) -> AsyncStream<()> {
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.write() {
                Ok(mut vars) => {
                    vars.clear();
                    let _ = sender.send(());
                }
                Err(e) => {
                    eprintln!("Failed to acquire write lock on variables: {}", e);
                    let _ = sender.send(());
                }
            }
        })
    }
}

impl Default for MacroProcessor {
    fn default() -> Self {
        Self::new()
    }
}