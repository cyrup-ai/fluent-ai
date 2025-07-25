//! Action processing and command handling for macros
//!
//! This module provides specialized handlers for different types of macro actions,
//! including message sending, command execution, and flow control operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_skiplist::SkipMap;
// AsyncStream, emit, handle_error - removed unused imports
use uuid::Uuid;

use super::types::*;
use crate::types::candle_chat::chat::commands::ImmutableChatCommand;

/// Action handler registry for processing different action types
pub struct ActionHandlerRegistry {
    /// Registered action handlers
    handlers: SkipMap<String, ActionHandler>,
    /// Handler execution statistics
    stats: ActionHandlerStats,
    /// Registry configuration
    config: ActionHandlerConfig}

impl std::fmt::Debug for ActionHandlerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActionHandlerRegistry")
            .field("handlers", &format!("[{} handlers]", self.handlers.len()))
            .field("stats", &self.stats)
            .field("config", &self.config)
            .finish()
    }
}

/// Configuration for action handler behavior
#[derive(Debug, Clone)]
pub struct ActionHandlerConfig {
    /// Enable action validation
    pub enable_validation: bool,
    /// Enable action queuing
    pub enable_queuing: bool,
    /// Maximum action execution time
    pub max_execution_time: Duration,
    /// Enable action logging
    pub enable_logging: bool,
    /// Maximum concurrent actions
    pub max_concurrent_actions: usize}

/// Statistics for action handler performance
#[derive(Debug)]
pub struct ActionHandlerStats {
    /// Total actions processed
    pub total_actions: AtomicUsize,
    /// Successful actions
    pub successful_actions: AtomicUsize,
    /// Failed actions
    pub failed_actions: AtomicUsize,
    /// Average execution time per action type
    pub execution_times: SkipMap<String, AtomicUsize>}

/// Action handler function type
pub type ActionHandler = Box<dyn Fn(&MacroAction, &mut MacroExecutionContext) -> Result<ActionExecutionResult, MacroSystemError> + Send + Sync>;

/// Message action processor for sending messages
#[derive(Debug)]
pub struct MessageActionProcessor {
    /// Configuration for message processing
    config: MessageProcessorConfig,
    /// Message queue for processing
    message_queue: Arc<crossbeam_queue::SegQueue<QueuedMessage>>,
    /// Processing statistics
    stats: MessageProcessorStats}

/// Configuration for message action processing
#[derive(Debug, Clone)]
pub struct MessageProcessorConfig {
    /// Enable message queuing
    pub enable_queuing: bool,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Message send timeout
    pub send_timeout: Duration,
    /// Enable message validation
    pub enable_validation: bool}

/// Statistics for message processing
#[derive(Debug)]
pub struct MessageProcessorStats {
    /// Messages sent
    pub messages_sent: AtomicUsize,
    /// Messages queued
    pub messages_queued: AtomicUsize,
    /// Send failures
    pub send_failures: AtomicUsize,
    /// Average send time
    pub average_send_time: AtomicUsize}

/// Queued message for processing
#[derive(Debug, Clone)]
pub struct QueuedMessage {
    /// Message ID
    pub id: Uuid,
    /// Message content
    pub content: String,
    /// Message type
    pub message_type: String,
    /// Execution context
    pub context_id: Uuid,
    /// Queue timestamp
    pub queued_at: Instant,
    /// Priority level
    pub priority: MessagePriority}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Low priority message
    Low,
    /// Normal priority message
    Normal,
    /// High priority message
    High,
    /// Critical priority message
    Critical}

/// Command action processor for executing commands
#[derive(Debug)]
pub struct CommandActionProcessor {
    /// Configuration for command processing
    config: CommandProcessorConfig,
    /// Command execution history
    history: Arc<SkipMap<Uuid, CommandExecution>>,
    /// Processing statistics
    stats: CommandProcessorStats}

/// Configuration for command action processing
#[derive(Debug, Clone)]
pub struct CommandProcessorConfig {
    /// Enable command validation
    pub enable_validation: bool,
    /// Command execution timeout
    pub execution_timeout: Duration,
    /// Enable command history
    pub enable_history: bool,
    /// Maximum history entries
    pub max_history_entries: usize}

/// Statistics for command processing
#[derive(Debug)]
pub struct CommandProcessorStats {
    /// Commands executed
    pub commands_executed: AtomicUsize,
    /// Command failures
    pub command_failures: AtomicUsize,
    /// Average execution time
    pub average_execution_time: AtomicUsize}

/// Command execution record
#[derive(Debug, Clone)]
pub struct CommandExecution {
    /// Execution ID
    pub id: Uuid,
    /// Command that was executed
    pub command: ImmutableChatCommand,
    /// Execution context
    pub context_id: Uuid,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Execution result
    pub result: Option<CommandExecutionResult>,
    /// Error message if failed
    pub error: Option<String>}

/// Result of command execution
#[derive(Debug, Clone)]
pub enum CommandExecutionResult {
    /// Command executed successfully
    Success(String),
    /// Command failed with error
    Failed(String),
    /// Command timed out
    Timeout,
    /// Command was cancelled
    Cancelled}

/// Variable action processor for variable operations
#[derive(Debug)]
pub struct VariableActionProcessor {
    /// Configuration for variable processing
    config: VariableProcessorConfig,
    /// Variable change history
    history: Arc<SkipMap<String, Vec<VariableChange>>>,
    /// Processing statistics
    stats: VariableProcessorStats}

/// Configuration for variable action processing
#[derive(Debug, Clone)]
pub struct VariableProcessorConfig {
    /// Enable variable validation
    pub enable_validation: bool,
    /// Enable change history
    pub enable_history: bool,
    /// Maximum history per variable
    pub max_history_per_variable: usize,
    /// Enable change notifications
    pub enable_notifications: bool}

/// Statistics for variable processing
#[derive(Debug)]
pub struct VariableProcessorStats {
    /// Variables set
    pub variables_set: AtomicUsize,
    /// Variables retrieved
    pub variables_retrieved: AtomicUsize,
    /// Variable changes
    pub variable_changes: AtomicUsize}

/// Variable change record
#[derive(Debug, Clone)]
pub struct VariableChange {
    /// Change timestamp
    pub timestamp: Instant,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: String,
    /// Context where change occurred
    pub context_id: Uuid}

impl ActionHandlerRegistry {
    /// Create a new action handler registry
    pub fn new() -> Self {
        let mut registry = Self {
            handlers: SkipMap::new(),
            stats: ActionHandlerStats::default(),
            config: ActionHandlerConfig::default()};

        // Register default handlers
        registry.register_default_handlers();
        registry
    }

    /// Create a new action handler registry with custom configuration
    pub fn with_config(config: ActionHandlerConfig) -> Self {
        let mut registry = Self {
            handlers: SkipMap::new(),
            stats: ActionHandlerStats::default(),
            config};

        // Register default handlers
        registry.register_default_handlers();
        registry
    }

    /// Register default action handlers
    fn register_default_handlers(&mut self) {
        // Message action handler
        self.register_handler(
            "send_message".to_string(),
            Box::new(|action, _context| {
                if let MacroAction::SendMessage { content, message_type, .. } = action {
                    // Process message sending
                    println!("Sending message: {} (type: {})", content, message_type);
                    Ok(ActionExecutionResult::Success)
                } else {
                    Err(MacroSystemError::ExecutionError("Invalid action type for message handler".to_string()))
                }
            }),
        );

        // Command action handler
        self.register_handler(
            "execute_command".to_string(),
            Box::new(|action, _context| {
                if let MacroAction::ExecuteCommand { command, .. } = action {
                    // Process command execution
                    println!("Executing command: {:?}", command);
                    Ok(ActionExecutionResult::Success)
                } else {
                    Err(MacroSystemError::ExecutionError("Invalid action type for command handler".to_string()))
                }
            }),
        );

        // Wait action handler
        self.register_handler(
            "wait".to_string(),
            Box::new(|action, _context| {
                if let MacroAction::Wait { duration, .. } = action {
                    Ok(ActionExecutionResult::Wait(*duration))
                } else {
                    Err(MacroSystemError::ExecutionError("Invalid action type for wait handler".to_string()))
                }
            }),
        );

        // Variable action handler
        self.register_handler(
            "set_variable".to_string(),
            Box::new(|action, context| {
                if let MacroAction::SetVariable { name, value, .. } = action {
                    context.variables.insert(name.clone(), value.clone());
                    Ok(ActionExecutionResult::Success)
                } else {
                    Err(MacroSystemError::ExecutionError("Invalid action type for variable handler".to_string()))
                }
            }),
        );
    }

    /// Register a custom action handler
    pub fn register_handler(&mut self, action_type: String, handler: ActionHandler) {
        self.handlers.insert(action_type, handler);
    }

    /// Unregister an action handler
    pub fn unregister_handler(&mut self, action_type: &str) -> bool {
        self.handlers.remove(action_type).is_some()
    }

    /// Process an action using the appropriate handler
    pub fn process_action(
        &self,
        action: &MacroAction,
        context: &mut MacroExecutionContext,
    ) -> Result<ActionExecutionResult, MacroSystemError> {
        let action_type = self.get_action_type(action);
        
        let start_time = Instant::now();
        let result = if let Some(handler_entry) = self.handlers.get(&action_type) {
            let handler = handler_entry.value();
            handler(action, context)
        } else {
            Err(MacroSystemError::ExecutionError(format!("No handler registered for action type: {}", action_type)))
        };

        let execution_time = start_time.elapsed();

        // Update statistics
        self.update_stats(&action_type, &result, execution_time);

        result
    }

    /// Get action type string for handler lookup
    fn get_action_type(&self, action: &MacroAction) -> String {
        match action {
            MacroAction::SendMessage { .. } => "send_message".to_string(),
            MacroAction::ExecuteCommand { .. } => "execute_command".to_string(),
            MacroAction::Wait { .. } => "wait".to_string(),
            MacroAction::SetVariable { .. } => "set_variable".to_string(),
            MacroAction::Conditional { .. } => "conditional".to_string(),
            MacroAction::Loop { .. } => "loop".to_string()}
    }

    /// Update execution statistics
    fn update_stats(
        &self,
        action_type: &str,
        result: &Result<ActionExecutionResult, MacroSystemError>,
        execution_time: Duration,
    ) {
        self.stats.total_actions.fetch_add(1, Ordering::Relaxed);

        match result {
            Ok(_) => {
                self.stats.successful_actions.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats.failed_actions.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update execution time (stored as microseconds)
        let execution_micros = execution_time.as_micros() as usize;
        if let Some(time_entry) = self.stats.execution_times.get(action_type) {
            // Simple moving average (in a real implementation, this would be more sophisticated)
            let current_avg = time_entry.value().load(Ordering::Relaxed);
            let new_avg = (current_avg + execution_micros) / 2;
            time_entry.value().store(new_avg, Ordering::Relaxed);
        } else {
            self.stats.execution_times.insert(action_type.to_string(), AtomicUsize::new(execution_micros));
        }
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_actions".to_string(), self.stats.total_actions.load(Ordering::Relaxed));
        stats.insert("successful_actions".to_string(), self.stats.successful_actions.load(Ordering::Relaxed));
        stats.insert("failed_actions".to_string(), self.stats.failed_actions.load(Ordering::Relaxed));
        stats
    }

    /// List registered handler types
    pub fn list_handlers(&self) -> Vec<String> {
        self.handlers.iter().map(|entry| entry.key().clone()).collect()
    }
}

impl MessageActionProcessor {
    /// Create a new message action processor
    pub fn new() -> Self {
        Self {
            config: MessageProcessorConfig::default(),
            message_queue: Arc::new(crossbeam_queue::SegQueue::new()),
            stats: MessageProcessorStats::default()}
    }

    /// Process a message action
    pub fn process_message(
        &self,
        content: &str,
        message_type: &str,
        context: &MacroExecutionContext,
    ) -> Result<ActionExecutionResult, MacroSystemError> {
        let start_time = Instant::now();

        if self.config.enable_validation {
            self.validate_message(content, message_type)?;
        }

        if self.config.enable_queuing {
            self.queue_message(content, message_type, context)?;
        } else {
            self.send_message_direct(content, message_type)?;
        }

        let execution_time = start_time.elapsed();
        self.update_stats(execution_time);

        Ok(ActionExecutionResult::Success)
    }

    /// Validate message content and type
    fn validate_message(&self, content: &str, message_type: &str) -> Result<(), MacroSystemError> {
        if content.is_empty() {
            return Err(MacroSystemError::ExecutionError("Message content cannot be empty".to_string()));
        }

        if message_type.is_empty() {
            return Err(MacroSystemError::ExecutionError("Message type cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Queue message for processing
    fn queue_message(
        &self,
        content: &str,
        message_type: &str,
        context: &MacroExecutionContext,
    ) -> Result<(), MacroSystemError> {
        let message = QueuedMessage {
            id: Uuid::new_v4(),
            content: content.to_string(),
            message_type: message_type.to_string(),
            context_id: context.execution_id,
            queued_at: Instant::now(),
            priority: MessagePriority::Normal};

        self.message_queue.push(message);
        self.stats.messages_queued.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Send message directly
    fn send_message_direct(&self, content: &str, message_type: &str) -> Result<(), MacroSystemError> {
        // In a real implementation, this would interface with the actual chat system
        println!("Sending message: {} (type: {})", content, message_type);
        self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Update processing statistics
    fn update_stats(&self, execution_time: Duration) {
        let execution_micros = execution_time.as_micros() as usize;
        let current_avg = self.stats.average_send_time.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0 {
            execution_micros
        } else {
            (current_avg + execution_micros) / 2
        };
        self.stats.average_send_time.store(new_avg, Ordering::Relaxed);
    }
}

// Default implementations
impl Default for ActionHandlerConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_queuing: false,
            max_execution_time: Duration::from_secs(30),
            enable_logging: true,
            max_concurrent_actions: 100}
    }
}

impl Default for ActionHandlerStats {
    fn default() -> Self {
        Self {
            total_actions: AtomicUsize::new(0),
            successful_actions: AtomicUsize::new(0),
            failed_actions: AtomicUsize::new(0),
            execution_times: SkipMap::new()}
    }
}

impl Default for MessageProcessorConfig {
    fn default() -> Self {
        Self {
            enable_queuing: true,
            max_queue_size: 1000,
            send_timeout: Duration::from_secs(10),
            enable_validation: true}
    }
}

impl Default for MessageProcessorStats {
    fn default() -> Self {
        Self {
            messages_sent: AtomicUsize::new(0),
            messages_queued: AtomicUsize::new(0),
            send_failures: AtomicUsize::new(0),
            average_send_time: AtomicUsize::new(0)}
    }
}

impl Default for ActionHandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MessageActionProcessor {
    fn default() -> Self {
        Self::new()
    }
}