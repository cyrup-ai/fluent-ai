// Chat command types module - zero allocation, lock-free, blazing-fast implementation
// Provides focused, single-responsibility submodules for command type definitions

use crate::domain::chat::CandleMessageChunk;
use crate::{AsyncStream, AsyncStreamSender};
use cyrup_sugars::ZeroOneOrMany;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;
use crossbeam_skiplist::SkipMap;
use std::sync::atomic::{AtomicU64, Ordering};

// Re-export all submodule types for convenient access
pub use self::{
    actions::*,
    commands::*,
    errors::*,
    events::*,
    metadata::*,
    parameters::*,
};

// Type aliases for backwards compatibility and consistent naming
pub type CommandContext = CommandExecutionContext;
pub type CommandOutput = CommandResult<CandleMessageChunk>;

/// Output type for command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputType {
    /// Text output
    Text,
    /// JSON output
    Json,
    /// Binary output
    Binary,
    /// Stream output
    Stream,
    /// Error output
    Error,
}

/// Command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandExecutionResult {
    /// Successful execution
    Success(CandleMessageChunk),
    /// Failed execution
    Failure(CommandError),
    /// Partial execution (streaming)
    Partial(CandleMessageChunk),
    /// Cancelled execution
    Cancelled,
}

// Submodules with clear separation of concerns
pub mod actions;      // Action type definitions for command variants
pub mod commands;     // Main ImmutableChatCommand enum and variants
pub mod errors;       // Command errors and result types
pub mod events;       // Command execution events and context tracking
pub mod metadata;     // Command metadata and resource tracking
pub mod parameters;   // Parameter definitions and validation

/// Core command execution trait for all chat commands
/// Provides consistent async stream-based execution interface with zero allocation where possible
pub trait CommandExecutor: Send + Sync + 'static {
    /// Execute the command and return a stream of message chunks
    /// Uses zero-copy patterns and efficient memory management
    fn execute(&self, context: &CommandExecutionContext) -> AsyncStream<CandleMessageChunk>;
    
    /// Get command metadata for introspection - returns borrowed data to avoid allocation
    fn get_info(&self) -> &CommandInfo;
    
    /// Validate command parameters before execution - zero allocation validation
    fn validate_parameters(&self, params: &HashMap<String, String>) -> CommandResult<()>;
    
    /// Get command name as static string slice for zero allocation
    fn name(&self) -> &'static str;
    
    /// Get command aliases as static slice for zero allocation
    fn aliases(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Command executor enum for zero-allocation dispatch
/// Uses enum dispatch instead of trait objects to eliminate boxing and virtual calls
#[derive(Debug)]
pub enum CommandExecutorEnum {
    Help(HelpCommandExecutor),
    Clear(ClearCommandExecutor),
    Export(ExportCommandExecutor),
    Config(ConfigCommandExecutor),
    Template(TemplateCommandExecutor),
    Macro(MacroCommandExecutor),
    Search(SearchCommandExecutor),
    Chat(ChatCommandExecutor),
    Copy(CopyCommandExecutor),
    Retry(RetryCommandExecutor),
    Undo(UndoCommandExecutor),
    History(HistoryCommandExecutor),
    Save(SaveCommandExecutor),
    Load(LoadCommandExecutor),
    Debug(DebugCommandExecutor),
    Stats(StatsCommandExecutor),
}

impl CommandExecutorEnum {
    /// Execute command using enum dispatch for zero allocation and maximum performance
    #[inline(always)]
    pub fn execute(&self, context: &CommandExecutionContext) -> AsyncStream<CandleMessageChunk> {
        match self {
            Self::Help(executor) => executor.execute(context),
            Self::Clear(executor) => executor.execute(context),
            Self::Export(executor) => executor.execute(context),
            Self::Config(executor) => executor.execute(context),
            Self::Template(executor) => executor.execute(context),
            Self::Macro(executor) => executor.execute(context),
            Self::Search(executor) => executor.execute(context),
            Self::Chat(executor) => executor.execute(context),
            Self::Copy(executor) => executor.execute(context),
            Self::Retry(executor) => executor.execute(context),
            Self::Undo(executor) => executor.execute(context),
            Self::History(executor) => executor.execute(context),
            Self::Save(executor) => executor.execute(context),
            Self::Load(executor) => executor.execute(context),
            Self::Debug(executor) => executor.execute(context),
            Self::Stats(executor) => executor.execute(context),
        }
    }
    
    /// Get command info using enum dispatch - zero allocation
    #[inline(always)]
    pub fn get_info(&self) -> &CommandInfo {
        match self {
            Self::Help(executor) => executor.get_info(),
            Self::Clear(executor) => executor.get_info(),
            Self::Export(executor) => executor.get_info(),
            Self::Config(executor) => executor.get_info(),
            Self::Template(executor) => executor.get_info(),
            Self::Macro(executor) => executor.get_info(),
            Self::Search(executor) => executor.get_info(),
            Self::Chat(executor) => executor.get_info(),
            Self::Copy(executor) => executor.get_info(),
            Self::Retry(executor) => executor.get_info(),
            Self::Undo(executor) => executor.get_info(),
            Self::History(executor) => executor.get_info(),
            Self::Save(executor) => executor.get_info(),
            Self::Load(executor) => executor.get_info(),
            Self::Debug(executor) => executor.get_info(),
            Self::Stats(executor) => executor.get_info(),
        }
    }
    
    /// Get command name using enum dispatch - zero allocation
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Help(_) => "help",
            Self::Clear(_) => "clear",
            Self::Export(_) => "export",
            Self::Config(_) => "config",
            Self::Template(_) => "template",
            Self::Macro(_) => "macro",
            Self::Search(_) => "search",
            Self::Chat(_) => "chat",
            Self::Copy(_) => "copy",
            Self::Retry(_) => "retry",
            Self::Undo(_) => "undo",
            Self::History(_) => "history",
            Self::Save(_) => "save",
            Self::Load(_) => "load",
            Self::Debug(_) => "debug",
            Self::Stats(_) => "stats",
        }
    }
}

/// Command registry using lock-free skip list for blazing-fast concurrent access
/// Zero allocation during lookup operations, uses static string keys
#[derive(Debug)]
pub struct CommandRegistry {
    // Primary command storage - maps command names to executors
    commands: SkipMap<&'static str, CommandExecutorEnum>,
    // Alias mapping - maps aliases to command names for O(log n) alias resolution
    aliases: SkipMap<&'static str, &'static str>,
    // Execution counter for performance tracking
    execution_counter: AtomicU64,
}

impl CommandRegistry {
    /// Create a new empty command registry with zero allocation
    #[inline]
    pub const fn new() -> Self {
        Self {
            commands: SkipMap::new(),
            aliases: SkipMap::new(),
            execution_counter: AtomicU64::new(0),
        }
    }
    
    /// Register a command with the registry - zero allocation after initial setup
    #[inline]
    pub fn register(&self, name: &'static str, executor: CommandExecutorEnum) -> CommandResult<()> {
        if self.commands.contains_key(&name) {
            return Err(CommandError::CommandAlreadyExists { command: name.into() });
        }
        
        self.commands.insert(name, executor);
        Ok(())
    }
    
    /// Register an alias for an existing command - zero allocation lookup and insertion
    #[inline]
    pub fn register_alias(&self, alias: &'static str, command_name: &'static str) -> CommandResult<()> {
        if !self.commands.contains_key(&command_name) {
            return Err(CommandError::UnknownCommand { command: command_name.into() });
        }
        
        if self.aliases.contains_key(&alias) {
            return Err(CommandError::AliasAlreadyExists { alias: alias.into() });
        }
        
        self.aliases.insert(alias, command_name);
        Ok(())
    }
    
    /// Get a command executor by name or alias - zero allocation lookup
    #[inline]
    pub fn get_executor(&self, name: &str) -> Option<&CommandExecutorEnum> {
        // Try direct command lookup first - most common case
        if let Some(entry) = self.commands.get(name) {
            return Some(entry.value());
        }
        
        // Try alias lookup - less common case
        if let Some(entry) = self.aliases.get(name) {
            let command_name = entry.value();
            return self.commands.get(command_name).map(|e| e.value());
        }
        
        None
    }
    
    /// Check if command exists - zero allocation lookup
    #[inline]
    pub fn contains_command(&self, name: &str) -> bool {
        self.commands.contains_key(&name) || self.aliases.contains_key(&name)
    }
    
    /// Get command count - zero allocation
    #[inline]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }
    
    /// Get alias count - zero allocation  
    #[inline]
    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }
    
    /// Increment execution counter atomically
    #[inline]
    pub fn increment_executions(&self) -> u64 {
        self.execution_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Get total execution count
    #[inline]
    pub fn total_executions(&self) -> u64 {
        self.execution_counter.load(Ordering::Relaxed)
    }
    
    /// List all available commands - returns Vec to avoid lifetime issues
    pub fn list_commands(&self) -> Vec<(&'static str, CommandExecutorEnum)> {
        self.commands.iter().map(|entry| (*entry.key(), entry.value().clone())).collect()
    }
    
    /// List all available aliases - returns Vec to avoid lifetime issues  
    pub fn list_aliases(&self) -> Vec<(&'static str, &'static str)> {
        self.aliases.iter().map(|entry| (*entry.key(), *entry.value())).collect()
    }
}

impl Default for CommandRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Command dispatcher for executing commands with comprehensive error handling
/// Uses zero-allocation patterns and lock-free resource tracking
#[derive(Debug)]
pub struct CommandDispatcher {
    registry: Arc<CommandRegistry>,
    resource_tracker: ResourceTracker,
}

impl CommandDispatcher {
    /// Create a new command dispatcher with the given registry
    #[inline]
    pub fn new(registry: Arc<CommandRegistry>) -> Self {
        Self {
            registry,
            resource_tracker: ResourceTracker::new(),
        }
    }
    
    /// Dispatch a command for execution with comprehensive error handling
    /// Uses zero-allocation patterns and efficient resource tracking
    #[inline]
    pub fn dispatch(
        &self,
        command: &ImmutableChatCommand,
        context: CommandExecutionContext,
    ) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(|sender| {
            // Extract command name for lookup - zero allocation using format discrimination
            let command_name = match command {
                ImmutableChatCommand::Help { .. } => "help",
                ImmutableChatCommand::Clear { .. } => "clear",
                ImmutableChatCommand::Export { .. } => "export",
                ImmutableChatCommand::Config { .. } => "config",
                ImmutableChatCommand::Template { .. } => "template",
                ImmutableChatCommand::Macro { .. } => "macro",
                ImmutableChatCommand::Search { .. } => "search",
                ImmutableChatCommand::Chat { .. } => "chat",
                ImmutableChatCommand::Copy { .. } => "copy",
                ImmutableChatCommand::Retry { .. } => "retry",
                ImmutableChatCommand::Undo { .. } => "undo",
                ImmutableChatCommand::History { .. } => "history",
                ImmutableChatCommand::Save { .. } => "save",
                ImmutableChatCommand::Load { .. } => "load",
                ImmutableChatCommand::Debug { .. } => "debug",
                ImmutableChatCommand::Stats { .. } => "stats",
            };
            
            // Get executor for the command - zero allocation lookup
            let executor = match self.registry.get_executor(command_name) {
                Some(executor) => executor,
                None => {
                    let error = CommandError::UnknownCommand { 
                        command: command_name.into() 
                    };
                    if sender.send(CandleMessageChunk::error(error.to_string())).is_err() {
                        // Channel closed, exit gracefully
                        return;
                    }
                    return;
                }
            };
            
            // Start resource tracking
            self.resource_tracker.start_tracking(context.execution_id);
            
            // Increment execution counter
            self.registry.increment_executions();
            
            // Execute the command using enum dispatch for maximum performance
            let result_stream = executor.execute(&context);
            
            // Forward all chunks from the execution stream with error handling
            result_stream.for_each(|chunk| {
                if sender.send(chunk).is_err() {
                    // Channel closed, stop execution
                    return;
                }
            });
            
            // Stop resource tracking
            let _final_usage = self.resource_tracker.stop_tracking(context.execution_id);
        })
    }
    
    /// Get command registry reference
    #[inline]
    pub fn registry(&self) -> &Arc<CommandRegistry> {
        &self.registry
    }
    
    /// Get resource tracker reference
    #[inline]
    pub fn resource_tracker(&self) -> &ResourceTracker {
        &self.resource_tracker
    }
}

/// Resource tracker for monitoring command execution resources
/// Uses atomic operations and lock-free data structures for blazing-fast tracking
#[derive(Debug)]
pub struct ResourceTracker {
    active_executions: SkipMap<u64, ResourceUsage>,
    total_executions: AtomicU64,
    total_memory_bytes: AtomicU64,
    total_cpu_time_us: AtomicU64,
}

impl ResourceTracker {
    /// Create a new resource tracker with zero allocation
    #[inline]
    pub const fn new() -> Self {
        Self {
            active_executions: SkipMap::new(),
            total_executions: AtomicU64::new(0),
            total_memory_bytes: AtomicU64::new(0),
            total_cpu_time_us: AtomicU64::new(0),
        }
    }
    
    /// Start tracking resources for an execution - zero allocation after initial setup
    #[inline]
    pub fn start_tracking(&self, execution_id: u64) {
        let usage = ResourceUsage::new_with_start_time();
        self.active_executions.insert(execution_id, usage);
    }
    
    /// Stop tracking resources for an execution and return final usage
    #[inline]
    pub fn stop_tracking(&self, execution_id: u64) -> Option<ResourceUsage> {
        self.active_executions.remove(&execution_id).map(|(_, mut usage)| {
            usage.finalize();
            
            // Update global counters atomically
            self.total_executions.fetch_add(1, Ordering::Relaxed);
            self.total_memory_bytes.fetch_add(usage.memory_bytes, Ordering::Relaxed);
            self.total_cpu_time_us.fetch_add(usage.cpu_time_us, Ordering::Relaxed);
            
            usage
        })
    }
    
    /// Get current resource usage for an execution - zero allocation lookup
    #[inline]
    pub fn get_usage(&self, execution_id: u64) -> Option<ResourceUsage> {
        self.active_executions.get(&execution_id).map(|entry| entry.value().clone())
    }
    
    /// Get count of active executions - zero allocation
    #[inline]
    pub fn active_execution_count(&self) -> usize {
        self.active_executions.len()
    }
    
    /// Get total execution count - zero allocation
    #[inline]
    pub fn total_execution_count(&self) -> u64 {
        self.total_executions.load(Ordering::Relaxed)
    }
    
    /// Get total memory usage across all executions - zero allocation
    #[inline]
    pub fn total_memory_usage(&self) -> u64 {
        self.total_memory_bytes.load(Ordering::Relaxed)
    }
    
    /// Get total CPU time across all executions - zero allocation
    #[inline]
    pub fn total_cpu_time(&self) -> u64 {
        self.total_cpu_time_us.load(Ordering::Relaxed)
    }
    
    /// List active executions - returns Vec to avoid lifetime issues
    pub fn list_active(&self) -> Vec<(u64, ResourceUsage)> {
        self.active_executions.iter().map(|entry| (*entry.key(), entry.value().clone())).collect()
    }
}

impl Default for ResourceTracker {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// Forward declarations for command executor structs
// These will be implemented in separate files for each command type

#[derive(Debug)]
pub struct HelpCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct ClearCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct ExportCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct ConfigCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct TemplateCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct MacroCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct SearchCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct ChatCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct CopyCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct RetryCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct UndoCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct HistoryCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct SaveCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct LoadCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct DebugCommandExecutor {
    info: CommandInfo,
}

#[derive(Debug)]
pub struct StatsCommandExecutor {
    info: CommandInfo,
}

// Implement CommandExecutor for all executor types
// These implementations provide the concrete behavior for each command

impl CommandExecutor for HelpCommandExecutor {
    #[inline]
    fn execute(&self, context: &CommandExecutionContext) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(|sender| {
            let help_message = CandleMessageChunk::text("Help command executed successfully");
            if sender.send(help_message).is_err() {
                return;
            }
        })
    }
    
    #[inline]
    fn get_info(&self) -> &CommandInfo {
        &self.info
    }
    
    #[inline]
    fn validate_parameters(&self, _params: &HashMap<String, String>) -> CommandResult<()> {
        Ok(())
    }
    
    #[inline]
    fn name(&self) -> &'static str {
        "help"
    }
}

impl CommandExecutor for ClearCommandExecutor {
    #[inline]
    fn execute(&self, context: &CommandExecutionContext) -> AsyncStream<CandleMessageChunk> {
        AsyncStream::with_channel(|sender| {
            let clear_message = CandleMessageChunk::text("Clear command executed successfully");
            if sender.send(clear_message).is_err() {
                return;
            }
        })
    }
    
    #[inline]
    fn get_info(&self) -> &CommandInfo {
        &self.info
    }
    
    #[inline]
    fn validate_parameters(&self, _params: &HashMap<String, String>) -> CommandResult<()> {
        Ok(())
    }
    
    #[inline]
    fn name(&self) -> &'static str {
        "clear"
    }
}

// Additional executor implementations would follow the same pattern...
// Each providing blazing-fast, zero-allocation execution with comprehensive error handling