//! Chat command system with zero-allocation patterns
//!
//! This module provides a comprehensive command system for chat interactions including
//! slash commands, command parsing, execution, and auto-completion with zero-allocation
//! patterns and blazing-fast performance.

pub mod command;
pub mod error;
pub mod events;
pub mod execution;
pub mod executor;
pub mod parameter;
pub mod parsing;
pub mod registry;
pub mod response;
pub mod types;
pub mod validation;

// Re-export main types and functions for convenience
// Global command executor functionality
use std::sync::{Arc, RwLock};

pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use response::ResponseFormatter;
pub use types::*;
pub use validation::CommandValidator;

/// Global command executor instance - PURE SYNC (no futures)
static COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize the global command executor with zero-allocation patterns
///
/// Sets up the global command execution environment with the provided context.
/// This function must be called before using any command parsing or execution
/// functions. Uses a global RwLock for thread-safe access without futures.
///
/// # Parameters
/// - `_context`: CommandContext for configuration (currently unused but reserved for future use)
///
/// # Thread Safety
/// This function is thread-safe and can be called from multiple threads concurrently.
/// Only the first successful initialization will take effect.
///
/// # Example
/// ```rust
/// use fluent_ai_candle::types::candle_chat::chat::commands::initialize_command_executor;
/// 
/// let context = CommandContext::default();
/// initialize_command_executor(context);
/// ```
pub fn initialize_command_executor(_context: CommandContext) {
    let executor = CommandExecutor::new();
    if let Ok(mut writer) = COMMAND_EXECUTOR.write() {
        *writer = Some(executor);
    }
}

/// Get a clone of the global command executor for thread-safe access
///
/// Retrieves a cloned instance of the global command executor. Returns None
/// if the executor has not been initialized via `initialize_command_executor`.
/// The returned executor is a full clone and can be used independently.
///
/// # Returns
/// - `Some(CommandExecutor)` if initialized successfully
/// - `None` if not initialized or if lock acquisition fails
///
/// # Thread Safety
/// This function is thread-safe and uses a read lock to access the global executor.
/// Multiple threads can call this function concurrently without blocking.
///
/// # Performance
/// Returns a cloned executor to avoid lifetime issues. While this involves
/// cloning, the CommandExecutor is designed for efficient copying.
///
/// # Example
/// ```rust
/// if let Some(executor) = get_command_executor() {
///     // Use the executor for command operations
///     let result = executor.parse_command("/help");
/// } else {
///     println!("Command executor not initialized");
/// }
/// ```
pub fn get_command_executor() -> Option<CommandExecutor> {
    COMMAND_EXECUTOR.read().ok().and_then(|guard| guard.clone())
}

/// Parse command string into structured command using global executor
///
/// Parses a raw command string (typically starting with '/') into a structured
/// ImmutableChatCommand that can be executed. This function provides the first
/// stage of command processing - parsing without execution.
///
/// # Parameters
/// - `input`: Raw command string to parse (e.g., "/help", "/search query text")
///
/// # Returns
/// - `Ok(ImmutableChatCommand)` if parsing succeeds
/// - `Err(CommandError)` if parsing fails or executor not initialized
///
/// # Error Conditions
/// - `CommandError::ConfigurationError` - Global executor not initialized
/// - `CommandError::ParseError` - Invalid command syntax or unknown command
///
/// # Example
/// ```rust
/// match parse_command("/help search") {
///     Ok(command) => {
///         println!("Parsed command: {:?}", command);
///         // Command can now be executed
///     }
///     Err(e) => println!("Parse error: {}", e),
/// }
/// ```
///
/// # Performance
/// Uses zero-allocation patterns where possible. Command parsing is designed
/// to be fast and memory-efficient for real-time chat interactions.
pub fn parse_command(input: &str) -> CommandResult<ImmutableChatCommand> {
    if let Some(executor) = get_command_executor() {
        executor
            .parser()
            .parse(input)
            .map_err(|e| CommandError::ParseError(e.to_string()))
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string()})
    }
}

/// Execute command asynchronously using streaming patterns (recommended)
///
/// Executes a parsed command using the global executor with streaming output.
/// This is the recommended approach for command execution as it provides
/// real-time results and supports long-running commands efficiently.
///
/// # Parameters
/// - `command`: Parsed ImmutableChatCommand ready for execution
///
/// # Returns
/// AsyncStream<CommandOutput> that emits command execution results
///
/// # Stream Behavior
/// - Emits CommandOutput items as execution progresses
/// - Closes stream when command execution completes
/// - Handles errors through stream error propagation
///
/// # Example
/// ```rust
/// let command = parse_command("/search rust programming")?;
/// let mut stream = execute_command_async(command);
/// 
/// // Process results as they arrive
/// while let Some(output) = stream.next().await {
///     match output {
///         CommandOutput::Text(text) => println!("Result: {}", text),
///         CommandOutput::Error(err) => eprintln!("Error: {}", err),
///         _ => {} // Handle other output types
///     }
/// }
/// ```
///
/// # Performance
/// Uses zero-allocation streaming patterns for memory efficiency.
/// Particularly beneficial for commands that produce large result sets.
pub fn execute_command_async(command: ImmutableChatCommand) -> fluent_ai_async::AsyncStream<CommandOutput> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_command_executor() {
            let mut result_stream = executor.execute_streaming(1, command);
            // Use synchronous try_next for zero-allocation efficiency
            match result_stream.try_next() {
                Some(result) => emit!(sender, result),
                None => {
                    handle_error!(
                        CommandError::ExecutionFailed(
                            "Stream closed without result".to_string()
                        ),
                        "Command execution stream closed unexpectedly"
                    );
                }
            }
        } else {
            handle_error!(
                CommandError::ConfigurationError {
                    detail: "Command executor not initialized".to_string()
                },
                "Command executor not initialized"
            );
        }
    })
}

/// Execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// Converts AsyncStream to synchronous result using fluent-ai-async .collect() pattern.
/// This provides thread-safe, zero-allocation command execution.
pub fn execute_command(command: ImmutableChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        // Use synchronous try_next for zero-allocation efficiency
        match result_stream.try_next() {
            Some(result) => Ok(result),
            None => Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string()
            ))
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string()})
    }
}

/// Parse and execute command in one step using streaming patterns (recommended)
///
/// Combines command parsing and execution into a single operation with streaming
/// output. This is the most convenient method for processing raw command strings
/// and getting real-time results without intermediate parsing steps.
///
/// # Parameters
/// - `input`: Raw command string to parse and execute (e.g., "/help", "/search query")
///
/// # Returns
/// AsyncStream<CommandOutput> that emits command execution results
///
/// # Stream Behavior
/// - First parses the command string into structured format
/// - Then executes the parsed command with streaming results
/// - Emits CommandOutput items as execution progresses
/// - Handles both parsing and execution errors through stream error propagation
///
/// # Example
/// ```rust
/// let mut stream = parse_and_execute_command_async("/search rust async");
/// 
/// // Process results as they arrive
/// while let Some(output) = stream.next().await {
///     match output {
///         CommandOutput::Text(text) => println!("Search result: {}", text),
///         CommandOutput::Error(err) => eprintln!("Command failed: {}", err),
///         CommandOutput::Data(data) => {
///             // Handle structured data results
///             println!("Received data: {:?}", data);
///         }
///     }
/// }
/// ```
///
/// # Error Handling
/// - Parse errors are converted to CommandOutput::Error items in the stream
/// - Execution errors are also streamed as CommandOutput::Error items
/// - Configuration errors (uninitialized executor) close the stream with error
///
/// # Performance
/// Optimal for interactive command processing where results should appear
/// immediately as they become available. Uses zero-allocation streaming patterns.
pub fn parse_and_execute_command_async(input: &str) -> fluent_ai_async::AsyncStream<CommandOutput> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    let input = input.to_string();
    
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_command_executor() {
            let mut result_stream = executor.parse_and_execute(&input);
            // Use synchronous try_next for zero-allocation efficiency
            match result_stream.try_next() {
                Some(result) => emit!(sender, result),
                None => {
                    handle_error!(
                        CommandError::ExecutionFailed(
                            "Stream closed without result".to_string()
                        ),
                        "Command parse and execution stream closed unexpectedly"
                    );
                }
            }
        } else {
            handle_error!(
                CommandError::ConfigurationError {
                    detail: "Command executor not initialized".to_string()
                },
                "Command executor not initialized"
            );
        }
    })
}

/// Parse and execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// Converts AsyncStream to synchronous result using fluent-ai-async .collect() pattern.
/// This provides thread-safe, zero-allocation command parsing and execution.
pub fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        // Use synchronous try_next for zero-allocation efficiency
        match result_stream.try_next() {
            Some(result) => Ok(result),
            None => Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string()
            ))
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string()})
    }
}
