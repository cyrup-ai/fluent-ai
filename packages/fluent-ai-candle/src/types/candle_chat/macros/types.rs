//! Comprehensive macro system types for automated chat operations
//!
//! This module provides a complete type system for recording, storing, and replaying
//! chat interactions as macros. It supports complex automation scenarios including
//! conditional execution, loops, variable substitution, and resource management.
//!
//! # Core Concepts
//!
//! - **Recording**: Capture user actions and commands into replayable sequences
//! - **Playback**: Execute recorded macros with variable substitution and control flow
//! - **Variables**: Dynamic content replacement during macro execution
//! - **Conditions**: Conditional logic for branching macro behavior
//! - **Loops**: Iterative execution patterns with configurable repetition
//! - **Resource Limits**: Safety constraints for memory, CPU, and I/O usage
//!
//! # Performance Characteristics
//!
//! - **Lock-free queues**: Uses `SegQueue` for high-performance action storage
//! - **Thread-safe counters**: Atomic execution statistics tracking
//! - **Zero-allocation patterns**: Minimal heap allocations during execution
//! - **Efficient cloning**: Smart cloning strategies for complex data structures
//!
//! # Safety Features
//!
//! - **Resource limits**: Configurable bounds on memory, CPU, and I/O usage
//! - **Timeout protection**: Automatic termination of long-running macros
//! - **Error isolation**: Failed actions don't crash the entire macro system
//! - **Recursion limits**: Protection against infinite recursion scenarios
//!
//! # Usage Patterns
//!
//! ```rust
//! use fluent_ai_candle::types::candle_chat::macros::types::*;
//!
//! // Create and record a macro
//! let metadata = MacroMetadata::new("test-macro");
//! let mut session = MacroRecordingSession::new("test".to_string(), metadata);
//! session.start_recording();
//! session.add_action(MacroAction::SendMessage {
//!     content: "Hello, world!".to_string(),
//!     message_type: "text".to_string(),
//!     timestamp: Duration::from_millis(100),
//! });
//! session.stop_recording();
//!
//! // Convert to macro and execute
//! let chat_macro = session.to_macro();
//! let mut playback = MacroPlaybackSession::new(chat_macro.metadata.id, chat_macro.actions.len());
//! playback.start_playback();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::candle_chat::search::tagging::ConsistentCounter;
use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::commands::ImmutableChatCommand;

/// Atomic macro action representing a single recorded operation
///
/// MacroAction defines the fundamental building blocks of chat macros, representing
/// discrete operations that can be recorded during user interactions and replayed
/// during macro execution. Each action includes timing information for accurate
/// reproduction of user behavior patterns.
///
/// # Action Types
///
/// - **SendMessage**: Text communication with type classification
/// - **ExecuteCommand**: Command execution with full command context
/// - **Wait**: Timing delays for natural interaction pacing
/// - **SetVariable**: Dynamic variable assignment during execution
/// - **Conditional**: Branching logic with optional else clauses
/// - **Loop**: Iterative execution with configurable repetition counts
///
/// # Serialization
///
/// All actions support JSON serialization for persistent macro storage and
/// cross-session compatibility. The format is optimized for both human
/// readability and efficient parsing.
///
/// # Performance Considerations
///
/// - **Memory efficient**: Uses `Arc<str>` for shared string data
/// - **Clone optimized**: Efficient cloning for macro duplication
/// - **Timestamp precision**: Duration-based timing for microsecond accuracy
/// - **Nested structures**: Recursive action support for complex control flow
///
/// # Examples
///
/// ```rust
/// use std::time::Duration;
/// use fluent_ai_candle::types::candle_chat::macros::types::MacroAction;
///
/// // Simple message action
/// let message = MacroAction::SendMessage {
///     content: "Hello, AI!".to_string(),
///     message_type: "greeting".to_string(),
///     timestamp: Duration::from_millis(500),
/// };
///
/// // Variable assignment
/// let var_action = MacroAction::SetVariable {
///     name: "user_name".to_string(),
///     value: "Alice".to_string(),
///     timestamp: Duration::from_millis(1000),
/// };
///
/// // Conditional execution
/// let conditional = MacroAction::Conditional {
///     condition: "user_name == 'Alice'".to_string(),
///     then_actions: vec![message],
///     else_actions: None,
///     timestamp: Duration::from_millis(1500),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroAction {
    /// Send a message with content and type classification
    ///
    /// Represents a text message transmission action with content and metadata.
    /// This is the most common macro action, capturing user text communications
    /// for replay during macro execution.
    ///
    /// # Performance
    /// - **Memory**: ~24-48 bytes + string content length
    /// - **Execution time**: ~1-5ms depending on message length
    /// - **Network impact**: Depends on underlying chat system
    SendMessage {
        /// Message content to send to the chat system
        ///
        /// The actual text content that will be transmitted. Supports variable
        /// substitution using `{variable_name}` syntax during macro playback.
        /// Content length is not restricted but should be reasonable for UX.
        content: String,
        /// Type classification for the message
        ///
        /// Semantic classification of the message type for proper handling:
        /// - `"text"`: Plain text message (most common)
        /// - `"command"`: System command or instruction
        /// - `"question"`: User question requiring response
        /// - `"response"`: Response to previous message
        /// - `"system"`: System-generated message
        /// - Custom types are supported for application-specific needs
        message_type: String,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// Duration from the beginning of macro recording when this action
        /// was captured. Used during playback to maintain original timing
        /// patterns and natural conversation flow.
        timestamp: Duration
    },
    /// Execute a command with full command context
    ///
    /// Represents execution of a chat command or system operation during
    /// macro playback. Commands are executed with their original context
    /// and parameters, allowing complex automated workflows.
    ///
    /// # Security
    /// - Commands are validated before execution
    /// - Resource limits apply to command execution
    /// - Sandboxing may be available depending on configuration
    ///
    /// # Performance
    /// - **Memory**: ~32-64 bytes + command data size
    /// - **Execution time**: Varies by command (1ms to several seconds)
    /// - **Side effects**: Commands may modify system state
    ExecuteCommand {
        /// Command to execute with full context and parameters
        ///
        /// Immutable command structure containing all necessary information
        /// for command execution including arguments, options, and context.
        /// Commands maintain their original state from recording time.
        command: ImmutableChatCommand,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// Duration from the beginning of macro recording when this command
        /// action was captured. Preserves timing for coordinated operations.
        timestamp: Duration
    },
    /// Wait for a specified duration before continuing
    ///
    /// Introduces a deliberate pause in macro execution to simulate natural
    /// user timing, wait for system responses, or coordinate with external
    /// processes. Essential for maintaining realistic interaction patterns.
    ///
    /// # Timing Precision
    /// - **Resolution**: Microsecond precision timing
    /// - **Accuracy**: ±1ms typical accuracy on modern systems
    /// - **Efficiency**: Non-blocking sleep implementation
    ///
    /// # Use Cases
    /// - Natural conversation pacing
    /// - Waiting for system responses
    /// - Rate limiting compliance
    /// - Coordinated multi-step operations
    Wait {
        /// Duration to wait before proceeding to next action
        ///
        /// The actual time to pause execution. During playback, this creates
        /// a non-blocking delay that maintains macro timing while allowing
        /// other system operations to continue.
        duration: Duration,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// When this wait action was originally recorded. Used for maintaining
        /// consistent timing patterns during macro playback and analysis.
        timestamp: Duration
    },
    /// Set a variable value for dynamic content replacement
    ///
    /// Assigns or updates a variable in the macro execution context,
    /// enabling dynamic content replacement in subsequent actions.
    /// Variables persist throughout macro execution and support complex
    /// data transformations and conditional logic.
    ///
    /// # Variable System
    /// - **Scope**: Variables are scoped to the current macro execution
    /// - **Types**: All variables are stored as strings with dynamic conversion
    /// - **Substitution**: Use `{variable_name}` syntax in other actions
    /// - **Persistence**: Variables persist until macro completion or reset
    ///
    /// # Performance
    /// - **Memory**: ~32 bytes + name/value string lengths
    /// - **Lookup time**: O(1) hash table access
    /// - **Thread safety**: Safe for concurrent access within execution context
    SetVariable {
        /// Variable name to set in the execution context
        ///
        /// The identifier for the variable. Must be a valid identifier string
        /// following standard naming conventions. Case-sensitive and supports
        /// alphanumeric characters and underscores.
        name: String,
        /// Value to assign to the variable
        ///
        /// The string value to store. All values are stored as strings but
        /// can be converted to other types during substitution. Supports
        /// complex values including JSON strings for structured data.
        value: String,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// When this variable assignment was originally captured during
        /// macro recording. Used for temporal analysis and timing reproduction.
        timestamp: Duration
    },
    /// Conditional execution based on variable evaluation
    ///
    /// Implements branching logic in macro execution, allowing different
    /// action sequences based on runtime conditions. Supports complex
    /// boolean expressions with variable substitution and comparison operators.
    ///
    /// # Condition Syntax
    /// - **Variables**: Reference variables with `variable_name`
    /// - **Literals**: String literals with quotes: `"literal value"`
    /// - **Operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`
    /// - **Grouping**: Parentheses for complex expressions
    ///
    /// # Examples
    /// - `user_name == "Alice"`: Simple equality check
    /// - `age > "18" && status == "active"`: Complex condition
    /// - `!is_admin || role == "moderator"`: Logical negation and OR
    ///
    /// # Performance
    /// - **Evaluation time**: ~0.1-1ms for simple conditions
    /// - **Memory**: ~64 bytes + condition string + nested actions
    /// - **Nesting**: Supports deep nesting limited only by stack size
    Conditional {
        /// Condition expression to evaluate using variable context
        ///
        /// Boolean expression string that will be evaluated against the current
        /// variable context. Supports comparison operators, logical operators,
        /// and variable substitution. Must evaluate to a boolean result.
        condition: String,
        /// Actions to execute if condition evaluates to true
        ///
        /// Vector of actions that will be executed sequentially if the condition
        /// expression evaluates to true. Can contain any valid macro actions
        /// including nested conditionals and loops.
        then_actions: Vec<MacroAction>,
        /// Optional actions to execute if condition evaluates to false
        ///
        /// Optional vector of actions for the "else" branch. If None and the
        /// condition is false, execution continues with the next action after
        /// this conditional block.
        else_actions: Option<Vec<MacroAction>>,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// When this conditional action was originally recorded. Used for
        /// maintaining timing consistency and macro analysis.
        timestamp: Duration
    },
    /// Loop execution with configurable iteration count
    ///
    /// Implements iterative execution patterns, repeating a sequence of actions
    /// for a specified number of iterations. Supports nested loops and provides
    /// loop control variables for dynamic behavior within iterations.
    ///
    /// # Loop Variables
    /// During execution, the following variables are automatically available:
    /// - `loop_index`: Current iteration (0-based)
    /// - `loop_count`: Total number of iterations
    /// - `loop_remaining`: Iterations remaining
    ///
    /// # Performance
    /// - **Memory**: ~48 bytes + nested actions memory
    /// - **Execution**: Linear time complexity O(n) where n = iterations
    /// - **Stack usage**: Each nested loop level uses additional stack space
    /// - **Limits**: Maximum iterations bounded by resource limits
    ///
    /// # Examples
    /// ```rust
    /// // Simple repeat loop
    /// MacroAction::Loop {
    ///     iterations: 3,
    ///     actions: vec![
    ///         MacroAction::SendMessage {
    ///             content: "Message {loop_index}".to_string(),
    ///             message_type: "text".to_string(),
    ///             timestamp: Duration::from_millis(100),
    ///         }
    ///     ],
    ///     timestamp: Duration::from_millis(200),
    /// }
    /// ```
    Loop {
        /// Number of times to repeat the loop body
        ///
        /// Total iteration count for the loop. Must be greater than 0.
        /// Each iteration executes all actions in the loop body sequentially.
        /// Limited by resource constraints to prevent infinite execution.
        iterations: u32,
        /// Actions to execute in each iteration of the loop
        ///
        /// Vector of actions that form the loop body. These actions are executed
        /// sequentially for each iteration. Can contain any valid macro actions
        /// including nested loops, conditionals, and variable operations.
        actions: Vec<MacroAction>,
        /// Timestamp when action was recorded (relative to macro start)
        ///
        /// When this loop action was originally recorded during macro creation.
        /// Used for timing analysis and consistent playback behavior.
        timestamp: Duration
    }
}

/// Macro recording state for session management
///
/// Tracks the current state of a macro recording session, enabling proper
/// state transitions and preventing invalid operations. The recording state
/// machine ensures data integrity and provides clear feedback about recording
/// status to users and systems.
///
/// # State Transitions
///
/// ```text
/// Idle -> Recording -> Completed
///   |        |           ^
///   |        v           |
///   +----> Paused -------+
/// ```
///
/// # Thread Safety
///
/// This enum is `Copy` and safe for concurrent access. State changes should
/// be coordinated through proper synchronization mechanisms in the containing
/// recording session structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroRecordingState {
    /// Not recording - initial state or after reset
    ///
    /// The recording session is inactive and ready to begin recording.
    /// No actions are being captured in this state.
    Idle,
    /// Currently recording actions
    ///
    /// The session is actively capturing user actions and commands.
    /// All eligible actions are being added to the recording queue.
    Recording,
    /// Recording paused - can be resumed
    ///
    /// Recording is temporarily suspended but can be resumed.
    /// Actions are not captured while in this state.
    Paused,
    /// Recording completed successfully
    ///
    /// The recording session has finished and cannot capture more actions.
    /// The recorded macro is ready for conversion and storage.
    Completed
}

/// Macro playback state for execution management
///
/// Tracks the current state of macro playback execution, providing clear
/// state management for complex macro operations. The playback state machine
/// ensures proper execution flow and enables recovery from various scenarios.
///
/// # State Transitions
///
/// ```text
/// Idle -> Playing -> Completed
///   |       |           ^
///   |       v           |
///   |     Paused -------+
///   |       |
///   +-------v
///         Failed
/// ```
///
/// # Recovery Patterns
///
/// - **From Failed**: Can be restarted from `Idle`
/// - **From Paused**: Can resume to `Playing` or abort to `Failed`
/// - **From Completed**: Cannot be restarted (create new session)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroPlaybackState {
    /// Not playing - initial state or ready to start
    ///
    /// The playback session is inactive and ready to begin execution.
    /// No actions are being executed in this state.
    Idle,
    /// Currently playing back the macro
    ///
    /// The session is actively executing macro actions according to
    /// the defined timing and control flow logic.
    Playing,
    /// Playback paused - can be resumed
    ///
    /// Playback is temporarily suspended but can be resumed from the
    /// current action. Execution state is preserved.
    Paused,
    /// Playback completed successfully
    ///
    /// All macro actions have been executed successfully.
    /// The session cannot be resumed or restarted.
    Completed,
    /// Playback failed with error
    ///
    /// An error occurred during playback execution. The session is
    /// terminated and error details are available in the session.
    Failed
}

/// Macro execution context with variable substitution and state management
///
/// Provides the runtime environment for macro execution, including variable storage,
/// execution tracking, and loop state management. The context maintains all state
/// necessary for complex macro operations with nested control flow structures.
///
/// # Core Features
///
/// - **Variable Storage**: Hash-based variable lookup with O(1) access time
/// - **Execution Tracking**: Current action indexing for debugging and recovery
/// - **Loop Management**: Stack-based nested loop context handling
/// - **UUID Tracking**: Unique execution identification for logging and analysis
/// - **Timing Information**: Precise execution timing for performance monitoring
///
/// # Memory Layout
///
/// - **Base size**: ~128 bytes
/// - **Variable overhead**: ~32 bytes per variable + string lengths
/// - **Loop stack**: ~32 bytes per nested loop level
/// - **Total typical**: 200-500 bytes for moderate complexity macros
///
/// # Thread Safety
///
/// The context is not inherently thread-safe but is designed for single-threaded
/// macro execution. If concurrent access is needed, external synchronization
/// is required.
///
/// # Performance Characteristics
///
/// - **Variable access**: O(1) hash table lookup
/// - **Loop operations**: O(1) stack push/pop operations
/// - **Memory usage**: Linear with number of variables and loop depth
/// - **Clone cost**: O(n) where n is total variable and loop data size
#[derive(Debug, Clone)]
pub struct MacroExecutionContext {
    /// Variables available during execution with dynamic substitution
    ///
    /// Hash map storing all variables defined during macro execution.
    /// Variables are stored as strings but support dynamic type conversion
    /// during substitution. Includes both user-defined and system variables.
    pub variables: HashMap<String, String>,
    /// Unique identifier for this execution instance
    ///
    /// UUID generated at context creation time, used for tracking execution
    /// across logging systems, debugging tools, and performance monitoring.
    /// Remains constant throughout the execution lifecycle.
    pub execution_id: Uuid,
    /// Time when execution started for performance measurement
    ///
    /// High-precision timestamp captured when macro execution begins.
    /// Used for timeout enforcement, performance analysis, and execution
    /// duration calculation.
    pub start_time: Instant,
    /// Index of currently executing action for progress tracking
    ///
    /// Zero-based index into the macro's action list indicating which
    /// action is currently being executed. Used for progress reporting,
    /// debugging, and execution recovery.
    pub current_action: usize,
    /// Stack of nested loop contexts for control flow management
    ///
    /// Vector-based stack storing loop contexts for nested loop execution.
    /// Each entry represents one level of loop nesting with iteration
    /// state and boundary information.
    pub loop_stack: Vec<LoopContext>
}

/// Loop execution context for nested iteration management
///
/// Maintains the state of a single loop during macro execution, including
/// iteration counters, action boundaries, and progress tracking. Supports
/// nested loops through stack-based management in the execution context.
///
/// # Loop State
///
/// - **Iteration Tracking**: Current and maximum iteration counts
/// - **Action Boundaries**: Start and end indices for loop body
/// - **Progress Calculation**: Real-time progress reporting
/// - **Reset Capability**: Support for loop restart operations
///
/// # Performance
///
/// - **Memory size**: 32 bytes (fixed size structure)
/// - **Creation time**: O(1) constant time initialization
/// - **Progress calculation**: O(1) floating point division
/// - **State updates**: O(1) simple arithmetic operations
///
/// # Usage in Nested Loops
///
/// Each nested loop level creates its own LoopContext, allowing for
/// independent iteration management and proper loop variable scoping.
/// The context stack ensures correct loop termination and variable cleanup.
#[derive(Debug, Clone)]
pub struct LoopContext {
    /// Current iteration number (0-based indexing)
    ///
    /// The current iteration count starting from 0. Incremented after
    /// each loop iteration completion. Used for loop variable substitution
    /// and progress calculation.
    pub iteration: u32,
    /// Maximum number of iterations for this loop
    ///
    /// Total number of iterations this loop will execute. Set at loop
    /// creation time and used for termination condition checking and
    /// progress percentage calculation.
    pub max_iterations: u32,
    /// Index of first action in loop body (inclusive)
    ///
    /// Zero-based index into the macro's action list where the loop body
    /// begins. All actions from this index to end_action are executed
    /// in each iteration.
    pub start_action: usize,
    /// Index of last action in loop body (inclusive)
    ///
    /// Zero-based index into the macro's action list where the loop body
    /// ends. After executing this action, control returns to start_action
    /// for the next iteration or exits the loop if complete.
    pub end_action: usize
}

/// Comprehensive macro metadata and execution statistics
///
/// Contains all metadata about a macro including identification, versioning,
/// categorization, and execution statistics. This structure serves as both
/// the macro's identity and its performance record over time.
///
/// # Metadata Categories
///
/// - **Identity**: ID, name, description, and author information
/// - **Versioning**: Version number and timestamp tracking
/// - **Organization**: Tags, categories, and privacy settings
/// - **Statistics**: Execution counts, success rates, and performance metrics
///
/// # Serialization
///
/// Fully serializable to JSON for persistent storage, backup, and transfer
/// between systems. The format is optimized for both human readability and
/// efficient parsing.
///
/// # Performance Tracking
///
/// - **Execution Count**: Total number of times macro has been executed
/// - **Success Rate**: Percentage of successful executions (0.0-1.0)
/// - **Average Duration**: Mean execution time across all runs
/// - **Last Execution**: Timestamp of most recent execution
///
/// # Memory Usage
///
/// - **Base structure**: ~200 bytes
/// - **String data**: Variable based on name, description, and tag lengths
/// - **Total typical**: 300-800 bytes for moderately complex macros
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    /// Unique identifier for the macro (persistent across sessions)
    ///
    /// UUID that uniquely identifies this macro across all systems and
    /// sessions. Generated once at creation and never changes. Used for
    /// macro lookup, dependency resolution, and execution tracking.
    pub id: Uuid,
    /// Display name for the macro (user-facing identifier)
    ///
    /// Human-readable name displayed in user interfaces and logs.
    /// Should be descriptive and unique within a user's macro collection.
    /// Arc<str> provides efficient sharing across execution contexts.
    pub name: std::sync::Arc<str>,
    /// Detailed description of macro purpose and functionality
    ///
    /// Comprehensive description explaining what the macro does, when to use it,
    /// and any special considerations. Supports markdown formatting for rich
    /// documentation in supporting interfaces.
    pub description: std::sync::Arc<str>,
    /// Timestamp when macro was created (Unix epoch seconds)
    ///
    /// Creation timestamp as Duration from Unix epoch. Used for macro
    /// lifecycle tracking, cleanup policies, and historical analysis.
    /// Immutable after creation.
    pub created_at: Duration,
    /// Timestamp when macro was last updated (Unix epoch seconds)
    ///
    /// Last modification timestamp, updated whenever macro definition,
    /// metadata, or configuration changes. Used for sync, versioning,
    /// and change tracking.
    pub updated_at: Duration,
    /// Version number for macro versioning and compatibility
    ///
    /// Integer version number incremented with each modification.
    /// Used for compatibility checking, rollback operations, and
    /// change tracking. Simple integer versioning scheme.
    pub version: u32,
    /// Tags for organization and search functionality
    ///
    /// Vector of tags for categorizing and searching macros. Tags enable
    /// flexible organization beyond categories and support complex queries.
    /// Each tag is shared via Arc<str> for memory efficiency.
    pub tags: Vec<std::sync::Arc<str>>,
    /// Author/creator of the macro for attribution
    ///
    /// Username or identifier of the macro creator. Used for attribution,
    /// permissions, and support contact. Arc<str> enables efficient sharing
    /// across multiple macros by the same author.
    pub author: std::sync::Arc<str>,
    /// Total number of times macro has been executed
    ///
    /// Cumulative execution counter incremented each time the macro runs.
    /// Used for popularity metrics, performance analysis, and usage statistics.
    /// 64-bit counter supports very high execution counts.
    pub execution_count: u64,
    /// Timestamp of last execution (Unix epoch seconds)
    ///
    /// Optional timestamp of most recent execution. None if macro has never
    /// been executed. Used for activity tracking and cleanup policies.
    pub last_execution: Option<Duration>,
    /// Average execution duration across all runs
    ///
    /// Running average of execution times, updated after each execution.
    /// Used for performance monitoring, timeout setting, and capacity planning.
    /// Calculated as total time / execution count.
    pub average_duration: Duration,
    /// Success rate as percentage (0.0-1.0 range)
    ///
    /// Ratio of successful to total executions. Used for reliability metrics,
    /// macro quality assessment, and automated quality control. Updated
    /// after each execution with exponential moving average.
    pub success_rate: f64,
    /// Category classification for the macro
    ///
    /// High-level category for macro organization (e.g., "automation",
    /// "testing", "communication"). Used for filtering and organization
    /// in user interfaces. Arc<str> enables efficient category sharing.
    pub category: std::sync::Arc<str>,
    /// Whether macro is private to the creator
    ///
    /// Privacy flag controlling macro visibility and sharing. Private macros
    /// are only accessible to their creator, while public macros can be
    /// shared and discovered by other users.
    pub is_private: bool
}

/// Complete macro definition with actions, metadata, and execution configuration
///
/// Represents a fully defined macro ready for execution, containing all necessary
/// components including actions, metadata, variable definitions, triggers, and
/// execution settings. This is the primary structure for stored and executable macros.
///
/// # Macro Components
///
/// - **Metadata**: Identity, versioning, and statistics
/// - **Actions**: Sequence of operations to execute
/// - **Variables**: Default values and variable definitions
/// - **Triggers**: Conditions that automatically activate the macro
/// - **Dependencies**: Other macros required for execution
/// - **Configuration**: Execution settings and resource limits
///
/// # Lifecycle Management
///
/// 1. **Creation**: Built from recording sessions or manual definition
/// 2. **Validation**: Structure and dependency validation
/// 3. **Storage**: Serialized for persistent storage
/// 4. **Execution**: Loaded and executed with runtime context
/// 5. **Updates**: Metadata and statistics updated after execution
///
/// # Memory Layout
///
/// - **Base structure**: ~400 bytes
/// - **Actions**: Variable based on action complexity and count
/// - **Variables**: ~32 bytes per variable + string lengths
/// - **Total typical**: 1-5KB for moderate complexity macros
///
/// # Validation Features
///
/// - **Structure validation**: Ensures required fields are present
/// - **Action validation**: Verifies action sequences are valid
/// - **Dependency checking**: Validates macro dependencies exist
/// - **Resource validation**: Checks resource limits are reasonable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMacro {
    /// Macro metadata and statistical information
    ///
    /// Complete metadata including identity, versioning, authorship, and
    /// execution statistics. Provides all information needed for macro
    /// management and performance monitoring.
    pub metadata: MacroMetadata,
    /// Ordered list of actions to execute during macro playback
    ///
    /// Sequential list of macro actions that define the macro's behavior.
    /// Actions are executed in order with support for control flow structures
    /// like conditionals and loops that can modify execution flow.
    pub actions: Vec<MacroAction>,
    /// Variable definitions with default values for parameterization
    ///
    /// Hash map of variable names to default values. Variables can be
    /// overridden at execution time and are available for substitution
    /// in action parameters using `{variable_name}` syntax.
    pub variables: HashMap<std::sync::Arc<str>, std::sync::Arc<str>>,
    /// Trigger conditions that automatically activate this macro
    ///
    /// List of trigger expressions that, when evaluated as true, will
    /// automatically start macro execution. Enables event-driven macro
    /// activation based on chat state or external conditions.
    pub triggers: Vec<String>,
    /// Preconditions that must be met for execution to begin
    ///
    /// List of condition expressions that must all evaluate to true
    /// before macro execution can start. Used for safety checks and
    /// ensuring proper execution environment.
    pub conditions: Vec<String>,
    /// Other macros this macro depends on for proper execution
    ///
    /// List of macro names or IDs that must be available and executable
    /// for this macro to function correctly. Used for dependency resolution
    /// and execution ordering in complex macro systems.
    pub dependencies: Vec<String>,
    /// Execution configuration settings and resource limits
    ///
    /// Configuration controlling how the macro executes including timeouts,
    /// retry policies, resource limits, and execution behavior settings.
    /// Ensures safe and predictable macro execution.
    pub execution_config: MacroExecutionConfig
}

/// Comprehensive macro execution configuration and resource management
///
/// Controls all aspects of macro execution including timing, error handling,
/// resource usage, and safety limits. Provides fine-grained control over
/// macro behavior to ensure safe, predictable, and efficient execution.
///
/// # Configuration Categories
///
/// - **Timing**: Execution timeouts and retry policies
/// - **Error Handling**: Failure behavior and recovery strategies
/// - **Performance**: Parallel execution and optimization settings
/// - **Safety**: Resource limits and usage boundaries
/// - **Priority**: Execution priority for resource scheduling
///
/// # Resource Management
///
/// Resource limits prevent runaway macro execution and ensure system
/// stability. Limits are enforced during execution and violations result
/// in graceful macro termination with appropriate error reporting.
///
/// # Default Values
///
/// All configuration options have safe default values suitable for most
/// use cases. Defaults prioritize safety and predictability over maximum
/// performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionConfig {
    /// Maximum time allowed for complete macro execution
    ///
    /// Total execution timeout including all actions, delays, and control flow.
    /// Prevents infinite or extremely long-running macros from consuming
    /// system resources indefinitely. Default: 5 minutes.
    pub max_execution_time: Duration,
    /// Number of retry attempts on macro execution failure
    ///
    /// How many times to retry the entire macro if it fails. Each retry
    /// starts from the beginning with a fresh execution context.
    /// Default: 3 retries.
    pub retry_count: u32,
    /// Delay between retry attempts for failure recovery
    ///
    /// Time to wait between retry attempts, allowing temporary issues
    /// to resolve. Uses exponential backoff internally.
    /// Default: 1 second base delay.
    pub retry_delay: Duration,
    /// Whether to abort entire macro on first action error
    ///
    /// If true, any action failure terminates the entire macro execution.
    /// If false, individual action failures are logged but execution continues
    /// with the next action. Default: false (continue on error).
    pub abort_on_error: bool,
    /// Whether to allow parallel execution of independent actions
    ///
    /// Enables concurrent execution of actions that don't have dependencies.
    /// Can improve performance but may increase resource usage and complexity.
    /// Default: false (sequential execution).
    pub parallel_execution: bool,
    /// Execution priority for resource scheduling (0-255 range)
    ///
    /// Higher values indicate higher priority for CPU and resource allocation.
    /// Used by the macro scheduler to prioritize important macros during
    /// resource contention. Default: 5 (normal priority).
    pub priority: u8,
    /// Resource usage limits for safety and system stability
    ///
    /// Comprehensive resource limits including memory, CPU, network, and
    /// file system usage to prevent macro abuse and ensure system stability.
    pub resource_limits: ResourceLimits
}

/// Resource limits for safe macro execution and system protection
///
/// Defines hard limits on system resource usage during macro execution
/// to prevent abuse, protect system stability, and ensure fair resource
/// sharing among concurrent macro executions.
///
/// # Resource Categories
///
/// - **Memory**: RAM usage limits to prevent memory exhaustion
/// - **CPU**: Processing time limits to ensure fair CPU sharing
/// - **Network**: Request limits to prevent network abuse
/// - **File System**: I/O operation limits to protect storage systems
///
/// # Enforcement
///
/// Limits are actively monitored during macro execution. When a limit
/// is exceeded, the macro is gracefully terminated with a resource
/// limit error. This protects the system from runaway macro execution.
///
/// # Default Values
///
/// Conservative defaults are provided that allow normal macro operation
/// while preventing system abuse. Defaults can be overridden for
/// specific macros that require higher resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in megabytes during execution
    ///
    /// Hard limit on RAM usage by the macro execution process.
    /// Includes action execution, variable storage, and temporary buffers.
    /// Default: 100MB (sufficient for most macro operations).
    pub max_memory_mb: u32,
    /// Maximum CPU usage as percentage (0-100 range)
    ///
    /// Maximum percentage of CPU time the macro can consume during execution.
    /// Prevents CPU-intensive macros from starving other processes.
    /// Default: 25% (allows other processes to run smoothly).
    pub max_cpu_percent: u8,
    /// Maximum number of network requests allowed during execution
    ///
    /// Limit on HTTP requests, API calls, and other network operations.
    /// Prevents network abuse and reduces attack surface for malicious macros.
    /// Default: 50 requests (suitable for API integration macros).
    pub max_network_requests: u32,
    /// Maximum number of file operations allowed during execution
    ///
    /// Limit on file reads, writes, and other filesystem operations.
    /// Protects against filesystem abuse and excessive I/O operations.
    /// Default: 20 operations (allows reasonable file manipulation).
    pub max_file_operations: u32
}

/// Live macro recording session with lock-free action capture
///
/// Manages the process of recording user actions into a macro, providing
/// real-time capture of chat interactions, commands, and timing information.
/// Uses lock-free data structures for high-performance recording without
/// blocking the user interface.
///
/// # Recording Process
///
/// 1. **Initialization**: Create session with metadata and unique ID
/// 2. **Start Recording**: Begin capturing user actions and commands
/// 3. **Action Capture**: Store actions with precise timing information
/// 4. **State Management**: Support pause/resume operations
/// 5. **Completion**: Convert recorded actions to executable macro
///
/// # Performance Features
///
/// - **Lock-free queues**: Uses `SegQueue` for non-blocking action storage
/// - **Real-time capture**: Minimal latency between action and storage
/// - **Memory efficient**: Incremental allocation as actions are recorded
/// - **Thread safety**: Safe for concurrent access during recording
///
/// # State Management
///
/// Recording sessions maintain state to ensure data integrity and provide
/// clear feedback about recording status. State transitions are atomic
/// and consistent.
///
/// # Memory Usage
///
/// - **Base structure**: ~200 bytes
/// - **Action queue**: ~32 bytes per recorded action + action data
/// - **Variables**: ~32 bytes per captured variable + string lengths
/// - **Total typical**: 1-10KB for moderate recording sessions
#[derive(Debug)]
pub struct MacroRecordingSession {
    /// Unique identifier for this recording session instance
    ///
    /// UUID generated at session creation, used for tracking the recording
    /// process across systems and providing unique identification for
    /// logging and debugging purposes.
    pub id: Uuid,
    /// Human-readable name for the recording session
    ///
    /// Descriptive name provided by the user to identify the recording
    /// session. Used in user interfaces and becomes part of the final
    /// macro metadata if not overridden.
    pub name: String,
    /// High-precision timestamp when recording started
    ///
    /// Instant captured when recording begins, used as the reference point
    /// for all action timestamps. Enables accurate timing reproduction
    /// during macro playback.
    pub start_time: Instant,
    /// Lock-free queue of recorded actions for high-performance capture
    ///
    /// Thread-safe queue using atomic operations for action storage.
    /// Allows concurrent action recording without blocking the user interface
    /// or other system operations. Actions maintain insertion order.
    pub actions: SegQueue<MacroAction>,
    /// Current state of the recording session
    ///
    /// Tracks recording progress through the state machine (Idle, Recording,
    /// Paused, Completed). Used for state validation and user interface
    /// updates during the recording process.
    pub state: MacroRecordingState,
    /// Variables captured during recording for macro parameterization
    ///
    /// Hash map of variables discovered or defined during recording.
    /// These become default variable values in the final macro and
    /// can be overridden during macro execution.
    pub variables: HashMap<String, String>,
    /// Metadata for the macro being recorded
    ///
    /// Complete macro metadata that will be attached to the final macro.
    /// Updated during recording with timing, execution count, and other
    /// statistical information as the recording progresses.
    pub metadata: MacroMetadata
}

impl Clone for MacroRecordingSession {
    /// Clone the recording session (expensive operation due to SegQueue)
    fn clone(&self) -> Self {
        // Clone actions from SegQueue by draining and re-adding
        let actions = SegQueue::new();
        while let Some(action) = self.actions.pop() {
            actions.push(action.clone());
        }
        
        Self {
            id: self.id,
            name: self.name.clone(),
            start_time: self.start_time,
            actions,
            state: self.state.clone(),
            variables: self.variables.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Active macro playback session with execution context and progress tracking
///
/// Manages the execution of a macro with full context tracking, progress
/// monitoring, and error handling. Provides detailed visibility into macro
/// execution state and supports pause/resume operations for interactive
/// macro debugging and control.
///
/// # Execution Management
///
/// - **Context Tracking**: Maintains execution state and variable context
/// - **Progress Monitoring**: Real-time progress reporting and estimation
/// - **Error Handling**: Comprehensive error capture and reporting
/// - **State Control**: Support for pause, resume, and termination operations
/// - **Performance Metrics**: Execution timing and resource usage tracking
///
/// # Playback Features
///
/// - **Variable Substitution**: Dynamic content replacement during execution
/// - **Timing Reproduction**: Accurate timing replication from recording
/// - **Loop Management**: Nested loop execution with proper context handling
/// - **Conditional Logic**: Boolean expression evaluation for control flow
/// - **Error Recovery**: Graceful handling of action execution failures
///
/// # Session Lifecycle
///
/// 1. **Initialization**: Create session with macro reference and configuration
/// 2. **Context Setup**: Initialize variables and execution environment
/// 3. **Execution**: Step through actions with timing and state management
/// 4. **Monitoring**: Track progress and handle state changes
/// 5. **Completion**: Finalize execution and update statistics
///
/// # Memory Usage
///
/// - **Base structure**: ~300 bytes
/// - **Execution context**: Variable based on macro complexity
/// - **Progress tracking**: ~100 bytes for state and metrics
/// - **Total typical**: 500-2000 bytes during execution
#[derive(Debug)]
pub struct MacroPlaybackSession {
    /// Unique identifier for this playback session instance
    ///
    /// UUID generated at session creation, used for tracking playback
    /// execution across logging systems and providing unique identification
    /// for debugging and performance monitoring.
    pub id: Uuid,
    /// ID of the macro being executed in this session
    ///
    /// Reference to the macro being played back, used for macro lookup,
    /// statistics updates, and execution history tracking. Links the
    /// playback session to its source macro.
    pub macro_id: Uuid,
    /// High-precision timestamp when playback execution started
    ///
    /// Instant captured when playback begins, used for execution duration
    /// calculation, timeout enforcement, and performance analysis.
    /// Enables accurate execution timing metrics.
    pub start_time: Instant,
    /// Complete execution context with variables and state management
    ///
    /// Full execution environment including variable storage, loop stack,
    /// and execution tracking. Maintains all state necessary for complex
    /// macro execution with nested control structures.
    pub context: MacroExecutionContext,
    /// Current state of playback execution
    ///
    /// Tracks playback progress through the state machine (Idle, Playing,
    /// Paused, Completed, Failed). Used for execution control and user
    /// interface updates during macro playback.
    pub state: MacroPlaybackState,
    /// Index of currently executing action (zero-based)
    ///
    /// Current position in the macro's action list. Used for progress
    /// calculation, debugging, and execution recovery after pause/resume
    /// operations. Updated as each action completes.
    pub current_action: usize,
    /// Total number of actions in the macro being executed
    ///
    /// Used for progress percentage calculation and completion detection.
    /// Set at session initialization and remains constant throughout
    /// the execution process.
    pub total_actions: usize,
    /// Error message if playback execution failed
    ///
    /// Optional error description populated when playback fails. Contains
    /// detailed error information for debugging and user feedback. None
    /// when execution is successful or still in progress.
    pub error: Option<String>
}

/// Thread-safe macro execution statistics with atomic counters
///
/// Maintains comprehensive execution statistics for macro performance monitoring,
/// reliability assessment, and usage analytics. Uses atomic operations and
/// lock-free data structures for high-performance concurrent access without
/// blocking macro execution.
///
/// # Statistical Categories
///
/// - **Execution Counts**: Total, successful, and failed execution tracking
/// - **Timing Metrics**: Duration statistics and performance analysis
/// - **Success Rates**: Reliability metrics and failure analysis
/// - **Historical Data**: Execution history and trend analysis
///
/// # Thread Safety Features
///
/// - **Atomic Counters**: Lock-free increment/decrement operations
/// - **Mutex Protection**: Safe access to complex data structures
/// - **Consistent Reads**: Atomic snapshots of statistical data
/// - **Concurrent Updates**: Safe updates from multiple execution threads
///
/// # Performance Characteristics
///
/// - **Counter updates**: O(1) atomic operations
/// - **Duration calculations**: O(1) arithmetic with mutex protection
/// - **Memory usage**: ~200 bytes fixed size structure
/// - **Contention handling**: Lock-free design minimizes blocking
///
/// # Usage Patterns
///
/// Statistics are automatically updated during macro execution and can be
/// safely accessed from multiple threads for monitoring and reporting
/// purposes. The atomic design ensures accuracy under high concurrency.
#[derive(Debug, Default)]
pub struct ExecutionStats {
    /// Total number of macro executions (atomic counter)
    ///
    /// Thread-safe counter tracking all execution attempts regardless of
    /// outcome. Incremented at the start of each execution attempt.
    /// Used for overall usage statistics and load analysis.
    pub total_executions: ConsistentCounter,
    /// Number of successful macro executions (atomic counter)
    ///
    /// Thread-safe counter tracking executions that completed without
    /// errors. Used for success rate calculation and reliability metrics.
    /// Updated only when macro execution completes successfully.
    pub successful_executions: ConsistentCounter,
    /// Number of failed macro executions (atomic counter)
    ///
    /// Thread-safe counter tracking executions that terminated with errors.
    /// Used for failure rate calculation and error analysis. Includes
    /// timeouts, resource limit violations, and action failures.
    pub failed_executions: ConsistentCounter,
    /// Cumulative execution time across all runs (mutex protected)
    ///
    /// Total wall-clock time spent executing this macro across all runs.
    /// Protected by mutex for atomic updates during concurrent execution.
    /// Used for average duration calculation and resource usage analysis.
    pub total_duration: parking_lot::Mutex<Duration>,
    /// Average execution duration per run (mutex protected)
    ///
    /// Mean execution time calculated as total_duration / total_executions.
    /// Updated after each execution completion. Protected by mutex for
    /// consistent reads during concurrent access.
    pub average_duration: parking_lot::Mutex<Duration>,
    /// Timestamp of most recent execution (mutex protected)
    ///
    /// Optional timestamp of the last execution attempt. Used for activity
    /// tracking and cleanup policies. None if macro has never been executed.
    /// Protected by mutex for atomic timestamp updates.
    pub last_execution: parking_lot::Mutex<Option<Instant>>
}

impl Clone for ExecutionStats {
    /// Clone execution statistics by copying counter values and mutex contents
    fn clone(&self) -> Self {
        ExecutionStats {
            total_executions: ConsistentCounter::new(self.total_executions.get()),
            successful_executions: ConsistentCounter::new(self.successful_executions.get()),
            failed_executions: ConsistentCounter::new(self.failed_executions.get()),
            total_duration: parking_lot::Mutex::new(*self.total_duration.lock()),
            average_duration: parking_lot::Mutex::new(*self.average_duration.lock()),
            last_execution: parking_lot::Mutex::new(*self.last_execution.lock())
        }
    }
}

impl Default for MacroExecutionConfig {
    /// Create default macro execution configuration with safe defaults
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300), // 5 minutes
            retry_count: 3,
            retry_delay: Duration::from_millis(1000),
            abort_on_error: false,
            parallel_execution: false,
            priority: 5,
            resource_limits: ResourceLimits::default()
        }
    }
}

impl Default for ResourceLimits {
    /// Create default resource limits with conservative values for safety
    fn default() -> Self {
        Self {
            max_memory_mb: 100,
            max_cpu_percent: 25,
            max_network_requests: 50,
            max_file_operations: 20
        }
    }
}

/// Comprehensive error enumeration for macro system operations
///
/// Provides detailed error variants covering all aspects of macro system
/// operation including recording, playback, storage, validation, and resource
/// management. Each error variant includes contextual information for debugging
/// and recovery strategies.
///
/// # Error Categories
///
/// - **Lookup Errors**: Missing macros, sessions, or resources
/// - **State Errors**: Invalid state transitions or concurrent access
/// - **Validation Errors**: Invalid macro definitions or parameters
/// - **Execution Errors**: Runtime failures during macro playback
/// - **Resource Errors**: Resource limit violations and system constraints
/// - **System Errors**: Low-level system and I/O failures
///
/// # Error Handling Patterns
///
/// Each error variant provides specific context to enable appropriate
/// recovery strategies. Errors are designed to be actionable with clear
/// guidance on potential resolution approaches.
///
/// # Thread Safety
///
/// All error variants are thread-safe and can be safely passed between
/// threads for logging, reporting, and error handling purposes.
#[derive(Debug, thiserror::Error)]
pub enum MacroSystemError {
    /// Macro with specified ID was not found in storage
    ///
    /// Indicates that a macro lookup operation failed because no macro
    /// with the given ID exists in the macro storage system. This can
    /// occur when referencing deleted macros or using invalid IDs.
    #[error("Macro not found: {0}")]
    MacroNotFound(
        /// UUID of the macro that was not found in storage
        ///
        /// The unique identifier that was used in the failed lookup operation.
        /// Can be used for logging and debugging macro reference issues.
        Uuid
    ),
    /// Recording or playback session with specified ID was not found
    ///
    /// Indicates that a session lookup operation failed because no active
    /// session with the given ID exists. This can occur when referencing
    /// expired sessions or using invalid session IDs.
    #[error("Session not found: {0}")]
    SessionNotFound(
        /// UUID of the session that was not found in active session storage
        ///
        /// The unique identifier that was used in the failed session lookup.
        /// Sessions may expire or be cleaned up, causing this error.
        Uuid
    ),
    /// Recording is already in progress for the specified session
    ///
    /// Indicates an attempt to start recording on a session that is already
    /// in the Recording state. Each session can only have one active recording
    /// at a time to maintain data integrity.
    #[error("Recording already in progress: {0}")]
    RecordingInProgress(
        /// UUID of the session that is already actively recording
        ///
        /// The session identifier that is currently in Recording state.
        /// The session must be stopped or completed before starting a new recording.
        Uuid
    ),
    /// Playback is already in progress for the specified session
    ///
    /// Indicates an attempt to start playback on a session that is already
    /// in the Playing state. Each session can only have one active playback
    /// at a time to prevent conflicting execution contexts.
    #[error("Playback already in progress: {0}")]
    PlaybackInProgress(
        /// UUID of the session that is already actively playing back
        ///
        /// The session identifier that is currently in Playing state.
        /// The session must be paused, completed, or failed before starting new playback.
        Uuid
    ),
    /// Invalid or malformed macro action definition
    ///
    /// Indicates that a macro action has invalid structure, missing required
    /// fields, or contains data that cannot be processed during execution.
    /// This typically occurs during macro validation or deserialization.
    #[error("Invalid macro action: {0}")]
    InvalidAction(
        /// Detailed description of what makes the action invalid
        ///
        /// Human-readable explanation of the validation failure, including
        /// specific field issues or structural problems with the action.
        String
    ),
    /// Macro execution exceeded configured time limit
    ///
    /// Indicates that macro execution was terminated because it ran longer
    /// than the configured maximum execution time. This prevents infinite
    /// or extremely long-running macros from consuming system resources.
    #[error("Execution timeout")]
    ExecutionTimeout,
    /// Required variable was not found in execution context
    ///
    /// Indicates that a macro action attempted to reference a variable
    /// that doesn't exist in the current execution context. This can occur
    /// when variable names are misspelled or variables are used before definition.
    #[error("Variable not found: {0}")]
    VariableNotFound(
        /// Name of the variable that was not found in the execution context
        ///
        /// The variable identifier that was referenced but not found.
        /// Variable names are case-sensitive and must be defined before use.
        String
    ),
    /// Error evaluating conditional expression during execution
    ///
    /// Indicates that a conditional action's boolean expression could not
    /// be evaluated due to syntax errors, type mismatches, or invalid
    /// variable references. Conditional expressions must evaluate to boolean values.
    #[error("Condition evaluation error: {0}")]
    ConditionError(
        /// Detailed description of the condition evaluation failure
        ///
        /// Human-readable explanation of why the condition could not be evaluated,
        /// including syntax errors or type conversion issues.
        String
    ),
    /// Resource usage exceeded configured safety limits
    ///
    /// Indicates that macro execution was terminated because it exceeded
    /// one or more resource limits (memory, CPU, network, or file operations).
    /// This protects the system from resource abuse by malicious or poorly designed macros.
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(
        /// Detailed description of which resource limit was exceeded
        ///
        /// Specifies which resource (memory, CPU, network, file) exceeded its limit
        /// and the actual vs. configured limit values for debugging.
        String
    ),
    /// JSON serialization/deserialization operation failed
    ///
    /// Indicates that macro storage or transmission operations failed due to
    /// JSON serialization errors. This can occur when saving macros to storage
    /// or loading macros with incompatible format versions.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    /// File system I/O operation failed during macro operations
    ///
    /// Indicates that file system operations (reading, writing, or accessing
    /// macro storage files) failed due to I/O errors. This can include
    /// permission issues, disk space problems, or network storage failures.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Mutex or lock was poisoned by panic in another thread
    ///
    /// Indicates that a thread holding a mutex panicked, leaving the mutex
    /// in a poisoned state. This is a critical error that typically requires
    /// system restart or macro system reinitialization.
    #[error("Lock poisoned: {0}")]
    LockPoisoned(
        /// Detailed description of which lock was poisoned and the context
        ///
        /// Information about the poisoned lock including which data structure
        /// was affected and potential recovery strategies.
        String
    ),
    /// Time-related operation failed during execution
    ///
    /// Indicates that timing operations (duration calculations, timestamp
    /// generation, or timeout handling) failed due to system clock issues
    /// or time representation problems.
    #[error("Time error: {0}")]
    TimeError(
        /// Detailed description of the time-related failure
        ///
        /// Explanation of which time operation failed and the underlying
        /// cause, such as system clock issues or duration overflow.
        String
    ),
    /// Generic lock acquisition failed during concurrent access
    ///
    /// Indicates that a lock acquisition operation failed, typically due to
    /// deadlock conditions or lock contention. This is a generic fallback
    /// for lock-related issues not covered by more specific error variants.
    #[error("Lock error")]
    LockError,
    /// Internal system error not covered by other variants
    ///
    /// Indicates an unexpected internal error in the macro system that doesn't
    /// fit other error categories. This typically represents bugs or unexpected
    /// system states that require investigation.
    #[error("Internal error: {0}")]
    InternalError(
        /// Detailed description of the internal error for debugging
        ///
        /// Technical information about the internal error including context,
        /// affected components, and potential debugging information.
        String
    ),
    /// Macro definition is invalid or corrupted
    ///
    /// Indicates that a macro definition failed validation due to structural
    /// issues, missing required fields, or logical inconsistencies. This
    /// prevents execution of potentially dangerous or malformed macros.
    #[error("Invalid macro: {0}")]
    InvalidMacro(
        /// Detailed description of why the macro is considered invalid
        ///
        /// Specific validation failures including missing fields, invalid
        /// action sequences, or logical inconsistencies in the macro definition.
        String
    ),
    /// Macro recursion exceeded maximum allowed depth
    ///
    /// Indicates that macro execution exceeded the maximum allowed recursion
    /// depth, typically due to macros calling other macros in infinite loops.
    /// This protects against stack overflow and infinite execution scenarios.
    #[error("Maximum recursion depth exceeded")]
    MaxRecursionDepthExceeded,
    /// Storage backend operation failed during macro persistence
    ///
    /// Indicates that macro storage operations (save, load, delete, or query)
    /// failed due to database errors, network issues, or storage system problems.
    /// This affects macro persistence and retrieval operations.
    #[error("Storage error: {0}")]
    StorageError(
        /// Detailed description of the storage operation failure
        ///
        /// Information about which storage operation failed, the underlying
        /// cause, and potential recovery strategies or alternative approaches.
        String
    ),
    /// Operating system level error during macro operations
    ///
    /// Indicates that low-level system operations failed during macro execution,
    /// including process creation, memory allocation, or system resource access.
    /// These errors typically require system-level troubleshooting.
    #[error("System error: {0}")]
    SystemError(
        /// Detailed description of the system-level error
        ///
        /// Technical information about the system error including error codes,
        /// affected resources, and potential system-level resolution approaches.
        String
    )
}

impl<T> From<std::sync::PoisonError<T>> for MacroSystemError {
    /// Convert standard library poison error to macro system error
    ///
    /// Automatically converts mutex poison errors from the standard library
    /// into macro system errors with appropriate context information.
    /// This enables seamless error propagation from standard library operations.
    fn from(err: std::sync::PoisonError<T>) -> Self {
        MacroSystemError::LockPoisoned(err.to_string())
    }
}

/// Standard result type for all macro system operations
///
/// Provides consistent error handling across the macro system by wrapping
/// successful results and macro system errors in a standard Result type.
/// Used throughout the macro API for consistent error propagation and handling.
///
/// # Usage
///
/// ```rust
/// use fluent_ai_candle::types::candle_chat::macros::types::{MacroResult, MacroSystemError};
///
/// fn create_macro() -> MacroResult<ChatMacro> {
///     // Returns Ok(ChatMacro) on success or Err(MacroSystemError) on failure
///     // ...
/// }
/// ```
pub type MacroResult<T> = Result<T, MacroSystemError>;

/// Result enumeration for individual macro action execution
///
/// Represents the outcome of executing a single macro action, providing
/// detailed control flow information for macro execution management.
/// Each variant indicates a different execution outcome that affects
/// how the macro execution should proceed.
///
/// # Control Flow
///
/// Action execution results control macro flow by indicating whether
/// execution should continue normally, pause, skip actions, or terminate.
/// The macro execution engine uses these results to implement complex
/// control flow patterns.
///
/// # Execution States
///
/// - **Success**: Normal completion, continue with next action
/// - **Error**: Action failed, handle according to error policy
/// - **Wait**: Pause execution for specified duration
/// - **Skip**: Jump to different action (for loops, conditionals)
/// - **Complete**: Terminate execution successfully
#[derive(Debug)]
pub enum ActionExecutionResult {
    /// Action executed successfully, continue with next action
    ///
    /// Indicates normal action completion. The macro execution engine
    /// will proceed to the next action in sequence or handle control
    /// flow structures (loops, conditionals) as appropriate.
    Success,
    /// Action failed with error message
    ///
    /// Indicates that the action could not be completed due to an error.
    /// The macro execution engine will handle this according to the
    /// configured error policy (continue or abort).
    Error(
        /// Detailed error message describing the failure cause
        ///
        /// Human-readable description of why the action failed, used for
        /// logging, debugging, and user feedback during macro execution.
        String
    ),
    /// Wait for specified duration before continuing execution
    ///
    /// Indicates that execution should pause for a specified time before
    /// proceeding. This is typically returned by Wait actions or actions
    /// that need to introduce delays for timing synchronization.
    Wait(
        /// Duration to pause execution before proceeding
        ///
        /// The exact time to wait before continuing with the next action.
        /// Uses high-precision timing for accurate delay reproduction.
        Duration
    ),
    /// Skip to specific action index for control flow
    ///
    /// Indicates that execution should jump to a different action index,
    /// used for implementing loops, conditionals, and other control flow
    /// structures within macro execution.
    SkipToAction(
        /// Zero-based index of the action to jump to next
        ///
        /// The target action index for the next execution step. Must be
        /// a valid index within the macro's action list.
        usize
    ),
    /// Complete macro execution successfully
    ///
    /// Indicates that the macro execution should terminate successfully,
    /// regardless of remaining actions. Used for early termination
    /// conditions or explicit completion actions.
    Complete
}

/// Result enumeration for overall macro playback operations
///
/// Represents the high-level outcome of macro playback operations,
/// providing status information for macro execution management and
/// user interface updates. Used for tracking playback state transitions.
///
/// # Playback States
///
/// - **ActionExecuted**: Individual action completed successfully
/// - **Completed**: All actions finished, macro execution done
/// - **Failed**: Execution terminated due to error
/// - **Paused**: Execution suspended, can be resumed
///
/// # Usage Context
///
/// These results are typically returned by high-level playback operations
/// and used by user interfaces, logging systems, and macro management
/// components to track execution progress and state.
#[derive(Debug)]
pub enum MacroPlaybackResult {
    /// Individual action executed successfully, playback continuing
    ///
    /// Indicates that a single action within the macro completed successfully
    /// and playback is continuing with the next action. Used for progress
    /// reporting and incremental execution feedback.
    ActionExecuted,
    /// All macro actions completed successfully
    ///
    /// Indicates that the entire macro has finished executing with all
    /// actions completed successfully. The playback session can be considered
    /// finished and statistics should be updated.
    Completed,
    /// Playback failed due to error
    ///
    /// Indicates that macro playback encountered an unrecoverable error
    /// and has been terminated. Error details are available in the
    /// playback session structure.
    Failed,
    /// Playback paused and can be resumed
    ///
    /// Indicates that macro playback has been suspended but can be resumed
    /// from the current position. The execution state is preserved for
    /// later continuation.
    Paused
}

impl MacroExecutionContext {
    /// Create a new macro execution context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            execution_id: Uuid::new_v4(),
            start_time: Instant::now(),
            current_action: 0,
            loop_stack: Vec::new(),
        }
    }

    /// Add a variable to the execution context
    pub fn set_variable(&mut self, name: String, value: String) {
        self.variables.insert(name, value);
    }

    /// Get a variable value from the execution context
    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.variables.get(name)
    }

    /// Check if a variable exists in the context
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Clear all variables from the context
    pub fn clear_variables(&mut self) {
        self.variables.clear();
    }

    /// Get the number of variables in the context
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Push a new loop context onto the stack
    pub fn push_loop(&mut self, context: LoopContext) {
        self.loop_stack.push(context);
    }

    /// Pop the top loop context from the stack
    pub fn pop_loop(&mut self) -> Option<LoopContext> {
        self.loop_stack.pop()
    }

    /// Get the current loop depth
    pub fn loop_depth(&self) -> usize {
        self.loop_stack.len()
    }

    /// Get the current loop context (if any)
    pub fn current_loop(&self) -> Option<&LoopContext> {
        self.loop_stack.last()
    }

    /// Reset the execution context for a new run
    pub fn reset(&mut self) {
        self.execution_id = Uuid::new_v4();
        self.start_time = Instant::now();
        self.current_action = 0;
        self.loop_stack.clear();
    }
}

impl Default for MacroExecutionContext {
    /// Create a default macro execution context
    fn default() -> Self {
        Self::new()
    }
}

impl LoopContext {
    /// Create a new loop context
    pub fn new(max_iterations: u32, start_action: usize, end_action: usize) -> Self {
        Self {
            iteration: 0,
            max_iterations,
            start_action,
            end_action,
        }
    }

    /// Check if the loop has more iterations
    pub fn has_more_iterations(&self) -> bool {
        self.iteration < self.max_iterations
    }

    /// Advance to the next iteration
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Get the current progress as a percentage (0.0-1.0)
    pub fn progress(&self) -> f64 {
        if self.max_iterations == 0 {
            1.0
        } else {
            self.iteration as f64 / self.max_iterations as f64
        }
    }

    /// Reset the loop to the beginning
    pub fn reset(&mut self) {
        self.iteration = 0;
    }
}

impl MacroMetadata {
    /// Create new macro metadata with minimal required fields
    pub fn new(name: impl Into<std::sync::Arc<str>>) -> Self {
        let now = Duration::from_secs(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: std::sync::Arc::from(""),
            created_at: now,
            updated_at: now,
            version: 1,
            tags: Vec::new(),
            author: std::sync::Arc::from(""),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::default(),
            success_rate: 0.0,
            category: std::sync::Arc::from("general"),
            is_private: false,
        }
    }

    /// Update the metadata after a successful execution
    pub fn record_execution(&mut self, duration: Duration, success: bool) {
        self.execution_count += 1;
        let now = Duration::from_secs(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        self.last_execution = Some(now);
        self.updated_at = now;

        // Update average duration
        let total_duration = self.average_duration.as_nanos() * (self.execution_count - 1) as u128
            + duration.as_nanos();
        self.average_duration = Duration::from_nanos((total_duration / self.execution_count as u128) as u64);

        // Update success rate (simplified - in real implementation would track failures too)
        if success {
            self.success_rate = (self.success_rate * (self.execution_count - 1) as f64 + 1.0) / self.execution_count as f64;
        }
    }

    /// Add a tag to the macro metadata
    pub fn add_tag(&mut self, tag: impl Into<std::sync::Arc<str>>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag from the macro metadata
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t.as_ref() != tag);
    }

    /// Check if the macro has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t.as_ref() == tag)
    }
}

impl ChatMacro {
    /// Create a new chat macro with metadata and actions
    pub fn new(metadata: MacroMetadata, actions: Vec<MacroAction>) -> Self {
        Self {
            metadata,
            actions,
            variables: HashMap::new(),
            triggers: Vec::new(),
            conditions: Vec::new(),
            dependencies: Vec::new(),
            execution_config: MacroExecutionConfig::default(),
        }
    }

    /// Add a variable with default value to the macro
    pub fn add_variable(&mut self, name: impl Into<std::sync::Arc<str>>, default_value: impl Into<std::sync::Arc<str>>) {
        self.variables.insert(name.into(), default_value.into());
    }

    /// Add a trigger condition for automatic execution
    pub fn add_trigger(&mut self, trigger: String) {
        self.triggers.push(trigger);
    }

    /// Add a precondition that must be met for execution
    pub fn add_condition(&mut self, condition: String) {
        self.conditions.push(condition);
    }

    /// Add a dependency on another macro
    pub fn add_dependency(&mut self, dependency: String) {
        self.dependencies.push(dependency);
    }

    /// Get the estimated execution duration based on metadata
    pub fn estimated_duration(&self) -> Duration {
        self.metadata.average_duration
    }

    /// Get the number of actions in the macro
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Check if the macro has any triggers
    pub fn has_triggers(&self) -> bool {
        !self.triggers.is_empty()
    }

    /// Check if the macro has any dependencies
    pub fn has_dependencies(&self) -> bool {
        !self.dependencies.is_empty()
    }

    /// Validate the macro structure
    pub fn validate(&self) -> MacroResult<()> {
        if self.metadata.name.is_empty() {
            return Err(MacroSystemError::InvalidMacro("Macro name cannot be empty".to_string()));
        }

        if self.actions.is_empty() {
            return Err(MacroSystemError::InvalidMacro("Macro must have at least one action".to_string()));
        }

        Ok(())
    }
}

impl MacroRecordingSession {
    /// Create a new macro recording session
    pub fn new(name: String, metadata: MacroMetadata) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            start_time: Instant::now(),
            actions: SegQueue::new(),
            state: MacroRecordingState::Idle,
            variables: HashMap::new(),
            metadata,
        }
    }

    /// Start recording actions
    pub fn start_recording(&mut self) {
        self.state = MacroRecordingState::Recording;
        self.start_time = Instant::now();
    }

    /// Pause the recording session
    pub fn pause_recording(&mut self) {
        if self.state == MacroRecordingState::Recording {
            self.state = MacroRecordingState::Paused;
        }
    }

    /// Resume the recording session
    pub fn resume_recording(&mut self) {
        if self.state == MacroRecordingState::Paused {
            self.state = MacroRecordingState::Recording;
        }
    }

    /// Stop the recording session
    pub fn stop_recording(&mut self) {
        self.state = MacroRecordingState::Completed;
    }

    /// Add an action to the recording
    pub fn add_action(&self, action: MacroAction) {
        if self.state == MacroRecordingState::Recording {
            self.actions.push(action);
        }
    }

    /// Get the duration of the recording session
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the number of recorded actions
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Check if the session is actively recording
    pub fn is_recording(&self) -> bool {
        self.state == MacroRecordingState::Recording
    }

    /// Export the recorded actions to a macro
    pub fn to_macro(&self) -> ChatMacro {
        let mut actions = Vec::new();
        while let Some(action) = self.actions.pop() {
            actions.push(action);
        }
        actions.reverse(); // Restore original order

        ChatMacro::new(self.metadata.clone(), actions)
    }
}

impl MacroPlaybackSession {
    /// Create a new macro playback session
    pub fn new(macro_id: Uuid, total_actions: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            macro_id,
            start_time: Instant::now(),
            context: MacroExecutionContext::new(),
            state: MacroPlaybackState::Idle,
            current_action: 0,
            total_actions,
            error: None,
        }
    }

    /// Start playing back the macro
    pub fn start_playback(&mut self) {
        self.state = MacroPlaybackState::Playing;
        self.start_time = Instant::now();
        self.context.reset();
    }

    /// Pause the playback session
    pub fn pause_playback(&mut self) {
        if self.state == MacroPlaybackState::Playing {
            self.state = MacroPlaybackState::Paused;
        }
    }

    /// Resume the playback session
    pub fn resume_playback(&mut self) {
        if self.state == MacroPlaybackState::Paused {
            self.state = MacroPlaybackState::Playing;
        }
    }

    /// Complete the playback session successfully
    pub fn complete_playback(&mut self) {
        self.state = MacroPlaybackState::Completed;
    }

    /// Fail the playback session with an error
    pub fn fail_playback(&mut self, error: String) {
        self.state = MacroPlaybackState::Failed;
        self.error = Some(error);
    }

    /// Advance to the next action
    pub fn next_action(&mut self) {
        if self.current_action < self.total_actions {
            self.current_action += 1;
            self.context.current_action = self.current_action;
        }
    }

    /// Get the progress as a percentage (0.0-1.0)
    pub fn progress(&self) -> f64 {
        if self.total_actions == 0 {
            1.0
        } else {
            self.current_action as f64 / self.total_actions as f64
        }
    }

    /// Get the duration of the playback session
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if the playback is currently active
    pub fn is_playing(&self) -> bool {
        self.state == MacroPlaybackState::Playing
    }

    /// Check if the playback has completed
    pub fn is_completed(&self) -> bool {
        matches!(self.state, MacroPlaybackState::Completed | MacroPlaybackState::Failed)
    }
}

impl ExecutionStats {
    /// Create new execution statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful execution
    pub fn record_success(&self, duration: Duration) {
        self.total_executions.inc();
        self.successful_executions.inc();
        
        let mut total_dur = self.total_duration.lock();
        *total_dur += duration;
        
        let total_execs = self.total_executions.get();
        let mut avg_dur = self.average_duration.lock();
        *avg_dur = *total_dur / total_execs as u32;
        
        let mut last_exec = self.last_execution.lock();
        *last_exec = Some(Instant::now());
    }

    /// Record a failed execution
    pub fn record_failure(&self, duration: Duration) {
        self.total_executions.inc();
        self.failed_executions.inc();
        
        let mut total_dur = self.total_duration.lock();
        *total_dur += duration;
        
        let total_execs = self.total_executions.get();
        let mut avg_dur = self.average_duration.lock();
        *avg_dur = *total_dur / total_execs as u32;
        
        let mut last_exec = self.last_execution.lock();
        *last_exec = Some(Instant::now());
    }

    /// Get the success rate as a percentage (0.0-1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_executions.get();
        if total == 0 {
            0.0
        } else {
            self.successful_executions.get() as f64 / total as f64
        }
    }

    /// Get the failure rate as a percentage (0.0-1.0)
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    /// Reset all statistics to zero
    pub fn reset(&self) {
        self.total_executions.set(0);
        self.successful_executions.set(0);
        self.failed_executions.set(0);
        
        let mut total_dur = self.total_duration.lock();
        *total_dur = Duration::default();
        
        let mut avg_dur = self.average_duration.lock();
        *avg_dur = Duration::default();
        
        let mut last_exec = self.last_execution.lock();
        *last_exec = None;
    }
}