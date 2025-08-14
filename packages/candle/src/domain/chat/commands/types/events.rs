//! Command execution events and context tracking with zero allocation patterns
//!
//! Provides blazing-fast event streaming and execution context management
//! with owned strings allocated once for maximum performance. No Arc usage, no locking.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::commands::{CommandExecutionResult, ImmutableChatCommand, OutputType};
use super::metadata::ResourceUsage;
use crate::AsyncStreamSender;

/// Command execution context with owned strings allocated once for performance
#[derive(Debug)]
pub struct CommandExecutionContext {
    /// Unique identifier for this execution
    pub execution_id: u64,
    /// Command name being executed (owned string allocated once)
    pub command_name: String,
    /// Start time in microseconds since epoch
    pub start_time: u64,
    /// Resource usage tracking
    pub resource_usage: ResourceUsage,
    /// User identifier (owned string allocated once)
    pub user_id: Option<String>,
    /// Session identifier (owned string allocated once)
    pub session_id: Option<String>,
    /// Execution environment (owned string allocated once)
    pub environment: String,
    /// Execution priority (1-255, lower is higher priority)
    pub priority: u8,
    /// Maximum allowed execution time in milliseconds
    pub timeout_ms: u64,
    /// Execution metadata
    pub metadata: Option<serde_json::Value>,
    /// Atomic execution counter for sequence numbering
    execution_counter: AtomicU64,
    /// Atomic event counter for event sequence numbering
    event_counter: AtomicUsize,
}

impl CommandExecutionContext {
    /// Create new execution context with current timestamp
    #[inline]
    pub fn new(
        execution_id: u64,
        command_name: impl Into<String>,
        environment: impl Into<String>,
    ) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Self {
            execution_id,
            command_name: command_name.into(),
            start_time,
            resource_usage: ResourceUsage::new_with_start_time(),
            user_id: None,
            session_id: None,
            environment: environment.into(),
            priority: 128,      // Default medium priority
            timeout_ms: 30_000, // Default 30 second timeout
            metadata: None,
            execution_counter: AtomicU64::new(0),
            event_counter: AtomicUsize::new(0),
        }
    }

    /// Set user identifier - builder pattern for fluent API
    #[inline]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set session identifier - builder pattern for fluent API
    #[inline]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set execution priority - builder pattern for fluent API
    #[inline]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set execution timeout - builder pattern for fluent API
    #[inline]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set metadata - builder pattern for fluent API
    #[inline]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get elapsed execution time in microseconds
    #[inline]
    pub fn elapsed_time_us(&self) -> u64 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(self.start_time);
        current_time.saturating_sub(self.start_time)
    }

    /// Check if execution has timed out
    #[inline]
    pub fn is_timed_out(&self) -> bool {
        self.elapsed_time_us() > (self.timeout_ms * 1000)
    }

    /// Get remaining time in microseconds before timeout
    #[inline]
    pub fn remaining_time_us(&self) -> u64 {
        let timeout_us = self.timeout_ms * 1000;
        let elapsed = self.elapsed_time_us();
        timeout_us.saturating_sub(elapsed)
    }

    /// Get next execution sequence number (atomic operation)
    #[inline]
    pub fn next_execution_id(&self) -> u64 {
        self.execution_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get next event sequence number (atomic operation)
    #[inline]
    pub fn next_event_id(&self) -> usize {
        self.event_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get elapsed time in microseconds (alias for elapsed_time_us for compatibility)
    #[inline]
    pub fn elapsed_time(&self) -> u64 {
        self.elapsed_time_us()
    }
}

/// Command execution event for streaming with zero allocation where possible
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandEvent {
    /// Command started executing
    Started {
        /// The command that was started
        command: ImmutableChatCommand,
        /// Unique identifier for this execution
        execution_id: u64,
        /// Start time in microseconds since epoch
        timestamp_us: u64,
    },
    /// Command execution progress
    Progress {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Progress percentage (0-100)
        progress: f32,
        /// Status message
        message: String,
        /// Timestamp
        timestamp: u64,
    },
    /// Command produced output
    Output {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Output content
        content: String,
        /// Type/format of the output
        output_type: OutputType,
        /// Current timestamp in microseconds since epoch
        timestamp_us: u64,
    },
    /// Command completed successfully
    Completed {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Final result of the execution
        result: CommandExecutionResult,
        /// Total execution time in microseconds
        duration_us: u64,
        /// Final resource usage
        resource_usage: ResourceUsage,
        /// Completion timestamp in microseconds since epoch
        timestamp_us: u64,
    },
    /// Command execution failed
    Failed {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Error message (owned string allocated once)
        error: String,
        /// Error code for categorization
        error_code: u32,
        /// Total execution time in microseconds
        duration_us: u64,
        /// Resource usage at failure
        resource_usage: ResourceUsage,
        /// Failure timestamp in microseconds since epoch
        timestamp_us: u64,
    },
    /// Command execution was cancelled
    Cancelled {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Cancellation reason (owned string allocated once)
        reason: String,
        /// Partial execution time in microseconds
        duration_us: u64,
        /// Resource usage at cancellation
        resource_usage: ResourceUsage,
        /// Cancellation timestamp in microseconds since epoch
        timestamp_us: u64,
    },
    /// Command execution warning
    Warning {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Warning message (owned string allocated once)
        message: String,
        /// Warning severity (1=low, 2=medium, 3=high, 4=critical)
        severity: u8,
        /// Warning timestamp in microseconds since epoch
        timestamp_us: u64,
    },
    /// Resource threshold exceeded
    ResourceAlert {
        /// Unique identifier for this execution
        execution_id: u64,
        /// Resource type that exceeded threshold (owned string allocated once)
        resource_type: String,
        /// Current usage value
        current_value: u64,
        /// Threshold value that was exceeded
        threshold_value: u64,
        /// Alert timestamp in microseconds since epoch
        timestamp_us: u64,
    },
}

impl CommandEvent {
    /// Create started event with zero allocation constructor
    #[inline]
    pub fn started(command: ImmutableChatCommand, execution_id: u64) -> Self {
        Self::Started {
            command,
            execution_id,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create progress event with zero allocation constructor
    #[inline]
    pub fn progress(execution_id: u64, progress: f32, message: String) -> Self {
        Self::Progress {
            execution_id,
            progress,
            message,
            timestamp: Self::current_timestamp_us(),
        }
    }

    /// Create output event with zero allocation constructor
    #[inline]
    pub fn output(execution_id: u64, output: impl Into<String>, output_type: OutputType) -> Self {
        Self::Output {
            execution_id,
            content: output.into(),
            output_type,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create completed event with zero allocation constructor
    #[inline]
    pub fn completed(
        execution_id: u64,
        result: CommandExecutionResult,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) -> Self {
        Self::Completed {
            execution_id,
            result,
            duration_us,
            resource_usage,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create failed event with zero allocation constructor
    #[inline]
    pub fn failed(
        execution_id: u64,
        error: impl Into<String>,
        error_code: u32,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) -> Self {
        Self::Failed {
            execution_id,
            error: error.into(),
            error_code,
            duration_us,
            resource_usage,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create cancelled event with zero allocation constructor
    #[inline]
    pub fn cancelled(
        execution_id: u64,
        reason: impl Into<String>,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) -> Self {
        Self::Cancelled {
            execution_id,
            reason: reason.into(),
            duration_us,
            resource_usage,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create warning event with zero allocation constructor
    #[inline]
    pub fn warning(execution_id: u64, message: impl Into<String>, severity: u8) -> Self {
        Self::Warning {
            execution_id,
            message: message.into(),
            severity,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Create resource alert event with zero allocation constructor
    #[inline]
    pub fn resource_alert(
        execution_id: u64,
        resource_type: impl Into<String>,
        current_value: u64,
        threshold_value: u64,
    ) -> Self {
        Self::ResourceAlert {
            execution_id,
            resource_type: resource_type.into(),
            current_value,
            threshold_value,
            timestamp_us: Self::current_timestamp_us(),
        }
    }

    /// Get current timestamp in microseconds since epoch
    #[inline]
    fn current_timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    /// Get execution ID from any event type
    #[inline]
    pub const fn execution_id(&self) -> u64 {
        match self {
            Self::Started { execution_id, .. }
            | Self::Progress { execution_id, .. }
            | Self::Output { execution_id, .. }
            | Self::Completed { execution_id, .. }
            | Self::Failed { execution_id, .. }
            | Self::Cancelled { execution_id, .. }
            | Self::Warning { execution_id, .. }
            | Self::ResourceAlert { execution_id, .. } => *execution_id,
        }
    }

    /// Get timestamp from any event type
    #[inline]
    pub const fn timestamp_us(&self) -> u64 {
        match self {
            Self::Started { timestamp_us, .. }
            | Self::Output { timestamp_us, .. }
            | Self::Completed { timestamp_us, .. }
            | Self::Failed { timestamp_us, .. }
            | Self::Cancelled { timestamp_us, .. }
            | Self::Warning { timestamp_us, .. }
            | Self::ResourceAlert { timestamp_us, .. } => *timestamp_us,
            Self::Progress { timestamp, .. } => *timestamp,
        }
    }

    /// Check if event indicates completion (success, failure, or cancellation)
    #[inline]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled { .. }
        )
    }

    /// Check if event indicates success
    #[inline]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Check if event indicates failure
    #[inline]
    pub const fn is_failure(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// Check if event indicates cancellation
    #[inline]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }

    /// Get event severity level for filtering and monitoring
    #[inline]
    pub const fn severity(&self) -> u8 {
        match self {
            Self::Started { .. } => 0,
            Self::Progress { .. } => 0,
            Self::Output { .. } => 0,
            Self::Completed { .. } => 1,
            Self::Warning { severity, .. } => *severity,
            Self::ResourceAlert { .. } => 2,
            Self::Failed { .. } => 3,
            Self::Cancelled { .. } => 2,
        }
    }
}

/// Streaming command executor with atomic state tracking for zero allocation
pub struct StreamingCommandExecutor {
    /// Execution counter (atomic) - thread-safe incrementing
    execution_counter: AtomicU64,
    /// Active executions (atomic) - current running count
    active_executions: AtomicUsize,
    /// Total executions (atomic) - lifetime total
    total_executions: AtomicU64,
    /// Successful executions (atomic) - success count
    successful_executions: AtomicU64,
    /// Failed executions (atomic) - failure count
    failed_executions: AtomicU64,
    /// Event stream sender for broadcasting events
    event_sender: Option<AsyncStreamSender<CommandEvent>>,
}

impl StreamingCommandExecutor {
    /// Create new streaming command executor with zero allocation
    #[inline]
    pub const fn new() -> Self {
        Self {
            execution_counter: AtomicU64::new(0),
            active_executions: AtomicUsize::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            event_sender: None,
        }
    }

    /// Create with event sender for streaming events
    #[inline]
    pub fn with_event_sender(event_sender: AsyncStreamSender<CommandEvent>) -> Self {
        Self {
            execution_counter: AtomicU64::new(0),
            active_executions: AtomicUsize::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            event_sender: Some(event_sender),
        }
    }

    /// Start new command execution and return unique ID
    #[inline]
    pub fn start_execution(&self, command: &ImmutableChatCommand) -> u64 {
        let execution_id = self.execution_counter.fetch_add(1, Ordering::Relaxed);
        self.active_executions.fetch_add(1, Ordering::Relaxed);
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        // Send started event if sender available
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::started(command.clone(), execution_id);
            if sender.send(event).is_err() {
                // Event channel closed, continue execution
            }
        }

        execution_id
    }

    /// Mark execution as completed successfully
    #[inline]
    pub fn complete_execution(
        &self,
        execution_id: u64,
        result: CommandExecutionResult,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) {
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
        self.successful_executions.fetch_add(1, Ordering::Relaxed);

        // Send completed event if sender available
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::completed(execution_id, result, duration_us, resource_usage);
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Mark execution as failed
    #[inline]
    pub fn fail_execution(
        &self,
        execution_id: u64,
        error: impl Into<String>,
        error_code: u32,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) {
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
        self.failed_executions.fetch_add(1, Ordering::Relaxed);

        // Send failed event if sender available
        if let Some(ref sender) = self.event_sender {
            let event =
                CommandEvent::failed(execution_id, error, error_code, duration_us, resource_usage);
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Cancel command execution
    #[inline]
    pub fn cancel_execution(
        &self,
        execution_id: u64,
        reason: impl Into<String>,
        duration_us: u64,
        resource_usage: ResourceUsage,
    ) {
        self.active_executions.fetch_sub(1, Ordering::Relaxed);

        // Send cancelled event if sender available
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::cancelled(execution_id, reason, duration_us, resource_usage);
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Send progress update
    #[inline]
    pub fn send_progress(&self, execution_id: u64, progress: f32, message: Option<String>) {
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::progress(execution_id, progress, message.unwrap_or_default());
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Send output data
    #[inline]
    pub fn send_output(
        &self,
        execution_id: u64,
        output: impl Into<String>,
        output_type: OutputType,
    ) {
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::output(execution_id, output, output_type);
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Send warning
    #[inline]
    pub fn send_warning(&self, execution_id: u64, message: impl Into<String>, severity: u8) {
        if let Some(ref sender) = self.event_sender {
            let event = CommandEvent::warning(execution_id, message, severity);
            if sender.send(event).is_err() {
                // Event channel closed, continue
            }
        }
    }

    /// Get execution statistics (atomic reads) - zero allocation
    #[inline]
    pub fn stats(&self) -> CommandExecutorStats {
        CommandExecutorStats {
            active_executions: self.active_executions.load(Ordering::Relaxed) as u64,
            total_executions: self.total_executions.load(Ordering::Relaxed),
            successful_executions: self.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.failed_executions.load(Ordering::Relaxed),
        }
    }

    /// Get next execution ID without starting execution
    #[inline]
    pub fn peek_next_execution_id(&self) -> u64 {
        self.execution_counter.load(Ordering::Relaxed)
    }
}

impl Default for StreamingCommandExecutor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Command executor statistics with zero allocation accessors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandExecutorStats {
    /// Number of currently active executions
    pub active_executions: u64,
    /// Total number of executions started
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
}

impl CommandExecutorStats {
    /// Calculate success rate as percentage - zero allocation
    #[inline]
    pub const fn success_rate(&self) -> f64 {
        let completed = self.successful_executions + self.failed_executions;
        if completed == 0 {
            0.0
        } else {
            (self.successful_executions as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage - zero allocation
    #[inline]
    pub const fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }

    /// Get completion rate (completed vs total) - zero allocation
    #[inline]
    pub const fn completion_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            let completed = self.successful_executions + self.failed_executions;
            (completed as f64 / self.total_executions as f64) * 100.0
        }
    }

    /// Check if any executions are currently active
    #[inline]
    pub const fn has_active_executions(&self) -> bool {
        self.active_executions > 0
    }

    /// Check if system is idle (no active executions)
    #[inline]
    pub const fn is_idle(&self) -> bool {
        self.active_executions == 0
    }
}

impl std::fmt::Debug for StreamingCommandExecutor {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingCommandExecutor")
            .field(
                "execution_counter",
                &self.execution_counter.load(Ordering::Relaxed),
            )
            .field(
                "active_executions",
                &self.active_executions.load(Ordering::Relaxed),
            )
            .field(
                "total_executions",
                &self.total_executions.load(Ordering::Relaxed),
            )
            .field(
                "successful_executions",
                &self.successful_executions.load(Ordering::Relaxed),
            )
            .field(
                "failed_executions",
                &self.failed_executions.load(Ordering::Relaxed),
            )
            .field("event_sender", &self.event_sender.is_some())
            .finish()
    }
}
