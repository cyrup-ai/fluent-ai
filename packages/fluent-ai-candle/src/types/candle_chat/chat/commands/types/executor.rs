//! Streaming command executor with atomic state tracking
//!
//! Provides high-performance command execution with atomic counters,
//! event streaming, and zero-allocation patterns for maximum throughput.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{AsyncStream, AsyncStreamSender};

use super::command::ImmutableChatCommand;
use super::core::CommandResult;
use super::events::{CommandEvent, CommandExecutionResult};

/// Streaming command executor with atomic state tracking
pub struct StreamingCommandExecutor {
    /// Execution counter (atomic)
    execution_counter: AtomicU64,
    /// Active executions (atomic)
    active_executions: AtomicUsize,
    /// Total executions (atomic)
    total_executions: AtomicU64,
    /// Successful executions (atomic)
    successful_executions: AtomicU64,
    /// Failed executions (atomic)
    failed_executions: AtomicU64,
    /// Event stream sender
    event_sender: Option<AsyncStreamSender<CommandEvent>>,
}

impl StreamingCommandExecutor {
    /// Create new streaming command executor
    #[inline]
    pub fn new() -> Self {
        Self {
            execution_counter: AtomicU64::new(0),
            active_executions: AtomicUsize::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            event_sender: None,
        }
    }

    /// Create executor with event streaming
    #[inline]
    pub fn with_streaming() -> (Self, AsyncStream<CommandEvent>) {
        // Create a channel for storing the sender
        let (tx, rx) = std::sync::mpsc::channel();

        // Create AsyncStream that will receive the sender
        let stream = AsyncStream::with_channel(move |sender| {
            // Send the sender through the channel so we can store it
            let _ = tx.send(sender);
            // Keep the thread alive but don't emit any events initially
            std::thread::park();
        });

        // Get the sender from the channel
        let event_sender = rx.recv().ok();

        let executor = Self {
            execution_counter: AtomicU64::new(0),
            active_executions: AtomicUsize::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            event_sender,
        };

        (executor, stream)
    }

    /// Execute command with streaming events
    #[inline]
    pub fn execute_command(&self, command: ImmutableChatCommand) -> CommandResult<u64> {
        // Validate command first
        command.validate()?;

        // Generate execution ID
        let execution_id = self.execution_counter.fetch_add(1, Ordering::Relaxed);

        // Update counters
        self.active_executions.fetch_add(1, Ordering::Relaxed);
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        // Send started event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Started {
                command: command.clone(),
                execution_id,
                timestamp_nanos: Self::current_timestamp_nanos(),
            });
        }

        // TODO: Implement actual command execution logic here
        // This would integrate with the command system to execute commands

        Ok(execution_id)
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get execution statistics (atomic reads)
    #[inline]
    pub fn stats(&self) -> CommandExecutorStats {
        CommandExecutorStats {
            active_executions: self.active_executions.load(Ordering::Relaxed) as u64,
            total_executions: self.total_executions.load(Ordering::Relaxed),
            successful_executions: self.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.failed_executions.load(Ordering::Relaxed),
        }
    }

    /// Cancel command execution
    #[inline]
    pub fn cancel_execution(&self, execution_id: u64, reason: impl Into<String>) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Cancelled {
                execution_id,
                reason: reason.into(),
            });
        }
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Default for StreamingCommandExecutor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Command executor statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandExecutorStats {
    pub active_executions: u64,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
}

impl CommandExecutorStats {
    /// Calculate success rate as percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_executions + self.failed_executions;
        if completed == 0 {
            0.0
        } else {
            (self.successful_executions as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }
}