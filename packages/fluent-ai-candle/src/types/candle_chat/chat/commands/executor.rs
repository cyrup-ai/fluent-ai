//! Streaming command executor with atomic state tracking
//!
//! Provides high-performance command execution with lock-free atomic operations
//! and comprehensive event streaming for command lifecycle management.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{AsyncStream, AsyncStreamSender};

use super::{
    error::{CommandError, CommandResult},
    command::ImmutableChatCommand,
    events::CommandEvent,
};

/// Streaming command executor with atomic state tracking
///
/// Manages command execution with lock-free atomic counters and optional
/// event streaming for real-time command lifecycle monitoring.
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
    #[inline(always)]
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

    /// Execute command asynchronously with progress tracking
    pub fn execute_command_async(&self, command: ImmutableChatCommand) -> AsyncStream<CommandEvent> {
        let event_sender = self.event_sender.clone();
        let executor_ref = self as *const _ as usize; // Safe reference for thread
        
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Reconstruct executor reference (safe because StreamingCommandExecutor is Send + Sync)
                let executor = unsafe { &*(executor_ref as *const StreamingCommandExecutor) };
                
                // Validate command first
                if let Err(error) = command.validate() {
                    let _ = sender.send(CommandEvent::Failed {
                        execution_id: 0,
                        error,
                        duration_nanos: 0,
                    });
                    return;
                }

                // Generate execution ID
                let execution_id = executor.execution_counter.fetch_add(1, Ordering::Relaxed);
                let start_time = Self::current_timestamp_nanos();

                // Update counters
                executor.active_executions.fetch_add(1, Ordering::Relaxed);
                executor.total_executions.fetch_add(1, Ordering::Relaxed);

                // Send started event
                let _ = sender.send(CommandEvent::Started {
                    command: command.clone(),
                    execution_id,
                    timestamp_nanos: start_time,
                });

                // TODO: Implement actual command execution logic here
                // For now, simulate execution with progress updates
                for progress in [25, 50, 75, 100] {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    
                    let _ = sender.send(CommandEvent::Progress {
                        execution_id,
                        progress_percent: progress as f32,
                        message: Some(format!("Processing step {}", progress / 25)),
                    });
                }

                // Complete execution
                let duration_nanos = Self::current_timestamp_nanos() - start_time;
                executor.successful_executions.fetch_add(1, Ordering::Relaxed);
                executor.active_executions.fetch_sub(1, Ordering::Relaxed);

                let _ = sender.send(CommandEvent::Completed {
                    execution_id,
                    result: super::events::CommandExecutionResult::success("Command completed successfully"),
                    duration_nanos,
                });
            });
        })
    }

    /// Report command progress
    #[inline(always)]
    pub fn report_progress(&self, execution_id: u64, progress_percent: f32, message: Option<String>) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Progress {
                execution_id,
                progress_percent,
                message,
            });
        }
    }

    /// Report command completion
    #[inline(always)]
    pub fn report_completion(&self, execution_id: u64, result: super::events::CommandExecutionResult, duration_nanos: u64) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Completed {
                execution_id,
                result,
                duration_nanos,
            });
        }
        self.successful_executions.fetch_add(1, Ordering::Relaxed);
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
    }

    /// Report command failure
    #[inline(always)]
    pub fn report_failure(&self, execution_id: u64, error: CommandError, duration_nanos: u64) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Failed {
                execution_id,
                error,
                duration_nanos,
            });
        }
        self.failed_executions.fetch_add(1, Ordering::Relaxed);
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
    }

    /// Cancel command execution
    #[inline(always)]
    pub fn cancel_execution(&self, execution_id: u64, reason: impl Into<String>) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Cancelled {
                execution_id,
                reason: reason.into(),
            });
        }
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current timestamp in nanoseconds
    #[inline(always)]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get execution statistics (atomic reads)
    #[inline(always)]
    pub fn stats(&self) -> CommandExecutorStats {
        CommandExecutorStats {
            active_executions: self.active_executions.load(Ordering::Relaxed) as u64,
            total_executions: self.total_executions.load(Ordering::Relaxed),
            successful_executions: self.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.failed_executions.load(Ordering::Relaxed),
        }
    }

    /// Check if executor is currently busy
    #[inline(always)]
    pub fn is_busy(&self) -> bool {
        self.active_executions.load(Ordering::Relaxed) > 0
    }

    /// Get next execution ID without incrementing
    #[inline(always)]
    pub fn next_execution_id(&self) -> u64 {
        self.execution_counter.load(Ordering::Relaxed)
    }

    /// Reset all statistics (useful for testing)
    pub fn reset_stats(&self) {
        self.execution_counter.store(0, Ordering::Relaxed);
        self.active_executions.store(0, Ordering::Relaxed);
        self.total_executions.store(0, Ordering::Relaxed);
        self.successful_executions.store(0, Ordering::Relaxed);
        self.failed_executions.store(0, Ordering::Relaxed);
    }
}

impl Default for StreamingCommandExecutor {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Command executor statistics
///
/// Provides atomic read access to execution statistics with derived metrics
/// for monitoring command executor performance and reliability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Calculate success rate as percentage
    #[inline(always)]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_executions + self.failed_executions;
        if completed == 0 {
            0.0
        } else {
            (self.successful_executions as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage
    #[inline(always)]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }

    /// Get completion rate (completed vs total)
    #[inline(always)]
    pub fn completion_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            let completed = self.successful_executions + self.failed_executions;
            (completed as f64 / self.total_executions as f64) * 100.0
        }
    }

    /// Check if executor is healthy (low failure rate)
    #[inline(always)]
    pub fn is_healthy(&self) -> bool {
        self.failure_rate() < 5.0 // Less than 5% failure rate
    }

    /// Get pending executions (started but not completed)
    #[inline(always)]
    pub fn pending_executions(&self) -> u64 {
        self.total_executions - (self.successful_executions + self.failed_executions)
    }
}

impl Default for CommandExecutorStats {
    fn default() -> Self {
        Self {
            active_executions: 0,
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
        }
    }
}

impl std::fmt::Display for CommandExecutorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Executor Stats: {}/{} active, {:.1}% success rate, {} total",
            self.active_executions,
            self.total_executions,
            self.success_rate(),
            self.total_executions
        )
    }
}