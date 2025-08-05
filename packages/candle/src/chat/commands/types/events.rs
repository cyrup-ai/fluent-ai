//! Command execution events and context
//!
//! This module provides event types and execution context for streaming
//! command execution with zero-allocation patterns.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use serde::{Deserialize, Serialize};
use crate::{AsyncStream, AsyncStreamSender};
use super::{errors::CommandResult, metadata::ResourceUsage};

/// Command execution context with streaming capabilities
#[derive(Debug)]
pub struct CommandExecutionContext {
    /// Unique execution ID
    pub execution_id: u64,
    /// Command name
    pub command_name: String,
    /// Start time in microseconds since epoch
    pub start_time: u64,
    /// Resource usage tracking
    pub resource_usage: ResourceUsage,
    /// Execution counter
    execution_counter: AtomicU64,
    /// Event counter
    event_counter: AtomicUsize,
}

impl CommandExecutionContext {
    /// Create new execution context
    pub fn new(command_name: String) -> Self {
        let execution_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            execution_id,
            command_name,
            start_time,
            resource_usage: ResourceUsage::new(),
            execution_counter: AtomicU64::new(0),
            event_counter: AtomicUsize::new(0),
        }
    }

    /// Get next execution sequence number
    pub fn next_execution_id(&self) -> u64 {
        self.execution_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get next event sequence number  
    pub fn next_event_id(&self) -> usize {
        self.event_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_time(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        now.saturating_sub(self.start_time)
    }
}

/// Command execution events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandEvent {
    /// Command execution started
    Started {
        /// Execution ID
        execution_id: u64,
        /// Command name
        command: String,
        /// Timestamp
        timestamp: u64,
    },
    /// Progress update
    Progress {
        /// Execution ID
        execution_id: u64,
        /// Progress percentage (0-100)
        progress: f32,
        /// Status message
        message: String,
        /// Timestamp
        timestamp: u64,
    },
    /// Output generated
    Output {
        /// Execution ID
        execution_id: u64,
        /// Output content
        content: String,
        /// Output type (stdout, stderr, etc.)
        output_type: String,
        /// Timestamp
        timestamp: u64,
    },
    /// Error occurred
    Error {
        /// Execution ID
        execution_id: u64,
        /// Error message
        error: String,
        /// Error category
        category: String,
        /// Timestamp
        timestamp: u64,
    },
    /// Command execution completed
    Completed {
        /// Execution ID
        execution_id: u64,
        /// Success status
        success: bool,
        /// Execution time in microseconds
        execution_time_us: u64,
        /// Resource usage
        resource_usage: ResourceUsage,
        /// Timestamp
        timestamp: u64,
    },
}

impl CommandEvent {
    /// Create a Started event
    pub fn started(execution_id: u64, command: String) -> Self {
        Self::Started {
            execution_id,
            command,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Create a Progress event
    pub fn progress(execution_id: u64, progress: f32, message: String) -> Self {
        Self::Progress {
            execution_id,
            progress,
            message,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Create an Output event
    pub fn output(execution_id: u64, content: String, output_type: String) -> Self {
        Self::Output {
            execution_id,
            content,
            output_type,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Create an Error event
    pub fn error(execution_id: u64, error: String, category: String) -> Self {
        Self::Error {
            execution_id,
            error,
            category,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Create a Completed event
    pub fn completed(execution_id: u64, success: bool, execution_time_us: u64, resource_usage: ResourceUsage) -> Self {
        Self::Completed {
            execution_id,
            success,
            execution_time_us,
            resource_usage,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Get event execution ID
    pub fn execution_id(&self) -> u64 {
        match self {
            CommandEvent::Started { execution_id, .. }
            | CommandEvent::Progress { execution_id, .. }
            | CommandEvent::Output { execution_id, .. }
            | CommandEvent::Error { execution_id, .. }
            | CommandEvent::Completed { execution_id, .. } => *execution_id,
        }
    }

    /// Get event timestamp
    pub fn timestamp(&self) -> u64 {
        match self {
            CommandEvent::Started { timestamp, .. }
            | CommandEvent::Progress { timestamp, .. }
            | CommandEvent::Output { timestamp, .. }
            | CommandEvent::Error { timestamp, .. }
            | CommandEvent::Completed { timestamp, .. } => *timestamp,
        }
    }
}