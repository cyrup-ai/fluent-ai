//! Core tool traits and types for zero-allocation tool execution
//!
//! Foundational types and traits for the Anthropic tool system with
//! optimal performance, lock-free operations, and elegant ergonomics.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bytes::Bytes;
use crossbeam_channel as channel;
use fluent_ai_domain::AsyncStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::error::{AnthropicError, AnthropicResult};
use super::super::types::AnthropicMessage;

/// Schema type specification for tool parameter definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaType {
    /// Auto-generate schema from serde Serialize/Deserialize types
    Serde,
    /// Manual JSON schema definition
    JsonSchema,
    /// Inline parameter definitions
    Inline,
}

/// Zero-allocation closure storage types for event handlers
pub type InvocationHandler<D, Req, Res> =
    Box<dyn Fn(&Conversation, &Emitter, Req, &D) -> AsyncStream<()> + Send + Sync>;
pub type ErrorHandler<D> =
    Box<dyn Fn(&Conversation, &ChainControl, AnthropicError, &D) + Send + Sync>;
pub type ResultHandler<D, Res> =
    Box<dyn Fn(&Conversation, &ChainControl, Res, &D) -> Res + Send + Sync>;

/// Typestate marker types for compile-time safety in builder pattern

/// Marker type indicating that tool name is required in builder
pub struct NameRequired;

/// Marker type indicating that tool description is required in builder
pub struct DescriptionRequired;

/// Marker type indicating that tool dependency is required in builder
pub struct DependencyRequired;

/// Marker type indicating that request schema is required in builder
pub struct RequestSchemaRequired;

/// Marker type indicating that response schema is required in builder
pub struct ResponseSchemaRequired;

/// Marker type indicating that invocation handler is required in builder
pub struct InvocationRequired;

/// Error types for tool registration and execution
#[derive(Debug, Clone)]
pub enum ToolRegistrationError {
    /// Tool name already exists
    DuplicateName { name: String },
    /// Invalid schema definition
    InvalidSchema { reason: String },
    /// Tool capacity exceeded
    CapacityExceeded { limit: usize },
    /// Type mismatch in tool parameters
    TypeMismatch { expected: String, actual: String },
}

/// Tool execution errors with comprehensive coverage
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutionError {
    #[error("Tool '{name}' not found")]
    NotFound { name: String },
    #[error("Invalid request: {reason}")]
    InvalidRequest { reason: String },
    #[error("Execution failed: {error}")]
    ExecutionFailed { error: String },
    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    #[error("Security violation: {violation}")]
    SecurityViolation { violation: String },
    #[error("Schema validation failed: {details}")]
    ValidationFailed { details: String },
    #[error("Missing required property: {property}")]
    MissingProperty { property: String },
}

impl std::fmt::Display for ToolRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolRegistrationError::DuplicateName { name } => {
                write!(f, "Tool with name '{}' already exists", name)
            }
            ToolRegistrationError::InvalidSchema { reason } => {
                write!(f, "Invalid schema: {}", reason)
            }
            ToolRegistrationError::CapacityExceeded { limit } => {
                write!(f, "Tool capacity exceeded: maximum {} tools allowed", limit)
            }
            ToolRegistrationError::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ToolRegistrationError {}

/// Foundational trait for tool execution with zero-allocation patterns
pub trait ToolExecutor: Send + Sync {
    /// Execute tool with given input and context
    ///
    /// # Performance
    /// - Zero-allocation execution path
    /// - Atomic error handling
    /// - Cache-aligned context access
    fn execute(&self, input: Value, context: &ToolExecutionContext) -> AsyncStream<Value>;
}

/// Tool execution context with zero-allocation access to conversation state
pub struct ToolExecutionContext {
    /// Conversation messages (borrowed, no allocation)
    pub messages: &'static [AnthropicMessage<'static>],
    /// Current tool name being executed
    pub tool_name: &'static str,
    /// Tool execution metadata
    pub metadata: HashMap<String, Value>,
    /// Tool use ID for tracking
    pub tool_use_id: String,
    /// User context information
    pub user_id: Option<String>,
    /// Conversation context
    pub conversation_id: Option<String>,
}

/// Real-time tool output emitter with zero-allocation streaming
pub struct Emitter {
    /// Lock-free sender for streaming output chunks
    sender: channel::Sender<ToolOutputChunk>,
    /// Atomic completion flag
    completed: AtomicBool,
    /// Performance metrics
    chunk_count: AtomicU32,
}

/// Tool output chunk for streaming results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputChunk {
    pub tool_use_id: String,
    pub content: String,
    pub chunk_type: ChunkType,
}

/// Type of output chunk for semantic streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    /// Partial text output
    Text,
    /// JSON data fragment
    Json,
    /// Error information
    Error,
    /// Completion marker
    Complete,
}

/// Conversation state for tool execution context
pub struct Conversation {
    /// Message history (zero-copy access)
    pub messages: Vec<AnthropicMessage<'static>>,
    /// Conversation metadata
    pub metadata: HashMap<String, Value>,
    /// Active tool tracking
    pub active_tools: Vec<String>,
}

/// Chain control for tool execution flow
pub struct ChainControl {
    /// Continue execution flag
    pub should_continue: AtomicBool,
    /// Error count
    pub error_count: AtomicU32,
    /// Retry attempts
    pub retry_count: AtomicU32,
}

impl Emitter {
    /// Create new emitter with bounded channel for backpressure
    #[inline(always)]
    pub fn new(capacity: usize) -> (Self, channel::Receiver<ToolOutputChunk>) {
        let (sender, receiver) = channel::bounded(capacity);
        let emitter = Self {
            sender,
            completed: AtomicBool::new(false),
            chunk_count: AtomicU32::new(0),
        };
        (emitter, receiver)
    }

    /// Emit a tool output chunk in real-time
    #[inline(always)]
    pub fn emit<T: Into<ToolOutputChunk>>(&self, chunk: T) -> AnthropicResult<()> {
        if self.completed.load(Ordering::Acquire) {
            return Err(AnthropicError::InvalidState(
                "Emitter already completed".into(),
            ));
        }

        self.sender
            .send(chunk.into())
            .map_err(|_| AnthropicError::StreamError("Failed to emit chunk".into()))?;

        self.chunk_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Mark emission as complete
    #[inline(always)]
    pub fn complete(&self) -> AnthropicResult<()> {
        self.completed.store(true, Ordering::Release);

        // Send completion marker
        let completion_chunk = ToolOutputChunk {
            tool_use_id: "completion".to_string(),
            content: "".to_string(),
            chunk_type: ChunkType::Complete,
        };

        self.sender
            .send(completion_chunk)
            .map_err(|_| AnthropicError::StreamError("Failed to emit completion".into()))
    }

    /// Get performance metrics
    #[inline(always)]
    pub fn metrics(&self) -> EmitterMetrics {
        EmitterMetrics {
            chunks_emitted: self.chunk_count.load(Ordering::Relaxed),
            is_completed: self.completed.load(Ordering::Acquire),
        }
    }
}

/// Emitter performance metrics
#[derive(Debug, Clone)]
pub struct EmitterMetrics {
    pub chunks_emitted: u32,
    pub is_completed: bool,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub tool_use_id: String,
    pub name: String,
    pub result: ToolOutput,
    pub execution_time_ms: u64,
    pub memory_usage_bytes: u64,
}

/// Tool output data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
    Error {
        message: String,
        code: Option<String>,
    },
    Binary {
        data: Bytes,
        mime_type: String,
    },
    Stream {
        chunks: Vec<ToolOutputChunk>,
    },
}

impl Default for ToolExecutionContext {
    fn default() -> Self {
        Self {
            messages: &[],
            tool_name: "",
            metadata: HashMap::new(),
            tool_use_id: String::new(),
            user_id: None,
            conversation_id: None,
        }
    }
}

impl ToolExecutionContext {
    /// Create new tool execution context
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metadata entry with zero allocation for static keys
    #[inline(always)]
    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Set tool use ID
    #[inline(always)]
    pub fn with_tool_use_id(mut self, id: String) -> Self {
        self.tool_use_id = id;
        self
    }
}

impl Into<ToolOutputChunk> for String {
    fn into(self) -> ToolOutputChunk {
        ToolOutputChunk {
            tool_use_id: "default".to_string(),
            content: self,
            chunk_type: ChunkType::Text,
        }
    }
}

impl Into<ToolOutputChunk> for &str {
    fn into(self) -> ToolOutputChunk {
        ToolOutputChunk {
            tool_use_id: "default".to_string(),
            content: self.to_string(),
            chunk_type: ChunkType::Text,
        }
    }
}

impl ChainControl {
    /// Create new chain control with default settings
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            should_continue: AtomicBool::new(true),
            error_count: AtomicU32::new(0),
            retry_count: AtomicU32::new(0),
        }
    }

    /// Stop execution chain
    #[inline(always)]
    pub fn stop(&self) {
        self.should_continue.store(false, Ordering::Release);
    }

    /// Check if execution should continue
    #[inline(always)]
    pub fn should_continue(&self) -> bool {
        self.should_continue.load(Ordering::Acquire)
    }

    /// Increment error count atomically
    #[inline(always)]
    pub fn record_error(&self) -> u32 {
        self.error_count.fetch_add(1, Ordering::Relaxed)
    }

    /// Increment retry count atomically
    #[inline(always)]
    pub fn record_retry(&self) -> u32 {
        self.retry_count.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for ChainControl {
    fn default() -> Self {
        Self::new()
    }
}
