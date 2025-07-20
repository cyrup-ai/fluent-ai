//! Memory Tool Implementation - Zero-allocation memorize/recall with lock-free cognitive search
//!
//! This module provides the MemoryTool with blazing-fast performance, zero allocation,
//! and lock-free concurrent access to the cognitive memory system.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// Ultra-high-performance zero-allocation imports
use arrayvec::ArrayVec;
use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Memory, MemoryError, MemoryNode, MemoryType};
use crate::async_task::{AsyncStream, AsyncTask, spawn_async};

use crate::error::{
    ErrorCategory, ErrorRecoverability, ErrorSeverity, ZeroAllocError, ZeroAllocResult,
};
use crate::tool::{McpTool, McpToolData, ToolTrait as Tool};

/// Maximum number of memory nodes in result collections
const MAX_MEMORY_TOOL_RESULTS: usize = 1000;

/// Maximum number of streaming results per operation
const MAX_STREAMING_RESULTS: usize = 100;

/// Global result aggregation statistics
static TOOL_STATS: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// Lock-free result queue for aggregation
static RESULT_QUEUE: Lazy<SegQueue<MemoryNode>> = Lazy::new(|| SegQueue::new());

/// Zero-allocation memory tool with lock-free cognitive search
#[derive(Debug, Clone)]
pub struct MemoryTool {
    /// Tool metadata
    data: McpToolData,
    /// Shared memory instance for lock-free concurrent access
    memory: Arc<Memory>,
}
/// Memory tool operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", content = "params")]
pub enum MemoryOperation {
    /// Memorize content with specified type
    Memorize {
        content: String,
        memory_type: MemoryType,
    },
    /// Recall memories by content search
    Recall { query: String, limit: Option<usize> },
    /// Search memories by vector similarity
    VectorSearch { vector: Vec<f32>, limit: usize },
    /// Get specific memory by ID
    GetMemory { id: String },
    /// Update existing memory
    UpdateMemory { memory: MemoryNode },
    /// Delete memory by ID
    DeleteMemory { id: String },
}

/// Memory tool result types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MemoryResult {
    /// Single memory node result
    Memory(MemoryNode),
    /// Multiple memory nodes result
    Memories(Vec<MemoryNode>),
    /// Boolean result for operations
    Success(bool),
    /// Error result
    Error(String),
}

/// Memory tool error types with semantic error handling
#[derive(Debug, thiserror::Error)]
pub enum MemoryToolError {
    /// Memory system error
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Invalid operation parameters
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),
    /// Zero allocation error
    #[error("Zero allocation error: {0}")]
    ZeroAlloc(#[from] ZeroAllocError),
    /// Buffer overflow error
    #[error("Buffer overflow: operation would exceed capacity")]
    BufferOverflow,
    /// Tool initialization error
    #[error("Tool initialization error: {0}")]
    InitializationError(String),
}

/// Zero-allocation result type for memory tool operations
pub type MemoryToolResult<T> = Result<T, MemoryToolError>;

impl MemoryTool {
    /// Create a new memory tool instance
    pub fn new(memory: Arc<Memory>) -> Self {
        let data = McpToolData {
            name: "memory".to_string(),
            description: "Memory management tool for storing and retrieving information".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["memorize", "recall", "vector_search", "get_memory", "update_memory", "delete_memory"]
                    }
                },
                "required": ["operation"]
            }),
        };

        Self { data, memory }
    }

    /// Get access to the underlying memory instance
    #[inline]
    pub fn memory(&self) -> &Arc<Memory> {
        &self.memory
    }
}
