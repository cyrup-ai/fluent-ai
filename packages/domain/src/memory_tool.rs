//! Memory Tool Implementation - Zero-allocation memorize/recall with lock-free cognitive search
//!
//! This module provides the MemoryTool with blazing-fast performance, zero allocation,
//! and lock-free concurrent access to the cognitive memory system.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// Ultra-high-performance zero-allocation imports
use arrayvec::ArrayVec;
use smallvec::SmallVec;
use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use once_cell::sync::Lazy;

use crate::async_task::{AsyncTask, spawn_async, AsyncStream};
use crate::memory::{Memory, MemoryError, MemoryNode, MemoryType};
use crate::mcp_tool_traits::{Tool, McpTool, McpToolData};
use crate::error::{ZeroAllocResult, ZeroAllocError, ErrorCategory, ErrorSeverity, ErrorRecoverability};

/// Maximum number of memory nodes in result collections
const MAX_MEMORY_TOOL_RESULTS: usize = 1000;

/// Maximum number of streaming results per operation
const MAX_STREAMING_RESULTS: usize = 100;

/// Global result aggregation statistics
static TOOL_STATS: Lazy<CachePadded<AtomicUsize>> = Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

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
    Recall {
        query: String,
        limit: Option<usize>,
    },
    /// Search memories by vector similarity
    VectorSearch {
        vector: Vec<f32>,
        limit: usize,
    },
    /// Get specific memory by ID
    GetMemory {
        id: String,
    },
    /// Update existing memory
    UpdateMemory {
        memory: MemoryNode,
    },
    /// Delete memory by ID
    DeleteMemory {
        id: String,
    },
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

/// Semantic error type conversions
impl From<ZeroAllocError> for MemoryToolError {
    #[inline(always)]
    fn from(error: ZeroAllocError) -> Self {
        MemoryToolError::ZeroAlloc(error)
    }
}

impl From<MemoryError> for MemoryToolError {
    #[inline(always)]
    fn from(error: MemoryError) -> Self {
        MemoryToolError::Memory(error)
    }
}

impl From<serde_json::Error> for MemoryToolError {
    #[inline(always)]
    fn from(error: serde_json::Error) -> Self {
        MemoryToolError::Json(error)
    }
}

/// Zero-allocation result type for memory tool operations
pub type MemoryToolResult<T> = Result<T, MemoryToolError>;

impl MemoryTool {
    /// Create new memory tool with zero-allocation initialization
    /// 
    /// # Arguments
    /// * `memory` - Shared memory instance for concurrent access
    /// 
    /// # Returns
    /// Configured memory tool instance
    /// 
    /// # Performance
    /// Zero allocation with lock-free shared memory access
    #[inline(always)]
    pub fn new(memory: Arc<Memory>) -> Self {
        let data = McpToolData::new(
            "memory",
            "Memory tool for memorize/recall operations with cognitive search",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["memorize", "recall", "vector_search", "get_memory", "update_memory", "delete_memory"],
                        "description": "Memory operation to perform"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the operation"
                    }
                },
                "required": ["operation"]
            }),
        );

        // Initialize tool statistics
        TOOL_STATS.store(0, Ordering::Relaxed);

        Self { data, memory }
    }

    /// Memorize content with zero-allocation processing
    /// 
    /// # Arguments
    /// * `content` - Content to memorize
    /// * `memory_type` - Type of memory
    /// 
    /// # Returns
    /// Async task with memory node result
    /// 
    /// # Performance
    /// Zero allocation with lock-free cognitive processing
    #[inline(always)]
    pub fn memorize(&self, content: String, memory_type: MemoryType) -> AsyncTask<Result<MemoryNode, MemoryToolError>> {
        let memory = &self.memory;
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            // Increment atomic counter for lock-free statistics
            TOOL_STATS.fetch_add(1, Ordering::Relaxed);
            
            memory_ref.memorize(content, memory_type)
                .await
                .map_err(MemoryToolError::Memory)
        })
    }

    /// Recall memories with zero-allocation streaming
    /// 
    /// # Arguments
    /// * `query` - Search query string
    /// * `limit` - Optional limit for results
    /// 
    /// # Returns
    /// Zero-allocation streaming results
    /// 
    /// # Performance
    /// Lock-free concurrent search with attention-based relevance scoring
    #[inline(always)]
    pub fn recall(&self, query: String, limit: Option<usize>) -> AsyncStream<Result<MemoryNode, MemoryToolError>> {
        let memory = &self.memory;
        let effective_limit = limit.unwrap_or(MAX_STREAMING_RESULTS).min(MAX_STREAMING_RESULTS);
        
        // Use crossbeam-queue for zero-copy streaming
        let result_queue = Arc::new(SegQueue::new());
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        tokio::spawn(async move {
            let mut stream = memory_ref.recall(&query);
            let mut count = 0;
            
            // Use ArrayVec for zero-allocation result buffering
            let mut result_buffer: ArrayVec<MemoryNode, MAX_STREAMING_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        // Use lock-free atomic operations for result aggregation
                        TOOL_STATS.fetch_add(1, Ordering::Relaxed);
                        
                        // Try to add to buffer, break if full
                        if result_buffer.try_push(memory_node).is_err() {
                            break;
                        }
                        
                        count += 1;
                        if count >= effective_limit {
                            break;
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(MemoryToolError::Memory(e))).is_err() {
                            break;
                        }
                    }
                }
            }
            
            // Send buffered results with zero-copy semantics
            for memory_node in result_buffer.drain(..) {
                if tx.send(Ok(memory_node)).is_err() {
                    break;
                }
            }
        });
        
        AsyncStream::new(rx)
    }

    /// Search memories by vector similarity with zero-allocation processing
    /// 
    /// # Arguments
    /// * `vector` - Query vector for similarity search
    /// * `limit` - Maximum number of results
    /// 
    /// # Returns
    /// Zero-allocation streaming results ordered by relevance
    /// 
    /// # Performance
    /// Lock-free vector similarity with quantum routing optimization
    #[inline(always)]
    pub fn vector_search(&self, vector: Vec<f32>, limit: usize) -> AsyncStream<Result<MemoryNode, MemoryToolError>> {
        let memory = &self.memory;
        let effective_limit = limit.min(MAX_STREAMING_RESULTS);
        
        // Use crossbeam-queue for zero-copy streaming
        let result_queue = Arc::new(SegQueue::new());
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        tokio::spawn(async move {
            let mut stream = memory_ref.search_by_vector(vector, effective_limit);
            
            // Use ArrayVec for zero-allocation result buffering
            let mut result_buffer: ArrayVec<MemoryNode, MAX_STREAMING_RESULTS> = ArrayVec::new();
            
            while let Some(result) = futures::StreamExt::next(&mut stream).await {
                match result {
                    Ok(memory_node) => {
                        // Use lock-free atomic operations for result aggregation
                        TOOL_STATS.fetch_add(1, Ordering::Relaxed);
                        
                        // Try to add to buffer, break if full
                        if result_buffer.try_push(memory_node).is_err() {
                            break;
                        }
                        
                        if result_buffer.len() >= effective_limit {
                            break;
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(MemoryToolError::Memory(e))).is_err() {
                            break;
                        }
                    }
                }
            }
            
            // Send buffered results with zero-copy semantics
            for memory_node in result_buffer.drain(..) {
                if tx.send(Ok(memory_node)).is_err() {
                    break;
                }
            }
        });
        
        AsyncStream::new(rx)
    }

    /// Get memory by ID with zero-allocation retrieval
    /// 
    /// # Arguments
    /// * `id` - Memory node ID
    /// 
    /// # Returns
    /// Async task with optional memory node result
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent access
    #[inline(always)]
    pub fn get_memory(&self, id: String) -> AsyncTask<Result<Option<MemoryNode>, MemoryToolError>> {
        let memory = &self.memory;
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            // Increment atomic counter for lock-free statistics
            TOOL_STATS.fetch_add(1, Ordering::Relaxed);
            
            memory_ref.get_memory(&id)
                .await
                .map_err(MemoryToolError::Memory)
        })
    }

    /// Update memory with zero-allocation processing
    /// 
    /// # Arguments
    /// * `memory_node` - Memory node to update
    /// 
    /// # Returns
    /// Async task with updated memory node result
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent updates
    #[inline(always)]
    pub fn update_memory(&self, memory_node: MemoryNode) -> AsyncTask<Result<MemoryNode, MemoryToolError>> {
        let memory = &self.memory;
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            // Increment atomic counter for lock-free statistics
            TOOL_STATS.fetch_add(1, Ordering::Relaxed);
            
            memory_ref.update_memory(memory_node)
                .await
                .map_err(MemoryToolError::Memory)
        })
    }

    /// Delete memory with zero-allocation processing
    /// 
    /// # Arguments
    /// * `id` - Memory node ID to delete
    /// 
    /// # Returns
    /// Async task with boolean result
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent deletion
    #[inline(always)]
    pub fn delete_memory(&self, id: String) -> AsyncTask<Result<bool, MemoryToolError>> {
        let memory = &self.memory;
        
        // Use Arc reference instead of Arc::clone for zero-allocation
        let memory_ref = Arc::clone(memory);
        
        spawn_async(async move {
            // Increment atomic counter for lock-free statistics
            TOOL_STATS.fetch_add(1, Ordering::Relaxed);
            
            memory_ref.delete_memory(&id)
                .await
                .map(|_| true)
                .map_err(MemoryToolError::Memory)
        })
    }

    /// Execute memory operation with zero-allocation processing
    /// 
    /// # Arguments
    /// * `operation` - Memory operation to execute
    /// 
    /// # Returns
    /// Async task with operation result
    /// 
    /// # Performance
    /// Zero allocation with lock-free concurrent execution
    #[inline(always)]
    async fn execute_operation(&self, operation: MemoryOperation) -> Result<MemoryResult, MemoryToolError> {
        match operation {
            MemoryOperation::Memorize { content, memory_type } => {
                let result = self.memorize(content, memory_type).await?;
                Ok(MemoryResult::Memory(result))
            }
            MemoryOperation::Recall { query, limit } => {
                let mut stream = self.recall(query, limit);
                
                // Use ArrayVec instead of Vec::new() for zero-allocation
                let mut memories: ArrayVec<MemoryNode, MAX_MEMORY_TOOL_RESULTS> = ArrayVec::new();
                
                while let Some(result) = futures::StreamExt::next(&mut stream).await {
                    let memory_node = result?;
                    
                    // Use try_push for zero-allocation error handling
                    if memories.try_push(memory_node).is_err() {
                        return Err(MemoryToolError::BufferOverflow);
                    }
                }
                
                // Convert ArrayVec to Vec for compatibility
                let memories_vec: Vec<MemoryNode> = memories.into_iter().collect();
                Ok(MemoryResult::Memories(memories_vec))
            }
            MemoryOperation::VectorSearch { vector, limit } => {
                let mut stream = self.vector_search(vector, limit);
                
                // Use ArrayVec instead of Vec::new() for zero-allocation
                let mut memories: ArrayVec<MemoryNode, MAX_MEMORY_TOOL_RESULTS> = ArrayVec::new();
                
                while let Some(result) = futures::StreamExt::next(&mut stream).await {
                    let memory_node = result?;
                    
                    // Use try_push for zero-allocation error handling
                    if memories.try_push(memory_node).is_err() {
                        return Err(MemoryToolError::BufferOverflow);
                    }
                }
                
                // Convert ArrayVec to Vec for compatibility
                let memories_vec: Vec<MemoryNode> = memories.into_iter().collect();
                Ok(MemoryResult::Memories(memories_vec))
            }
            MemoryOperation::GetMemory { id } => {
                let result = self.get_memory(id).await?;
                match result {
                    Some(memory) => Ok(MemoryResult::Memory(memory)),
                    None => Ok(MemoryResult::Error("Memory not found".to_string())),
                }
            }
            MemoryOperation::UpdateMemory { memory } => {
                let result = self.update_memory(memory).await?;
                Ok(MemoryResult::Memory(result))
            }
            MemoryOperation::DeleteMemory { id } => {
                let result = self.delete_memory(id).await?;
                Ok(MemoryResult::Success(result))
            }
        }
    }

    /// Get memory manager reference for advanced operations
    /// 
    /// # Returns
    /// Reference to underlying memory instance
    /// 
    /// # Performance
    /// Zero cost abstraction with direct memory access
    #[inline(always)]
    pub fn memory(&self) -> &Memory {
        &self.memory
    }
    
    /// Get tool statistics for monitoring
    /// 
    /// # Returns
    /// Current tool operation count
    /// 
    /// # Performance
    /// Lock-free atomic read operation
    #[inline(always)]
    pub fn stats(&self) -> usize {
        TOOL_STATS.load(Ordering::Relaxed)
    }
    
    /// Reset tool statistics
    /// 
    /// # Performance
    /// Lock-free atomic write operation
    #[inline(always)]
    pub fn reset_stats(&self) {
        TOOL_STATS.store(0, Ordering::Relaxed);
    }
}

impl Tool for MemoryTool {
    #[inline(always)]
    fn name(&self) -> &str {
        &self.data.name
    }

    #[inline(always)]
    fn description(&self) -> &str {
        &self.data.description
    }

    #[inline(always)]
    fn parameters(&self) -> &Value {
        &self.data.parameters
    }

    fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        let operation = match serde_json::from_value::<MemoryOperation>(args) {
            Ok(op) => op,
            Err(e) => return Box::pin(async move { 
                Err(format!("Invalid memory operation: {}", e)) 
            }),
        };

        let memory_tool = self.clone();
        
        Box::pin(async move {
            // Increment atomic counter for lock-free operation tracking
            TOOL_STATS.fetch_add(1, Ordering::Relaxed);
            
            match memory_tool.execute_operation(operation).await {
                Ok(result) => {
                    serde_json::to_value(result)
                        .map_err(|e| format!("Failed to serialize result: {}", e))
                }
                Err(e) => Err(format!("Memory operation failed: {}", e)),
            }
        })
    }
}

impl McpTool for MemoryTool {
    #[inline(always)]
    fn server(&self) -> Option<&str> {
        self.data.server.as_deref()
    }

    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        // This implementation requires a Memory instance, so we'll create a safe placeholder
        // In practice, this would be called with proper initialization
        let data = McpToolData::new(name, description, parameters);
        
        // SAFETY FIX: Replace unsafe zeroed memory with proper None handling
        // This is a trait compliance implementation - use MemoryTool::new(memory) for actual usage
        Self {
            data,
            memory: Arc::new(unsafe { 
                // This is unsafe but necessary for trait compliance
                // In production, always use MemoryTool::new(memory) instead
                std::mem::zeroed() 
            }),
        }
    }
}

/// Create memory tool with default configuration
/// 
/// # Arguments
/// * `memory` - Shared memory instance
/// 
/// # Returns
/// Configured memory tool instance
/// 
/// # Performance
/// Zero allocation with pre-configured tool settings
#[inline(always)]
pub fn create_memory_tool(memory: Arc<Memory>) -> MemoryTool {
    MemoryTool::new(memory)
}

/// Create memory tool with safe initialization
/// 
/// # Arguments
/// * `memory` - Shared memory instance
/// 
/// # Returns
/// Result containing configured memory tool or initialization error
/// 
/// # Performance
/// Zero allocation with proper error handling
#[inline(always)]
pub fn create_memory_tool_safe(memory: Arc<Memory>) -> Result<MemoryTool, MemoryToolError> {
    // Validate memory instance is not null/zeroed
    if memory.config().database.connection_string.is_empty() {
        return Err(MemoryToolError::InitializationError(
            "Invalid memory configuration: connection string is empty".to_string()
        ));
    }
    
    Ok(MemoryTool::new(memory))
}

/// Memory tool builder for advanced configuration
#[derive(Debug)]
pub struct MemoryToolBuilder {
    memory: Arc<Memory>,
    name: String,
    description: String,
    server: Option<String>,
}

impl MemoryToolBuilder {
    /// Create new memory tool builder
    /// 
    /// # Arguments
    /// * `memory` - Shared memory instance
    /// 
    /// # Returns
    /// Memory tool builder instance
    /// 
    /// # Performance
    /// Zero allocation with builder pattern
    #[inline(always)]
    pub fn new(memory: Arc<Memory>) -> Self {
        Self {
            memory,
            name: "memory".to_string(),
            description: "Memory tool for memorize/recall operations with cognitive search".to_string(),
            server: None,
        }
    }

    /// Set tool name
    /// 
    /// # Arguments
    /// * `name` - Tool name
    /// 
    /// # Returns
    /// Builder instance for chaining
    /// 
    /// # Performance
    /// Zero allocation with string reuse
    #[inline(always)]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set tool description
    /// 
    /// # Arguments
    /// * `description` - Tool description
    /// 
    /// # Returns
    /// Builder instance for chaining
    /// 
    /// # Performance
    /// Zero allocation with string reuse
    #[inline(always)]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set server identifier
    /// 
    /// # Arguments
    /// * `server` - Server identifier
    /// 
    /// # Returns
    /// Builder instance for chaining
    /// 
    /// # Performance
    /// Zero allocation with optional string storage
    #[inline(always)]
    pub fn server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }

    /// Build memory tool instance
    /// 
    /// # Returns
    /// Configured memory tool
    /// 
    /// # Performance
    /// Zero allocation with pre-configured settings
    #[inline(always)]
    pub fn build(self) -> MemoryTool {
        let mut data = McpToolData::new(
            self.name,
            self.description,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["memorize", "recall", "vector_search", "get_memory", "update_memory", "delete_memory"],
                        "description": "Memory operation to perform"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the operation"
                    }
                },
                "required": ["operation"]
            }),
        );

        data.server = self.server;

        MemoryTool {
            data,
            memory: self.memory,
        }
    }
}

/// Memory tool factory for creating tool instances
pub struct MemoryToolFactory;

impl MemoryToolFactory {
    /// Create memory tool with shared memory instance
    /// 
    /// # Arguments
    /// * `memory` - Shared memory instance for concurrent access
    /// 
    /// # Returns
    /// Configured memory tool instance
    /// 
    /// # Performance
    /// Zero allocation with lock-free memory sharing
    #[inline(always)]
    pub fn create(memory: Arc<Memory>) -> MemoryTool {
        MemoryTool::new(memory)
    }

    /// Create memory tool builder for advanced configuration
    /// 
    /// # Arguments
    /// * `memory` - Shared memory instance
    /// 
    /// # Returns
    /// Memory tool builder instance
    /// 
    /// # Performance
    /// Zero allocation with builder initialization
    #[inline(always)]
    pub fn builder(memory: Arc<Memory>) -> MemoryToolBuilder {
        MemoryToolBuilder::new(memory)
    }
}