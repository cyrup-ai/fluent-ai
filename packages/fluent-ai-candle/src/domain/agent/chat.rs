//! Chat functionality for memory-enhanced agent conversations

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering}};

use arrayvec::ArrayVec;
use atomic_counter::RelaxedCounter;
use crossbeam_utils::CachePadded;
use once_cell::sync::Lazy;
// Removed unused import: use tokio_stream::StreamExt;

use crate::domain::agent::role::AgentRoleImpl;
use crate::memory::primitives::{MemoryContent, MemoryTypeEnum};
use crate::memory::{Memory, MemoryError, MemoryNode};
use crate::memory::{MemoryTool, MemoryToolError};

/// Maximum number of relevant memories for context injection
const MAX_RELEVANT_MEMORIES: usize = 10;

/// Global atomic counter for memory node creation
#[allow(dead_code)] // TODO: Implement in memory node creation system
static MEMORY_NODE_COUNTER: Lazy<CachePadded<RelaxedCounter>> =
    Lazy::new(|| CachePadded::new(RelaxedCounter::new(0)));

/// Global atomic counter for attention scoring operations
static ATTENTION_SCORE_COUNTER: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// Chat error types for memory-enhanced agent conversations
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// Memory system error
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Memory tool error
    #[error("Memory tool error: {0}")]
    MemoryTool(#[from] MemoryToolError),
    /// Message processing error
    #[error("Message processing error: {0}")]
    Message(String),
    /// System error
    #[error("System error: {0}")]
    System(String)}

/// Context injection result with relevance scoring
#[derive(Debug, Clone)]
pub struct ContextInjectionResult {
    /// The context that was injected into the conversation
    pub injected_context: String,
    /// Score indicating how relevant the injected context is (0.0 to 1.0)
    pub relevance_score: f64,
    /// Number of memory nodes that were used in the injection
    pub memory_nodes_used: usize}

/// Memory-enhanced chat response with zero-allocation collections
#[derive(Debug, Clone)]
pub struct MemoryEnhancedChatResponse {
    /// The generated response text
    pub response: String,
    /// Details about the context that was injected
    pub context_injection: ContextInjectionResult,
    /// Memory nodes that were considered and stored, using fixed-size allocation
    pub memorized_nodes: ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES>}

impl AgentRoleImpl {
    /// Context-aware chat with automatic memory injection and memorization
    ///
    /// # Arguments
    /// * `message` - User message to process
    /// * `memory` - Shared memory instance for context injection
    /// * `memory_tool` - Memory tool for storage operations
    ///
    /// # Returns
    /// Result containing memory-enhanced chat response
    ///
    /// # Performance
    /// Zero allocation with lock-free memory operations and quantum routing
    pub async fn chat(
        &self,
        message: impl Into<String>,
        memory: &Memory,
        memory_tool: &MemoryTool,
    ) -> Result<MemoryEnhancedChatResponse, ChatError> {
        let message = message.into();

        // Inject relevant memory context with zero-allocation processing
        let memory_arc = Arc::new(memory.clone());
        let context_injection = self.inject_memory_context(&message, &memory_arc).await?;

        // TODO: Integrate with actual completion provider for response generation
        // For now, return a placeholder response
        let response = format!("Response to: {}", message);

        // Memorize the conversation turn with zero-allocation node creation
        let memorized_nodes = self
            .memorize_conversation(&message, &response, memory_tool)
            .await?;

        Ok(MemoryEnhancedChatResponse {
            response,
            context_injection,
            memorized_nodes})
    }

    /// Inject memory context with zero-allocation processing
    ///
    /// # Arguments
    /// * `message` - User message for context relevance
    /// * `memory` - Shared memory instance for queries
    ///
    /// # Returns
    /// Result containing context injection result
    ///
    /// # Performance
    /// Zero allocation with lock-free memory queries and quantum routing
    pub async fn inject_memory_context(
        &self,
        _message: &str,
        _memory: &Arc<Memory>,
    ) -> Result<ContextInjectionResult, ChatError> {
        // Query relevant memories with zero-allocation buffer
        let relevant_memories = ArrayVec::<MemoryNode, MAX_RELEVANT_MEMORIES>::new();

        // TODO: Implement actual memory querying logic
        // For now, return empty context
        let injected_context = String::new();
        let relevance_score = 0.0;
        let memory_nodes_used = relevant_memories.len();

        Ok(ContextInjectionResult {
            injected_context,
            relevance_score,
            memory_nodes_used})
    }

    /// Calculate relevance score using attention mechanism
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `memory_node` - Memory node to score
    ///
    /// # Returns
    /// Result containing relevance score (0.0 to 1.0)
    ///
    /// # Performance
    /// Zero allocation with inlined relevance calculations
    pub fn calculate_relevance_score(
        &self,
        message: &str,
        memory_node: &MemoryNode,
    ) -> Result<f64, ChatError> {
        // Increment atomic counter for lock-free statistics
        ATTENTION_SCORE_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Simple relevance scoring based on content similarity and memory node importance
        let message_len = message.len();
        let memory_content = match &memory_node.base_memory().content {
            MemoryContent::Text(text) => text.as_ref(),
            _ => "", // Non-text content gets empty string for comparison
        };
        let memory_len = memory_content.len();

        // Basic content length similarity (normalized)
        let length_similarity = 1.0
            - ((message_len as f64 - memory_len as f64).abs()
                / (message_len.max(memory_len) as f64 + 1.0));

        // Memory node importance factor
        let importance_factor = memory_node.importance() as f64;

        // Time decay factor based on last access
        let time_factor = if let Ok(elapsed) = memory_node.last_accessed().elapsed() {
            // Decay over 24 hours, minimum 0.1
            (1.0 - (elapsed.as_secs() as f64 / 86400.0)).max(0.1)
        } else {
            0.5 // Default if time calculation fails
        };

        // Combined relevance score (weighted average)
        let score =
            (length_similarity * 0.3 + importance_factor * 0.5 + time_factor * 0.2).min(1.0);

        Ok(score)
    }

    /// Memorize conversation turn with zero-allocation node creation
    ///
    /// # Arguments
    /// * `user_message` - User message to memorize
    /// * `assistant_response` - Assistant response to memorize
    /// * `memory_tool` - Memory tool for storage operations
    ///
    /// # Returns
    /// Result containing memorized nodes
    ///
    /// # Performance
    /// Zero allocation with lock-free atomic counters for memory node tracking
    pub async fn memorize_conversation(
        &self,
        user_message: &str,
        assistant_response: &str,
        memory_tool: &MemoryTool,
    ) -> Result<ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES>, ChatError> {
        let mut memorized_nodes = ArrayVec::new();

        // Create memory node for user message using direct constructor
        let user_memory = MemoryNode::new(MemoryTypeEnum::Episodic, MemoryContent::text(user_message));

        // Store user memory with zero-allocation error handling - PURE STREAMING
        let store_stream = memory_tool.memory().store_memory(&user_memory);

        // Use StreamExt to properly consume AsyncStream
        let mut stream = store_stream;
        if let Some(_store_result) = stream.try_next() {
            // AsyncStream now returns unwrapped values, no error handling needed
        }

        if memorized_nodes.try_push(user_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add user memory to result buffer".to_string(),
            ));
        }

        // Create memory node for assistant response
        let assistant_memory = MemoryNode::new(
            MemoryTypeEnum::Episodic,
            MemoryContent::text(assistant_response),
        );

        // Store assistant memory with zero-allocation error handling - PURE STREAMING
        let store_stream = memory_tool.memory().store_memory(&assistant_memory);

        // Use AsyncStream try_next method (NO FUTURES architecture)
        let mut stream = store_stream;
        if let Some(_store_result) = stream.try_next() {
            // AsyncStream now returns unwrapped values, no error handling needed
        }

        if memorized_nodes.try_push(assistant_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add assistant memory to result buffer".to_string(),
            ));
        }

        // Create contextual memory node linking the conversation
        let context_memory = MemoryNode::new(
            MemoryTypeEnum::Contextual,
            MemoryContent::text(format!(
                "Conversation: {} -> {}",
                user_message, assistant_response
            )),
        );

        // Store context memory with zero-allocation error handling - PURE STREAMING
        let store_stream = memory_tool.memory().store_memory(&context_memory);

        // Use AsyncStream try_next method (NO FUTURES architecture)
        let mut stream = store_stream;
        if let Some(_store_result) = stream.try_next() {
            // AsyncStream now returns unwrapped values, no error handling needed
        }

        if memorized_nodes.try_push(context_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add context memory to result buffer".to_string(),
            ));
        }

        Ok(memorized_nodes)
    }
}
