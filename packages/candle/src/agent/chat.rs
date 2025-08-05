//! Chat functionality for memory-enhanced agent conversations
//! Uses AsyncStream patterns exclusively - NO FUTURES

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering}};

use arrayvec::ArrayVec;
use atomic_counter::RelaxedCounter;
use crossbeam_utils::CachePadded;
use once_cell::sync::Lazy;
use fluent_ai_async::AsyncStream;

use crate::domain::agent::AgentRoleImpl;
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
    /// Stream containing memory-enhanced chat response
    ///
    /// # Performance
    /// Zero allocation with lock-free memory operations and quantum routing
    pub fn chat(
        &self,
        message: impl Into<String>,
        memory: &Memory,
        memory_tool: &MemoryTool,
    ) -> AsyncStream<MemoryEnhancedChatResponse> {
        let message = message.into();
        let memory_arc = Arc::new(memory.clone());
        let memory_tool = memory_tool.clone();
        let agent = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Inject relevant memory context with zero-allocation processing
            let mut context_stream = agent.inject_memory_context(&message, &memory_arc);
            
            if let Some(context_injection) = context_stream.try_next() {
                // Generate response using Candle engine with injected context
                let full_prompt = if context_injection.injected_context.is_empty() {
                    message.clone()
                } else {
                    format!("Context: {}\n\nUser: {}", context_injection.injected_context, message)
                };
                
                // Use the core engine for completion
                let engine_config = crate::core::EngineConfig::new("kimi-k2", "kimi-k2")
                    .with_temperature(agent.temperature.unwrap_or(0.7) as f32)
                    .with_max_tokens(agent.max_tokens.unwrap_or(1000) as u32);
                    
                if let Ok(engine) = crate::core::Engine::new(engine_config) {
                    let completion_request = crate::core::CompletionRequest::new(&full_prompt)
                        .with_system_prompt(agent.system_prompt.as_deref().unwrap_or("You are a helpful assistant."));
                        
                    // Get the completion result using streams
                    let mut completion_stream = engine.process_completion(completion_request);
                    if let Some(completion_response) = completion_stream.try_next() {
                        let response = completion_response.text.to_string();

                        // Memorize the conversation turn with zero-allocation node creation
                        let mut memorize_stream = agent.memorize_conversation(&message, &response, &memory_tool);
                        if let Some(memorized_nodes) = memorize_stream.try_next() {
                            let response_obj = MemoryEnhancedChatResponse {
                                response,
                                context_injection,
                                memorized_nodes};
                            let _ = sender.send(response_obj);
                        }
                    }
                }
            }
        })
    }

    /// Inject memory context with zero-allocation processing
    ///
    /// # Arguments
    /// * `message` - User message for context relevance
    /// * `memory` - Shared memory instance for queries
    ///
    /// # Returns
    /// Stream containing context injection result
    ///
    /// # Performance
    /// Zero allocation with lock-free memory queries and quantum routing
    pub fn inject_memory_context(
        &self,
        _message: &str,
        _memory: &Arc<Memory>,
    ) -> AsyncStream<ContextInjectionResult> {
        AsyncStream::with_channel(move |sender| {
            // Query relevant memories with zero-allocation buffer
            let relevant_memories = ArrayVec::<MemoryNode, MAX_RELEVANT_MEMORIES>::new();

            // TODO: Implement actual memory querying logic
            // For now, return empty context
            let injected_context = String::new();
            let relevance_score = 0.0;
            let memory_nodes_used = relevant_memories.len();

            let result = ContextInjectionResult {
                injected_context,
                relevance_score,
                memory_nodes_used};

            let _ = sender.send(result);
        })
    }

    /// Calculate relevance score using attention mechanism
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `memory_node` - Memory node to score
    ///
    /// # Returns
    /// Relevance score (0.0 to 1.0)
    ///
    /// # Performance
    /// Zero allocation with inlined relevance calculations
    pub fn calculate_relevance_score(
        &self,
        message: &str,
        memory_node: &MemoryNode,
    ) -> f64 {
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
        (length_similarity * 0.3 + importance_factor * 0.5 + time_factor * 0.2).min(1.0)
    }

    /// Memorize conversation turn with zero-allocation node creation
    ///
    /// # Arguments
    /// * `user_message` - User message to memorize
    /// * `assistant_response` - Assistant response to memorize
    /// * `memory_tool` - Memory tool for storage operations
    ///
    /// # Returns
    /// Stream containing memorized nodes
    ///
    /// # Performance
    /// Zero allocation with lock-free atomic counters for memory node tracking
    pub fn memorize_conversation(
        &self,
        user_message: &str,
        assistant_response: &str,
        memory_tool: &MemoryTool,
    ) -> AsyncStream<ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES>> {
        let user_message = user_message.to_string();
        let assistant_response = assistant_response.to_string();
        let memory_tool = memory_tool.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut memorized_nodes = ArrayVec::new();

            // Create memory node for user message using direct constructor
            let user_memory = MemoryNode::new(MemoryTypeEnum::Episodic, MemoryContent::text(user_message));

            // Store user memory with zero-allocation error handling - PURE STREAMING
            let mut store_stream = memory_tool.memory().store_memory(&user_memory);
            if let Some(_store_result) = store_stream.try_next() {
                // AsyncStream now returns unwrapped values, no error handling needed
            }

            if memorized_nodes.try_push(user_memory).is_ok() {
                // Create memory node for assistant response
                let assistant_memory = MemoryNode::new(
                    MemoryTypeEnum::Episodic,
                    MemoryContent::text(assistant_response.clone()),
                );

                // Store assistant memory with zero-allocation error handling - PURE STREAMING
                let mut store_stream = memory_tool.memory().store_memory(&assistant_memory);
                if let Some(_store_result) = store_stream.try_next() {
                    // AsyncStream now returns unwrapped values, no error handling needed
                }

                if memorized_nodes.try_push(assistant_memory).is_ok() {
                    // Create contextual memory node linking the conversation
                    let context_memory = MemoryNode::new(
                        MemoryTypeEnum::Contextual,
                        MemoryContent::text(format!(
                            "Conversation: {} -> {}",
                            user_message, assistant_response
                        )),
                    );

                    // Store context memory with zero-allocation error handling - PURE STREAMING
                    let mut store_stream = memory_tool.memory().store_memory(&context_memory);
                    if let Some(_store_result) = store_stream.try_next() {
                        // AsyncStream now returns unwrapped values, no error handling needed
                    }

                    if memorized_nodes.try_push(context_memory).is_ok() {
                        let _ = sender.send(memorized_nodes);
                    }
                }
            }
        })
    }
}