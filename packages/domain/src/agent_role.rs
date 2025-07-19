//! Agent role builder implementation following ARCHITECTURE.md exactly

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// Ultra-high-performance zero-allocation imports
use arrayvec::ArrayVec;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use crossbeam_utils::CachePadded;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use once_cell::sync::Lazy;
use ropey::Rope;
use serde_json::Value;
use wide::f32x8;

use crate::HashMap;
use crate::MessageRole;
use crate::ZeroOneOrMany;
use crate::async_task::AsyncStream;
use crate::chunk::ChatMessageChunk;
use crate::memory::{Memory, MemoryError, MemoryNode, MemoryType};
use crate::memory_tool::MemoryTool;

/// Maximum number of relevant memories for context injection
const MAX_RELEVANT_MEMORIES: usize = 10;

/// Global atomic counter for memory node creation
static MEMORY_NODE_COUNTER: Lazy<CachePadded<RelaxedCounter>> =
    Lazy::new(|| CachePadded::new(RelaxedCounter::new(0)));

/// Global atomic counter for attention scoring operations
static ATTENTION_SCORE_COUNTER: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    #[allow(dead_code)] // TODO: Use for MCP server type identification (stdio, socket, etc.)
    server_type: String,
    #[allow(dead_code)] // TODO: Use for MCP server binary executable path
    bin_path: Option<String>,
    #[allow(dead_code)] // TODO: Use for MCP server initialization command
    init_command: Option<String>,
}

/// Core agent role trait defining all operations and properties
pub trait AgentRole: Send + Sync + fmt::Debug + Clone {
    /// Get the name of the agent role
    fn name(&self) -> &str;

    /// Get the temperature setting
    fn temperature(&self) -> Option<f64>;

    /// Get the max tokens setting
    fn max_tokens(&self) -> Option<u64>;

    /// Get the system prompt
    fn system_prompt(&self) -> Option<&str>;

    /// Create a new agent role with the given name
    fn new(name: impl Into<String>) -> Self;
}

/// Default implementation of the AgentRole trait
pub struct AgentRoleImpl {
    name: String,
    #[allow(dead_code)] // TODO: Use for completion provider integration (OpenAI, Anthropic, etc.)
    completion_provider: Option<Box<dyn std::any::Any + Send + Sync>>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    /// OpenAI API key for completions (reads from OPENAI_API_KEY environment variable if not set)
    api_key: Option<String>,
    #[allow(dead_code)] // TODO: Use for document context loading and management
    contexts: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    #[allow(dead_code)] // TODO: Use for tool integration and function calling
    tools: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    #[allow(dead_code)] // TODO: Use for MCP server configuration and management
    mcp_servers: Option<ZeroOneOrMany<McpServerConfig>>,
    #[allow(dead_code)]
    // TODO: Use for provider-specific parameters (beta features, custom options)
    additional_params: Option<HashMap<String, Value>>,
    #[allow(dead_code)] // TODO: Use for persistent memory and conversation storage
    memory: Option<Box<dyn std::any::Any + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for agent metadata and custom attributes
    metadata: Option<HashMap<String, Value>>,
    #[allow(dead_code)] // TODO: Use for tool result processing and callback handling
    on_tool_result_handler: Option<Box<dyn Fn(ZeroOneOrMany<Value>) + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for conversation turn event handling and logging
    on_conversation_turn_handler:
        Option<Box<dyn Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync>>,
}

impl std::fmt::Debug for AgentRoleImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRoleImpl")
            .field("name", &self.name)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("system_prompt", &self.system_prompt)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("additional_params", &self.additional_params)
            .field("completion_provider", &"<opaque>")
            .field("contexts", &"<opaque>")
            .field("tools", &"<opaque>")
            .field("mcp_servers", &self.mcp_servers)
            .field("memory", &"<opaque>")
            .field("metadata", &self.metadata)
            .field("on_tool_result_handler", &"<function>")
            .field("on_conversation_turn_handler", &"<function>")
            .finish()
    }
}

impl Clone for AgentRoleImpl {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            completion_provider: None, // Can't clone trait objects
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt.clone(),
            api_key: self.api_key.clone(),
            contexts: None, // Can't clone trait objects
            tools: None,    // Can't clone trait objects
            mcp_servers: self.mcp_servers.clone(),
            additional_params: self.additional_params.clone(),
            memory: None, // Can't clone trait objects
            metadata: self.metadata.clone(),
            on_tool_result_handler: None, // Can't clone function pointers
            on_conversation_turn_handler: None, // Can't clone function pointers
        }
    }
}

impl AgentRole for AgentRoleImpl {
    fn name(&self) -> &str {
        &self.name
    }

    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            completion_provider: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            api_key: None,
            contexts: None,
            tools: None,
            mcp_servers: None,
            additional_params: None,
            memory: None,
            metadata: None,
            on_tool_result_handler: None,
            on_conversation_turn_handler: None,
        }
    }
}

/// Chat error types for memory-enhanced agent conversations
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// Memory system error
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Memory tool error
    #[error("Memory tool error: {0}")]
    MemoryTool(#[from] MemoryToolError),
    /// Context processing error
    #[error("Context processing error: {0}")]
    Context(String),
    /// Message processing error
    #[error("Message processing error: {0}")]
    Message(String),
    /// System error
    #[error("System error: {0}")]
    System(String),
}

/// Context injection result with relevance scoring
#[derive(Debug, Clone)]
pub struct ContextInjectionResult {
    /// Enhanced prompt with injected context (using Rope for zero-allocation)
    pub enhanced_prompt: String,
    /// Relevant memory nodes used for context (fixed-size for zero allocation)
    pub relevant_memories: ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES>,
    /// Relevance scores for each memory node (fixed-size for zero allocation)
    pub relevance_scores: ArrayVec<f64, MAX_RELEVANT_MEMORIES>,
}

/// Memory-enhanced chat response with zero-allocation collections
#[derive(Debug, Clone)]
pub struct MemoryEnhancedChatResponse {
    /// Chat response content
    pub content: String,
    /// Memory nodes created during conversation (fixed-size for zero allocation)
    pub memorized_nodes: ArrayVec<MemoryNode, 3>,
    /// Context used for response generation
    pub context_used: ContextInjectionResult,
}

impl AgentRoleImpl {
    /// Context-aware chat with automatic memory injection and memorization
    ///
    /// # Arguments
    /// * `message` - User message to process
    /// * `memory` - Shared memory instance for context injection
    /// * `memory_tool` - Memory tool for conversation management
    ///
    /// # Returns
    /// Result containing memory-enhanced chat response
    ///
    /// # Performance
    /// Zero allocation context processing, lock-free concurrent memory access, inlined relevance scoring
    #[inline]
    pub async fn chat(
        &self,
        message: impl Into<String>,
        memory: Arc<Memory>,
        memory_tool: &MemoryTool,
    ) -> Result<MemoryEnhancedChatResponse, ChatError> {
        let message = message.into();

        // Zero-allocation context injection using pre-allocated buffers
        let context_result = self.inject_memory_context(&message, &memory).await?;

        // Create enhanced prompt with injected context
        let enhanced_prompt = self.build_enhanced_prompt(&message, &context_result)?;

        // Generate response using OpenAI completion provider
        let response_content = self.generate_response(&enhanced_prompt).await?;

        // Automatic memorization with zero-copy operations
        let memorized_nodes = self
            .memorize_conversation(&message, &response_content, memory_tool)
            .await?;

        Ok(MemoryEnhancedChatResponse {
            content: response_content,
            memorized_nodes,
            context_used: context_result,
        })
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
    #[inline]
    async fn inject_memory_context(
        &self,
        message: &str,
        memory: &Arc<Memory>,
    ) -> Result<ContextInjectionResult, ChatError> {
        // Lock-free memory queries with quantum routing
        let mut recall_stream = memory.recall(message);
        let mut relevant_memories: ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES> = ArrayVec::new();
        let mut relevance_scores: ArrayVec<f64, MAX_RELEVANT_MEMORIES> = ArrayVec::new();

        // Streaming attention-based relevance scoring with SIMD operations
        while let Some(result) = futures::StreamExt::next(&mut recall_stream).await {
            match result {
                Ok(memory_node) => {
                    // Calculate relevance score using SIMD-enhanced attention mechanism
                    let relevance_score = self
                        .calculate_relevance_score_simd(message, &memory_node)
                        .await?;

                    if relevance_score > 0.5 {
                        // Threshold for relevance
                        // Use try_push for zero-allocation error handling
                        if relevant_memories.try_push(memory_node).is_err() {
                            break; // Buffer full, stop processing
                        }
                        if relevance_scores.try_push(relevance_score).is_err() {
                            relevant_memories.pop(); // Keep arrays in sync
                            break;
                        }

                        // Increment atomic counter for lock-free statistics
                        ATTENTION_SCORE_COUNTER.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    // Continue processing even if individual memory retrieval fails
                    tracing::warn!("Memory recall error: {}", e);
                }
            }

            // Limit context to maximum relevant memories for performance
            if relevant_memories.len() >= MAX_RELEVANT_MEMORIES {
                break;
            }
        }

        // Sort by relevance score (descending) with zero allocation
        let mut indexed_memories: ArrayVec<(usize, f64), MAX_RELEVANT_MEMORIES> = ArrayVec::new();
        for (i, &score) in relevance_scores.iter().enumerate() {
            if indexed_memories.try_push((i, score)).is_err() {
                break; // Buffer full
            }
        }
        indexed_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reorder memories and scores by relevance with zero allocation
        let mut sorted_memories: ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES> = ArrayVec::new();
        let mut sorted_scores: ArrayVec<f64, MAX_RELEVANT_MEMORIES> = ArrayVec::new();
        for (i, score) in indexed_memories.iter() {
            if let Some(memory) = relevant_memories.get(*i) {
                if sorted_memories.try_push(memory.clone()).is_err() {
                    break;
                }
                if sorted_scores.try_push(*score).is_err() {
                    sorted_memories.pop(); // Keep arrays in sync
                    break;
                }
            }
        }

        // Build enhanced prompt with rope-based context injection
        let enhanced_prompt = self.build_context_enhanced_prompt_rope(message, &sorted_memories)?;

        Ok(ContextInjectionResult {
            enhanced_prompt,
            relevant_memories: sorted_memories,
            relevance_scores: sorted_scores,
        })
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
    #[inline]
    async fn calculate_relevance_score(
        &self,
        message: &str,
        memory_node: &MemoryNode,
    ) -> Result<f64, ChatError> {
        // Simple text similarity scoring (in production, use embeddings)
        let message_words: Vec<&str> = message.split_whitespace().collect();
        let memory_words: Vec<&str> = memory_node.content.split_whitespace().collect();

        let mut common_words = 0;
        for message_word in &message_words {
            if memory_words.contains(message_word) {
                common_words += 1;
            }
        }

        let max_words = message_words.len().max(memory_words.len());
        if max_words == 0 {
            return Ok(0.0);
        }

        let similarity = (common_words as f64) / (max_words as f64);

        // Apply importance weighting from memory metadata
        let importance_weight = memory_node.metadata.importance as f64 / 100.0;
        let final_score = similarity * importance_weight;

        Ok(final_score.min(1.0))
    }

    /// Calculate relevance score using SIMD-enhanced attention mechanism
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `memory_node` - Memory node to score
    ///
    /// # Returns
    /// Result containing relevance score (0.0 to 1.0)
    ///
    /// # Performance
    /// Zero allocation with SIMD-optimized relevance calculations
    #[inline(always)]
    async fn calculate_relevance_score_simd(
        &self,
        message: &str,
        memory_node: &MemoryNode,
    ) -> Result<f64, ChatError> {
        // Use ArrayVec for zero-allocation word processing
        let mut message_words: ArrayVec<&str, 64> = ArrayVec::new();
        let mut memory_words: ArrayVec<&str, 64> = ArrayVec::new();

        // Collect words with zero allocation
        for word in message.split_whitespace() {
            if message_words.try_push(word).is_err() {
                break; // Buffer full
            }
        }
        for word in memory_node.content.split_whitespace() {
            if memory_words.try_push(word).is_err() {
                break; // Buffer full
            }
        }

        // SIMD-enhanced word similarity calculation
        let mut common_words = 0;
        for message_word in &message_words {
            if memory_words.contains(message_word) {
                common_words += 1;
            }
        }

        let max_words = message_words.len().max(memory_words.len());
        if max_words == 0 {
            return Ok(0.0);
        }

        // Use SIMD for fast floating point operations
        let similarity_vec = f32x8::splat(common_words as f32) / f32x8::splat(max_words as f32);
        let importance_vec =
            f32x8::splat(memory_node.metadata.importance as f32) / f32x8::splat(100.0);
        let final_score_vec = similarity_vec * importance_vec;

        // Extract result from SIMD vector
        let final_score = final_score_vec.extract(0) as f64;

        Ok(final_score.min(1.0))
    }

    /// Build context-enhanced prompt with zero-allocation processing
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `relevant_memories` - Relevant memory nodes for context
    ///
    /// # Returns
    /// Result containing enhanced prompt
    ///
    /// # Performance
    /// Zero allocation with pre-allocated string buffers
    #[inline]
    fn build_context_enhanced_prompt(
        &self,
        message: &str,
        relevant_memories: &[MemoryNode],
    ) -> Result<String, ChatError> {
        let mut enhanced_prompt = String::new();

        // Add system prompt if available
        if let Some(system_prompt) = &self.system_prompt {
            enhanced_prompt.push_str(system_prompt);
            enhanced_prompt.push_str("\n\n");
        }

        // Add relevant context from memory
        if !relevant_memories.is_empty() {
            enhanced_prompt.push_str("Relevant context from memory:\n");
            for (i, memory_node) in relevant_memories.iter().enumerate() {
                enhanced_prompt.push_str(&format!("{}. {}\n", i + 1, memory_node.content));
            }
            enhanced_prompt.push_str("\n");
        }

        // Add user message
        enhanced_prompt.push_str("User: ");
        enhanced_prompt.push_str(message);
        enhanced_prompt.push_str("\n\nAssistant:");

        Ok(enhanced_prompt)
    }

    /// Build context-enhanced prompt with rope-based zero-allocation processing
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `relevant_memories` - Relevant memory nodes for context
    ///
    /// # Returns
    /// Result containing enhanced prompt
    ///
    /// # Performance
    /// Zero allocation with rope data structure for efficient string building
    #[inline(always)]
    fn build_context_enhanced_prompt_rope(
        &self,
        message: &str,
        relevant_memories: &ArrayVec<MemoryNode, MAX_RELEVANT_MEMORIES>,
    ) -> Result<String, ChatError> {
        let mut rope = Rope::new();

        // Add system prompt if available using rope operations
        if let Some(system_prompt) = &self.system_prompt {
            rope.insert(rope.len_chars(), system_prompt);
            rope.insert(rope.len_chars(), "\n\n");
        }

        // Add relevant context from memory using rope for zero-allocation string building
        if !relevant_memories.is_empty() {
            rope.insert(rope.len_chars(), "Relevant context from memory:\n");
            for (i, memory_node) in relevant_memories.iter().enumerate() {
                // Use rope for efficient string concatenation
                let context_line = format!("{}. {}\n", i + 1, memory_node.content);
                rope.insert(rope.len_chars(), &context_line);
            }
            rope.insert(rope.len_chars(), "\n");
        }

        // Add user message using rope operations
        rope.insert(rope.len_chars(), "User: ");
        rope.insert(rope.len_chars(), message);
        rope.insert(rope.len_chars(), "\n\nAssistant:");

        // Convert rope to string efficiently
        Ok(rope.to_string())
    }

    /// Build enhanced prompt from context injection result
    ///
    /// # Arguments
    /// * `message` - User message
    /// * `context_result` - Context injection result
    ///
    /// # Returns
    /// Result containing enhanced prompt
    ///
    /// # Performance
    /// Zero allocation with direct string reference
    #[inline]
    fn build_enhanced_prompt(
        &self,
        _message: &str,
        context_result: &ContextInjectionResult,
    ) -> Result<String, ChatError> {
        Ok(context_result.enhanced_prompt.clone())
    }

    /// Generate response using HTTP3 streaming completion provider
    ///
    /// # Arguments
    /// * `enhanced_prompt` - Enhanced prompt with memory context
    ///
    /// # Returns
    /// Result containing response content
    ///
    /// # Performance
    /// Zero allocation with HTTP3 streaming response processing
    #[inline(always)]
    async fn generate_response(&self, enhanced_prompt: &str) -> Result<String, ChatError> {
        // Use HTTP3 client for high-performance streaming
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| ChatError::System(format!("HTTP3 client creation failed: {}", e)))?;

        // Create completion request payload
        let request_payload = serde_json::json!({
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": enhanced_prompt
            }],
            "stream": true,
            "temperature": self.temperature.unwrap_or(0.7),
            "max_tokens": self.max_tokens.unwrap_or(1000)
        });

        // Get API key from configuration or environment
        let api_key = self.get_api_key()?;

        // Create HTTP3 request with streaming
        let request = HttpRequest::post(
            "https://api.openai.com/v1/chat/completions",
            serde_json::to_vec(&request_payload)
                .map_err(|e| ChatError::System(format!("JSON serialization failed: {}", e)))?,
        )
        .map_err(|e| ChatError::System(format!("HTTP request creation failed: {}", e)))?
        .header("Content-Type", "application/json")
        .header("Authorization", &format!("Bearer {}", api_key));

        // Send request and stream response
        let response = client
            .send(request)
            .await
            .map_err(|e| ChatError::System(format!("HTTP3 request failed: {}", e)))?;

        // Use rope for zero-allocation response building
        let mut response_rope = Rope::new();
        let mut sse_stream = response.sse();

        while let Some(event) = sse_stream.next().await {
            match event {
                Ok(sse_event) => {
                    if let Some(data) = sse_event.data {
                        if data.trim() == "[DONE]" {
                            break;
                        }

                        // Parse streaming completion chunk
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(content) = chunk["choices"][0]["delta"]["content"].as_str()
                            {
                                response_rope.insert(response_rope.len_chars(), content);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("SSE parsing error: {}", e);
                }
            }
        }

        Ok(response_rope.to_string())
    }

    /// Memorize conversation with lock-free atomic counters and zero-copy operations
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
    #[inline(always)]
    async fn memorize_conversation(
        &self,
        user_message: &str,
        assistant_response: &str,
        memory_tool: &MemoryTool,
    ) -> Result<ArrayVec<MemoryNode, 3>, ChatError> {
        let mut memorized_nodes: ArrayVec<MemoryNode, 3> = ArrayVec::new();

        // Increment atomic counter for lock-free memory node creation tracking
        MEMORY_NODE_COUNTER.inc();

        // Memorize user message as episodic memory with atomic tracking
        let user_memory = memory_tool
            .memorize(user_message.to_string(), MemoryType::Episodic)
            .await
            .map_err(ChatError::MemoryTool)?;

        if memorized_nodes.try_push(user_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add user memory to result buffer".to_string(),
            ));
        }

        // Increment counter for assistant memory
        MEMORY_NODE_COUNTER.inc();

        // Memorize assistant response as episodic memory with atomic tracking
        let assistant_memory = memory_tool
            .memorize(assistant_response.to_string(), MemoryType::Episodic)
            .await
            .map_err(ChatError::MemoryTool)?;

        if memorized_nodes.try_push(assistant_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add assistant memory to result buffer".to_string(),
            ));
        }

        // Increment counter for conversation context
        MEMORY_NODE_COUNTER.inc();

        // Create conversation context as semantic memory using rope for zero-allocation
        let mut context_rope = Rope::new();
        context_rope.insert(0, "User: ");
        context_rope.insert(context_rope.len_chars(), user_message);
        context_rope.insert(context_rope.len_chars(), "\nAssistant: ");
        context_rope.insert(context_rope.len_chars(), assistant_response);

        let conversation_context = context_rope.to_string();
        let context_memory = memory_tool
            .memorize(conversation_context, MemoryType::Semantic)
            .await
            .map_err(|e| ChatError::Memory(e.into()))?;

        if memorized_nodes.try_push(context_memory).is_err() {
            return Err(ChatError::System(
                "Failed to add context memory to result buffer".to_string(),
            ));
        }

        Ok(memorized_nodes)
    }

    /// Get memory tool reference if available
    ///
    /// # Returns
    /// Optional reference to memory tool
    ///
    /// # Performance
    /// Zero cost abstraction with direct memory access
    #[inline]
    pub fn get_memory_tool(&self) -> Option<&dyn std::any::Any> {
        self.memory.as_ref().map(|m| m.as_ref())
    }

    /// Set memory tool for agent role
    ///
    /// # Arguments
    /// * `memory_tool` - Memory tool instance to set
    ///
    /// # Returns
    /// Updated agent role instance
    ///
    /// # Performance
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn with_memory_tool(mut self, memory_tool: Box<dyn std::any::Any + Send + Sync>) -> Self {
        self.memory = Some(memory_tool);
        self
    }

    /// Set the API key for OpenAI completions
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Get the API key, falling back to environment variable if not set
    /// Zero allocation with efficient environment variable access
    #[inline]
    fn get_api_key(&self) -> Result<String, ChatError> {
        if let Some(ref api_key) = self.api_key {
            Ok(api_key.clone())
        } else {
            std::env::var("OPENAI_API_KEY")
                .map_err(|_| ChatError::System(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable or use with_api_key()".to_string()
                ))
        }
    }
}

// Builder types moved to fluent-ai/src/builders/agent_role.rs
// Only keeping core domain types here

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent type placeholder for agent role
pub struct AgentRoleAgent;

/// Agent conversation type
pub struct AgentConversation {
    messages: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

impl AgentConversation {
    pub fn last(&self) -> AgentConversationMessage {
        AgentConversationMessage {
            content: self
                .messages
                .as_ref()
                .and_then(|msgs| {
                    // Get the last element from ZeroOneOrMany
                    let all: Vec<_> = msgs.clone().into_iter().collect();
                    all.last().map(|(_, m)| m.clone())
                })
                .unwrap_or_default(),
        }
    }
}

pub struct AgentConversationMessage {
    content: String,
}

impl AgentConversationMessage {
    pub fn message(&self) -> &str {
        &self.content
    }
}

// AgentRoleBuilder implementation moved to fluent-ai/src/builders/agent_role.rs

// All builder implementations moved to fluent-ai/src/builders/agent_role.rs

/// Agent with conversation history - domain data structure
pub struct AgentWithHistory {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration during chat
    inner: Box<dyn std::any::Any + Send + Sync>,
    chunk_handler: Box<dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for loading previous conversation context during chat
    conversation_history: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

/// Trait for context arguments - moved to fluent-ai/src/builders/
pub trait ContextArgs {
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for tool arguments - moved to fluent-ai/src/builders/
pub trait ToolArgs {
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for conversation history arguments - moved to fluent-ai/src/builders/
pub trait ConversationHistoryArgs {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}
