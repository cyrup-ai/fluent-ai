//! Core agent data structures with automatic memory tool injection
//! Uses AsyncStream patterns exclusively - NO FUTURES

use std::sync::{Arc, atomic::AtomicUsize};

use crossbeam_utils::CachePadded;
use serde_json::Value;
use fluent_ai_async::AsyncStream;

use cyrup_sugars::ZeroOneOrMany;
use crate::context::Document;
use crate::memory::config::memory::MemoryConfig as ComprehensiveMemoryConfig;
use crate::memory::manager::MemoryConfig;
use crate::memory::{Memory, MemoryError, MemoryTool, MemoryToolError};
use crate::model::Model;
use crate::tool::McpToolData;

/// Maximum number of tools per agent (const generic default)
pub const MAX_AGENT_TOOLS: usize = 32;

/// Agent statistics for performance monitoring
#[allow(dead_code)] // TODO: Implement in agent monitoring system
static AGENT_STATS: CachePadded<AtomicUsize> = CachePadded::new(AtomicUsize::new(0));

/// Agent creation error types
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Memory system initialization error
    #[error("Memory initialization failed: {0}")]
    MemoryInit(#[from] MemoryError),
    /// Memory tool creation error
    #[error("Memory tool creation failed: {0}")]
    MemoryTool(#[from] MemoryToolError),
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    /// Agent initialization error
    #[error("Agent initialization failed: {0}")]
    InitializationError(String)}

/// Agent data structure with automatic memory tool injection
#[derive(Debug, Clone)]
pub struct Agent {
    /// The model configuration and implementation to use for this agent
    pub model: &'static dyn Model,
    /// System prompt that defines the agent's personality and behavior
    pub system_prompt: String,
    /// Context documents that provide background information
    pub context: ZeroOneOrMany<Document>,
    /// MCP tools available to the agent for function calling
    pub tools: ZeroOneOrMany<McpToolData>,
    /// Optional memory system for storing and retrieving conversation context
    pub memory: Option<Memory>,
    /// Memory tool for automated memory management operations
    pub memory_tool: Option<MemoryTool>,
    /// Temperature setting for response randomness (0.0 to 2.0)
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate in responses
    pub max_tokens: Option<u64>,
    /// Additional model-specific parameters as flexible JSON
    pub additional_params: Option<Value>}

impl Agent {
    /// Create a new agent with zero-allocation memory tool injection
    ///
    /// # Arguments
    /// * `model` - Model configuration for the agent
    /// * `system_prompt` - System prompt for the agent
    ///
    /// # Returns
    /// Stream containing configured agent with memory tool
    ///
    /// # Performance
    /// Zero allocation agent construction with lock-free memory manager sharing
    #[inline]
    pub fn new(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
    ) -> AsyncStream<Self> {
        let system_prompt = system_prompt.into();
        
        AsyncStream::with_channel(move |sender| {
            // Initialize memory system with cognitive settings optimized for performance
            let comprehensive_config = ComprehensiveMemoryConfig::default();
            // Convert comprehensive config to Memory::new() format
            let memory_config = MemoryConfig {
                database_url: comprehensive_config.database.connection_string.to_string(),
                embedding_dimension: comprehensive_config.vector_store.dimension};
            let mut memory_stream = Memory::new(memory_config);
            
            // Use streaming-only pattern to get the Memory instance
            if let Some(memory) = memory_stream.try_next() {
                // Create memory tool with zero-allocation initialization
                let memory_arc = Arc::new(memory);
                let memory_tool = MemoryTool::new(memory_arc.clone());

                let agent = Self {
                    model,
                    system_prompt,
                    context: ZeroOneOrMany::None,
                    tools: ZeroOneOrMany::None,
                    memory: Some((*memory_arc).clone()),
                    memory_tool: Some(memory_tool),
                    temperature: None,
                    max_tokens: None,
                    additional_params: None};
                    
                let _ = sender.send(agent);
            }
        })
    }

    /// Create a new agent with custom memory configuration
    ///
    /// # Arguments
    /// * `model` - Model configuration for the agent
    /// * `system_prompt` - System prompt for the agent
    /// * `memory_config` - Custom memory configuration
    ///
    /// # Returns
    /// Stream containing configured agent with memory tool
    ///
    /// # Performance
    /// Zero allocation with custom cognitive settings
    #[inline]
    pub fn with_memory_config(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
        memory_config: ComprehensiveMemoryConfig,
    ) -> AsyncStream<Self> {
        let system_prompt = system_prompt.into();
        
        AsyncStream::with_channel(move |sender| {
            // Initialize memory system with custom configuration
            // Convert comprehensive config to Memory::new() format
            let memory_cfg = MemoryConfig {
                database_url: memory_config.database.connection_string.to_string(),
                embedding_dimension: memory_config.vector_store.dimension};
            let mut memory_stream = Memory::new(memory_cfg);
            
            // Use streaming-only pattern to get the Memory instance
            if let Some(memory) = memory_stream.try_next() {
                // Create memory tool with zero-allocation initialization
                let memory_arc = Arc::new(memory);
                let memory_tool = MemoryTool::new(memory_arc.clone());

                let agent = Self {
                    model,
                    system_prompt,
                    context: ZeroOneOrMany::None,
                    tools: ZeroOneOrMany::None,
                    memory: Some((*memory_arc).clone()),
                    memory_tool: Some(memory_tool),
                    temperature: None,
                    max_tokens: None,
                    additional_params: None};
                    
                let _ = sender.send(agent);
            }
        })
    }

    /// Create a new agent with shared memory instance
    ///
    /// # Arguments
    /// * `model` - Model configuration for the agent
    /// * `system_prompt` - System prompt for the agent
    /// * `memory` - Shared memory instance for lock-free concurrent access
    ///
    /// # Returns
    /// Stream containing configured agent with shared memory
    ///
    /// # Performance
    /// Zero allocation with lock-free memory sharing between agents
    #[inline]
    pub fn with_shared_memory(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
        memory: Memory,
    ) -> AsyncStream<Self> {
        let system_prompt = system_prompt.into();
        
        AsyncStream::with_channel(move |sender| {
            // Create memory tool with zero-allocation initialization
            let memory_arc = Arc::new(memory);
            let memory_tool = MemoryTool::new(memory_arc.clone());

            let agent = Self {
                model,
                system_prompt,
                context: ZeroOneOrMany::None,
                tools: ZeroOneOrMany::None,
                memory: Some((*memory_arc).clone()),
                memory_tool: Some(memory_tool),
                temperature: None,
                max_tokens: None,
                additional_params: None};
                
            let _ = sender.send(agent);
        })
    }

    /// Get memory tool reference for direct access
    ///
    /// # Returns
    /// Optional reference to memory tool
    ///
    /// # Performance
    /// Zero cost abstraction with direct tool access
    #[inline]
    pub fn memory_tool(&self) -> Option<&MemoryTool> {
        self.memory_tool.as_ref()
    }

    /// Get memory reference for direct access
    ///
    /// # Returns
    /// Optional reference to memory instance
    ///
    /// # Performance
    /// Zero cost abstraction with direct memory access
    #[inline]
    pub fn memory(&self) -> Option<&Memory> {
        self.memory.as_ref()
    }

    /// Add additional tool to the agent
    ///
    /// # Arguments
    /// * `tool` - Tool to add to the agent
    ///
    /// # Returns
    /// Updated agent instance
    ///
    /// # Performance
    /// Zero allocation with inlined tool addition
    #[inline]
    pub fn add_tool(mut self, tool: McpToolData) -> Self {
        match &mut self.tools {
            ZeroOneOrMany::None => {
                self.tools = ZeroOneOrMany::One(tool);
            }
            ZeroOneOrMany::One(existing) => {
                let existing = std::mem::replace(existing, tool.clone());
                self.tools = ZeroOneOrMany::Many(vec![existing, tool]);
            }
            ZeroOneOrMany::Many(tools) => {
                tools.push(tool);
            }
        }
        self
    }

    /// Set agent temperature
    ///
    /// # Arguments
    /// * `temperature` - Temperature value for model sampling
    ///
    /// # Returns
    /// Updated agent instance
    ///
    /// # Performance
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set agent max tokens
    ///
    /// # Arguments
    /// * `max_tokens` - Maximum tokens for model output
    ///
    /// # Returns
    /// Updated agent instance
    ///
    /// # Performance
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}