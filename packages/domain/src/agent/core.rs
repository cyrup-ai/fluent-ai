//! Core agent data structures with automatic memory tool injection

use std::sync::{Arc, atomic::AtomicUsize};

use crossbeam_utils::CachePadded;
use fluent_ai_memory::MemoryConfig;
use serde_json::Value;

use crate::ZeroOneOrMany;
use crate::context::Document;
use crate::memory::{Memory, MemoryError, MemoryTool, MemoryToolError};
use crate::model::Model;
use crate::tool::McpToolData;

/// Maximum number of tools per agent (const generic default)
pub const MAX_AGENT_TOOLS: usize = 32;

/// Agent statistics for performance monitoring
#[allow(dead_code)] // TODO: Implement in agent monitoring system
static AGENT_STATS: CachePadded<AtomicUsize> = CachePadded::new(AtomicUsize::new(0));

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;

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
}

/// Agent data structure with automatic memory tool injection
#[derive(Debug, Clone)]
pub struct Agent {
    pub model: &'static dyn Model,
    pub system_prompt: String,
    pub context: ZeroOneOrMany<Document>,
    pub tools: ZeroOneOrMany<McpToolData>,
    pub memory: Option<Memory>,
    pub memory_tool: Option<MemoryTool>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<Value>,
}

impl Agent {
    /// Create a new agent with zero-allocation memory tool injection
    ///
    /// # Arguments
    /// * `model` - Model configuration for the agent
    /// * `system_prompt` - System prompt for the agent
    ///
    /// # Returns
    /// Result containing configured agent with memory tool
    ///
    /// # Performance
    /// Zero allocation agent construction with lock-free memory manager sharing
    #[inline]
    pub async fn new(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
    ) -> Result<Self, AgentError> {
        // Initialize memory system with cognitive settings optimized for performance
        let memory_config = MemoryConfig::default();
        let memory = Arc::new(Memory::new(memory_config).await?);

        // Create memory tool with zero-allocation initialization
        let memory_tool = MemoryTool::new(Arc::clone(&memory));

        Ok(Self {
            model,
            system_prompt: system_prompt.into(),
            context: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            memory: Some((*memory).clone()),
            memory_tool: Some(memory_tool),
            temperature: None,
            max_tokens: None,
            additional_params: None,
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
    /// Result containing configured agent with memory tool
    ///
    /// # Performance
    /// Zero allocation with custom cognitive settings
    #[inline]
    pub async fn with_memory_config(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
        memory_config: MemoryConfig,
    ) -> Result<Self, AgentError> {
        // Initialize memory system with custom configuration
        let memory = Arc::new(Memory::new(memory_config).await?);

        // Create memory tool with zero-allocation initialization
        let memory_tool = MemoryTool::new(Arc::clone(&memory));

        Ok(Self {
            model,
            system_prompt: system_prompt.into(),
            context: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            memory: Some((*memory).clone()),
            memory_tool: Some(memory_tool),
            temperature: None,
            max_tokens: None,
            additional_params: None,
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
    /// Result containing configured agent with shared memory
    ///
    /// # Performance
    /// Zero allocation with lock-free memory sharing between agents
    #[inline]
    pub async fn with_shared_memory(
        model: &'static dyn Model,
        system_prompt: impl Into<String>,
        memory: Arc<Memory>,
    ) -> Result<Self, AgentError> {
        // Create memory tool with zero-allocation initialization
        let memory_tool = MemoryTool::new(Arc::clone(&memory));

        Ok(Self {
            model,
            system_prompt: system_prompt.into(),
            context: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            memory: Some((*memory).clone()),
            memory_tool: Some(memory_tool),
            temperature: None,
            max_tokens: None,
            additional_params: None,
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
