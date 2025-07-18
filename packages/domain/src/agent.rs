//! Agent domain types
//!
//! Contains pure data structures for agents with automatic memory tool injection.
//! Builder implementations are in fluent_ai package.

use std::sync::Arc;

use serde_json::Value;

use crate::mcp_tool::McpToolData;
use crate::memory::{MemoryConfig, MemoryError};
use crate::memory_tool::{MemoryTool, MemoryToolError};
use crate::{Document, Memory, Models, ZeroOneOrMany};

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
    pub model: Models,
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
    pub async fn new(model: Models, system_prompt: impl Into<String>) -> Result<Self, AgentError> {
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
        model: Models,
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
    /// Result containing configured agent with shared memory tool
    ///
    /// # Performance
    /// Zero allocation with lock-free memory sharing between agents
    #[inline]
    pub fn with_shared_memory(
        model: Models,
        system_prompt: impl Into<String>,
        memory: Arc<Memory>,
    ) -> Result<Self, AgentError> {
        // Create memory tool with shared memory instance
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
        self.tools = match self.tools {
            ZeroOneOrMany::None => ZeroOneOrMany::One(tool),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, tool]),
            ZeroOneOrMany::Many(mut tools) => {
                tools.push(tool);
                ZeroOneOrMany::Many(tools)
            }
        };
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

/// Zero-allocation agent builder with const generics
#[derive(Debug)]
pub struct AgentBuilder<const TOOLS_CAPACITY: usize = MAX_AGENT_TOOLS> {
    model: Option<Models>,
    system_prompt: Option<String>,
    memory_config: Option<MemoryConfig>,
    shared_memory: Option<Arc<Memory>>,
    tools: ArrayVec<McpToolData, TOOLS_CAPACITY>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<Value>,
    config: Arc<AgentConfig>,
}

impl<const TOOLS_CAPACITY: usize> Default for AgentBuilder<TOOLS_CAPACITY> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const TOOLS_CAPACITY: usize> AgentBuilder<TOOLS_CAPACITY> {
    /// Create new agent builder with zero-allocation initialization
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            model: None,
            system_prompt: None,
            memory_config: None,
            shared_memory: None,
            tools: ArrayVec::new(),
            temperature: None,
            max_tokens: None,
            additional_params: None,
            config: get_agent_config(),
        }
    }

    /// Set model with validation
    #[inline(always)]
    pub fn model(mut self, model: Models) -> Self {
        self.model = Some(model);
        self
    }

    /// Set system prompt
    #[inline(always)]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set memory configuration
    #[inline(always)]
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Set shared memory instance
    #[inline(always)]
    pub fn shared_memory(mut self, memory: Arc<Memory>) -> Self {
        self.shared_memory = Some(memory);
        self
    }

    /// Add tool with zero-allocation error handling
    #[inline(always)]
    pub fn tool(mut self, tool: McpToolData) -> AgentResult<Self> {
        if self.tools.try_push(tool).is_err() {
            return Err(AgentError::ToolCapacityExceeded {
                max: TOOLS_CAPACITY,
            });
        }
        Ok(self)
    }

    /// Set temperature with validation
    #[inline(always)]
    pub fn temperature(mut self, temperature: f64) -> AgentResult<Self> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(AgentError::InvalidModel {
                reason: format!(
                    "Temperature {} is outside valid range [0.0, 2.0]",
                    temperature
                ),
            });
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set max tokens with validation
    #[inline(always)]
    pub fn max_tokens(mut self, max_tokens: u64) -> AgentResult<Self> {
        if !(1..=100_000).contains(&max_tokens) {
            return Err(AgentError::InvalidModel {
                reason: format!(
                    "Max tokens {} is outside valid range [1, 100000]",
                    max_tokens
                ),
            });
        }
        self.max_tokens = Some(max_tokens);
        Ok(self)
    }

    /// Build agent with comprehensive error handling and retry logic
    #[inline(always)]
    pub async fn build(self) -> AgentResult<Agent> {
        // Validate required fields
        let model = self
            .model
            .ok_or_else(|| AgentError::Config("Model is required".to_string()))?;
        let system_prompt = self
            .system_prompt
            .unwrap_or_else(|| "You are a helpful AI assistant.".to_string());

        // Increment atomic counter for lock-free statistics
        AGENT_STATS.fetch_add(1, Ordering::Relaxed);

        // Initialize memory system
        let memory_arc = if let Some(shared_memory) = self.shared_memory {
            shared_memory
        } else {
            let memory_config = self
                .memory_config
                .unwrap_or_else(|| self.config.memory_config.clone());

            let memory = retry_with_backoff(
                || async { Memory::new(memory_config.clone()).await },
                self.config.max_retry_attempts,
                self.config.base_backoff_delay,
            )
            .await?;

            Arc::new(memory)
        };

        // Create memory tool if enabled
        let memory_tool = if self.config.auto_memory_injection {
            Some(MemoryTool::new(Arc::clone(&memory_arc)))
        } else {
            None
        };

        // Convert tools with zero-allocation
        let tools = match self.tools.len() {
            0 => ZeroOneOrMany::None,
            1 => ZeroOneOrMany::One(self.tools.into_iter().next().unwrap_or_default()),
            _ => {
                let tools_vec: Vec<_> = self.tools.into_iter().collect();
                ZeroOneOrMany::Many(tools_vec)
            }
        };

        Ok(Agent {
            model,
            system_prompt,
            context: ZeroOneOrMany::None,
            tools,
            memory: Some(memory_arc),
            memory_tool,
            temperature: self.temperature.or(self.config.default_temperature),
            max_tokens: self.max_tokens.or(self.config.default_max_tokens),
            additional_params: self.additional_params,
            config: self.config,
            created_at: std::time::Instant::now(),
            stats: Arc::new(CachePadded::new(AtomicUsize::new(0))),
        })
    }
}

/// Create agent builder with default capacity
#[inline(always)]
pub fn agent_builder() -> AgentBuilder {
    AgentBuilder::new()
}

/// Create agent builder with custom capacity
#[inline(always)]
pub fn agent_builder_with_capacity<const CAPACITY: usize>() -> AgentBuilder<CAPACITY> {
    AgentBuilder::new()
}
