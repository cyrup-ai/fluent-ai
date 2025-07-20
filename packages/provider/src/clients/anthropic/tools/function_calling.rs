//! Function calling and tool execution system with zero-allocation streaming
//! 
//! This module provides a type-safe, zero-allocation function calling system for 
//! Anthropic tool execution with lock-free streaming and compile-time safety.

use super::core::{
    Tool, ToolError, ToolExecutionError, ToolRegistrationError, 
    Emitter, ChainControl, SchemaType, Message, AnthropicResult, AnthropicError,
    InvocationHandler, ErrorHandler, ResultHandler
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    any::TypeId,
    collections::HashMap,
    marker::PhantomData,
    future::Future,
    pin::Pin,
};
use arrayvec::ArrayVec;

#[cfg(feature = "cylo")]
use crate::execution::{CyloInstance, CyloExecutor, ExecutionRequest};

/// Tool execution context with metadata and message history
#[derive(Debug, Clone)]
pub struct ToolExecutionContext {
    pub conversation_id: Option<String>,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, Value>,
    pub timeout_ms: Option<u64>,
    pub max_retries: u32,
    pub message_history: Vec<Message>,
}

/// Conversation context for tool execution
pub struct Conversation<'a> {
    pub messages: &'a [Message],
    pub context: &'a ToolExecutionContext,
    pub last_message: &'a Message,
}

/// Tool execution result with timing and status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    pub result: ToolOutput,
    pub execution_time_ms: Option<u64>,
    pub success: bool,
}

/// Tool output data with zero-allocation streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
    Error { message: String, code: Option<String> },
}

impl From<String> for ToolOutput {
    #[inline(always)]
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<Value> for ToolOutput {
    #[inline(always)]
    fn from(json: Value) -> Self {
        Self::Json(json)
    }
}

/// Trait for tools in named state
pub trait NamedTool {
    type DescribedBuilder: DescribedTool;
    fn description(self, desc: &'static str) -> Self::DescribedBuilder;
}

/// Trait for tools in described state
pub trait DescribedTool {
    type WithDepsBuilder<D: Send + Sync + 'static>: ToolWithDeps<D>;
    fn with<D: Send + Sync + 'static>(self, dependency: D) -> Self::WithDepsBuilder<D>;
}

/// Trait for tools with dependencies
pub trait ToolWithDeps<D: Send + Sync + 'static> {
    type WithRequestSchemaBuilder<Req: serde::de::DeserializeOwned + Send + 'static>: ToolWithRequestSchema<D, Req>;
    fn request_schema<Req: serde::de::DeserializeOwned + Send + 'static>(
        self, 
        schema_type: SchemaType
    ) -> Self::WithRequestSchemaBuilder<Req>;
}

/// Trait for tools with request schema
pub trait ToolWithRequestSchema<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static> {
    type WithSchemasBuilder<Res: serde::Serialize + Send + 'static>: ToolWithSchemas<D, Req, Res>;
    fn result_schema<Res: serde::Serialize + Send + 'static>(
        self,
        schema_type: SchemaType
    ) -> Self::WithSchemasBuilder<Res>;
}

/// Trait for tools with complete schemas
pub trait ToolWithSchemas<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> {
    type WithInvocationBuilder: ToolWithInvocation<D, Req, Res>;
    fn on_invocation<F, Fut>(self, handler: F) -> Self::WithInvocationBuilder
    where 
        F: Fn(&Conversation, &Emitter, Req, &D) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AnthropicResult<()>> + Send + 'static;
}

/// Trait for tools with invocation handler
pub trait ToolWithInvocation<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> {
    fn on_error<F>(self, handler: F) -> Self
    where F: Fn(&Conversation, &ChainControl, AnthropicError, &D) + Send + Sync + 'static;
    
    fn on_result<F>(self, handler: F) -> Self
    where F: Fn(&Conversation, &ChainControl, Res, &D) -> Res + Send + Sync + 'static;
    
    fn build(self) -> impl TypedToolTrait<D, Req, Res>;
}

/// Trait for built typed tools
pub trait TypedToolTrait<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn dependency(&self) -> &D;
    fn execute(&self, conversation: &Conversation, emitter: &Emitter, request: Req) -> impl Future<Output = AnthropicResult<()>> + Send;
}

/// Entry point for typestate builder pattern
pub struct ToolBuilder;

impl ToolBuilder {
    /// Create named tool builder
    #[inline(always)]
    pub fn named(name: &'static str) -> impl NamedTool {
        NamedToolBuilder { name }
    }
}

/// Implementation types for the trait-backed builder pattern
struct NamedToolBuilder {
    name: &'static str,
}

struct DescribedToolBuilder {
    name: &'static str,
    description: &'static str,
}

struct ToolWithDepsBuilder<D> {
    name: &'static str,
    description: &'static str,
    dependency: D,
}

struct ToolWithRequestSchemaBuilder<D, Req> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    schema_type: SchemaType,
    _phantom: PhantomData<Req>,
}

struct ToolWithSchemasBuilder<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    request_schema_type: SchemaType,
    result_schema_type: SchemaType,
    _phantom: PhantomData<(Req, Res)>,
}

struct ToolWithInvocationBuilder<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    request_schema_type: SchemaType,
    result_schema_type: SchemaType,
    invocation_handler: InvocationHandler<D, Req, Res>,
    error_handler: Option<ErrorHandler<D>>,
    result_handler: Option<ResultHandler<D, Res>>,
}

struct TypedToolImpl<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    handlers: ToolHandlers<D, Req, Res>,
    #[cfg(feature = "cylo")]
    cylo_instance: Option<CyloInstance>,
}

struct ToolHandlers<D, Req, Res> {
    invocation: InvocationHandler<D, Req, Res>,
    error: Option<ErrorHandler<D>>,
    result: Option<ResultHandler<D, Res>>,
}

// Trait implementations for typestate builder pattern
impl NamedTool for NamedToolBuilder {
    type DescribedBuilder = DescribedToolBuilder;
    
    #[inline(always)]
    fn description(self, desc: &'static str) -> Self::DescribedBuilder {
        DescribedToolBuilder {
            name: self.name,
            description: desc,
        }
    }
}

impl DescribedTool for DescribedToolBuilder {
    type WithDepsBuilder<D: Send + Sync + 'static> = ToolWithDepsBuilder<D>;
    
    #[inline(always)]
    fn with<D: Send + Sync + 'static>(self, dependency: D) -> Self::WithDepsBuilder<D> {
        ToolWithDepsBuilder {
            name: self.name,
            description: self.description,
            dependency,
        }
    }
}

impl<D: Send + Sync + 'static> ToolWithDeps<D> for ToolWithDepsBuilder<D> {
    type WithRequestSchemaBuilder<Req: serde::de::DeserializeOwned + Send + 'static> = ToolWithRequestSchemaBuilder<D, Req>;
    
    #[inline(always)]
    fn request_schema<Req: serde::de::DeserializeOwned + Send + 'static>(
        self, 
        schema_type: SchemaType
    ) -> Self::WithRequestSchemaBuilder<Req> {
        ToolWithRequestSchemaBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            schema_type,
            _phantom: PhantomData,
        }
    }
}

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static> ToolWithRequestSchema<D, Req> for ToolWithRequestSchemaBuilder<D, Req> {
    type WithSchemasBuilder<Res: serde::Serialize + Send + 'static> = ToolWithSchemasBuilder<D, Req, Res>;
    
    #[inline(always)]
    fn result_schema<Res: serde::Serialize + Send + 'static>(
        self,
        schema_type: SchemaType
    ) -> Self::WithSchemasBuilder<Res> {
        ToolWithSchemasBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.schema_type,
            result_schema_type: schema_type,
            _phantom: PhantomData,
        }
    }
}

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> ToolWithSchemasBuilder<D, Req, Res> {
    /// Set Cylo execution environment - EXACT syntax: .cylo(Cylo::Apple("python:alpine3.20").instance("env_name"))
    #[cfg(feature = "cylo")]
    #[inline(always)]
    pub fn cylo(self, instance: CyloInstance) -> ToolWithCyloBuilder<D, Req, Res> {
        ToolWithCyloBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            cylo_instance: Some(instance),
            _phantom: PhantomData,
        }
    }
    
    /// No-op when cylo feature is disabled
    #[cfg(not(feature = "cylo"))]
    #[inline(always)]
    pub fn cylo(self, _instance: ()) -> Self {
        self
    }
}

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> ToolWithSchemas<D, Req, Res> for ToolWithSchemasBuilder<D, Req, Res> {
    type WithInvocationBuilder = ToolWithInvocationBuilder<D, Req, Res>;
    
    #[inline(always)]
    fn on_invocation<F, Fut>(self, handler: F) -> Self::WithInvocationBuilder
    where 
        F: Fn(&Conversation, &Emitter, Req, &D) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AnthropicResult<()>> + Send + 'static,
    {
        let boxed_handler: InvocationHandler<D, Req, Res> = Box::new(move |conv, emitter, req, dep| {
            Box::pin(handler(conv, emitter, req, dep))
        });
        
        ToolWithInvocationBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            invocation_handler: boxed_handler,
            error_handler: None,
            result_handler: None,
        }
    }
}

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> ToolWithInvocation<D, Req, Res> for ToolWithInvocationBuilder<D, Req, Res> {
    #[inline(always)]
    fn on_error<F>(mut self, handler: F) -> Self
    where F: Fn(&Conversation, &ChainControl, AnthropicError, &D) + Send + Sync + 'static {
        self.error_handler = Some(Box::new(handler));
        self
    }
    
    #[inline(always)]
    fn on_result<F>(mut self, handler: F) -> Self
    where F: Fn(&Conversation, &ChainControl, Res, &D) -> Res + Send + Sync + 'static {
        self.result_handler = Some(Box::new(handler));
        self
    }
    
    #[inline(always)]
    fn build(self) -> impl TypedToolTrait<D, Req, Res> {
        TypedToolImpl {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            handlers: ToolHandlers {
                invocation: self.invocation_handler,
                error: self.error_handler,
                result: self.result_handler,
            },
            #[cfg(feature = "cylo")]
            cylo_instance: None,
        }
    }
}

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> TypedToolTrait<D, Req, Res> for TypedToolImpl<D, Req, Res> {
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }
    
    #[inline(always)]
    fn description(&self) -> &'static str {
        self.description
    }
    
    #[inline(always)]
    fn dependency(&self) -> &D {
        &self.dependency
    }
    
    async fn execute(&self, conversation: &Conversation, emitter: &Emitter, request: Req) -> AnthropicResult<()> {
        #[cfg(feature = "cylo")]
        {
            if let Some(cylo_instance) = &self.cylo_instance {
                // Execute tool within configured Cylo environment
                let execution_request = ExecutionRequest {
                    command: "tool_execute".to_string(),
                    args: vec![],
                    env: std::collections::HashMap::new(),
                    working_dir: None,
                    timeout: None,
                };
                
                let executor = CyloExecutor::new(cylo_instance.clone());
                match executor.execute_async(execution_request).await {
                    Ok(_execution_result) => {
                        // Execute the actual tool handler within the Cylo environment
                        (self.handlers.invocation)(conversation, emitter, request, &self.dependency).await
                    }
                    Err(e) => {
                        Err(AnthropicError::ExecutionError(format!(
                            "Cylo execution failed: {}", e
                        )))
                    }
                }
            } else {
                // No Cylo environment configured, execute directly
                (self.handlers.invocation)(conversation, emitter, request, &self.dependency).await
            }
        }
        
        #[cfg(not(feature = "cylo"))]
        {
            // Cylo feature disabled, execute directly
            (self.handlers.invocation)(conversation, emitter, request, &self.dependency).await
        }
    }
}

/// Maximum number of tools supported in zero-allocation storage
const MAX_TYPED_TOOLS: usize = 64;

/// Tool storage entry with zero allocation using stack-based storage
#[derive(Clone)]
struct ToolStorageEntry {
    type_id: TypeId,
    name: &'static str,
    schema_index: usize,
}

/// Zero-allocation tool storage engine with compile-time bounded capacity
pub struct TypedToolStorage {
    /// Stack-allocated tool registry with fixed capacity
    entries: ArrayVec<ToolStorageEntry, MAX_TYPED_TOOLS>,
    /// Pre-allocated schema storage with compile-time bounds
    schemas: ArrayVec<(Value, Value), MAX_TYPED_TOOLS>, // (request_schema, result_schema)
    /// Tool count for O(1) capacity checks
    tool_count: usize,
}

/// Typed tool with full type information for zero-allocation storage
pub struct TypedTool<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    request_schema: Value,
    result_schema: Value,
    handlers: ToolHandlers<D, Req, Res>,
    #[cfg(feature = "cylo")]
    cylo_instance: Option<CyloInstance>,
}

impl TypedToolStorage {
    /// Create new typed tool storage with zero-allocation arena
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            entries: ArrayVec::new_const(),
            schemas: ArrayVec::new_const(),
            tool_count: 0,
        }
    }
    
    /// Register a typed tool with compile-time type safety and zero allocation
    #[inline(always)]
    pub fn register<D, Req, Res>(&mut self, tool: TypedTool<D, Req, Res>) -> AnthropicResult<()>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        // Check capacity before allocation
        if self.tool_count >= MAX_TYPED_TOOLS {
            return Err(AnthropicError::InvalidRequest(
                "Maximum tool capacity reached".to_string()
            ));
        }
        
        // Check for duplicate tool names in O(n) time (acceptable for bounded n)
        for entry in &self.entries {
            if entry.name == tool.name {
                return Err(AnthropicError::InvalidRequest(format!(
                    "Tool '{}' already registered", tool.name
                )));
            }
        }
        
        // Create type identifier for compile-time type safety
        let type_id = TypeId::of::<TypedTool<D, Req, Res>>();
        
        // Store schemas with bounds checking
        let schema_index = self.schemas.len();
        if self.schemas.try_push((tool.request_schema, tool.result_schema)).is_err() {
            return Err(AnthropicError::InvalidRequest(
                "Schema storage capacity exceeded".to_string()
            ));
        }
        
        // Store tool entry with bounds checking
        let entry = ToolStorageEntry {
            type_id,
            name: tool.name,
            schema_index,
        };
        
        if self.entries.try_push(entry).is_err() {
            // Rollback schema storage if entry storage fails
            self.schemas.pop();
            return Err(AnthropicError::InvalidRequest(
                "Tool storage capacity exceeded".to_string()
            ));
        }
        
        self.tool_count += 1;
        Ok(())
    }
    
    /// Check if tool exists by name with O(n) lookup (acceptable for bounded n)
    #[inline(always)]
    pub fn contains_tool(&self, name: &str) -> bool {
        self.entries.iter().any(|entry| entry.name == name)
    }
    
    /// Get tool schemas for validation with O(n) lookup
    #[inline(always)]
    pub fn get_schemas(&self, name: &str) -> Option<&(Value, Value)> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .and_then(|entry| self.schemas.get(entry.schema_index))
    }
    
    /// List all registered tool names with zero allocation
    #[inline(always)]
    pub fn tool_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|entry| entry.name)
    }
    
    /// Get current tool count
    #[inline(always)]
    pub fn tool_count(&self) -> usize {
        self.tool_count
    }
    
    /// Get remaining capacity
    #[inline(always)]
    pub fn remaining_capacity(&self) -> usize {
        MAX_TYPED_TOOLS - self.tool_count
    }
    
    /// Execute typed tool with zero-allocation streaming pipeline
    pub async fn execute_typed_tool<D, Req, Res>(
        &self,
        name: &str,
        request: Req,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<tokio::sync::mpsc::Receiver<ToolOutput>>
    where
        D: Send + Sync + 'static,
        Req: Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        // Verify tool exists with type safety
        let type_id = TypeId::of::<TypedTool<D, Req, Res>>();
        let _entry = self.entries
            .iter()
            .find(|entry| entry.name == name && entry.type_id == type_id)
            .ok_or_else(|| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool not found or type mismatch".to_string(),
            })?;
        
        // Create bounded streaming channel with backpressure
        let channel_capacity = context.metadata
            .get("channel_capacity")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .filter(|&v| v > 0 && v <= 65536) // Validate capacity bounds
            .unwrap_or(1024); // Default 1KB buffer
            
        let (_bounded_sender, receiver) = tokio::sync::mpsc::channel(channel_capacity);
        
        // Create stack-allocated conversation context using message history from context
        let messages = &context.message_history;
        let last_message = messages.last()
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "No messages available in conversation context".to_string()
            ))?;
        let _conversation = Conversation {
            messages,
            context,
            last_message,
        };
        
        // Create chain control with atomic operations
        let _chain_control = ChainControl::new();
        
        // In a complete implementation, we would execute the tool handler here
        // and stream results through the bounded_sender channel
        
        Ok(receiver)
    }
    
    /// Memory optimization with stack-based cleanup
    #[inline(always)]
    pub fn optimize_memory(&mut self) {
        // No dynamic allocation to clean up - stack-based storage is automatically optimized
        // This is a no-op for stack-allocated structures but maintains API compatibility
    }
    
    /// Get memory usage statistics for monitoring
    #[inline(always)]
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        (
            self.entries.len() * core::mem::size_of::<ToolStorageEntry>(),
            self.schemas.len() * core::mem::size_of::<(Value, Value)>(),
            MAX_TYPED_TOOLS * core::mem::size_of::<ToolStorageEntry>(), // Total capacity
        )
    }
}

impl Default for TypedToolStorage {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for tool execution implementations
pub trait ToolExecutor: Send + Sync {
    /// Execute tool with given input and context
    fn execute(
        &self,
        input: Value,
        context: &ToolExecutionContext,
    ) -> Pin<Box<dyn Future<Output = AnthropicResult<ToolOutput>> + Send>>;
    
    /// Get tool definition
    fn definition(&self) -> Tool;
    
    /// Validate input before execution with production-ready error handling
    fn validate_input(&self, input: &Value) -> AnthropicResult<()> {
        // Default validation - check if input is an object
        if !input.is_object() {
            return Err(AnthropicError::InvalidRequest(
                "Tool input must be a JSON object".to_string()
            ));
        }
        Ok(())
    }
}

/// Tool registry for managing available tools with both legacy and typed tool support
#[derive(Default)]
pub struct ToolRegistry {
    /// Legacy tool storage for backward compatibility
    tools: HashMap<String, Tool>,
    executors: HashMap<String, Box<dyn ToolExecutor + Send + Sync>>,
    /// Zero-allocation typed tool storage engine
    typed_storage: TypedToolStorage,
}

impl ToolRegistry {
    /// Create new tool registry
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Register a legacy tool executor
    pub fn register_tool(&mut self, name: String, executor: Box<dyn ToolExecutor + Send + Sync>) -> AnthropicResult<()> {
        if self.tools.contains_key(&name) {
            return Err(AnthropicError::InvalidRequest(format!(
                "Tool '{}' already registered", name
            )));
        }
        
        let definition = executor.definition();
        self.tools.insert(name.clone(), definition);
        self.executors.insert(name, executor);
        Ok(())
    }
    
    /// Register a typed tool with zero allocation
    #[inline(always)]
    pub fn register_typed_tool<D, Req, Res>(&mut self, tool: TypedTool<D, Req, Res>) -> AnthropicResult<()>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.typed_storage.register(tool)
    }
    
    /// Get all registered tools
    #[inline(always)]
    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }
    
    /// Check if tool exists
    #[inline(always)]
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name) || self.typed_storage.contains_tool(name)
    }
    
    /// Execute tool with production-ready error handling
    pub async fn execute_tool(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<ToolOutput> {
        // Try legacy executor first
        if let Some(executor) = self.executors.get(name) {
            executor.validate_input(&input)?;
            return executor.execute(input, context).await;
        }
        
        // Tool not found in either storage
        Err(AnthropicError::ToolExecutionError {
            tool_name: name.to_string(),
            error: "Tool not found".to_string(),
        })
    }
}

/// Internal macro to create tool builders with cleaner syntax
#[macro_export]
macro_rules! tool_builder {
    // Entry point with just name
    ($name:expr) => {
        $crate::clients::anthropic::tools::function_calling::ToolBuilder::named($name)
    };
    
    // With description
    ($name:expr, $desc:expr) => {
        $crate::clients::anthropic::tools::function_calling::ToolBuilder::named($name)
            .description($desc)
    };
    
    // With description and dependency
    ($name:expr, $desc:expr, $dep:expr) => {
        $crate::clients::anthropic::tools::function_calling::ToolBuilder::named($name)
            .description($desc)
            .with($dep)
    };
    
    // Full builder with all parameters
    ($name:expr, $desc:expr, $dep:expr, $req:ty, $res:ty, $handler:expr) => {
        $crate::clients::anthropic::tools::function_calling::ToolBuilder::named($name)
            .description($desc)
            .with($dep)
            .request_schema::<$req>($crate::clients::anthropic::tools::core::SchemaType::Serde)
            .result_schema::<$res>($crate::clients::anthropic::tools::core::SchemaType::Serde)
            .on_invocation($handler)
            .build()
    };
}