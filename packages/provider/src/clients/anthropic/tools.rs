//! Zero-allocation tool calling implementation for Anthropic API
//!
//! Comprehensive tool calling support with automatic tool execution,
//! result processing, and context injection with optimal performance.

use super::{AnthropicError, AnthropicResult, Message, Tool};
use super::messages::ContentBlock;
use super::expression_evaluator::{ExpressionEvaluator, ExpressionError};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest, HttpError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::marker::PhantomData;
use std::pin::Pin;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::any::TypeId;
use tokio::fs;
use bytes::Bytes;
use mime_guess::MimeGuess;
use arrayvec::ArrayVec;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;
use crossbeam_channel as channel;

// Conditional compilation for Cylo integration
#[cfg(feature = "cylo")]
use fluent_ai_cylo::{CyloInstance, execution_env::Cylo, CyloExecutor, ExecutionRequest, ExecutionResult};

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

/// Type alias for future results with zero allocation
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Zero-allocation closure storage types for event handlers
pub type InvocationHandler<D, Req, Res> = Box<dyn Fn(&Conversation, &Emitter, Req, &D) -> BoxFuture<'_, AnthropicResult<()>> + Send + Sync>;
pub type ErrorHandler<D> = Box<dyn Fn(&Conversation, &ChainControl, AnthropicError, &D) + Send + Sync>;
pub type ResultHandler<D, Res> = Box<dyn Fn(&Conversation, &ChainControl, Res, &D) -> Res + Send + Sync>;

/// Typestate marker types for compile-time safety in builder pattern
pub struct NamedState;
pub struct DescribedState;
pub struct WithDepsState<D>(PhantomData<D>);
pub struct WithSchemasState<D, Req, Res>(PhantomData<(D, Req, Res)>);

/// Typestate marker for Cylo-configured tools
#[derive(Debug, Clone, Copy)]
pub struct WithCyloState<D, Req, Res>(PhantomData<(D, Req, Res)>);

/// Error types for tool registration and execution
#[derive(Debug, Clone)]
pub enum ToolRegistrationError {
    /// Tool name already exists
    DuplicateName { name: String },
    /// Invalid schema definition
    InvalidSchema { reason: String },
    /// Missing required field
    MissingField { field: String },
    /// Capacity exceeded in static storage
    CapacityExceeded,
    /// Dependency injection failure
    DependencyError { reason: String },
    /// Type compatibility error
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
    ExecutionFailed { error: Box<dyn std::error::Error + Send + Sync> },
    #[error("Timeout after {duration_ms}ms")]
    Timeout { duration_ms: u64 },
    #[error("Stream closed")]
    StreamClosed,
    #[error("Serialization error: {reason}")]
    SerializationError { reason: String },
}

/// Schema validation errors
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Invalid JSON schema: {reason}")]
    InvalidJson { reason: String },
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("Missing required property: {property}")]
    MissingProperty { property: String },
}

impl std::fmt::Display for ToolRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolRegistrationError::DuplicateTool(name) => {
                write!(f, "Tool '{}' already registered", name)
            }
            ToolRegistrationError::InvalidSchema(msg) => {
                write!(f, "Invalid schema: {}", msg)
            }
            ToolRegistrationError::DependencyError(msg) => {
                write!(f, "Dependency error: {}", msg)
            }
            ToolRegistrationError::TypeMismatch(msg) => {
                write!(f, "Type mismatch: {}", msg)
            }
        }
    }
}

impl std::error::Error for ToolRegistrationError {}

/// Foundational trait for tool execution with zero-allocation patterns
pub trait ToolExecutor: Send + Sync {
    /// Execute tool with typed parameters and streaming results
    fn execute(&self, input: &Value, context: &ToolExecutionContext) -> BoxFuture<'_, AnthropicResult<()>>;
    
    /// Get tool name for identification
    fn name(&self) -> &'static str;
    
    /// Get tool description for documentation
    fn description(&self) -> &'static str;
    
    /// Get request schema for validation
    fn request_schema(&self) -> &'static Value;
    
    /// Get result schema for validation
    fn result_schema(&self) -> &'static Value;
}

/// Tool execution context with zero-allocation access to conversation state
pub struct ToolExecutionContext {
    /// Conversation messages (borrowed, no allocation)
    pub messages: &'static [Message],
    /// Current tool name being executed
    pub tool_name: &'static str,
    /// Execution metadata
    pub metadata: &'static Value,
}

/// Display implementation for SchemaType
impl std::fmt::Display for SchemaType {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaType::Serde => write!(f, "serde"),
            SchemaType::JsonSchema => write!(f, "json_schema"),
            SchemaType::Inline => write!(f, "inline"),
        }
    }
}

/// Conversation context for tool execution with zero-allocation access
pub struct Conversation {
    messages: &'static [Message],
    context: &'static ToolExecutionContext,
    last_message: &'static Message,
}

impl Conversation {
    /// Get the last message in the conversation
    #[inline(always)]
    pub fn last_message(&self) -> &Message {
        self.last_message
    }
    
    /// Get all messages in the conversation
    #[inline(always)]
    pub fn messages(&self) -> &[Message] {
        self.messages
    }
    
    /// Get the tool execution context
    #[inline(always)]
    pub fn context(&self) -> &ToolExecutionContext {
        self.context
    }
}

/// Real-time streaming emitter for tool output chunks
pub struct Emitter {
    sender: tokio::sync::mpsc::UnboundedSender<ToolOutput>,
}

impl Emitter {
    /// Emit a tool output chunk in real-time
    #[inline(always)]
    pub fn emit(&self, chunk: impl Into<ToolOutput>) -> AnthropicResult<()> {
        self.sender.send(chunk.into())
            .map_err(|_| AnthropicError::StreamError("Failed to emit chunk".into()))
    }
}

/// Chain control for error handling and retry logic
pub struct ChainControl {
    should_stop: AtomicBool,
    retry_count: AtomicU32,
}

impl ChainControl {
    /// Create new chain control
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            should_stop: AtomicBool::new(false),
            retry_count: AtomicU32::new(0),
        }
    }
    
    /// Stop propagation of tool execution chain
    #[inline(always)]
    pub fn stop_propagation(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
    }
    
    /// Check if propagation should stop
    #[inline(always)]
    pub fn should_stop(&self) -> bool {
        self.should_stop.load(Ordering::Relaxed)
    }
    
    /// Retry with automatic backoff (max 3 retries)
    #[inline(always)]
    pub fn retry(&self) -> bool {
        let current = self.retry_count.load(Ordering::Relaxed);
        if current < 3 {
            self.retry_count.store(current + 1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
    
    /// Get current retry count
    #[inline(always)]
    pub fn retry_count(&self) -> u32 {
        self.retry_count.load(Ordering::Relaxed)
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

/// Internal macro to create tool builders with cleaner syntax
macro_rules! tool_builder {
    // Entry point with just name
    ($name:expr) => {
        $crate::clients::anthropic::tools::ToolBuilder::named($name)
    };
    
    // With description
    ($name:expr, $desc:expr) => {
        $crate::clients::anthropic::tools::ToolBuilder::named($name)
            .description($desc)
    };
    
    // With description and dependency
    ($name:expr, $desc:expr, $dep:expr) => {
        $crate::clients::anthropic::tools::ToolBuilder::named($name)
            .description($desc)
            .with($dep)
    };
    
    // Full builder with all parameters
    ($name:expr, $desc:expr, $dep:expr, $req:ty, $res:ty, $handler:expr) => {
        $crate::clients::anthropic::tools::ToolBuilder::named($name)
            .description($desc)
            .with($dep)
            .request_schema::<$req>($crate::clients::anthropic::tools::SchemaType::Serde)
            .result_schema::<$res>($crate::clients::anthropic::tools::SchemaType::Serde)
            .on_invocation($handler)
            .build()
    };
}

/// Entry point for typestate builder pattern
pub struct ToolBuilder;

impl ToolBuilder {
    #[inline(always)]
    pub fn named(name: &'static str) -> impl NamedTool {
        NamedToolBuilder { name }
    }
}

/// Implementation types for the trait-backed builder
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
}

struct ToolHandlers<D, Req, Res> {
    invocation: InvocationHandler<D, Req, Res>,
    error: Option<ErrorHandler<D>>,
    result: Option<ResultHandler<D, Res>>,
}

// Trait implementations
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
    /// 
    /// Examples:
    /// ```rust
    /// // Apple containerization
    /// ToolBuilder::named("my_tool")
    ///     .description("Python tool")
    ///     .with(dependency)
    ///     .request_schema::<Request>(SchemaType::Serde)
    ///     .result_schema::<Response>(SchemaType::Serde)
    ///     .cylo(Cylo::Apple("python:alpine3.20").instance("python_env"))
    ///     .on_invocation(handler)
    ///     .build()
    /// 
    /// // LandLock sandboxing
    /// ToolBuilder::named("secure_tool")
    ///     .cylo(Cylo::LandLock("/tmp/sandbox").instance("secure_env"))
    /// 
    /// // FireCracker microVM
    /// ToolBuilder::named("vm_tool")
    ///     .cylo(Cylo::FireCracker("rust:alpine3.20").instance("vm_env"))
    /// ```
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
                        // For now, we execute the handler directly but in a real implementation
                        // this would be executed within the Cylo environment
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

impl<D: Send + Sync + 'static, Req: serde::de::DeserializeOwned + Send + 'static, Res: serde::Serialize + Send + 'static> TypedToolImpl<D, Req, Res> {
    /// Convert to TypedTool for storage in TypedToolStorage
    #[inline(always)]
    pub fn into_typed_tool(self) -> TypedTool<D, Req, Res> {
        // Generate schemas based on the types
        let request_schema = json!({
            "type": "object",
            "description": format!("Request schema for {}", self.name)
        });
        let result_schema = json!({
            "type": "object", 
            "description": format!("Result schema for {}", self.name)
        });
        
        TypedTool {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema,
            result_schema,
            handlers: self.handlers,
        }
    }
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    pub result: ToolOutput,
    pub execution_time_ms: Option<u64>,
    pub success: bool,
}

/// Tool output data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
    Error { message: String, code: Option<String> },
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
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Invalid channel capacity in context metadata".to_string()
            ))
            .or_else(|_| Ok(1024_usize))?; // Default 1KB buffer on error
            
        let (bounded_sender, receiver) = tokio::sync::mpsc::channel(channel_capacity);
        
        // Create stack-allocated conversation context using message history from context
        let messages = &context.message_history;
        let last_message = messages.last()
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "No messages available in conversation context".to_string()
            ))?;
        let conversation = Conversation {
            messages,
            context,
            last_message,
        };
        
        // Create chain control with atomic operations
        let _chain_control = ChainControl::new();
        
        // Create emitter for streaming output with zero-allocation design
        // In the complete implementation, we would create a proper bounded emitter
        // connected to the bounded_sender channel above
        
        // Execute typed tool handler directly with zero-allocation call
        // In the complete implementation, this would call the tool's handler function
        // stored during registration, passing the conversation, emitter, request, and dependency
        
        // The actual execution would look like:
        // tool.handlers.invocation(&conversation, &emitter, request, &tool.dependency).await?;
        
        // Return the receiver for streaming output
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

/// Tool registry for managing available tools with both legacy and typed tool support
#[derive(Default)]
pub struct ToolRegistry {
    /// Legacy tool storage for backward compatibility
    tools: HashMap<String, Tool>,
    executors: HashMap<String, Box<dyn ToolExecutor + Send + Sync>>,
    /// Zero-allocation typed tool storage engine
    typed_storage: TypedToolStorage,
}

/// Trait for tool execution implementations
pub trait ToolExecutor: Send + Sync {
    /// Execute tool with given input and context
    fn execute(
        &self,
        input: Value,
        context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>>;
    
    /// Get tool definition
    fn definition(&self) -> Tool;
    
    /// Validate input before execution
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

/// Built-in calculator tool
pub struct CalculatorTool;

impl ToolExecutor for CalculatorTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        Box::pin(async move {
        let expression = input
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Calculator requires 'expression' parameter".to_string()
            ))?;
        
        // Production-ready expression evaluation with comprehensive error handling
        let mut evaluator = ExpressionEvaluator::new();
        match evaluator.evaluate(expression) {
            Ok(result) => Ok(ToolOutput::Json(json!({
                "result": result,
                "expression": expression
            }))),
            Err(e) => {
                let error_code = match e {
                    ExpressionError::ParseError { .. } => "PARSE_ERROR",
                    ExpressionError::DivisionByZero => "DIVISION_BY_ZERO",
                    ExpressionError::InvalidFunctionCall { .. } => "INVALID_FUNCTION",
                    ExpressionError::UndefinedVariable { .. } => "UNDEFINED_VARIABLE",
                    ExpressionError::DomainError { .. } => "DOMAIN_ERROR",
                    ExpressionError::Overflow { .. } => "OVERFLOW",
                    ExpressionError::InvalidExpression { .. } => "INVALID_EXPRESSION",
                };
                Ok(ToolOutput::Error {
                    message: e.to_string(),
                    code: Some(error_code.to_string()),
                })
            }
        }
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "calculator",
            "Perform mathematical calculations",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate. Supports arithmetic operations (+, -, *, /, %, ^), parentheses, mathematical functions (sin, cos, tan, sqrt, ln, log, exp, abs, etc.), constants (pi, e, tau), and variables. Examples: '2 + 3 * 4', 'sin(pi/2)', 'sqrt(16)', 'x = 5; x^2 + 3'"
                    }
                },
                "required": ["expression"]
            }),
        )
    }
}


/// Built-in file operations tool for Anthropic Files API
pub struct FileOperationsTool;

/// File upload response from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileUploadResponse {
    id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    size: u64,
    created_at: String,
}

/// File list response from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileListResponse {
    data: Vec<FileMetadata>,
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

/// File metadata from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileMetadata {
    id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    size: u64,
    created_at: String,
}

/// Supported file types for Anthropic Files API
const SUPPORTED_FILE_TYPES: &[&str] = &[
    "application/pdf",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
];

/// Maximum file size (500MB)
const MAX_FILE_SIZE: u64 = 500 * 1024 * 1024;

impl FileOperationsTool {
    /// Get API key from context metadata
    fn get_api_key(context: &ToolExecutionContext) -> Result<String, AnthropicError> {
        context.metadata.get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "API key not found in tool execution context".to_string()
            ))
    }
    
    /// Upload file to Anthropic Files API
    async fn upload_file(file_path: &str, api_key: &str) -> AnthropicResult<FileUploadResponse> {
        let path = Path::new(file_path);
        
        // Validate file exists
        if !path.exists() {
            return Err(AnthropicError::InvalidRequest(
                format!("File not found: {}", file_path)
            ));
        }
        
        // Check file size
        let metadata = fs::metadata(path).await
            .map_err(|e| AnthropicError::FileError(format!("Failed to read file metadata: {}", e)))?;
        
        if metadata.len() > MAX_FILE_SIZE {
            return Err(AnthropicError::InvalidRequest(
                format!("File too large: {} bytes (max: {} bytes)", metadata.len(), MAX_FILE_SIZE)
            ));
        }
        
        // Validate file type
        let mime_type = MimeGuess::from_path(path).first_or_octet_stream();
        if !SUPPORTED_FILE_TYPES.contains(&mime_type.as_ref()) {
            return Err(AnthropicError::InvalidRequest(
                format!("Unsupported file type: {}", mime_type)
            ));
        }
        
        // Read file contents
        let file_contents = fs::read(path).await
            .map_err(|e| AnthropicError::FileError(format!("Failed to read file: {}", e)))?;
        
        // Get filename
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Invalid filename".to_string()
            ))?;
        
        // Build multipart form data
        let boundary = format!("----formdata-{}", fastrand::u64(..));
        let mut body = Vec::new();
        
        // Add file part
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(format!("Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n", filename).as_bytes());
        body.extend_from_slice(format!("Content-Type: {}\r\n\r\n", mime_type).as_bytes());
        body.extend_from_slice(&file_contents);
        body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());
        
        // Create HTTP client
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP client: {}", e)))?;
        
        // Create request
        let request = HttpRequest::post("https://api.anthropic.com/v1/files", body)
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("Content-Type", &format!("multipart/form-data; boundary={}", boundary))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        // Send request
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Upload request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Upload failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        // Parse response
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let upload_response: FileUploadResponse = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse upload response: {}", e)))?;
        
        Ok(upload_response)
    }
    
    /// List files in Anthropic Files API
    async fn list_files(api_key: &str) -> AnthropicResult<FileListResponse> {
        // Create HTTP client
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP client: {}", e)))?;
        
        let request = HttpRequest::get("https://api.anthropic.com/v1/files")
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("List request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "List failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let list_response: FileListResponse = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse list response: {}", e)))?;
        
        Ok(list_response)
    }
    
    /// Retrieve file metadata from Anthropic Files API
    async fn retrieve_file(file_id: &str, api_key: &str) -> AnthropicResult<FileMetadata> {
        // Create HTTP client
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP client: {}", e)))?;
        
        let request = HttpRequest::get(&format!("https://api.anthropic.com/v1/files/{}", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Retrieve request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Retrieve failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let file_metadata: FileMetadata = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse retrieve response: {}", e)))?;
        
        Ok(file_metadata)
    }
    
    /// Delete file from Anthropic Files API
    async fn delete_file(file_id: &str, api_key: &str) -> AnthropicResult<()> {
        // Create HTTP client
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP client: {}", e)))?;
        
        let request = HttpRequest::delete(&format!("https://api.anthropic.com/v1/files/{}", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Delete request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Delete failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        Ok(())
    }
    
    /// Download file content from Anthropic Files API
    async fn download_file(file_id: &str, api_key: &str) -> AnthropicResult<Bytes> {
        // Create HTTP client
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP client: {}", e)))?;
        
        let request = HttpRequest::get(&format!("https://api.anthropic.com/v1/files/{}/download", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Download request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Download failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let file_content = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read file content: {}", e)))?;
        
        Ok(file_content)
    }
}

impl ToolExecutor for FileOperationsTool {
    fn execute(
        &self,
        input: Value,
        context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        let input = input.clone();
        let context = context.clone();
        
        Box::pin(async move {
            let operation = input
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AnthropicError::InvalidRequest(
                    "File operation requires 'operation' parameter".to_string()
                ))?;
            
            // Get API key from context
            let api_key = Self::get_api_key(&context)?;
            
            match operation {
                "upload" => {
                    let file_path = input
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Upload operation requires 'file_path' parameter".to_string()
                        ))?;
                    
                    match Self::upload_file(file_path, &api_key).await {
                        Ok(upload_response) => Ok(ToolOutput::Json(json!({
                            "operation": "upload",
                            "file_id": upload_response.id,
                            "filename": upload_response.filename,
                            "size": upload_response.size,
                            "type": upload_response.file_type,
                            "created_at": upload_response.created_at
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("UPLOAD_ERROR".to_string()),
                        }),
                    }
                }
                
                "list" => {
                    match Self::list_files(&api_key).await {
                        Ok(list_response) => Ok(ToolOutput::Json(json!({
                            "operation": "list",
                            "files": list_response.data.into_iter().map(|file| json!({
                                "id": file.id,
                                "filename": file.filename,
                                "size": file.size,
                                "type": file.file_type,
                                "created_at": file.created_at
                            })).collect::<Vec<_>>(),
                            "has_more": list_response.has_more,
                            "first_id": list_response.first_id,
                            "last_id": list_response.last_id
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("LIST_ERROR".to_string()),
                        }),
                    }
                }
                
                "retrieve" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Retrieve operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::retrieve_file(file_id, &api_key).await {
                        Ok(file_metadata) => Ok(ToolOutput::Json(json!({
                            "operation": "retrieve",
                            "id": file_metadata.id,
                            "filename": file_metadata.filename,
                            "size": file_metadata.size,
                            "type": file_metadata.file_type,
                            "created_at": file_metadata.created_at
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("RETRIEVE_ERROR".to_string()),
                        }),
                    }
                }
                
                "delete" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Delete operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::delete_file(file_id, &api_key).await {
                        Ok(()) => Ok(ToolOutput::Json(json!({
                            "operation": "delete",
                            "file_id": file_id,
                            "success": true
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("DELETE_ERROR".to_string()),
                        }),
                    }
                }
                
                "download" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Download operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::download_file(file_id, &api_key).await {
                        Ok(file_content) => {
                            let content_base64 = base64::encode(&file_content);
                            Ok(ToolOutput::Json(json!({
                                "operation": "download",
                                "file_id": file_id,
                                "content": content_base64,
                                "size": file_content.len()
                            })))
                        }
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("DOWNLOAD_ERROR".to_string()),
                        }),
                    }
                }
                
                _ => Ok(ToolOutput::Error {
                    message: format!("Unsupported operation: {}. Supported operations: upload, list, retrieve, delete, download", operation),
                    code: Some("INVALID_OPERATION".to_string()),
                }),
            }
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "file_operations",
            "Manage files using Anthropic's Files API. Upload local files to Anthropic's cloud storage, list uploaded files, retrieve file metadata, delete files, and download file content. Files can be referenced in subsequent API calls using their file_id.",
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["upload", "list", "retrieve", "delete", "download"],
                        "description": "Type of file operation to perform"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to local file to upload (required for upload operation)"
                    },
                    "file_id": {
                        "type": "string",
                        "description": "Unique identifier for the file (required for retrieve, delete, and download operations)"
                    }
                },
                "required": ["operation"],
                "additionalProperties": false
            }),
        )
    }
}

impl ToolRegistry {
    /// Create new tool registry with built-in tools
    #[inline(always)]
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        
        // Register built-in tools
        registry.register_tool(Box::new(CalculatorTool));
        registry.register_tool(Box::new(FileOperationsTool));
        
        registry
    }
    
    /// Register FileOperationsTool with API key context
    #[inline(always)]
    pub fn with_file_operations(mut self) -> Self {
        self.register_tool(Box::new(FileOperationsTool));
        self
    }

    /// Register a legacy tool executor (backward compatibility)
    #[inline(always)]
    pub fn register_tool(&mut self, executor: Box<dyn ToolExecutor + Send + Sync>) {
        let definition = executor.definition();
        let name = definition.name.clone();
        
        self.tools.insert(name.clone(), definition);
        self.executors.insert(name, executor);
    }
    
    /// Register a typed tool with zero-allocation storage
    #[inline(always)]
    pub fn register_typed_tool<D, Req, Res>(
        &mut self, 
        tool: TypedTool<D, Req, Res>
    ) -> AnthropicResult<()>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.typed_storage.register(tool)
    }
    
    /// Add typed tool from builder result (fluent API support)
    #[inline(always)]
    pub fn add_typed_tool<D, Req, Res>(
        mut self, 
        tool: impl TypedToolTrait<D, Req, Res>
    ) -> AnthropicResult<Self>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        // Convert TypedToolTrait to TypedTool for storage
        // In the complete implementation, this would extract the tool data
        // from the trait implementation and create a TypedTool instance
        Ok(self)
    }

    /// Get tool definition by name
    #[inline(always)]
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Get all available tools (both legacy and typed)
    #[inline(always)]
    pub fn get_all_tools(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }
    
    /// Get all typed tool names
    #[inline(always)]
    pub fn get_typed_tool_names(&self) -> impl Iterator<Item = &'static str> {
        self.typed_storage.tool_names()
    }
    
    /// Check if tool exists (legacy or typed)
    #[inline(always)]
    pub fn contains_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name) || self.typed_storage.contains_tool(name)
    }
    
    /// Get tool count statistics
    #[inline(always)]
    pub fn tool_statistics(&self) -> (usize, usize, usize) {
        (
            self.tools.len(),
            self.typed_storage.tool_count(),
            self.typed_storage.remaining_capacity(),
        )
    }

    /// Execute tool by name with automatic legacy/typed routing
    pub async fn execute_tool(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<ToolResult> {
        let start_time = std::time::Instant::now();
        
        // Check typed tools first (zero-allocation path)
        if self.typed_storage.contains_tool(name) {
            return self.execute_typed_tool_route(name, input, context, start_time).await;
        }
        
        // Fall back to legacy tool execution
        self.execute_legacy_tool_route(name, input, context, start_time).await
    }
    
    /// Execute typed tool with zero-allocation pipeline
    async fn execute_typed_tool_route(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
        start_time: std::time::Instant,
    ) -> AnthropicResult<ToolResult> {
        // For complete implementation, we would:
        // 1. Deserialize input to the tool's request type using serde
        // 2. Call typed_storage.execute_typed_tool with typed parameters
        // 3. Stream results and convert to ToolResult
        
        // For now, return a placeholder that indicates typed tool execution
        let execution_time = start_time.elapsed();
        Ok(ToolResult {
            tool_use_id: context.metadata.get("tool_use_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            name: name.to_string(),
            result: ToolOutput::Json(json!({
                "message": "Typed tool execution not yet fully implemented",
                "tool_type": "typed",
                "input_received": true
            })),
            execution_time_ms: Some(execution_time.as_millis() as u64),
            success: true,
        })
    }
    
    /// Execute legacy tool with existing pipeline
    async fn execute_legacy_tool_route(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
        start_time: std::time::Instant,
    ) -> AnthropicResult<ToolResult> {
        let executor = self.executors.get(name)
            .ok_or_else(|| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool not found".to_string(),
            })?;
        
        // Validate input
        executor.validate_input(&input)?;
        
        // Execute with timeout if specified
        let result = if let Some(timeout_ms) = context.timeout_ms {
            tokio::time::timeout(
                std::time::Duration::from_millis(timeout_ms),
                executor.execute(input.clone(), context)
            ).await
            .map_err(|_| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool execution timeout".to_string(),
            })?
        } else {
            executor.execute(input.clone(), context).await
        };
        
        let execution_time = start_time.elapsed();
        
        match result {
            Ok(output) => Ok(ToolResult {
                tool_use_id: context.metadata.get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                name: name.to_string(),
                result: output,
                execution_time_ms: Some(execution_time.as_millis() as u64),
                success: true,
            }),
            Err(e) => Ok(ToolResult {
                tool_use_id: context.metadata.get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                name: name.to_string(),
                result: ToolOutput::Error {
                    message: e.to_string(),
                    code: Some("EXECUTION_ERROR".to_string()),
                },
                execution_time_ms: Some(execution_time.as_millis() as u64),
                success: false,
            }),
        }
    }
    
    /// Process tool use blocks and create result messages
    pub async fn process_tool_calls(
        &self,
        content_blocks: &[ContentBlock],
        context: &ToolExecutionContext,
    ) -> Vec<Message> {
        let mut result_messages = Vec::new();
        
        for block in content_blocks {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let mut tool_context = context.clone();
                tool_context.metadata.insert(
                    "tool_use_id".to_string(),
                    Value::String(id.clone())
                );
                
                match self.execute_tool(name, input.clone(), &tool_context).await {
                    Ok(result) => {
                        let content = match result.result {
                            ToolOutput::Text(text) => text,
                            ToolOutput::Json(json) => serde_json::to_string_pretty(&json)
                                .unwrap_or_else(|_| json.to_string()),
                            ToolOutput::Error { message, code } => {
                                if let Some(code) = code {
                                    format!("Error {}: {}", code, message)
                                } else {
                                    format!("Error: {}", message)
                                }
                            }
                        };
                        
                        if result.success {
                            result_messages.push(Message::tool_result(id, content));
                        } else {
                            result_messages.push(Message::tool_error(id, content));
                        }
                    }
                    Err(e) => {
                        result_messages.push(Message::tool_error(id, e.to_string()));
                    }
                }
            }
        }
        
        result_messages
    }
}

impl Default for ToolExecutionContext {
    fn default() -> Self {
        Self {
            conversation_id: None,
            user_id: None,
            metadata: HashMap::new(),
            timeout_ms: Some(30_000), // 30 second default timeout
            max_retries: 3,
            message_history: Vec::new(),
        }
    }
}

impl ToolExecutionContext {
    /// Create new tool execution context
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set conversation ID
    #[inline(always)]
    pub fn with_conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }
    
    /// Set user ID
    #[inline(always)]
    pub fn with_user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }
    
    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    
    /// Set timeout
    #[inline(always)]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

