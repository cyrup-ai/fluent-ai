//! Function calling and tool execution system with zero-allocation streaming
//!
//! This module provides a type-safe, zero-allocation function calling system for
//! Anthropic tool execution with lock-free streaming and compile-time safety.

use std::{any::TypeId, collections::HashMap, marker::PhantomData};
use arrayvec::ArrayVec;
use std::collections::HashMap;

use fluent_ai_async::AsyncStream;
use fluent_ai_async::channel;
use fluent_ai_domain::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::core::{
    AnthropicError, AnthropicResult, ChainControl, Emitter, ErrorHandler, InvocationHandler,
    Message, ResultHandler, SchemaType};
#[cfg(feature = "cylo")]
use crate::execution::{CyloExecutor, CyloInstance, ExecutionRequest};

/// Tool execution context with metadata and message history
#[derive(Debug, Clone)]
pub struct ToolExecutionContext {
    pub conversation_id: Option<String>,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, Value>,
    pub timeout_ms: Option<u64>,
    pub max_retries: u32,
    pub message_history: Vec<Message>}

/// Conversation context for tool execution
pub struct Conversation<'a> {
    pub messages: &'a [Message],
    pub context: &'a ToolExecutionContext,
    pub last_message: &'a Message}

/// Tool execution result with timing and status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    pub result: ToolOutput,
    pub execution_time_ms: Option<u64>,
    pub success: bool}

/// Tool output data with zero-allocation streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
    Error {
        message: String,
        code: Option<String>}}

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
    type DependencyBuilder: WithDependency;
    fn with<D: Send + Sync + 'static>(self, dependency: D) -> Self::DependencyBuilder;
}

/// Trait for tools with dependency
pub trait WithDependency {
    type RequestSchemaBuilder: WithRequestSchema;
    fn request_schema<Req: Send + Sync + 'static>(
        self,
        schema_type: SchemaType,
    ) -> Self::RequestSchemaBuilder;
}

/// Trait for tools with request schema
pub trait WithRequestSchema {
    type ResultSchemaBuilder: WithResultSchema;
    fn result_schema<Res: Send + Sync + 'static>(
        self,
        schema_type: SchemaType,
    ) -> Self::ResultSchemaBuilder;
}

/// Trait for tools with result schema
pub trait WithResultSchema {
    type WithInvocationBuilder: WithInvocation;
    fn on_invocation<F>(self, handler: F) -> Self::WithInvocationBuilder;
}

/// Trait for tools with invocation handler
pub trait WithInvocation {
    type WithErrorBuilder: WithError;
    fn on_error<F>(self, handler: F) -> Self::WithErrorBuilder;
}

/// Trait for tools with error handler
pub trait WithError {
    type WithResultBuilder: WithResult;
    fn on_result<F>(self, handler: F) -> Self::WithResultBuilder;
}

/// Trait for tools with result handler
pub trait WithResult {
    type FinalBuilder: FinalTool;
    fn build(self) -> Self::FinalBuilder;
}

/// Trait for fully built tools
pub trait FinalTool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(
        &self,
        conversation: &Conversation,
        emitter: &Emitter,
        request: Value,
    ) -> AsyncStream<AnthropicResult<()>>;
}

/// Builder for creating tools with a fluent interface
pub struct ToolBuilder<State, D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    request_schema_type: SchemaType,
    result_schema_type: SchemaType,
    handlers: Handlers<D, Req, Res>,
    _state: PhantomData<State>}

/// Handlers for tool execution
struct Handlers<D, Req, Res> {
    invocation: InvocationHandler<D, Req, Res>,
    error: Option<ErrorHandler<D>>,
    result: Option<ResultHandler<D, Res>>}

impl<D, Req, Res> Clone for Handlers<D, Req, Res>
where
    D: Clone,
    Req: Clone,
    Res: Clone,
{
    fn clone(&self) -> Self {
        Self {
            invocation: self.invocation.clone(),
            error: self.error.clone(),
            result: self.result.clone()}
    }
}

impl<D, Req, Res> ToolBuilder<(), D, Req, Res> {
    /// Create a new tool builder with a name
    pub fn named(name: &'static str) -> ToolBuilder<Named, (), (), ()> {
        ToolBuilder {
            name,
            description: "",
            dependency: (),
            request_schema_type: SchemaType::Serde,
            result_schema_type: SchemaType::Serde,
            handlers: Handlers {
                invocation: Box::new(|_, _, _, _| panic!("Invocation handler not set")),
                error: None,
                result: None},
            _state: PhantomData}
    }
}

/// State marker for named tools
pub struct Named;

impl NamedTool for ToolBuilder<Named, (), (), ()> {
    type DescribedBuilder = ToolBuilder<Described, (), (), ()>;

    #[inline(always)]
    fn description(self, desc: &'static str) -> Self::DescribedBuilder {
        ToolBuilder {
            name: self.name,
            description: desc,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            handlers: self.handlers,
            _state: PhantomData}
    }
}

/// State marker for described tools
pub struct Described;

impl<D, Req, Res> DescribedTool for ToolBuilder<Described, D, Req, Res> {
    type DependencyBuilder = ToolBuilder<WithDependencyState, D, Req, Res>;

    #[inline(always)]
    fn with<NewD: Send + Sync + 'static>(
        self,
        dependency: NewD,
    ) -> ToolBuilder<WithDependencyState, NewD, Req, Res> {
        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            handlers: Handlers {
                invocation: Box::new(|_, _, _, _| panic!("Invocation handler not set")),
                error: None,
                result: None},
            _state: PhantomData}
    }
}

/// State marker for tools with dependency
pub struct WithDependencyState;

impl<D, Req, Res> WithDependency for ToolBuilder<WithDependencyState, D, Req, Res> {
    type RequestSchemaBuilder = ToolBuilder<WithRequestSchemaState, D, Req, Res>;

    #[inline(always)]
    fn request_schema<NewReq: Send + Sync + 'static>(
        self,
        schema_type: SchemaType,
    ) -> ToolBuilder<WithRequestSchemaState, D, NewReq, Res> {
        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: schema_type,
            result_schema_type: self.result_schema_type,
            handlers: Handlers {
                invocation: Box::new(|_, _, _, _| panic!("Invocation handler not set")),
                error: None,
                result: None},
            _state: PhantomData}
    }
}

/// State marker for tools with request schema
pub struct WithRequestSchemaState;

impl<D, Req, Res> WithRequestSchema for ToolBuilder<WithRequestSchemaState, D, Req, Res> {
    type ResultSchemaBuilder = ToolBuilder<WithResultSchemaState, D, Req, Res>;

    #[inline(always)]
    fn result_schema<NewRes: Send + Sync + 'static>(
        self,
        schema_type: SchemaType,
    ) -> ToolBuilder<WithResultSchemaState, D, Req, NewRes> {
        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: schema_type,
            handlers: Handlers {
                invocation: Box::new(|_, _, _, _| panic!("Invocation handler not set")),
                error: None,
                result: None},
            _state: PhantomData}
    }
}

/// State marker for tools with result schema
pub struct WithResultSchemaState;

impl<D, Req, Res> WithResultSchema for ToolBuilder<WithResultSchemaState, D, Req, Res> {
    type WithInvocationBuilder = ToolBuilder<WithInvocationState, D, Req, Res>;

    #[inline(always)]
    fn on_invocation<F>(self, handler: F) -> Self::WithInvocationBuilder
    where
        F: Fn(&Conversation, &Emitter, Req, &D) -> AsyncStream<()>
            + Send
            + Sync
            + 'static,
    {
        let boxed_handler: InvocationHandler<D, Req, Res> =
            Box::new(move |conv, emitter, req, dep| {
                let (tx, stream) = channel();
                let handler_stream = handler(conv, emitter, req, dep);
                tokio::spawn(async move {
                    use tokio_stream::StreamExt;
                    let mut handler_stream = handler_stream;
                    while let Some(res) = handler_stream.next().await {
                        let _ = tx.send(res);
                    }
                });
                stream
            });

        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            handlers: Handlers {
                invocation: boxed_handler,
                error: None,
                result: None},
            _state: PhantomData}
    }
}

/// State marker for tools with invocation handler
pub struct WithInvocationState;

impl<D, Req, Res> WithInvocation for ToolBuilder<WithInvocationState, D, Req, Res> {
    type WithErrorBuilder = ToolBuilder<WithErrorState, D, Req, Res>;

    #[inline(always)]
    fn on_error<F>(self, handler: F) -> Self::WithErrorBuilder
    where
        F: Fn(&Conversation, &Emitter, &AnthropicError, &D) -> ChainControl + Send + Sync + 'static,
    {
        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            handlers: Handlers {
                invocation: self.handlers.invocation,
                error: Some(Box::new(handler)),
                result: self.handlers.result},
            _state: PhantomData}
    }
}

/// State marker for tools with error handler
pub struct WithErrorState;

impl<D, Req, Res> WithError for ToolBuilder<WithErrorState, D, Req, Res> {
    type WithResultBuilder = ToolBuilder<WithResultState, D, Req, Res>;

    #[inline(always)]
    fn on_result<F>(self, handler: F) -> Self::WithResultBuilder
    where
        F: Fn(&Conversation, &Emitter, Res, &D) -> ChainControl + Send + Sync + 'static,
    {
        ToolBuilder {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            request_schema_type: self.request_schema_type,
            result_schema_type: self.result_schema_type,
            handlers: Handlers {
                invocation: self.handlers.invocation,
                error: self.handlers.error,
                result: Some(Box::new(handler))},
            _state: PhantomData}
    }
}

/// State marker for tools with result handler
pub struct WithResultState;

impl<D, Req, Res> WithResult for ToolBuilder<WithResultState, D, Req, Res> {
    type FinalBuilder = BuiltTool<D, Req, Res>;

    #[inline(always)]
    fn build(self) -> Self::FinalBuilder {
        BuiltTool {
            name: self.name,
            description: self.description,
            dependency: self.dependency,
            handlers: self.handlers,
            _req: PhantomData,
            _res: PhantomData}
    }
}

/// Final, built tool
pub struct BuiltTool<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    handlers: Handlers<D, Req, Res>,
    _req: PhantomData<Req>,
    _res: PhantomData<Res>}

impl<D, Req, Res> FinalTool for BuiltTool<D, Req, Res>
where
    D: Send + Sync + 'static,
    Req: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Res: Serialize + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn execute(
        &self,
        conversation: &Conversation,
        emitter: &Emitter,
        request: Value,
    ) -> AsyncStream<()> {
        let (tx, stream) = channel();
        let invocation_handler = self.handlers.invocation.clone();
        let dependency = self.dependency.clone();
        tokio::spawn(async move {
            let result = async move {
                let handler_stream =
                    invocation_handler(conversation, emitter, request, &dependency);
                let mut results = handler_stream.collect::<Vec<_>>().await;
                if let Some(result) = results.pop() {
                    result
                } else {
                    Ok(())
                }
            }
            .await;
            let _ = tx.send(result);
        });
        stream
    }
}

/// Storage for tools with zero-allocation retrieval
pub struct ToolStorage<const N: usize> {
    tools: ArrayVec<Box<dyn FinalTool + Send + Sync>, N>,
    by_name: HashMap<&'static str, usize>}

impl<const N: usize> ToolStorage<N> {
    /// Create new tool storage
    pub fn new() -> Self {
        Self {
            tools: ArrayVec::new(),
            by_name: HashMap::new()}
    }

    /// Add a tool to storage
    pub fn add<T: FinalTool + Send + Sync + 'static>(&mut self, tool: T) {
        let name = tool.name();
        let index = self.tools.len();
        self.tools.push(Box::new(tool));
        self.by_name.insert(name, index);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn FinalTool> {
        self.by_name.get(name).map(|&i| &*self.tools[i])
    }
}

/// Zero-allocation tool executor
pub struct ToolExecutor<const N: usize> {
    tools: ToolStorage<N>,
    #[cfg(feature = "cylo")]
    cylo: Option<CyloInstance>}

impl<const N: usize> ToolExecutor<N> {
    /// Create a new tool executor
    pub fn new() -> Self {
        Self {
            tools: ToolStorage::new(),
            cylo: None}
    }

    /// Add a tool to the executor
    pub fn with(mut self, tool: impl FinalTool + Send + Sync + 'static) -> Self {
        self.tools.add(tool);
        self
    }

    /// Set up Cylo for remote execution
    #[cfg(feature = "cylo")]
    pub fn with_cylo(mut self, instance: CyloInstance) -> Self {
        self.cylo = Some(instance);
        self
    }

    /// Execute a tool by name
    pub fn execute_tool(
        &self,
        name: String,
        input: Value,
    ) -> AsyncStream<ToolOutput> {
        let (tx, stream) = channel();
        if let Some(tool) = self.tools.get(&name) {
            let conversation = Conversation {
                messages: &[],
                context: &ToolExecutionContext::default(),
                last_message: &Message::default()};
            let emitter = Emitter::new(tx.clone());
            let mut stream = tool.execute(&conversation, &emitter, input);
            tokio::spawn(async move {
                while let Some(result) = stream.next().await {
                    // Handle result
                }
            });
        } else {
            #[cfg(feature = "cylo")]
            if let Some(cylo) = &self.cylo {
                let request = ExecutionRequest {
                    tool_name: name,
                    input,
                    context: None};
                let mut stream = cylo.execute(request);
                tokio::spawn(async move {
                    while let Some(result) = stream.next().await {
                        let result = if let Some(result) = results.pop() {
                            result
                        } else {
                            Err(AnthropicError::ToolExecutionError {
                                tool_name: name,
                                error: "No result from executor".to_string()})
                        };
                        let _ = tx.send(result);
                    }
                });
            } else {
                let _ = tx.send(Err(AnthropicError::ToolExecutionError {
                    tool_name: name,
                    error: "Cylo executor not available".to_string()}));
            }
        }

        stream
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
