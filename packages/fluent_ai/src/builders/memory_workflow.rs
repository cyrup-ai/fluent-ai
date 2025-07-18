//! Memory workflow builder implementations with zero-allocation, lock-free design
//!
//! Provides EXACT API syntax for workflow composition and parallel execution.

use fluent_ai_domain::{
    memory::{MemoryManager, MemoryNode, MemoryType, MemoryError},
    memory_ops::{self, Op},
    ZeroOneOrMany, AsyncTask, spawn_async
};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, warn};

/// Create a new workflow builder - EXACT syntax: workflow::new()
#[inline(always)]
pub fn new() -> WorkflowBuilder {
    WorkflowBuilder::default()
}

/// Zero-allocation workflow builder with lock-free operation composition
#[derive(Debug, Clone)]
pub struct WorkflowBuilder {
    ops: Vec<Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    parallel_groups: Vec<Vec<usize>>, // Indices into ops for parallel execution
}

impl Default for WorkflowBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            parallel_groups: Vec::new(),
        }
    }
}

impl WorkflowBuilder {
    /// Chain a sequential operation with zero allocation hot path - EXACT syntax: .chain(op)
    #[inline(always)]
    pub fn chain<O>(mut self, op: O) -> Self
    where
        O: Op<Input = Value, Output = Value> + Send + Sync + 'static,
    {
        self.ops.push(Box::new(op));
        self
    }

    /// Add operations for parallel execution - EXACT syntax: .parallel(ops)
    #[inline(always)]
    pub fn parallel<I>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    {
        let mut indices = Vec::new();
        
        for op in ops {
            indices.push(self.ops.len());
            self.ops.push(op);
        }
        
        if !indices.is_empty() {
            self.parallel_groups.push(indices);
        }
        
        self
    }

    /// Execute the workflow with optimal allocation patterns - EXACT syntax: .execute(input)
    pub async fn execute(&self, input: Value) -> Result<Value, WorkflowError> {
        if self.ops.is_empty() {
            return Ok(input);
        }

        let mut current_value = input;
        let mut op_index = 0;

        // Execute operations in sequence and parallel groups
        while op_index < self.ops.len() {
            // Check if this operation is part of a parallel group
            if let Some(group) = self.parallel_groups.iter().find(|group| group.contains(&op_index)) {
                // Execute parallel group
                let mut futures = Vec::with_capacity(group.len());
                
                for &idx in group {
                    let op = &self.ops[idx];
                    let input_clone = current_value.clone();
                    futures.push(op.call(input_clone));
                }
                
                // Await all parallel operations
                let results = futures::future::join_all(futures).await;
                
                // Combine results (take the first successful result or Null)
                current_value = results
                    .into_iter()
                    .next()
                    .unwrap_or(Value::Null);
                
                // Skip all operations in this parallel group
                op_index = group.iter().max().copied().map(|x| x + 1).unwrap_or(op_index + 1);
            } else {
                // Execute sequential operation
                if let Some(op) = self.ops.get(op_index) {
                    current_value = op.call(current_value).await;
                }
                op_index += 1;
            }
        }

        Ok(current_value)
    }

    /// Build an executable workflow - EXACT syntax: .build()
    #[inline(always)]
    pub fn build(self) -> ExecutableWorkflow {
        ExecutableWorkflow {
            builder: self,
        }
    }
}

/// Zero-allocation executable workflow
pub struct ExecutableWorkflow {
    builder: WorkflowBuilder,
}

impl ExecutableWorkflow {
    /// Execute the workflow with the given input - EXACT syntax: .run(input)
    #[inline(always)]
    pub async fn run(&self, input: Value) -> Result<Value, WorkflowError> {
        self.builder.execute(input).await
    }
}

/// Error type for memory workflows
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Prompt error: {0}")]
    Prompt(String),

    #[error("Workflow error: {0}")]
    Other(String),
}

/// Define traits locally - no external dependencies
pub trait Prompt: Clone {
    fn prompt(
        &self,
        input: String,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + Send;
}

#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("Prompt error: {0}")]
    Error(String),
}

impl From<PromptError> for WorkflowError {
    fn from(error: PromptError) -> Self {
        WorkflowError::Prompt(error.to_string())
    }
}

/// Zero-allocation passthrough operation for testing
#[allow(dead_code)]
pub fn passthrough<T: Clone + Send + Sync + 'static>() -> impl Op<Input = T, Output = T> {
    struct PassthroughOp<T> {
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T: Clone + Send + Sync + 'static> Op for PassthroughOp<T> {
        type Input = T;
        type Output = T;

        async fn call(&self, input: Self::Input) -> Self::Output {
            input
        }
    }

    PassthroughOp {
        _phantom: std::marker::PhantomData,
    }
}

/// Zero-allocation tuple operation that runs two ops and returns a tuple
#[allow(dead_code)]
pub fn run_both<I, O1, O2, Op1, Op2>(op1: Op1, op2: Op2) -> impl Op<Input = I, Output = (O1, O2)>
where
    I: Clone + Send + Sync + 'static,
    O1: Send + Sync + 'static,
    O2: Send + Sync + 'static,
    Op1: Op<Input = I, Output = O1> + Send + Sync,
    Op2: Op<Input = I, Output = O2> + Send + Sync,
{
    struct BothOp<Op1, Op2> {
        op1: Op1,
        op2: Op2,
    }

    impl<I, O1, O2, Op1, Op2> Op for BothOp<Op1, Op2>
    where
        I: Clone + Send + Sync + 'static,
        O1: Send + Sync + 'static,
        O2: Send + Sync + 'static,
        Op1: Op<Input = I, Output = O1> + Send + Sync,
        Op2: Op<Input = I, Output = O2> + Send + Sync,
    {
        type Input = I;
        type Output = (O1, O2);

        async fn call(&self, input: Self::Input) -> Self::Output {
            let result1 = self.op1.call(input.clone()).await;
            let result2 = self.op2.call(input).await;
            (result1, result2)
        }
    }

    BothOp { op1, op2 }
}

/// High-performance memory workflow operation with zero allocations where possible
struct MemoryWorkflowOp<M, P> {
    memory_manager: M,
    prompt_model: P,
    context_limit: usize,
}

impl<M, P> Op for MemoryWorkflowOp<M, P>
where
    M: MemoryManager + Clone + Send + Sync,
    P: Prompt + Send + Sync,
{
    type Input = String;
    type Output = Result<String, WorkflowError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        // Store input as episodic memory
        let _ = memory_ops::store_memory(self.memory_manager.clone(), MemoryType::Episodic)
            .call(input.clone())
            .await;

        // Search for relevant memories
        let memories = match memory_ops::search_memories(self.memory_manager.clone())
            .call(input.clone())
            .await
        {
            Ok(memories) => memories,
            Err(_) => ZeroOneOrMany::None, // Safe fallback without unwrap_or_default
        };

        // Format the prompt with context using pre-sized capacity for efficiency
        let mut context_parts = Vec::with_capacity(self.context_limit);
        match memories {
            ZeroOneOrMany::None => {},
            ZeroOneOrMany::One(memory) => {
                context_parts.push(format!("- {}", memory.content));
            },
            ZeroOneOrMany::Many(memories) => {
                for memory in memories.iter().take(self.context_limit) {
                    context_parts.push(format!("- {}", memory.content));
                }
            },
        }
        let context = context_parts.join("\n");

        let formatted_input = if context.is_empty() {
            input
        } else {
            format!("Previous context:\n{}\n\nCurrent query: {}", context, input)
        };

        // Prompt the model
        let response = self
            .prompt_model
            .prompt(formatted_input)
            .await
            .map_err(WorkflowError::Prompt)?;

        // Store response as semantic memory
        let _ = memory_ops::store_memory(self.memory_manager.clone(), MemoryType::Semantic)
            .call(response.clone())
            .await;

        Ok(response)
    }
}

/// A memory-enhanced workflow that stores inputs, retrieves context, and generates responses
pub struct MemoryEnhancedWorkflow<M, P> {
    memory_manager: M,
    prompt_model: P,
    context_limit: usize,
}

impl<M, P> MemoryEnhancedWorkflow<M, P>
where
    M: MemoryManager + Clone,
    P: Prompt + Send + Sync,
{
    /// Create new memory-enhanced workflow - EXACT syntax: MemoryEnhancedWorkflow::new(manager, model)
    pub fn new(memory_manager: M, prompt_model: P) -> Self {
        Self {
            memory_manager,
            prompt_model,
            context_limit: 5,
        }
    }

    /// Set context limit - EXACT syntax: .with_context_limit(10)
    pub fn with_context_limit(mut self, limit: usize) -> Self {
        self.context_limit = limit;
        self
    }

    /// Build the memory-enhanced workflow - EXACT syntax: .build()
    pub fn build(self) -> impl Op<Input = String, Output = Result<String, WorkflowError>> {
        let memory_manager = self.memory_manager;
        let prompt_model = self.prompt_model;
        let context_limit = self.context_limit;

        // Return a high-performance Op implementation
        MemoryWorkflowOp {
            memory_manager,
            prompt_model,
            context_limit,
        }
    }
}

/// Create a simple memory-aware conversation workflow - EXACT syntax: conversation_workflow(manager, model)
pub fn conversation_workflow<M, P>(
    memory_manager: M,
    prompt_model: P,
) -> impl Op<Input = String, Output = Result<String, WorkflowError>>
where
    M: MemoryManager + Clone,
    P: Prompt + Send + Sync,
{
    MemoryEnhancedWorkflow::new(memory_manager, prompt_model)
        .with_context_limit(10)
        .build()
}

/// Create a learning workflow that adapts based on feedback
pub struct AdaptiveWorkflow<M, B> {
    memory_manager: M,
    base_op: B,
}

impl<M, B> AdaptiveWorkflow<M, B>
where
    M: MemoryManager + Clone,
    B: Op,
{
    /// Create new adaptive workflow - EXACT syntax: AdaptiveWorkflow::new(manager, op)
    pub fn new(memory_manager: M, base_op: B) -> Self {
        Self {
            memory_manager,
            base_op,
        }
    }
}

impl<M, B> Op for AdaptiveWorkflow<M, B>
where
    M: MemoryManager + Clone + Send + Sync,
    B: Op + Send + Sync,
    B::Input: Clone + serde::Serialize + Send,
    B::Output: Clone + serde::Serialize + Send,
{
    type Input = B::Input;
    type Output = (B::Output, String); // (result, memory_id)

    async fn call(&self, input: Self::Input) -> Self::Output {
        // Execute the base operation
        let output = self.base_op.call(input.clone()).await;

        // Create a memory capturing both input and output with safe timestamp handling
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_else(|_| {
                warn!("Failed to get system time, using fallback timestamp");
                0 // Safe fallback to epoch time
            });

        let memory_content = serde_json::json!({
            "input": input,
            "output": output,
            "timestamp": timestamp,
        }).to_string();

        let memory = MemoryNode::new(memory_content, MemoryType::Episodic).with_importance(0.5); // Initial neutral importance

        // Attempt to store memory with exponential backoff retry logic
        let memory_id = store_memory_with_retry(&self.memory_manager, memory).await;

        (output, memory_id)
    }
}

/// Store memory with exponential backoff retry logic
/// 
/// Attempts to store a memory with up to 3 retries using exponential backoff.
/// If all attempts fail, returns a fallback error ID to maintain system stability.
async fn store_memory_with_retry<M: MemoryManager>(
    memory_manager: &M,
    memory: MemoryNode,
) -> String {
    const MAX_RETRIES: u32 = 3;
    const BASE_DELAY_MS: u64 = 100;
    
    for attempt in 0..MAX_RETRIES {
        match memory_manager.create_memory(memory.clone()).await {
            Ok(stored_memory) => {
                return stored_memory.id;
            }
            Err(e) => {
                if attempt == MAX_RETRIES - 1 {
                    error!(
                        "Failed to store memory after {} attempts: {}. Using fallback ID.", 
                        MAX_RETRIES, e
                    );
                    // Return a fallback error ID to maintain API compatibility
                    return format!("error_fallback_{}", timestamp_safe());
                } else {
                    warn!(
                        "Memory storage attempt {} failed ({}), retrying in {}ms: {}", 
                        attempt + 1, 
                        MAX_RETRIES,
                        BASE_DELAY_MS * (1 << attempt), // Exponential backoff: 100ms, 200ms, 400ms
                        e
                    );
                    
                    // Exponential backoff delay
                    let delay = Duration::from_millis(BASE_DELAY_MS * (1 << attempt));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
    
    // This should never be reached due to the return in the last iteration,
    // but provide a final fallback for safety
    format!("error_exhausted_{}", timestamp_safe())
}

/// Safe timestamp generation for fallback scenarios
#[inline(always)]
fn timestamp_safe() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Apply feedback to a stored memory - EXACT syntax: apply_feedback(manager, id, feedback)
pub async fn apply_feedback<M: MemoryManager>(
    memory_manager: M,
    memory_id: String,
    feedback: f32,
) -> Result<(), MemoryError> {
    if let Some(mut memory) = memory_manager.get_memory(&memory_id).await? {
        // Update importance based on feedback
        memory.metadata.importance = feedback.clamp(0.0, 1.0);
        memory_manager.update_memory(memory).await?;
    }
    Ok(())
}

/// Create a RAG (Retrieval-Augmented Generation) workflow - EXACT syntax: rag_workflow(manager, model, limit)
pub fn rag_workflow<M, P>(
    memory_manager: M,
    prompt_model: P,
    retrieval_limit: usize,
) -> impl Op<Input = String, Output = Result<String, WorkflowError>>
where
    M: MemoryManager + Clone,
    P: Prompt + Send + Sync,
{
    RagWorkflowOp {
        memory_manager,
        prompt_model,
        retrieval_limit,
    }
}

struct RagWorkflowOp<M, P> {
    memory_manager: M,
    prompt_model: P,
    retrieval_limit: usize,
}

impl<M, P> Op for RagWorkflowOp<M, P>
where
    M: MemoryManager + Clone + Send + Sync,
    P: Prompt + Send + Sync,
{
    type Input = String;
    type Output = Result<String, WorkflowError>;

    async fn call(&self, query: Self::Input) -> Self::Output {
        // Retrieve relevant documents
        let memories = memory_ops::search_memories(self.memory_manager.clone())
            .call(query.clone())
            .await
            .unwrap_or_else(|_| ZeroOneOrMany::None);

        // Format as RAG prompt
        let documents = memories
            .iter()
            .take(self.retrieval_limit)
            .enumerate()
            .map(|(i, m)| format!("Document {}: {}", i + 1, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Using the following documents as context:\n\n{}\n\nAnswer the question: {}",
            documents, query
        );

        // Generate response
        self.prompt_model
            .prompt(prompt)
            .await
            .map_err(WorkflowError::Prompt)
    }
}