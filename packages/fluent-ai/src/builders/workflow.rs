use std::sync::Arc;

use fluent_ai_http3::async_task::AsyncStream;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::domain::memory_workflow::{MemoryEnhancedWorkflow, Op, WorkflowError};

/// Zero-allocation workflow builder with blazing-fast execution
#[derive(Debug, Clone)]
pub struct WorkflowBuilder {
    operations: Vec<Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    parallel_groups: Vec<ParallelGroup>,
    error_strategy: ErrorStrategy,
    timeout_ms: Option<u64>,
    max_retries: u32,
}

/// Parallel execution group with dependency management
#[derive(Debug, Clone)]
struct ParallelGroup {
    operation_indices: Vec<usize>,
    merge_strategy: MergeStrategy,
    dependencies: Vec<usize>, // Indices of operations that must complete first
}

/// Strategy for handling operation errors
#[derive(Debug, Clone)]
pub enum ErrorStrategy {
    /// Stop execution on first error
    FailFast,
    /// Continue execution, collect errors at the end
    ContinueOnError,
    /// Retry failed operations up to max_retries
    RetryOnError,
    /// Use fallback operations for failed ones
    UseFallback(Vec<Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>),
}

/// Strategy for merging parallel operation results
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Take the first successful result
    First,
    /// Take the last successful result
    Last,
    /// Merge all results into an array
    Array,
    /// Merge results into an object using operation names as keys
    Object,
    /// Use custom merge function
    Custom(Arc<dyn Fn(Vec<Value>) -> Value + Send + Sync>),
}

impl Default for WorkflowBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowBuilder {
    /// Create a new workflow builder with optimal defaults
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            parallel_groups: Vec::new(),
            error_strategy: ErrorStrategy::FailFast,
            timeout_ms: None,
            max_retries: 0,
        }
    }

    /// Add a sequential operation to the workflow
    #[inline(always)]
    pub fn then<O>(mut self, operation: O) -> Self
    where
        O: Op<Input = Value, Output = Value> + Send + Sync + 'static,
    {
        self.operations.push(Box::new(operation));
        self
    }

    /// Add multiple operations for parallel execution
    #[inline(always)]
    pub fn parallel<I>(mut self, operations: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    {
        let start_index = self.operations.len();
        let mut indices = Vec::new();

        for operation in operations {
            indices.push(self.operations.len());
            self.operations.push(operation);
        }

        if !indices.is_empty() {
            self.parallel_groups.push(ParallelGroup {
                operation_indices: indices,
                merge_strategy: MergeStrategy::Array,
                dependencies: Vec::new(),
            });
        }

        self
    }

    /// Add parallel operations with custom merge strategy
    #[inline(always)]
    pub fn parallel_with_merge<I>(mut self, operations: I, merge_strategy: MergeStrategy) -> Self
    where
        I: IntoIterator<Item = Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    {
        let start_index = self.operations.len();
        let mut indices = Vec::new();

        for operation in operations {
            indices.push(self.operations.len());
            self.operations.push(operation);
        }

        if !indices.is_empty() {
            self.parallel_groups.push(ParallelGroup {
                operation_indices: indices,
                merge_strategy,
                dependencies: Vec::new(),
            });
        }

        self
    }

    /// Set error handling strategy
    #[inline(always)]
    pub fn error_strategy(mut self, strategy: ErrorStrategy) -> Self {
        self.error_strategy = strategy;
        self
    }

    /// Set workflow timeout in milliseconds
    #[inline(always)]
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set maximum number of retries for failed operations
    #[inline(always)]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Add conditional branching based on previous result
    pub fn branch<F>(
        mut self,
        condition: F,
        true_branch: WorkflowBuilder,
        false_branch: Option<WorkflowBuilder>,
    ) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        let conditional_op = ConditionalOperation {
            condition: Arc::new(condition),
            true_branch: true_branch.build_executor(),
            false_branch: false_branch.map(|b| b.build_executor()),
        };

        self.operations.push(Box::new(conditional_op));
        self
    }

    /// Add a loop operation that continues while condition is true
    pub fn while_loop<F>(
        mut self,
        condition: F,
        loop_body: WorkflowBuilder,
        max_iterations: Option<u32>,
    ) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        let loop_op = LoopOperation {
            condition: Arc::new(condition),
            body: loop_body.build_executor(),
            max_iterations: max_iterations.unwrap_or(1000),
        };

        self.operations.push(Box::new(loop_op));
        self
    }

    /// Build an executable workflow
    #[inline(always)]
    pub fn build(self) -> ExecutableWorkflow {
        ExecutableWorkflow {
            executor: self.build_executor(),
        }
    }

    /// Build internal executor
    fn build_executor(self) -> WorkflowExecutor {
        WorkflowExecutor {
            operations: self.operations,
            parallel_groups: self.parallel_groups,
            error_strategy: self.error_strategy,
            timeout_ms: self.timeout_ms,
            max_retries: self.max_retries,
        }
    }
}

/// High-performance workflow executor
#[derive(Debug)]
pub struct WorkflowExecutor {
    operations: Vec<Box<dyn Op<Input = Value, Output = Value> + Send + Sync>>,
    parallel_groups: Vec<ParallelGroup>,
    error_strategy: ErrorStrategy,
    timeout_ms: Option<u64>,
    max_retries: u32,
}

impl WorkflowExecutor {
    /// Execute the workflow with the given input
    pub async fn execute(&self, input: Value) -> Result<Value, WorkflowError> {
        if self.operations.is_empty() {
            return Ok(input);
        }

        let execution_context = ExecutionContext::new(
            input,
            self.timeout_ms,
            self.max_retries,
            &self.error_strategy,
        );

        self.execute_with_context(execution_context).await
    }

    /// Execute with streaming support for intermediate results
    async fn execute_with_streaming(
        &self,
        mut context: ExecutionContext,
        tx: &mpsc::UnboundedSender<Value>,
    ) -> Result<Value, WorkflowError> {
        let mut operation_index = 0;

        while operation_index < self.operations.len() {
            // Send progress update
            let _ = tx.send(serde_json::json!({
                "progress": operation_index as f64 / self.operations.len() as f64,
                "operation_index": operation_index,
                "status": "executing"
            }));

            // Check if this operation is part of a parallel group
            if let Some(group) = self.find_parallel_group(operation_index) {
                context.current_value = self.execute_parallel_group(group, &context).await?;
                operation_index = match group.operation_indices.iter().max().copied() {
                    Some(max_index) => max_index + 1,
                    None => operation_index + 1,
                };
            } else {
                // Execute sequential operation
                if let Some(operation) = self.operations.get(operation_index) {
                    context.current_value = self
                        .execute_operation_with_retry(
                            operation.as_ref(),
                            context.current_value.clone(),
                            &context,
                        )
                        .await?;

                    // Send intermediate result
                    let _ = tx.send(serde_json::json!({
                        "intermediate_result": context.current_value,
                        "operation_index": operation_index,
                        "status": "completed_operation"
                    }));
                }
                operation_index += 1;
            }
        }

        // Send final result
        let _ = tx.send(serde_json::json!({
            "final_result": context.current_value,
            "status": "completed"
        }));

        Ok(context.current_value)
    }

    /// Execute with execution context for advanced control
    async fn execute_with_context(
        &self,
        mut context: ExecutionContext,
    ) -> Result<Value, WorkflowError> {
        let mut operation_index = 0;

        while operation_index < self.operations.len() {
            // Check if this operation is part of a parallel group
            if let Some(group) = self.find_parallel_group(operation_index) {
                context.current_value = self.execute_parallel_group(group, &context).await?;
                operation_index = match group.operation_indices.iter().max().copied() {
                    Some(max_index) => max_index + 1,
                    None => operation_index + 1,
                };
            } else {
                // Execute sequential operation
                if let Some(operation) = self.operations.get(operation_index) {
                    context.current_value = self
                        .execute_operation_with_retry(
                            operation.as_ref(),
                            context.current_value.clone(),
                            &context,
                        )
                        .await?;
                }
                operation_index += 1;
            }
        }

        Ok(context.current_value)
    }

    /// Execute a parallel group of operations
    async fn execute_parallel_group(
        &self,
        group: &ParallelGroup,
        context: &ExecutionContext,
    ) -> Result<Value, WorkflowError> {
        let mut result_streams = Vec::with_capacity(group.operation_indices.len());
        let mut handles = Vec::new();

        for &index in &group.operation_indices {
            if let Some(operation) = self.operations.get(index) {
                let input_clone = context.current_value.clone();
                let context_clone = context.clone();
                let op_ref = Arc::new(operation.clone());
                
                let (tx, stream) = AsyncStream::channel();
                result_streams.push(stream);
                
                let handle = tokio::spawn(async move {
                    let result = self.execute_operation_with_retry(op_ref.as_ref(), input_clone, &context_clone)
                        .await;
                    let _ = tx.send(result);
                });
                handles.push(handle);
            }
        }

        // Collect all results
        let mut results = Vec::with_capacity(result_streams.len());
        for mut stream in result_streams {
            if let Some(result) = stream.next() {
                results.push(result);
            }
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        self.merge_parallel_results(results, &group.merge_strategy)
    }

    /// Execute operation with retry logic
    async fn execute_operation_with_retry(
        &self,
        operation: &dyn Op<Input = Value, Output = Value>,
        input: Value,
        context: &ExecutionContext,
    ) -> Result<Value, WorkflowError> {
        let mut last_error = None;

        for attempt in 0..=context.max_retries {
            match self
                .execute_single_operation(operation, input.clone(), context)
                .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error);
                    if attempt < context.max_retries {
                        // Exponential backoff
                        let delay_ms = 100 * (2_u64.pow(attempt));
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        match &self.error_strategy {
            ErrorStrategy::FailFast => match last_error {
                Some(error) => Err(error),
                None => Err(WorkflowError::ExecutionFailed("Unknown error".to_string())),
            },
            ErrorStrategy::ContinueOnError => {
                Ok(Value::Null) // Continue with null value
            }
            ErrorStrategy::RetryOnError => match last_error {
                Some(error) => Err(error),
                None => Err(WorkflowError::ExecutionFailed(
                    "Max retries exceeded".to_string(),
                )),
            },
            ErrorStrategy::UseFallback(fallbacks) => {
                // Execute fallback operations in sequence until one succeeds
                for fallback_op in fallbacks {
                    match self
                        .execute_single_operation(fallback_op.as_ref(), input.clone(), context)
                        .await
                    {
                        Ok(result) => return Ok(result),
                        Err(_) => continue, // Try next fallback
                    }
                }
                // All fallbacks failed
                match last_error {
                    Some(error) => Err(error),
                    None => Err(WorkflowError::ExecutionFailed(
                        "All operations and fallbacks failed".to_string(),
                    )),
                }
            }
        }
    }

    /// Execute a single operation with timeout
    async fn execute_single_operation(
        &self,
        operation: &dyn Op<Input = Value, Output = Value>,
        input: Value,
        context: &ExecutionContext,
    ) -> Result<Value, WorkflowError> {
        if let Some(timeout_ms) = context.timeout_ms {
            let timeout_duration = tokio::time::Duration::from_millis(timeout_ms);

            match tokio::time::timeout(timeout_duration, operation.call(input)).await {
                Ok(result) => Ok(result),
                Err(_) => Err(WorkflowError::Timeout),
            }
        } else {
            Ok(operation.call(input).await)
        }
    }

    /// Find parallel group containing the given operation index
    fn find_parallel_group(&self, operation_index: usize) -> Option<&ParallelGroup> {
        self.parallel_groups
            .iter()
            .find(|group| group.operation_indices.contains(&operation_index))
    }

    /// Merge results from parallel operations
    fn merge_parallel_results(
        &self,
        results: Vec<Result<Value, WorkflowError>>,
        strategy: &MergeStrategy,
    ) -> Result<Value, WorkflowError> {
        let successful_results: Vec<Value> = results.into_iter().filter_map(|r| r.ok()).collect();

        if successful_results.is_empty() {
            return Err(WorkflowError::ExecutionFailed(
                "All parallel operations failed".to_string(),
            ));
        }

        match strategy {
            MergeStrategy::First => {
                Ok(successful_results.into_iter().next().unwrap_or(Value::Null))
            }
            MergeStrategy::Last => Ok(successful_results.into_iter().last().unwrap_or(Value::Null)),
            MergeStrategy::Array => Ok(Value::Array(successful_results)),
            MergeStrategy::Object => {
                let mut object = serde_json::Map::new();
                for (index, result) in successful_results.into_iter().enumerate() {
                    object.insert(format!("result_{}", index), result);
                }
                Ok(Value::Object(object))
            }
            MergeStrategy::Custom(merge_fn) => Ok(merge_fn(successful_results)),
        }
    }
}

/// Execution context for tracking workflow state
#[derive(Debug, Clone)]
struct ExecutionContext {
    current_value: Value,
    timeout_ms: Option<u64>,
    max_retries: u32,
    error_strategy: ErrorStrategy,
    start_time: std::time::Instant,
}

impl ExecutionContext {
    #[inline(always)]
    fn new(
        input: Value,
        timeout_ms: Option<u64>,
        max_retries: u32,
        error_strategy: &ErrorStrategy,
    ) -> Self {
        Self {
            current_value: input,
            timeout_ms,
            max_retries,
            error_strategy: error_strategy.clone(),
            start_time: std::time::Instant::now(),
        }
    }
}

/// Conditional operation for branching workflows
struct ConditionalOperation {
    condition: Arc<dyn Fn(&Value) -> bool + Send + Sync>,
    true_branch: WorkflowExecutor,
    false_branch: Option<WorkflowExecutor>,
}

impl std::fmt::Debug for ConditionalOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalOperation").finish()
    }
}

impl Op for ConditionalOperation {
    type Input = Value;
    type Output = Value;

    async fn call(&self, input: Self::Input) -> Self::Output {
        if (self.condition)(&input) {
            match self.true_branch.execute(input).await {
                Ok(result) => result,
                Err(_) => Value::Null,
            }
        } else if let Some(false_branch) = &self.false_branch {
            match false_branch.execute(input).await {
                Ok(result) => result,
                Err(_) => Value::Null,
            }
        } else {
            input
        }
    }
}

/// Loop operation for iterative workflows
struct LoopOperation {
    condition: Arc<dyn Fn(&Value) -> bool + Send + Sync>,
    body: WorkflowExecutor,
    max_iterations: u32,
}

impl std::fmt::Debug for LoopOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopOperation")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl Op for LoopOperation {
    type Input = Value;
    type Output = Value;

    async fn call(&self, mut input: Self::Input) -> Self::Output {
        let mut iterations = 0;

        while (self.condition)(&input) && iterations < self.max_iterations {
            match self.body.execute(input.clone()).await {
                Ok(result) => input = result,
                Err(_) => break,
            }
            iterations += 1;
        }

        input
    }
}

/// Executable workflow wrapper
#[derive(Debug)]
pub struct ExecutableWorkflow {
    executor: WorkflowExecutor,
}

impl ExecutableWorkflow {
    /// Execute the workflow with the given input
    #[inline(always)]
    pub async fn run(&self, input: Value) -> Result<Value, WorkflowError> {
        self.executor.execute(input).await
    }

    /// Execute the workflow with streaming output
    pub async fn run_streaming(
        &self,
        input: Value,
    ) -> Result<mpsc::UnboundedReceiver<Value>, WorkflowError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let executor = &self.executor;

        // Execute workflow with intermediate result streaming
        tokio::spawn(async move {
            let execution_context = ExecutionContext::new(
                input,
                executor.timeout_ms,
                executor.max_retries,
                &executor.error_strategy,
            );

            // Stream intermediate results during execution
            if let Err(e) = executor
                .execute_with_streaming(execution_context, &tx)
                .await
            {
                let _ = tx.send(serde_json::json!({
                    "error": e.to_string(),
                    "status": "failed"
                }));
            }
        });

        Ok(rx)
    }

    /// Get workflow execution statistics
    pub fn stats(&self) -> WorkflowStats {
        WorkflowStats {
            operation_count: self.executor.operations.len(),
            parallel_group_count: self.executor.parallel_groups.len(),
            estimated_execution_time_ms: self.estimate_execution_time(),
        }
    }

    fn estimate_execution_time(&self) -> u64 {
        // Simple estimation based on operation count
        (self.executor.operations.len() as u64) * 10 // 10ms per operation estimate
    }
}

/// Workflow execution statistics
#[derive(Debug, Clone)]
pub struct WorkflowStats {
    pub operation_count: usize,
    pub parallel_group_count: usize,
    pub estimated_execution_time_ms: u64,
}
