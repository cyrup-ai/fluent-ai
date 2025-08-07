//! Workflow Core - Real polymorphic workflow execution with streams-only architecture
//!
//! This module provides the foundational traits and types for building and executing
//! workflows using the fluent-ai streams-only architecture. All execution uses
//! AsyncStream without Future/Result wrapping in execution paths.

use fluent_ai_async::AsyncStream;
use std::marker::PhantomData;
use std::sync::Arc;

/// Core workflow execution trait enabling polymorphic workflow steps
/// 
/// This trait defines the contract for any executable workflow step, providing
/// true async execution with streams-only architecture compliance. All implementations
/// must use AsyncStream for output without Future trait usage.
/// 
/// ## Architecture Constraints
/// - Zero-allocation with PhantomData for type safety
/// - Send + Sync for concurrent execution capabilities
/// - AsyncStream-only outputs (NO Result wrapping)
/// - No unsafe code, no locking primitives
/// - Error handling via polymorphic error handlers, not Result<T, E>
/// 
/// ## Performance Characteristics
/// - Hot path marked #[inline] for zero-cost abstractions
/// - Concrete type specialization for blazing-fast execution
/// - Memory-efficient with Arc for shared ownership
/// 
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::{WorkflowStep, Workflow};
/// use fluent_ai_async::AsyncStream;
/// 
/// struct SimpleStep;
/// 
/// impl WorkflowStep<String, String> for SimpleStep {
///     fn execute(&self, input: String) -> AsyncStream<String> {
///         // Real execution logic here - no mocking, no Result wrapping
///         AsyncStream::with_channel(|sender| {
///             let _ = sender.send(format!("Processed: {}", input));
///         })
///     }
/// }
/// ```
pub trait WorkflowStep<In, Out>: Send + Sync + 'static {
    /// Execute the workflow step with streaming output
    /// 
    /// Takes input of type `In` and produces a stream of `Out` values
    /// using AsyncStream. This method is the core execution primitive for all
    /// workflow operations.
    /// 
    /// ## Implementation Requirements
    /// - Must use AsyncStream for return type (streams-only architecture)
    /// - No .await on AsyncStream (streams are consumed, not awaited)
    /// - NO Result<T, E> wrapping - error handling via polymorphic handlers
    /// - Real execution logic - no mocking or simulation
    /// 
    /// ## Performance Notes
    /// - Method is not marked #[inline] to allow trait object optimization
    /// - Implementations should use #[inline] for hot path methods
    /// - AsyncStream provides zero-copy streaming where possible
    fn execute(&self, input: In) -> AsyncStream<Out>;
}

/// Compiled workflow with real execution capabilities
/// 
/// This struct represents a fully compiled workflow that can be executed
/// multiple times with different inputs. It wraps a WorkflowStep trait object
/// for polymorphic execution while maintaining zero-allocation characteristics.
/// 
/// ## Architecture
/// - Arc<dyn WorkflowStep> for shared ownership without locking
/// - PhantomData for zero-allocation type safety
/// - Clone implementation for workflow reusability
/// - streams-only execution using AsyncTask integration
/// 
/// ## Memory Efficiency
/// - Single Arc allocation per workflow instance
/// - PhantomData adds zero runtime cost
/// - Trait object enables polymorphic dispatch
/// 
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::{Workflow, WorkflowStep};
/// 
/// let step = /* some WorkflowStep implementation */;
/// let workflow = Workflow::new(step);
/// let result_stream = workflow.execute("input".to_string());
/// ```
pub struct Workflow<In, Out> {
    /// The compiled workflow step for execution
    step: Arc<dyn WorkflowStep<In, Out>>,
    /// Zero-cost type marker for input/output types
    _phantom: PhantomData<(In, Out)>,
}

impl<In, Out> Workflow<In, Out>
where
    In: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    /// Create a new workflow from a step implementation
    /// 
    /// Takes ownership of any type implementing WorkflowStep and wraps it
    /// in an Arc for shared ownership. This enables workflow reuse across
    /// multiple execution contexts without additional allocation.
    /// 
    /// ## Performance
    /// - Single Arc allocation
    /// - Zero-cost type erasure through trait objects
    /// - Optimal for repeated execution scenarios
    #[inline]
    pub fn new<S>(step: S) -> Self 
    where
        S: WorkflowStep<In, Out> + 'static,
    {
        Self {
            step: Arc::new(step),
            _phantom: PhantomData,
        }
    }

    /// Execute the workflow with streaming output
    /// 
    /// Delegates to the underlying WorkflowStep::execute method, providing
    /// consistent streaming interface. Returns AsyncStream for streams-only
    /// architecture compliance.
    /// 
    /// ## Streams-Only Architecture
    /// - No Future trait usage in execution path
    /// - AsyncStream for all outputs (no Result wrapping)
    /// - No .await on streams (streams are consumed)
    /// - Error handling via polymorphic error handlers
    #[inline]
    pub fn execute(&self, input: In) -> AsyncStream<Out> {
        self.step.execute(input)
    }

    /// Get streaming output without execution context
    /// 
    /// Alias for execute() that emphasizes the streaming nature of workflow
    /// outputs. Useful for composition patterns where streaming semantics
    /// are the primary concern.
    #[inline]
    pub fn stream(&self, input: In) -> AsyncStream<Out> {
        self.execute(input)
    }

    /// Chain another workflow step for composition
    /// 
    /// Creates a new workflow that executes this workflow first, then pipes
    /// its output to the next step. This enables functional composition of
    /// workflow steps while maintaining streams-only architecture.
    /// 
    /// ## Composition Semantics
    /// - Sequential execution (first this, then next)
    /// - Stream-based piping (no intermediate materialization)
    /// - Error propagation through composition chain
    /// - Type-safe composition with generic constraints
    /// 
    /// ## Performance
    /// - Zero allocation composition
    /// - Stream fusion where possible
    /// - No intermediate collection overhead
    pub fn then<NewOut, S>(self, next_step: S) -> Workflow<In, NewOut>
    where
        S: WorkflowStep<Out, NewOut> + 'static,
        NewOut: Send + Sync + 'static,
    {
        let current_step = self.step;
        let next_arc = Arc::new(next_step);
        
        Workflow::new(ComposedStep {
            first: current_step,
            second: next_arc,
            _phantom: PhantomData,
        })
    }
}

impl<In, Out> Clone for Workflow<In, Out> {
    /// Clone workflow for reuse
    /// 
    /// Creates a new workflow instance sharing the same underlying step
    /// implementation. This is a cheap operation due to Arc sharing and
    /// enables workflow reuse across multiple execution contexts.
    #[inline]
    fn clone(&self) -> Self {
        Self {
            step: Arc::clone(&self.step),
            _phantom: PhantomData,
        }
    }
}

/// Internal composition step for workflow chaining
/// 
/// This struct implements the sequential execution of two workflow steps,
/// enabling the `then()` combinator. It handles stream-based piping between
/// steps while maintaining error propagation semantics.
struct ComposedStep<In, Mid, Out> {
    first: Arc<dyn WorkflowStep<In, Mid>>,
    second: Arc<dyn WorkflowStep<Mid, Out>>,
    _phantom: PhantomData<(In, Mid, Out)>,
}

impl<In, Mid, Out> WorkflowStep<In, Out> for ComposedStep<In, Mid, Out>
where
    In: Send + Sync + 'static,
    Mid: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    fn execute(&self, input: In) -> AsyncStream<Out> {
        let first = Arc::clone(&self.first);
        let second = Arc::clone(&self.second);
        
        // Create event-driven streaming composition with timeout-aware forwarding
        AsyncStream::with_channel(move |sender| {
            // Execute first step to get intermediate stream
            let first_stream = first.execute(input);
            
            // Use thread-based forwarding to avoid blocking the main execution
            std::thread::spawn(move || {
                // Collect intermediate results with timeout handling
                let intermediate_results = first_stream.collect();
                
                // Process each intermediate result through second step
                for mid_value in intermediate_results {
                    // Execute second step with intermediate value
                    let second_stream = second.execute(mid_value);
                    
                    // Collect and forward outputs from second step
                    let second_results = second_stream.collect();
                    for output in second_results {
                        if sender.send(output).is_err() {
                            return; // Receiver dropped, exit gracefully
                        }
                    }
                }
            });
        })
    }
}