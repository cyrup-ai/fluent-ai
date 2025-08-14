//! Candle Workflow Core - Zero-allocation impl Trait workflow execution
//!
//! This module provides candle workflow traits and builders following fluent-ai
//! patterns exactly but with Candle prefixes. All execution uses unwrapped
//! AsyncStream<Out> with NO trait objects - only impl Trait patterns.

use std::marker::PhantomData;

use fluent_ai_async::AsyncStream;

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
/// - Error handling via handle_error! macro and stream termination
///
/// ## Performance Characteristics
/// - Hot path marked #[inline] for zero-cost abstractions
/// - Concrete type specialization for blazing-fast execution
/// - No trait objects - all impl Trait patterns for zero allocation
///
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::{CandleWorkflowStep, CandleWorkflow};
/// use fluent_ai_async::{AsyncStream, handle_error};
///
/// struct SimpleStep;
///
/// impl CandleWorkflowStep<String, String> for SimpleStep {
///     fn execute(&self, input: String) -> AsyncStream<String> {
///         // Real execution logic here - no mocking, no Result wrapping
///         AsyncStream::with_channel(|sender| {
///             match process_input(&input) {
///                 Ok(result) => { sender.send(format!("Processed: {}", result)); },
///                 Err(e) => handle_error!(e, "SimpleStep execution failed"),
///             }
///         })
///     }
/// }
/// ```
pub trait CandleWorkflowStep<In, Out>: Send + Sync + 'static {
    /// Execute the workflow step with streaming output
    ///
    /// Takes input of type `In` and produces a stream of `Out` values
    /// using AsyncStream. This method is the core execution primitive for all
    /// workflow operations.
    ///
    /// ## Implementation Requirements
    /// - Must use AsyncStream for return type (streams-only architecture)
    /// - No .await on AsyncStream (streams are consumed, not awaited)
    /// - NO Result<T, E> wrapping - error handling via handle_error! macro
    /// - Real execution logic - no mocking or simulation
    ///
    /// ## Performance Notes
    /// - Method is not marked #[inline] to allow concrete specialization
    /// - Implementations should use #[inline] for hot path methods
    /// - AsyncStream provides zero-copy streaming where possible
    fn execute(&self, input: In) -> AsyncStream<Out>;
}

/// Simple passthrough step for basic workflow functionality  
/// This is a concrete implementation that can be used as a starting point
#[derive(Clone)]
pub struct CandlePassthroughStep;

impl CandleWorkflowStep<serde_json::Value, serde_json::Value> for CandlePassthroughStep {
    #[inline]
    fn execute(&self, input: serde_json::Value) -> AsyncStream<serde_json::Value> {
        AsyncStream::with_channel(move |sender| {
            if sender.send(input).is_err() {
                return; // Receiver dropped, terminate gracefully
            }
        })
    }
}

/// Public constructor for creating candle workflows
/// Returns a simple executable workflow that passes through its input
pub fn candle_workflow() -> CandleExecutableWorkflow<CandlePassthroughStep> {
    CandleExecutableWorkflow {
        step: CandlePassthroughStep,
        _phantom: PhantomData,
    }
}

/// Compiled, executable candle workflow with concrete step type
///
/// This struct represents a fully built workflow that can be executed
/// with streaming output. It follows candle patterns with AsyncStream<Out>
/// unwrapped streams for error handling via handle_error! macro.
pub struct CandleExecutableWorkflow<S>
where
    S: CandleWorkflowStep<serde_json::Value, serde_json::Value>,
{
    step: S,
    _phantom: PhantomData<S>,
}

impl<S> CandleExecutableWorkflow<S>
where
    S: CandleWorkflowStep<serde_json::Value, serde_json::Value>,
{
    /// Execute the workflow with streaming output
    ///
    /// Takes input and produces a stream of outputs using streams-only architecture.
    /// Error handling uses handle_error! macro pattern for stream termination.
    ///
    /// ## Streams-Only Architecture  
    /// - Returns AsyncStream<serde_json::Value> (unwrapped)
    /// - NO Result<T,E> wrapping inside streams
    /// - Error handling via handle_error! macro and stream termination  
    /// - No Future trait usage in execution paths
    #[inline]
    pub fn execute(&self, input: serde_json::Value) -> AsyncStream<serde_json::Value> {
        self.step.execute(input)
    }

    /// Chain another step after this workflow
    /// Creates a new workflow that executes this step then the next step
    pub fn then<NextS>(
        self,
        next_step: NextS,
    ) -> CandleExecutableWorkflow<CandleSequentialStep<S, NextS>>
    where
        NextS: CandleWorkflowStep<serde_json::Value, serde_json::Value> + Clone,
        S: Clone,
    {
        CandleExecutableWorkflow {
            step: CandleSequentialStep {
                first: self.step,
                second: next_step,
                _phantom: PhantomData,
            },
            _phantom: PhantomData,
        }
    }
}

/// Sequential composition step that chains two steps together
/// This replaces the Arc<dyn> pattern with concrete generic composition
pub struct CandleSequentialStep<A, B>
where
    A: CandleWorkflowStep<serde_json::Value, serde_json::Value> + Clone,
    B: CandleWorkflowStep<serde_json::Value, serde_json::Value> + Clone,
{
    first: A,
    second: B,
    _phantom: PhantomData<(A, B)>,
}

impl<A, B> CandleWorkflowStep<serde_json::Value, serde_json::Value> for CandleSequentialStep<A, B>
where
    A: CandleWorkflowStep<serde_json::Value, serde_json::Value> + Clone,
    B: CandleWorkflowStep<serde_json::Value, serde_json::Value> + Clone,
{
    fn execute(&self, input: serde_json::Value) -> AsyncStream<serde_json::Value> {
        // Clone to avoid lifetime issues in the closure
        let second_clone = self.second.clone();

        // Execute first step and chain with second step
        let first_stream = self.first.execute(input);

        AsyncStream::with_channel(move |sender| {
            let first_results = first_stream.collect();
            for intermediate in first_results {
                let second_stream = second_clone.execute(intermediate);
                let second_results = second_stream.collect();
                for output in second_results {
                    if sender.send(output).is_err() {
                        return; // Receiver dropped, terminate gracefully
                    }
                }
            }
        })
    }
}
