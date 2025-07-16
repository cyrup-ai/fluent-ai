// ============================================================================
// File: src/workflow/workflow.rs
// ----------------------------------------------------------------------------
// Best-of-best workflow system combining polymorphic builders with high-performance ops
//
//  • `WorkflowStep` – polymorphic trait for flexible workflow construction
//  • `Workflow` – compiled, reusable workflow with error handling
//  • `Op` – zero-cost, async-aware transformation node for performance
//  • `Sequential` combinator for fluent chaining
//  • Helpers: `map`, `then`, `lookup`, `prompt`, `passthrough`
//
// Every combinator returns a concrete type (no boxing) so the compiler can
// fully optimise the call-graph.  All hot-path methods are `#[inline]`.
// ============================================================================

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use futures::future::BoxFuture;

#[allow(unused_imports)] // used in downstream macro expansion
use futures::join;
use futures::stream;

use crate::{completion, vector_store};
use crate::prelude::*;

// ================================================================
// 0. Polymorphic WorkflowStep trait for flexible workflow construction
// ================================================================

/// A single transformation step in a workflow
pub trait WorkflowStep<In, Out>: Send + Sync + 'static {
    /// Execute this step on the input
    fn execute(&self, input: In) -> BoxFuture<'static, Result<Out, String>>;
}

/// Implementation for async functions as workflow steps
impl<F, In, Out, Fut> WorkflowStep<In, Out> for F
where
    F: Fn(In) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Out, String>> + Send + 'static,
    In: Send + 'static,
    Out: Send + 'static,
{
    fn execute(&self, input: In) -> BoxFuture<'static, Result<Out, String>> {
        Box::pin(self(input))
    }
}

/// A compiled, reusable workflow that transforms In → Out
pub struct Workflow<In, Out> {
    step: Arc<dyn WorkflowStep<In, Out>>,
    _phantom: PhantomData<(In, Out)>,
}

impl<In, Out> Clone for Workflow<In, Out> {
    fn clone(&self) -> Self {
        Self {
            step: self.step.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<In, Out> Workflow<In, Out>
where
    In: Send + 'static,
    Out: Send + 'static,
{
    /// Execute the workflow once
    pub fn execute(&self, input: In) -> AsyncTask<Out> 
    where
        Out: crate::async_task::NotResult,
    {
        let step = self.step.clone();

        crate::async_task::spawn_async(async move {
            match step.execute(input).await {
                Ok(output) => output,
                Err(e) => panic!("Workflow error: {}", e), // Error handler was already called
            }
        })
    }

    /// Execute the workflow on a stream of inputs
    pub fn stream(&self, inputs: AsyncStream<In>) -> AsyncStream<Out>
    where
        In: 'static + crate::async_task::NotResult,
        Out: crate::async_task::NotResult,
    {
        let step = self.step.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = inputs;

            while let Some(input) = stream.next().await {
                let step = step.clone();
                match step.execute(input).await {
                    Ok(output) => {
                        if tx.send(output).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        // Error already handled by error handler
                        continue;
                    }
                }
            }
        });

        AsyncStream::new(rx)
    }

    /// Compose this workflow with another
    pub fn then<Out2>(self, other: Workflow<Out, Out2>) -> Workflow<In, Out2>
    where
        Out2: Send + 'static,
    {
        let step1 = self.step;
        let step2 = other.step;

        let composed = move |input: In| {
            let step1 = step1.clone();
            let step2 = step2.clone();

            async move {
                let result1 = step1.execute(input).await?;
                step2.execute(result1).await
            }
        };

        Workflow {
            step: Arc::new(composed),
            _phantom: PhantomData,
        }
    }
}

// ================================================================
// 1. Op trait – the universal node for high-performance operations
// ================================================================
pub trait Op: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;

    fn call(&self, input: Self::Input) -> impl Future<Output = Self::Output> + Send;

    /// Execute this op over an iterator of inputs with at most `n` concurrent
    /// in-flight tasks.
    fn batch_call<I>(&self, n: usize, input: I) -> impl Future<Output = Vec<Self::Output>> + Send
    where
        I: IntoIterator<Item = Self::Input> + Send,
        I::IntoIter: Send,
        Self: Sized,
    {
        use futures::stream::StreamExt;

        async move {
            stream::iter(input)
                .map(|v| self.call(v))
                .buffered(n)
                .collect()
                .await
        }
    }

    /* ---------------------------------------------------------------------
     * Fluent combinators
     * ------------------------------------------------------------------ */
    #[inline]
    fn map<F, Out2>(self, f: F) -> Sequential<Self, Map<F, Self::Output>>
    where
        Self: Sized,
        F: Fn(Self::Output) -> Out2 + Send + Sync,
        Out2: Send + Sync,
    {
        Sequential::new(self, Map::new(f))
    }

    #[inline]
    fn then<F, Fut>(self, f: F) -> Sequential<Self, Then<F, Fut::Output>>
    where
        Self: Sized,
        F: Fn(Self::Output) -> Fut + Send + Sync,
        Fut: Future + Send,
        Fut::Output: Send + Sync,
    {
        Sequential::new(self, Then::new(f))
    }

    #[inline]
    fn chain<O>(self, op: O) -> Sequential<Self, O>
    where
        Self: Sized,
        O: Op<Input = Self::Output>,
    {
        Sequential::new(self, op)
    }

    #[inline]
    fn lookup<Ix, Doc>(
        self,
        index: Ix,
        n: usize,
    ) -> Sequential<Self, crate::workflow::agent_ops::Lookup<Ix, Self::Output, Doc>>
    where
        Self: Sized,
        Ix: vector_store::VectorStoreIndexDyn,
        Doc: for<'a> serde::Deserialize<'a> + Send + Sync,
        Self::Output: Into<String>,
    {
        Sequential::new(self, crate::workflow::agent_ops::Lookup::new(index, n))
    }

    #[inline]
    fn prompt<P>(
        self,
        agent: P,
    ) -> Sequential<Self, crate::workflow::agent_ops::Prompt<P, Self::Output>>
    where
        Self: Sized,
        P: completion::Prompt,
        Self::Output: Into<String>,
    {
        Sequential::new(self, crate::workflow::agent_ops::Prompt::new(agent))
    }
}

// Blanket impl so `&Op` can be used interchangeably.
impl<T: Op> Op for &T {
    type Input = T::Input;
    type Output = T::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (**self).call(input).await
    }
}

// ================================================================
// 1. Sequential – glue two ops together
// ================================================================
pub struct Sequential<A, B> {
    a: A,
    b: B,
}

impl<A, B> Sequential<A, B> {
    #[inline(always)]
    pub(crate) fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Op for Sequential<A, B>
where
    A: Op,
    B: Op<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        let mid = self.a.call(input).await;
        self.b.call(mid).await
    }
}

// ================================================================
// 2. Primitive ops
// ================================================================
pub struct Map<F, In> {
    f: F,
    _pd: std::marker::PhantomData<In>,
}

impl<F, In> Map<F, In> {
    #[inline(always)]
    fn new(f: F) -> Self {
        Self {
            f,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<F, In, Out> Op for Map<F, In>
where
    F: Fn(In) -> Out + Send + Sync,
    In: Send + Sync,
    Out: Send + Sync,
{
    type Input = In;
    type Output = Out;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input)
    }
}

pub fn map<F, In, Out>(f: F) -> Map<F, In>
where
    F: Fn(In) -> Out + Send + Sync,
    In: Send + Sync,
    Out: Send + Sync,
{
    Map::new(f)
}

pub struct Then<F, In> {
    f: F,
    _pd: std::marker::PhantomData<In>,
}

impl<F, In> Then<F, In> {
    #[inline(always)]
    fn new(f: F) -> Self {
        Self {
            f,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<F, In, Fut> Op for Then<F, In>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    type Input = In;
    type Output = Fut::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input).await
    }
}

pub fn then<F, In, Fut>(f: F) -> Then<F, In>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    Then::new(f)
}

/// Identity node – forwards the input unchanged.
pub struct Passthrough<T>(std::marker::PhantomData<T>);

impl<T> Passthrough<T> {
    #[inline(always)]
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Send + Sync> Op for Passthrough<T> {
    type Input = T;
    type Output = T;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        input
    }
}

pub fn passthrough<T: Send + Sync>() -> Passthrough<T> {
    Passthrough::new()
}

// ================================================================
// 3. Workflow Builder for polymorphic construction
// ================================================================

/// Builder for constructing workflows
pub struct WorkflowBuilder<In, Out> {
    step: Arc<dyn WorkflowStep<In, Out>>,
    _phantom: PhantomData<(In, Out)>,
}

/// Builder with error handler - exposes terminal methods
pub struct WorkflowBuilderWithHandler<In, Out> {
    step: Arc<dyn WorkflowStep<In, Out>>,
    error_handler: Box<dyn FnMut(String) + Send + 'static>,
    _phantom: PhantomData<(In, Out)>,
}

impl<In, Out> WorkflowBuilder<In, Out>
where
    In: Send + 'static,
    Out: Send + 'static,
{
    fn new<S: WorkflowStep<In, Out>>(step: S) -> Self {
        Self {
            step: Arc::new(step),
            _phantom: PhantomData,
        }
    }

    /// Chain another step sequentially
    pub fn then<Out2, S>(self, step: S) -> WorkflowBuilder<In, Out2>
    where
        S: WorkflowStep<Out, Out2>,
        Out2: Send + 'static,
    {
        let prev_step = self.step;
        let next_step = Arc::new(step);

        let composed = move |input: In| {
            let prev = prev_step.clone();
            let next = next_step.clone();

            async move {
                let result = prev.execute(input).await?;
                next.execute(result).await
            }
        };

        WorkflowBuilder::new(composed)
    }

    /// Map the output through a synchronous function
    pub fn map<Out2, F>(self, f: F) -> WorkflowBuilder<In, Out2>
    where
        F: Fn(Out) -> Out2 + Send + Sync + Clone + 'static,
        Out2: Send + 'static,
    {
        let prev_step = self.step;

        let mapped = move |input: In| {
            let prev = prev_step.clone();
            let mapper = f.clone();

            async move {
                let result = prev.execute(input).await?;
                Ok(mapper(result))
            }
        };

        WorkflowBuilder::new(mapped)
    }

    /// Add error handler to enable terminal methods
    pub fn on_error<F>(self, error_handler: F) -> WorkflowBuilderWithHandler<In, Out>
    where
        F: FnMut(String) + Send + 'static,
    {
        WorkflowBuilderWithHandler {
            step: self.step,
            error_handler: Box::new(error_handler),
            _phantom: PhantomData,
        }
    }
}

impl<In, Out> WorkflowBuilderWithHandler<In, Out>
where
    In: Send + 'static,
    Out: Send + 'static,
{
    /// Build the final workflow
    pub fn build(self) -> Workflow<In, Out> {
        let step = self.step;
        let error_handler = self.error_handler;

        // Wrap the error handler in an Arc<Mutex> for sharing across async boundaries
        let error_handler = std::sync::Arc::new(parking_lot::Mutex::new(error_handler));

        let wrapped = move |input: In| {
            let step = step.clone();
            let handler = error_handler.clone();

            async move {
                match step.execute(input).await {
                    Ok(output) => Ok(output),
                    Err(e) => {
                        handler.lock()(e.clone());
                        Err(e)
                    }
                }
            }
        };

        Workflow {
            step: Arc::new(wrapped),
            _phantom: PhantomData,
        }
    }
}

/// Entry point for workflow construction
impl Workflow<(), ()> {
    /// Create a workflow from a single step
    pub fn from<In, Out, S>(step: S) -> WorkflowBuilder<In, Out>
    where
        S: WorkflowStep<In, Out>,
        In: Send + 'static,
        Out: Send + 'static,
    {
        WorkflowBuilder::new(step)
    }

    /// Create a workflow from an AsyncTask
    pub fn from_task<T>(task: AsyncTask<T>) -> WorkflowBuilder<(), T>
    where
        T: Send + 'static + Clone + crate::async_task::NotResult,
    {
        let task = Arc::new(std::sync::Mutex::new(Some(task)));
        let step = move |_: ()| {
            let task = task.clone();
            async move {
                let task = match task.lock() {
                    Ok(mut guard) => match guard.take() {
                        Some(t) => t,
                        None => return Err("Task already consumed".to_string()),
                    },
                    Err(_) => return Err("Failed to acquire task lock".to_string()),
                };
                
                // task.await returns T directly since AsyncTask handles errors internally
                Ok(task.await)
            }
        };
        WorkflowBuilder::new(step)
    }

    /// Create a workflow from an AsyncStream
    pub fn from_stream<T>(stream: AsyncStream<T>) -> WorkflowBuilder<(), Vec<T>>
    where
        T: Send + 'static + crate::async_task::NotResult,
    {
        let stream = Arc::new(std::sync::Mutex::new(Some(stream)));
        let step = move |_: ()| {
            let stream = stream.clone();
            async move {
                let stream = match stream.lock() {
                    Ok(mut guard) => match guard.take() {
                        Some(s) => s,
                        None => return Err("Stream already consumed".to_string()),
                    },
                    Err(_) => return Err("Failed to acquire stream lock".to_string()),
                };
                
                // Manually collect stream items since AsyncStream doesn't have collect()
                let mut items = Vec::new();
                use futures_util::StreamExt;
                let mut pinned_stream = std::pin::pin!(stream);
                while let Some(item) = pinned_stream.next().await {
                    items.push(item);
                }
                Ok(items)
            }
        };
        WorkflowBuilder::new(step)
    }
}

/// Helper function for creating simple steps
pub fn step<In, Out, F>(f: F) -> impl WorkflowStep<In, Out>
where
    F: Fn(In) -> Out + Send + Sync + Clone + 'static,
    In: Send + 'static,
    Out: Send + 'static,
{
    move |input: In| {
        let f = f.clone();
        async move { Ok(f(input)) }
    }
}

/// Helper function for creating async steps
pub fn async_step<In, Out, F, Fut>(f: F) -> impl WorkflowStep<In, Out>
where
    F: Fn(In) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Out, String>> + Send + 'static,
    In: Send + 'static,
    Out: Send + 'static,
{
    f
}

/* --------------------------------------------------------------------------
 * Tests
 * ----------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sequential_runs_in_order() {
        let pipeline = map(|x: i32| x + 1)
            .map(|x| x * 2)
            .then(|x| async move { x * 3 });

        assert_eq!(pipeline.call(1).await, 12);
    }

    #[tokio::test]
    async fn batch_processing() {
        let op = map(|x: i32| x + 1);
        let data = vec![1, 2, 3, 4];

        let out = op.batch_call(2, data).await;
        assert_eq!(out, vec![2, 3, 4, 5]);
    }
}
