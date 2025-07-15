//! Workflow system for composing operations in a fluent style
//!
//! A workflow is a reusable, composable unit of computation that transforms
//! input to output through a series of steps.

use crate::prelude::*;
use futures::future::BoxFuture;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

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

        AsyncTask::from_future(async move {
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

    /// Filter outputs based on a predicate
    pub fn filter<F>(self, predicate: F) -> WorkflowBuilder<In, Option<Out>>
    where
        F: Fn(&Out) -> bool + Send + Sync + Clone + 'static,
    {
        let prev_step = self.step;

        let filtered = move |input: In| {
            let prev = prev_step.clone();
            let pred = predicate.clone();

            async move {
                let result = prev.execute(input).await?;
                if pred(&result) {
                    Ok(Some(result))
                } else {
                    Ok(None)
                }
            }
        };

        WorkflowBuilder::new(filtered)
    }

    /// Tap into the output for side effects
    pub fn tap<F>(self, f: F) -> WorkflowBuilder<In, Out>
    where
        F: Fn(&Out) + Send + Sync + Clone + 'static,
        Out: Clone,
    {
        let prev_step = self.step;

        let tapped = move |input: In| {
            let prev = prev_step.clone();
            let tap_fn = f.clone();

            async move {
                let result = prev.execute(input).await?;
                tap_fn(&result);
                Ok(result)
            }
        };

        WorkflowBuilder::new(tapped)
    }

    /// Add parallel steps
    pub fn then_parallel(self) -> ParallelStepsBuilder<In, Out>
    where
        Out: Clone,
    {
        ParallelStepsBuilder::new(self)
    }

    /// Add conditional step
    pub fn then_if<F, S>(self, condition: F, step: S) -> WorkflowBuilder<In, Out>
    where
        F: Fn(&Out) -> bool + Send + Sync + Clone + 'static,
        S: WorkflowStep<Out, Out>,
        Out: Clone,
    {
        let prev_step = self.step;
        let cond_step = Arc::new(step);

        let conditional = move |input: In| {
            let prev = prev_step.clone();
            let cond = condition.clone();
            let step = cond_step.clone();

            async move {
                let result = prev.execute(input).await?;
                if cond(&result) {
                    step.execute(result).await
                } else {
                    Ok(result)
                }
            }
        };

        WorkflowBuilder::new(conditional)
    }

    /// Retry on failure
    pub fn retry(self, max_attempts: usize) -> WorkflowBuilder<In, Out>
    where
        In: Clone,
    {
        let prev_step = self.step;

        let retried = move |input: In| {
            let step = prev_step.clone();

            async move {
                if max_attempts == 0 {
                    return Err("No retry attempts configured".into());
                }

                let mut last_error = None;
                for _ in 0..max_attempts {
                    match step.execute(input.clone()).await {
                        Ok(result) => return Ok(result),
                        Err(e) => last_error = Some(e),
                    }
                }

                // Safe because last_error is guaranteed to be Some after the loop
                match last_error {
                    Some(error) => Err(error),
                    None => Err("Unknown error during retry attempts".into()),
                }
            }
        };

        WorkflowBuilder::new(retried)
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

/// Builder for parallel step execution
pub struct ParallelStepsBuilder<In, Out> {
    prev_builder: WorkflowBuilder<In, Out>,
    steps: Vec<Arc<dyn WorkflowStep<Out, Out>>>,
}

impl<In, Out> ParallelStepsBuilder<In, Out>
where
    In: Send + 'static,
    Out: Send + 'static + Clone,
{
    fn new(prev_builder: WorkflowBuilder<In, Out>) -> Self {
        Self {
            prev_builder,
            steps: Vec::new(),
        }
    }

    /// Add a step to execute in parallel
    pub fn add<S>(mut self, step: S) -> Self
    where
        S: WorkflowStep<Out, Out>,
    {
        self.steps.push(Arc::new(step));
        self
    }

    /// Join all parallel results into a Vec
    pub fn join(self) -> WorkflowBuilder<In, Vec<Out>> {
        let prev_step = self.prev_builder.step;
        let parallel_steps = self.steps;

        let joined = move |input: In| {
            let prev = prev_step.clone();
            let steps = parallel_steps.clone();

            async move {
                let initial_result = prev.execute(input).await?;

                let futures: Vec<_> = steps
                    .iter()
                    .map(|step| {
                        let step = step.clone();
                        let input = initial_result.clone();
                        async move { step.execute(input).await }
                    })
                    .collect();

                let results = futures::future::join_all(futures).await;

                // Collect successful results
                let mut outputs = Vec::new();
                for result in results {
                    outputs.push(result?);
                }
                Ok(outputs)
            }
        };

        WorkflowBuilder::new(joined)
    }

    /// Join with a custom combiner function
    pub fn join_with<Combined, F>(self, combiner: F) -> WorkflowBuilder<In, Combined>
    where
        F: Fn(Vec<Out>) -> Combined + Send + Sync + Clone + 'static,
        Combined: Send + 'static,
    {
        self.join().map(combiner)
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
                let task = match task.lock().unwrap().take() {
                    Some(t) => t,
                    None => return Err("Task already consumed".to_string()),
                };
                
                match task.await {
                    Ok(result) => Ok(result),
                    Err(e) => Err(e.to_string()),
                }
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
                let stream = match stream.lock().unwrap().take() {
                    Some(s) => s,
                    None => return Err("Stream already consumed".to_string()),
                };
                
                let items = stream.collect().await;
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
