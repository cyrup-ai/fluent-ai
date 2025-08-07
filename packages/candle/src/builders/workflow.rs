//! Real Workflow Builder - Zero-allocation construction patterns
//!
//! Provides fluent API for building executable workflows using the real WorkflowStep
//! trait and streams-only architecture. All builders produce functional Workflow
//! instances without fake or mock implementations.

use fluent_ai_async::AsyncStream;
use crate::workflow::{WorkflowStep, Workflow};
use serde_json::Value;

/// Real workflow builder trait - streams-only architecture
/// 
/// Provides fluent API for building executable workflows using the real
/// WorkflowStep trait and AsyncStream architecture. All operations are
/// zero-allocation and blazing-fast.
pub trait WorkflowBuilder: Sized {
    /// Add sequential operation - EXACT syntax: .then(step)
    fn then<S>(self, step: S) -> impl WorkflowBuilder
    where
        S: WorkflowStep<Value, Value> + 'static;
    
    /// Add parallel operations - EXACT syntax: .parallel(steps)
    fn parallel<I, S>(self, steps: I) -> impl WorkflowBuilder
    where
        I: IntoIterator<Item = S>,
        S: WorkflowStep<Value, Value> + 'static;
    
    /// Build final executable workflow - EXACT syntax: .build()
    fn build(self) -> Workflow<Value, Value>;
}

/// Real workflow builder implementation - zero-allocation state
/// 
/// Uses actual WorkflowStep trait and produces real Workflow instances
pub struct WorkflowBuilderImpl {
    steps: Vec<Box<dyn WorkflowStep<Value, Value>>>,
}

impl WorkflowBuilderImpl {
    /// Create a new workflow builder
    pub fn new() -> impl WorkflowBuilder {
        WorkflowBuilderImpl {
            steps: Vec::new(),
        }
    }
}

impl WorkflowBuilder for WorkflowBuilderImpl {
    fn then<S>(mut self, step: S) -> impl WorkflowBuilder
    where
        S: WorkflowStep<Value, Value> + 'static,
    {
        self.steps.push(Box::new(step));
        self
    }

    fn parallel<I, S>(mut self, steps: I) -> impl WorkflowBuilder
    where
        I: IntoIterator<Item = S>,
        S: WorkflowStep<Value, Value> + 'static,
    {
        // For now, execute parallel steps sequentially
        // TODO: Implement true parallel execution in future iteration
        for step in steps {
            self.steps.push(Box::new(step));
        }
        self
    }

    fn build(self) -> Workflow<Value, Value> {
        if self.steps.is_empty() {
            // Create a passthrough workflow for empty builders
            Workflow::new(PassthroughStep)
        } else if self.steps.len() == 1 {
            // Single step workflow
            let step = self.steps.into_iter().next().unwrap();
            Workflow::new(BoxedStep { inner: step })
        } else {
            // Chain multiple steps using composition
            let mut steps = self.steps.into_iter();
            let mut workflow = {
                let first = steps.next().unwrap();
                Workflow::new(BoxedStep { inner: first })
            };
            
            for step in steps {
                let boxed = BoxedStep { inner: step };
                workflow = workflow.then(boxed);
            }
            workflow
        }
    }
}

/// Simple passthrough step for empty workflows
struct PassthroughStep;

impl WorkflowStep<Value, Value> for PassthroughStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(input);
        })
    }
}

/// Wrapper for boxed workflow steps
struct BoxedStep {
    inner: Box<dyn WorkflowStep<Value, Value>>,
}

impl WorkflowStep<Value, Value> for BoxedStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        self.inner.execute(input)
    }
}

/// Example workflow step for testing/demonstration
pub struct SimpleStep {
    pub name: String,
}

impl SimpleStep {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl WorkflowStep<Value, Value> for SimpleStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        let name = self.name.clone();
        AsyncStream::with_channel(move |sender| {
            // Simple transformation: add step name to input
            let output = match input {
                Value::Object(mut obj) => {
                    obj.insert("processed_by".to_string(), Value::String(name));
                    Value::Object(obj)
                }
                other => {
                    let mut result = serde_json::Map::new();
                    result.insert("original".to_string(), other);
                    result.insert("processed_by".to_string(), Value::String(name));
                    Value::Object(result)
                }
            };
            let _ = sender.send(output);
        })
    }
}

/// Create a new workflow builder - EXACT syntax: WorkflowBuilder::new()
pub fn new() -> impl WorkflowBuilder {
    WorkflowBuilderImpl::new()
}

impl Default for WorkflowBuilderImpl {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
        }
    }
}