//! Real Memory Workflow Builder - Memory-enhanced workflow construction
//!
//! Provides fluent API for building memory-enhanced workflows that integrate
//! with RAG systems, adaptive learning, and cognitive processing patterns.
//! All builders use real WorkflowStep implementations.

use fluent_ai_async::AsyncStream;
use crate::workflow::{WorkflowStep, Workflow};
use serde_json::Value;

/// Real memory workflow builder trait - streams-only architecture
/// 
/// Extends WorkflowBuilder with memory-enhanced operations for RAG and
/// cognitive processing workflows. Uses real WorkflowStep implementations.
pub trait MemoryWorkflowBuilder: Sized {
    /// Chain sequential operation - EXACT syntax: .chain(step)
    fn chain<S>(self, step: S) -> impl MemoryWorkflowBuilder
    where
        S: WorkflowStep<Value, Value> + 'static;
    
    /// Add parallel operations - EXACT syntax: .parallel(steps)
    fn parallel<I, S>(self, steps: I) -> impl MemoryWorkflowBuilder
    where
        I: IntoIterator<Item = S>,
        S: WorkflowStep<Value, Value> + 'static;
    
    /// Build executable memory workflow - EXACT syntax: .build()
    fn build(self) -> Workflow<Value, Value>;
}

/// Real memory workflow builder implementation
pub struct MemoryWorkflowBuilderImpl {
    steps: Vec<Box<dyn WorkflowStep<Value, Value>>>,
}

impl MemoryWorkflowBuilderImpl {
    /// Create a new memory workflow builder with optimal defaults
    pub fn new() -> impl MemoryWorkflowBuilder {
        MemoryWorkflowBuilderImpl {
            steps: Vec::new(),
        }
    }
}

impl MemoryWorkflowBuilder for MemoryWorkflowBuilderImpl {
    fn chain<S>(mut self, step: S) -> impl MemoryWorkflowBuilder
    where
        S: WorkflowStep<Value, Value> + 'static,
    {
        self.steps.push(Box::new(step));
        self
    }

    fn parallel<I, S>(mut self, steps: I) -> impl MemoryWorkflowBuilder
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
            Workflow::new(MemoryPassthroughStep)
        } else if self.steps.len() == 1 {
            // Single step workflow
            let step = self.steps.into_iter().next().unwrap();
            Workflow::new(MemoryBoxedStep { inner: step })
        } else {
            // Chain multiple steps using composition
            let mut steps = self.steps.into_iter();
            let mut workflow = {
                let first = steps.next().unwrap();
                Workflow::new(MemoryBoxedStep { inner: first })
            };
            
            for step in steps {
                let boxed = MemoryBoxedStep { inner: step };
                workflow = workflow.then(boxed);
            }
            workflow
        }
    }
}

/// Memory-aware passthrough step
struct MemoryPassthroughStep;

impl WorkflowStep<Value, Value> for MemoryPassthroughStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        AsyncStream::with_channel(move |sender| {
            // Add memory processing metadata
            let output = match input {
                Value::Object(mut obj) => {
                    obj.insert("memory_processed".to_string(), Value::Bool(true));
                    Value::Object(obj)
                }
                other => other,
            };
            let _ = sender.send(output);
        })
    }
}

/// Wrapper for boxed memory workflow steps
struct MemoryBoxedStep {
    inner: Box<dyn WorkflowStep<Value, Value>>,
}

impl WorkflowStep<Value, Value> for MemoryBoxedStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        self.inner.execute(input)
    }
}

/// Example memory-enhanced workflow step
pub struct SimpleMemoryStep {
    pub name: String,
}

impl SimpleMemoryStep {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl WorkflowStep<Value, Value> for SimpleMemoryStep {
    fn execute(&self, input: Value) -> AsyncStream<Value> {
        let name = self.name.clone();
        AsyncStream::with_channel(move |sender| {
            // Memory-enhanced transformation
            let output = match input {
                Value::Object(mut obj) => {
                    obj.insert("memory_step".to_string(), Value::String(name));
                    obj.insert("memory_enhanced".to_string(), Value::Bool(true));
                    Value::Object(obj)
                }
                other => {
                    let mut result = serde_json::Map::new();
                    result.insert("original".to_string(), other);
                    result.insert("memory_step".to_string(), Value::String(name));
                    result.insert("memory_enhanced".to_string(), Value::Bool(true));
                    Value::Object(result)
                }
            };
            let _ = sender.send(output);
        })
    }
}

/// Create a new memory workflow builder
pub fn new() -> impl MemoryWorkflowBuilder {
    MemoryWorkflowBuilderImpl::new()
}

impl Default for MemoryWorkflowBuilderImpl {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
        }
    }
}