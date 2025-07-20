//! Memory Workflow System for Cognitive Processing
//!
//! This module provides workflow operations for memory-enhanced cognitive processing,
//! including RAG (Retrieval-Augmented Generation) and adaptive learning workflows.

use super::ops::Op;
use super::{MemoryError, MemoryManagerTrait as MemoryManager, MemoryNode, MemoryType};

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

/// Error type for memory workflows
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Prompt error: {0}")]
    Prompt(#[from] PromptError),

    #[error("Workflow error: {0}")]
    Other(String),
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
    pub fn new(memory_manager: M, prompt_model: P) -> Self {
        Self {
            memory_manager,
            prompt_model,
            context_limit: 10,
        }
    }

    pub fn with_context_limit(mut self, limit: usize) -> Self {
        self.context_limit = limit;
        self
    }

    /// Build the memory-enhanced workflow
    pub fn build(self) -> impl Op<Input = String, Output = Result<String, WorkflowError>> {
        MemoryWorkflowOp {
            memory_manager: self.memory_manager,
            prompt_model: self.prompt_model,
            context_limit: self.context_limit,
        }
    }
}
