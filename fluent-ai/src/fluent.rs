use crate::domain::CompletionModel;
use crate::domain::*;
use crate::{Models, memory, workflow};

/// Master builder for Fluent AI - semantic entry point for all builders
pub struct FluentAi;

impl FluentAi {
    /// Create an AI agent with persistent context and tools
    pub fn agent(model: Models) -> agent::AgentBuilder {
        agent::Agent::with_model(model)
    }

    /// Define a reusable agent role/persona
    pub fn agent_role(name: impl Into<String>) -> agent_role::AgentRoleBuilder {
        agent_role::AgentRoleBuilder::new(name)
    }

    /// Make a one-off completion request
    pub fn completion(system_prompt: impl Into<String>) -> completion::CompletionRequestBuilder {
        completion::CompletionRequest::prompt(system_prompt)
    }

    /// Extract structured data from unstructured text
    pub fn extract<
        T: serde::de::DeserializeOwned + Send + Sync + std::fmt::Debug + Clone + 'static,
        M: CompletionModel,
    >(
        model: M,
    ) -> extractor::ExtractorBuilder<T, M> {
        extractor::ExtractorImpl::<T>::extract_with(model)
    }

    /// Load files from various sources
    pub fn loader(pattern: &str) -> loader::LoaderBuilder<std::path::PathBuf> {
        loader::LoaderImpl::files_matching(pattern)
    }

    /// Access the memory system
    pub fn memory() -> memory::Memory {
        memory::Memory::new()
    }

    /// Create a conversation for message history management
    pub fn conversation() -> conversation::ConversationBuilder {
        conversation::ConversationBuilder::new()
    }

    /// Create a workflow from a step
    pub fn workflow<In, Out, S>(step: S) -> workflow::WorkflowBuilder<In, Out>
    where
        S: workflow::WorkflowStep<In, Out>,
        In: Send + 'static,
        Out: Send + 'static,
    {
        workflow::Workflow::from(step)
    }
}

// Convenience re-export
pub use FluentAi as Ai;
