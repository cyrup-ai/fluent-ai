use crate::domain::CompletionModel;
use crate::domain::*;
use crate::{Models, memory, workflow};
use termcolor::{ThemeConfig, set_global_theme};

/// Master builder for Fluent AI - semantic entry point for all builders
pub struct FluentAi;

/// Builder for configuring the global FluentAi environment
pub struct FluentAiBuilder {
    theme_config: ThemeConfig,
}

impl Default for FluentAiBuilder {
    fn default() -> Self {
        Self {
            theme_config: ThemeConfig::Default,
        }
    }
}

impl FluentAiBuilder {
    /// Set theme configuration (use Cyrup.ai default colors)
    pub fn theme(mut self, config: ThemeConfig) -> Self {
        self.theme_config = config;
        self
    }
    
    /// Build and apply global configuration
    pub fn build(self) -> FluentAi {
        // Apply theme configuration globally
        set_global_theme(self.theme_config);
        FluentAi
    }
}

impl FluentAi {
    /// Create a new FluentAi builder for global configuration
    pub fn builder() -> FluentAiBuilder {
        FluentAiBuilder::default()
    }
    
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
