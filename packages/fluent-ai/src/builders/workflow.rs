/// Workflow builder for creating complex processing pipelines
pub struct WorkflowBuilder;

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new() -> Self {
        WorkflowBuilder
    }

    /// Chain operations in the workflow
    pub fn chain<O>(self, op: O) -> O {
        op
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}