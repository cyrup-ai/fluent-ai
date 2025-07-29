//! Minimal CandleFluentAi entry point for testing
//!
//! This is a simplified version that provides just the basic entry point
//! without complex dependencies to verify the core pattern works.

/// CandleFluentAi entry point for creating agent roles
pub struct CandleFluentAi;

impl CandleFluentAi {
    /// Create a new Candle agent role builder - main entry point
    pub fn agent_role(name: impl Into<String>) -> CandleAgentRoleBuilder {
        CandleAgentRoleBuilder::new(name)
    }
}

/// Minimal agent role builder
pub struct CandleAgentRoleBuilder {
    name: String,
}

impl CandleAgentRoleBuilder {
    /// Create new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
        }
    }
    
    /// Get the agent name
    pub fn name(&self) -> &str {
        &self.name
    }
}