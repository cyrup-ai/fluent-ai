// Define traits locally - no external dependencies
// use serde_json::Value; // Commented out - unused
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ZeroOneOrMany;

/// A workflow step that can be stored and executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Unique identifier for this workflow step
    pub id: String,
    /// Human-readable name for this step
    pub name: String,
    /// Detailed description of what this step does
    pub description: String,
    /// The type and configuration of this step
    pub step_type: StepType,
    /// JSON parameters specific to this step type
    pub parameters: serde_json::Value,
    /// IDs of steps that must complete before this step can execute
    pub dependencies: ZeroOneOrMany<String>,
}

/// Defines the different types of workflow steps and their configurations.
///
/// Each step type has specific parameters that control its execution behavior.
/// The enum uses serde's tag attribute for clean JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StepType {
    /// A prompt step that generates text using a template
    Prompt {
        /// The template string with placeholders for dynamic content
        template: String,
    },
    /// A transformation step that processes data using a function
    Transform {
        /// The name or definition of the transformation function to apply
        function: String,
    },
    /// A conditional step that branches based on a condition
    Conditional {
        /// The condition expression to evaluate
        condition: String,
        /// The step ID to execute if the condition is true
        true_branch: String,
        /// The step ID to execute if the condition is false
        false_branch: String,
    },
    /// A parallel step that executes multiple branches concurrently
    Parallel {
        /// The step IDs to execute in parallel
        branches: ZeroOneOrMany<String>,
    },
    /// A loop step that repeats execution while a condition is true
    Loop {
        /// The condition expression that controls loop continuation
        condition: String,
        /// The step ID to execute in each loop iteration
        body: String,
    },
}

/// A complete workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Unique identifier for this workflow
    pub id: String,
    /// Human-readable name for this workflow
    pub name: String,
    /// Detailed description of what this workflow accomplishes
    pub description: String,
    /// All the steps that make up this workflow
    pub steps: ZeroOneOrMany<WorkflowStep>,
    /// ID of the first step to execute when starting this workflow
    pub entry_point: String,
    /// Additional metadata and configuration for this workflow
    pub metadata: HashMap<String, serde_json::Value>,
}
