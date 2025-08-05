//! Command metadata and resource tracking
//!
//! This module provides command information structures and resource usage tracking
//! for performance monitoring and help generation.

use serde::{Deserialize, Serialize};
use super::parameters::ParameterInfo;

/// Command information for command registry with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandInfo {
    /// Command name
    pub name: String,
    /// Command description
    pub description: String,
    /// Usage string
    pub usage: String,
    /// Command parameters
    pub parameters: Vec<ParameterInfo>,
    /// Command aliases
    pub aliases: Vec<String>,
    /// Command category
    pub category: String,
    /// Usage examples
    pub examples: Vec<String>,
}

impl CommandInfo {
    /// Create a new command info
    pub fn new(name: impl Into<String>, description: impl Into<String>, usage: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            usage: usage.into(),
            parameters: Vec::new(),
            aliases: Vec::new(),
            category: "General".to_string(),
            examples: Vec::new(),
        }
    }

    /// Add a parameter to this command
    pub fn with_parameter(mut self, parameter: ParameterInfo) -> Self {
        self.parameters.push(parameter);
        self
    }

    /// Add multiple parameters
    pub fn with_parameters(mut self, parameters: Vec<ParameterInfo>) -> Self {
        self.parameters.extend(parameters);
        self
    }

    /// Add an alias
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add multiple aliases
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases.extend(aliases);
        self
    }

    /// Set the category
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Add an example
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Add multiple examples
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples.extend(examples);
        self
    }

    /// Check if command has a specific alias
    pub fn has_alias(&self, alias: &str) -> bool {
        self.aliases.iter().any(|a| a == alias)
    }

    /// Get all names (primary name + aliases)
    pub fn all_names(&self) -> Vec<String> {
        let mut names = vec![self.name.clone()];
        names.extend(self.aliases.clone());
        names
    }
}

/// Resource usage tracking for command execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in microseconds
    pub cpu_time_us: u64,
    /// Number of network requests made
    pub network_requests: u32,
    /// Number of disk operations performed
    pub disk_operations: u32,
    /// Peak memory usage
    pub peak_memory_bytes: u64,
    /// Total execution time in microseconds
    pub total_time_us: u64,
}

impl ResourceUsage {
    /// Create new empty resource usage
    pub fn new() -> Self {
        Self::default()
    }

    /// Add memory usage
    pub fn add_memory(&mut self, bytes: u64) {
        self.memory_bytes = self.memory_bytes.saturating_add(bytes);
        self.peak_memory_bytes = self.peak_memory_bytes.max(self.memory_bytes);
    }

    /// Add CPU time
    pub fn add_cpu_time(&mut self, microseconds: u64) {
        self.cpu_time_us = self.cpu_time_us.saturating_add(microseconds);
    }

    /// Add network request
    pub fn add_network_request(&mut self) {
        self.network_requests = self.network_requests.saturating_add(1);
    }

    /// Add disk operation
    pub fn add_disk_operation(&mut self) {
        self.disk_operations = self.disk_operations.saturating_add(1);
    }

    /// Set total execution time
    pub fn set_total_time(&mut self, microseconds: u64) {
        self.total_time_us = microseconds;
    }

    /// Get formatted memory usage
    pub fn formatted_memory(&self) -> String {
        if self.memory_bytes > 1_048_576 {
            format!("{:.2} MB", self.memory_bytes as f64 / 1_048_576.0)
        } else if self.memory_bytes > 1024 {
            format!("{:.2} KB", self.memory_bytes as f64 / 1024.0)
        } else {
            format!("{} bytes", self.memory_bytes)
        }
    }

    /// Get formatted execution time
    pub fn formatted_time(&self) -> String {
        if self.total_time_us > 1_000_000 {
            format!("{:.2}s", self.total_time_us as f64 / 1_000_000.0)
        } else if self.total_time_us > 1000 {
            format!("{:.2}ms", self.total_time_us as f64 / 1000.0)
        } else {
            format!("{}μs", self.total_time_us)
        }
    }
}