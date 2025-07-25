//! SecureMcpTool builder implementations
//!
//! All secure MCP tool construction logic and builder patterns.

use fluent_ai_cylo::{Cylo, CyloInstance};
use fluent_ai_domain::secure_mcp_tool::SecureMcpTool;
use serde_json::Value;

/// Zero-allocation builder for high-performance MCP tools
///
/// Provides fluent API for creating MCP tools with container execution,
/// optimal resource configuration, and intelligent backend selection.
#[derive(Debug, Default)]
pub struct SecureMcpToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
    server: Option<String>,
    cylo_instance: Option<CyloInstance>,
    timeout_seconds: u64,
    memory_limit: Option<u64>,
    cpu_limit: Option<u32>}

impl SecureMcpToolBuilder {
    /// Create new builder with optimal defaults
    #[inline]
    pub fn new() -> Self {
        Self {
            timeout_seconds: 30,
            memory_limit: Some(512 * 1024 * 1024), // 512MB
            cpu_limit: Some(1),
            ..Default::default()
        }
    }

    /// Set tool identifier
    #[inline]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set tool description
    #[inline]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set parameter schema
    #[inline]
    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    /// Set MCP server identifier
    #[inline]
    pub fn server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }

    /// Set explicit Cylo container environment
    ///
    /// # Arguments
    /// * `instance` - Cylo container instance for execution
    #[inline]
    pub fn cylo(mut self, instance: CyloInstance) -> Self {
        self.cylo_instance = Some(instance);
        self
    }

    /// Use Apple containerization (macOS with Apple Silicon)
    ///
    /// # Arguments
    /// * `image` - Container image specification
    /// * `instance_name` - Named instance identifier
    #[inline]
    pub fn apple_container(
        mut self,
        image: impl Into<String>,
        instance_name: impl Into<String>,
    ) -> Self {
        let cylo_env = Cylo::Apple(image.into());
        self.cylo_instance = Some(cylo_env.instance(instance_name.into()));
        self
    }

    /// Use LandLock sandboxing (Linux with kernel security)
    ///
    /// # Arguments
    /// * `jail_path` - Sandbox directory path
    /// * `instance_name` - Named instance identifier
    #[inline]
    pub fn landlock_sandbox(
        mut self,
        jail_path: impl Into<String>,
        instance_name: impl Into<String>,
    ) -> Self {
        let cylo_env = Cylo::LandLock(jail_path.into());
        self.cylo_instance = Some(cylo_env.instance(instance_name.into()));
        self
    }

    /// Use FireCracker microVM (ultra-lightweight virtualization)
    ///
    /// # Arguments
    /// * `image` - Container image specification
    /// * `instance_name` - Named instance identifier
    #[inline]
    pub fn firecracker_vm(
        mut self,
        image: impl Into<String>,
        instance_name: impl Into<String>,
    ) -> Self {
        let cylo_env = Cylo::FireCracker(image.into());
        self.cylo_instance = Some(cylo_env.instance(instance_name.into()));
        self
    }

    /// Set execution timeout
    ///
    /// # Arguments
    /// * `seconds` - Maximum execution time
    #[inline]
    pub fn timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Set execution timeout - EXACT syntax: .with_timeout(seconds)
    #[inline]
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Set memory limit in bytes
    ///
    /// # Arguments
    /// * `bytes` - Maximum memory usage
    #[inline]
    pub fn memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    /// Set memory limit - EXACT syntax: .with_memory_limit(bytes)
    #[inline]
    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    /// Set memory limit in megabytes (convenience method)
    ///
    /// # Arguments
    /// * `mb` - Maximum memory in megabytes
    #[inline]
    pub fn memory_mb(mut self, mb: u64) -> Self {
        self.memory_limit = Some(mb * 1024 * 1024);
        self
    }

    /// Set CPU core limit
    ///
    /// # Arguments
    /// * `cores` - Maximum number of CPU cores
    #[inline]
    pub fn cpu_limit(mut self, cores: u32) -> Self {
        self.cpu_limit = Some(cores);
        self
    }

    /// Set CPU core limit - EXACT syntax: .with_cpu_limit(cores)
    #[inline]
    pub fn with_cpu_limit(mut self, cores: u32) -> Self {
        self.cpu_limit = Some(cores);
        self
    }

    /// Build the MCP tool with container execution
    ///
    /// # Returns
    /// Configured MCP tool ready for container-based execution
    pub fn build(self) -> SecureMcpTool {
        let name = self.name.unwrap_or_else(|| "container_tool".to_string());
        let description = self
            .description
            .unwrap_or_else(|| "Container-based tool execution".to_string());
        let parameters = self.parameters.unwrap_or_else(|| serde_json::json!({}));

        let mut tool = match self.server {
            Some(server) => SecureMcpTool::with_server(name, description, parameters, server),
            None => SecureMcpTool::new(name, description, parameters)};

        // Set the configured values
        tool.set_cylo_instance(self.cylo_instance);
        tool.set_timeout_seconds(self.timeout_seconds);
        tool.set_memory_limit(self.memory_limit);
        tool.set_cpu_limit(self.cpu_limit);

        tool
    }
}

/// Builder for fluent tool creation
///
/// # Returns
/// New tool builder with optimal defaults
#[inline]
pub fn tool_builder() -> SecureMcpToolBuilder {
    SecureMcpToolBuilder::new()
}
