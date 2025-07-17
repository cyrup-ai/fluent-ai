use crate::domain::secure_mcp_tool::SecureMcpTool;
use serde_json::Value;

/// Builder for SecureMcpTool objects
pub struct SecureMcpToolBuilder {
    name: String,
    description: String,
    parameters: Value,
    server: Option<String>,
    timeout_seconds: u64,
    memory_limit: Option<u64>,
    cpu_limit: Option<u32>,
}

impl SecureMcpToolBuilder {
    /// Create a new SecureMcpToolBuilder
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Value::Object(Default::default()),
            server: None,
            timeout_seconds: 30,
            memory_limit: Some(512 * 1024 * 1024), // 512MB default
            cpu_limit: Some(1),
        }
    }

    /// Set the parameters schema
    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set the server identifier
    pub fn server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }

    /// Set the timeout in seconds
    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = timeout;
        self
    }

    /// Set the memory limit in bytes
    pub fn memory_limit(mut self, limit: u64) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Set the CPU core limit
    pub fn cpu_limit(mut self, limit: u32) -> Self {
        self.cpu_limit = Some(limit);
        self
    }

    /// Build the SecureMcpTool object
    pub fn build(self) -> SecureMcpTool {
        if let Some(server) = self.server {
            SecureMcpTool::with_server(self.name, self.description, self.parameters, server)
                .with_timeout(self.timeout_seconds)
                .with_memory_limit(self.memory_limit.unwrap_or(512 * 1024 * 1024))
                .with_cpu_limit(self.cpu_limit.unwrap_or(1))
        } else {
            SecureMcpTool::new(self.name, self.description, self.parameters)
                .with_timeout(self.timeout_seconds)
                .with_memory_limit(self.memory_limit.unwrap_or(512 * 1024 * 1024))
                .with_cpu_limit(self.cpu_limit.unwrap_or(1))
        }
    }
}

impl SecureMcpTool {
    /// Create a builder for SecureMcpTool
    pub fn builder(name: impl Into<String>, description: impl Into<String>) -> SecureMcpToolBuilder {
        SecureMcpToolBuilder::new(name, description)
    }
}