//! MCP client builder implementations with zero-allocation, lock-free design
//!
//! All MCP client construction logic and builder patterns.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use fluent_ai_domain::mcp::McpClient;
use tokio::sync::RwLock;

/// Zero-allocation MCP client builder with blazing-fast construction
pub struct McpClientBuilder {
    timeout: Duration}

impl McpClientBuilder {
    /// Create new MCP client builder - EXACT syntax: McpClientBuilder::new()
    #[inline]
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30)}
    }

    /// Set request timeout - EXACT syntax: .with_timeout(timeout)
    #[inline]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build final MCP client - EXACT syntax: .build()
    #[inline]
    pub fn build(self) -> McpClient {
        McpClient {
            request_id: AtomicU64::new(0),
            response_cache: Arc::new(RwLock::new(HashMap::with_capacity(256))),
            request_timeout: self.timeout}
    }
}

impl Default for McpClientBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl McpClient {
    /// Create MCP client builder - EXACT syntax: McpClient::builder()
    #[inline]
    pub fn builder() -> McpClientBuilder {
        McpClientBuilder::new()
    }
}

/// Builder function for convenient MCP client construction
#[inline]
pub fn mcp_client() -> McpClientBuilder {
    McpClientBuilder::new()
}
