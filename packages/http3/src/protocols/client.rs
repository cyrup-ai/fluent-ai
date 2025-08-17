//! Unified HTTP client with protocol strategy pattern
//!
//! Provides a single interface for HTTP/2, HTTP/3, and QUIC protocols with
//! automatic fallback and zero-allocation streaming architecture.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fluent_ai_async::prelude::*;

use crate::protocols::core::{HttpProtocol, HttpRequestBuilder, HttpVersion};
use crate::protocols::strategy::{HttpProtocolStrategy, ProtocolConfigs, H2Config, H3Config, QuicheConfig};
use crate::error::HttpClientError;
use crate::protocols::h2::H2Connection;
use crate::protocols::h3::H3Connection;
use crate::protocols::quiche::QuicheConnection;

/// Unified HTTP client supporting multiple protocols with strategy pattern
pub struct HttpClient {
    strategy: HttpProtocolStrategy,
    active_protocol: Box<dyn HttpProtocol>,
    created_at: Instant,
}

impl HttpClient {
    /// Create client with specific strategy
    pub fn new(strategy: HttpProtocolStrategy) -> Result<Self, HttpClientError> {
        let active_protocol = Self::create_protocol(&strategy)?;
        Ok(Self {
            strategy,
            active_protocol,
            created_at: Instant::now(),
        })
    }
    
    /// Create optimized client for AI workloads
    pub fn ai_optimized() -> Result<Self, HttpClientError> {
        Self::new(HttpProtocolStrategy::ai_optimized())
    }
    
    /// Create streaming-optimized client for real-time data
    pub fn streaming_optimized() -> Result<Self, HttpClientError> {
        Self::new(HttpProtocolStrategy::streaming_optimized())
    }
    
    /// Create low-latency client for interactive applications
    pub fn low_latency() -> Result<Self, HttpClientError> {
        Self::new(HttpProtocolStrategy::low_latency())
    }
    
    /// Get request builder using active protocol
    pub fn request(&self) -> impl HttpRequestBuilder {
        self.active_protocol.request_builder()
    }
    
    /// Get active protocol version
    pub fn version(&self) -> HttpVersion {
        self.active_protocol.version()
    }
    
    /// Check if connection is ready for requests
    pub fn is_ready(&self) -> bool {
        self.active_protocol.connection_state().is_ready()
    }
    
    /// Check if connection is closed
    pub fn is_closed(&self) -> bool {
        self.active_protocol.connection_state().is_closed()
    }
    
    /// Get connection state information
    #[inline]
    pub fn connection_state(&self) -> Box<dyn ConnectionState> {
        self.active_protocol.connection_state()
    }

    /// Get statistics snapshot for monitoring and debugging
    #[inline]
    pub fn stats_snapshot(&self) -> ClientStats {
        ClientStats {
            strategy: format!("{:?}", self.strategy),
            protocol_version: self.version(),
            created_at: self.created_at,
            uptime: self.created_at.elapsed(),
            connection_state: format!("{:?}", self.connection_state()),
        }
    }

    /// Create a new HttpClient with direct protocol specification (bypassing strategy)
    #[inline]
    pub fn new_direct(protocol: Box<dyn HttpProtocol>) -> Result<Self, HttpClientError> {
        // Determine strategy from protocol version
        let version = protocol.version();
        let strategy = match version {
            HttpVersion::Http2 => HttpProtocolStrategy::Http2(H2Config::default()),
            HttpVersion::Http3 => HttpProtocolStrategy::Http3(H3Config::default()),
            HttpVersion::Quiche => HttpProtocolStrategy::Quiche(QuicheConfig::default()),
        };

        Ok(HttpClient {
            strategy,
            active_protocol: protocol,
            created_at: Instant::now(),
        })
    }
    
    /// Get connection error if any
    pub fn connection_error(&self) -> Option<&str> {
        self.active_protocol.connection_state().error_message()
    }
    
    /// Get client uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
    
    /// Get current strategy
    pub fn strategy(&self) -> &HttpProtocolStrategy {
        &self.strategy
    }
    
    /// Attempt protocol upgrade or fallback
    pub fn try_protocol_change(&mut self, target_version: HttpVersion) -> Result<(), HttpClientError> {
        let new_strategy = match target_version {
            HttpVersion::Http2 => {
                if let HttpProtocolStrategy::Auto { configs, .. } = &self.strategy {
                    HttpProtocolStrategy::Http2(configs.h2.clone())
                } else {
                    return Err(HttpClientError::UnsupportedProtocolChange);
                }
            }
            HttpVersion::Http3 => {
                if let HttpProtocolStrategy::Auto { configs, .. } = &self.strategy {
                    HttpProtocolStrategy::Http3(configs.h3.clone())
                } else {
                    return Err(HttpClientError::UnsupportedProtocolChange);
                }
            }
            HttpVersion::Http11 => {
                return Err(HttpClientError::UnsupportedProtocol(HttpVersion::Http11));
            }
        };
        
        let new_protocol = Self::create_protocol(&new_strategy)?;
        self.active_protocol = new_protocol;
        self.strategy = new_strategy;
        Ok(())
    }
    
    fn create_protocol(strategy: &HttpProtocolStrategy) -> Result<Box<dyn HttpProtocol>, HttpClientError> {
        match strategy {
            HttpProtocolStrategy::Http2(config) => {
                let connection = H2Connection::with_config(config.clone())
                    .map_err(|e| HttpClientError::ProtocolInitialization(format!("H2: {}", e)))?;
                Ok(Box::new(connection))
            }
            HttpProtocolStrategy::Http3(config) => {
                let connection = H3Connection::with_config(config.clone())
                    .map_err(|e| HttpClientError::ProtocolInitialization(format!("H3: {}", e)))?;
                Ok(Box::new(connection))
            }
            HttpProtocolStrategy::Quiche(config) => {
                let connection = QuicheConnection::with_config(config.clone())
                    .map_err(|e| HttpClientError::ProtocolInitialization(format!("Quiche: {}", e)))?;
                Ok(Box::new(connection))
            }
            HttpProtocolStrategy::Auto { prefer, fallback_chain, configs } => {
                // Try protocols in preference order
                for &version in prefer {
                    if let Ok(protocol) = Self::try_create_protocol_version(version, configs) {
                        return Ok(protocol);
                    }
                }
                
                // Try fallback chain
                for &version in fallback_chain {
                    if let Ok(protocol) = Self::try_create_protocol_version(version, configs) {
                        return Ok(protocol);
                    }
                }
                
                Err(HttpClientError::NoSupportedProtocol)
            }
        }
    }
    
    fn try_create_protocol_version(
        version: HttpVersion,
        configs: &ProtocolConfigs,
    ) -> Result<Box<dyn HttpProtocol>, HttpClientError> {
        match version {
            HttpVersion::Http2 => {
                let connection = H2Connection::with_config(configs.h2.clone())
                    .map_err(|e| HttpClientError::ProtocolInitialization(format!("H2: {}", e)))?;
                Ok(Box::new(connection))
            }
            HttpVersion::Http3 => {
                let connection = H3Connection::with_config(configs.h3.clone())
                    .map_err(|e| HttpClientError::ProtocolInitialization(format!("H3: {}", e)))?;
                Ok(Box::new(connection))
            }
            HttpVersion::Http11 => {
                Err(HttpClientError::UnsupportedProtocol(HttpVersion::Http11))
            }
        }
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::ai_optimized().unwrap_or_else(|_| {
            // Fallback to basic HTTP/2 if AI optimization fails
            Self::new(HttpProtocolStrategy::Http2(Default::default()))
                .expect("Default HTTP/2 client creation should never fail")
        })
    }
}

/// HTTP client errors
#[derive(Debug, thiserror::Error)]
pub enum HttpClientError {
    #[error("Protocol initialization failed: {0}")]
    ProtocolInitialization(String),
    
    #[error("No supported protocol available")]
    NoSupportedProtocol,
    
    #[error("Unsupported protocol: {0:?}")]
    UnsupportedProtocol(HttpVersion),
    
    #[error("Protocol change not supported for current strategy")]
    UnsupportedProtocolChange,
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Request builder factory for unified interface
pub struct UnifiedRequestBuilder {
    client: Arc<HttpClient>,
}

impl UnifiedRequestBuilder {
    pub fn new(client: Arc<HttpClient>) -> Self {
        Self { client }
    }
    
    /// Create request builder using active protocol
    pub fn build(&self) -> impl HttpRequestBuilder {
        self.client.request()
    }
}

/// Protocol selection result for auto-negotiation
#[derive(Debug, Clone)]
pub struct ProtocolSelection {
    pub selected: HttpVersion,
    pub attempted: Vec<HttpVersion>,
    pub selection_time_ms: u64,
}

/// Connection pool manager for protocol-specific connections
pub struct ProtocolConnectionPool {
    h2_connections: Vec<H2Connection>,
    h3_connections: Vec<H3Connection>,
    quiche_connections: Vec<QuicheConnection>,
    max_connections_per_protocol: usize,
}

impl ProtocolConnectionPool {
    pub fn new(max_connections_per_protocol: usize) -> Self {
        Self {
            h2_connections: Vec::with_capacity(max_connections_per_protocol),
            h3_connections: Vec::with_capacity(max_connections_per_protocol),
            quiche_connections: Vec::with_capacity(max_connections_per_protocol),
            max_connections_per_protocol,
        }
    }
    
    pub fn get_or_create_h2(&mut self) -> Result<&mut H2Connection, HttpClientError> {
        if self.h2_connections.is_empty() || self.h2_connections.len() < self.max_connections_per_protocol {
            let connection = H2Connection::new()
                .map_err(|e| HttpClientError::ConnectionFailed(e.to_string()))?;
            self.h2_connections.push(connection);
        }
        
        // Find available connection
        for connection in &mut self.h2_connections {
            if connection.is_ready() {
                return Ok(connection);
            }
        }
        
        // All connections busy, return first one
        self.h2_connections.first_mut()
            .ok_or(HttpClientError::NoSupportedProtocol)
    }
    
    pub fn get_or_create_h3(&mut self) -> Result<&mut H3Connection, HttpClientError> {
        if self.h3_connections.is_empty() || self.h3_connections.len() < self.max_connections_per_protocol {
            let connection = H3Connection::new()
                .map_err(|e| HttpClientError::ConnectionFailed(e.to_string()))?;
            self.h3_connections.push(connection);
        }
        
        // Find available connection
        for connection in &mut self.h3_connections {
            if connection.is_ready() {
                return Ok(connection);
            }
        }
        
        // All connections busy, return first one
        self.h3_connections.first_mut()
            .ok_or(HttpClientError::NoSupportedProtocol)
    }
    
    pub fn cleanup_closed_connections(&mut self) {
        self.h2_connections.retain(|conn| !conn.is_closed());
        self.h3_connections.retain(|conn| !conn.is_closed());
        self.quiche_connections.retain(|conn| !conn.is_closed());
    }
}
