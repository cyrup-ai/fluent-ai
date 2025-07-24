//! Core integration types and utilities
//!
//! This module provides fundamental data structures for integration management
//! with zero-allocation streaming patterns and lock-free operations.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Configuration for external integrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Integration ID
    pub id: Uuid,
    /// Integration name
    pub name: Arc<str>,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Integration endpoint URL
    pub endpoint: String,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Integration settings
    pub settings: HashMap<String, serde_json::Value>,
    /// Enabled flag
    pub enabled: bool,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Timeout configuration
    pub timeout_config: TimeoutConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
    /// Integration metadata
    pub metadata: HashMap<String, String>,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Type of integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationType {
    /// Webhook integration
    Webhook,
    /// REST API integration
    RestApi,
    /// GraphQL integration
    GraphQL,
    /// Plugin integration
    Plugin,
    /// Database integration
    Database,
    /// Message queue integration
    MessageQueue,
    /// File system integration
    FileSystem,
    /// Custom integration
    Custom(String),
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Authentication credentials
    pub credentials: HashMap<String, String>,
    /// Token refresh settings
    pub token_refresh: Option<TokenRefreshConfig>,
    /// Authentication scope
    pub scope: Vec<String>,
    /// Custom auth parameters
    pub custom_params: HashMap<String, String>,
}

/// Authentication type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// Bearer token authentication
    Bearer,
    /// Basic authentication
    Basic,
    /// OAuth 2.0 authentication
    OAuth2,
    /// JWT authentication
    JWT,
    /// Custom authentication
    Custom(String),
}

/// Token refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshConfig {
    /// Refresh endpoint URL
    pub refresh_url: String,
    /// Refresh token field name
    pub refresh_token_field: String,
    /// Access token field name
    pub access_token_field: String,
    /// Token expiry buffer in seconds
    pub expiry_buffer_seconds: u64,
    /// Auto-refresh enabled
    pub auto_refresh: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second limit
    pub requests_per_second: f64,
    /// Burst limit
    pub burst_limit: usize,
    /// Rate limit window in seconds
    pub window_seconds: u64,
    /// Rate limit strategy
    pub strategy: RateLimitStrategy,
    /// Custom rate limit rules
    pub custom_rules: Vec<RateLimitRule>,
}

/// Rate limiting strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitStrategy {
    /// Token bucket algorithm
    TokenBucket,
    /// Sliding window algorithm
    SlidingWindow,
    /// Fixed window algorithm
    FixedWindow,
    /// Leaky bucket algorithm
    LeakyBucket,
}

/// Custom rate limit rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Rate limit for this rule
    pub limit: f64,
    /// Rule priority
    pub priority: i32,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection timeout in seconds
    pub connection_timeout_seconds: u64,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Read timeout in seconds
    pub read_timeout_seconds: u64,
    /// Write timeout in seconds
    pub write_timeout_seconds: u64,
    /// Keep-alive timeout in seconds
    pub keep_alive_timeout_seconds: u64,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable jitter in retry delays
    pub enable_jitter: bool,
    /// Retry on specific HTTP status codes
    pub retry_on_status_codes: Vec<u16>,
    /// Retry on timeout
    pub retry_on_timeout: bool,
}

/// Integration request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationRequest {
    /// Request ID
    pub id: Uuid,
    /// Integration ID
    pub integration_id: Uuid,
    /// Request method
    pub method: HttpMethod,
    /// Request path
    pub path: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body
    pub body: Option<serde_json::Value>,
    /// Query parameters
    pub query_params: HashMap<String, String>,
    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request timeout
    pub timeout_seconds: Option<u64>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// HTTP method enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET method
    GET,
    /// POST method
    POST,
    /// PUT method
    PUT,
    /// PATCH method
    PATCH,
    /// DELETE method
    DELETE,
    /// HEAD method
    HEAD,
    /// OPTIONS method
    OPTIONS,
}

/// Integration response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResponse {
    /// Response ID
    pub id: Uuid,
    /// Request ID this response corresponds to
    pub request_id: Uuid,
    /// HTTP status code
    pub status_code: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body
    pub body: Option<serde_json::Value>,
    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Response duration in milliseconds
    pub duration_ms: u64,
    /// Response size in bytes
    pub size_bytes: usize,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Integration ID
    pub integration_id: Uuid,
    /// Total requests made
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total data transferred in bytes
    pub total_bytes_transferred: usize,
    /// Last request timestamp
    pub last_request: Option<chrono::DateTime<chrono::Utc>>,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Requests per minute
    pub requests_per_minute: f64,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// External integration definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalIntegration {
    /// Integration configuration
    pub config: IntegrationConfig,
    /// Integration statistics
    pub stats: IntegrationStats,
    /// Connection status
    pub status: IntegrationStatus,
    /// Last health check
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Integration capabilities
    pub capabilities: Vec<IntegrationCapability>,
    /// Custom properties
    pub custom_properties: HashMap<String, serde_json::Value>,
}

/// Integration status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IntegrationStatus {
    /// Integration is active and healthy
    Active,
    /// Integration is inactive
    Inactive,
    /// Integration is experiencing issues
    Degraded,
    /// Integration has failed
    Failed,
    /// Integration is being initialized
    Initializing,
    /// Integration is being updated
    Updating,
}

/// Integration capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationCapability {
    /// Can send data
    Send,
    /// Can receive data
    Receive,
    /// Can stream data
    Stream,
    /// Can batch operations
    Batch,
    /// Can handle webhooks
    Webhook,
    /// Can authenticate
    Auth,
    /// Can cache responses
    Cache,
    /// Custom capability
    Custom(String),
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: Arc::from("Default Integration"),
            integration_type: IntegrationType::RestApi,
            endpoint: "https://api.example.com".to_string(),
            auth_config: AuthConfig::default(),
            settings: HashMap::new(),
            enabled: true,
            rate_limit: RateLimitConfig::default(),
            timeout_config: TimeoutConfig::default(),
            retry_config: RetryConfig::default(),
            custom_headers: HashMap::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            auth_type: AuthType::None,
            credentials: HashMap::new(),
            token_refresh: None,
            scope: Vec::new(),
            custom_params: HashMap::new(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 10.0,
            burst_limit: 20,
            window_seconds: 60,
            strategy: RateLimitStrategy::TokenBucket,
            custom_rules: Vec::new(),
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connection_timeout_seconds: 10,
            request_timeout_seconds: 30,
            read_timeout_seconds: 30,
            write_timeout_seconds: 30,
            keep_alive_timeout_seconds: 60,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            enable_jitter: true,
            retry_on_status_codes: vec![429, 500, 502, 503, 504],
            retry_on_timeout: true,
        }
    }
}

impl Default for IntegrationStats {
    fn default() -> Self {
        Self {
            integration_id: Uuid::new_v4(),
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            total_bytes_transferred: 0,
            last_request: None,
            error_rate: 0.0,
            requests_per_minute: 0.0,
            performance_metrics: HashMap::new(),
        }
    }
}

impl Default for ExternalIntegration {
    fn default() -> Self {
        Self {
            config: IntegrationConfig::default(),
            stats: IntegrationStats::default(),
            status: IntegrationStatus::Inactive,
            last_health_check: None,
            health_check_interval_seconds: 300,
            capabilities: vec![IntegrationCapability::Send, IntegrationCapability::Receive],
            custom_properties: HashMap::new(),
        }
    }
}