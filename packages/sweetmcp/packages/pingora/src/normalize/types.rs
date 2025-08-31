//! Protocol types and context definitions
//!
//! This module provides core protocol types and context structures
//! for protocol normalization with zero allocation patterns and
//! blazing-fast performance.

use std::collections::HashMap;

use async_graphql::parser::types::{FragmentDefinition, TypeCondition};
use serde::{Deserialize, Serialize};

/// Supported protocol types for normalization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Proto {
    /// GraphQL protocol
    GraphQL,
    /// JSON-RPC 2.0 protocol
    JsonRpc,
    /// Cap'n Proto binary protocol
    Capnp,
    /// MCP Streamable HTTP protocol
    McpStreamableHttp,
}

impl Proto {
    /// Check if protocol is binary
    pub fn is_binary(&self) -> bool {
        matches!(self, Proto::Capnp)
    }

    /// Check if protocol supports streaming
    pub fn supports_streaming(&self) -> bool {
        matches!(self, Proto::McpStreamableHttp)
    }

    /// Get protocol name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Proto::GraphQL => "graphql",
            Proto::JsonRpc => "json-rpc",
            Proto::Capnp => "capnp",
            Proto::McpStreamableHttp => "mcp-streamable-http",
        }
    }

    /// Parse protocol from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "graphql" => Some(Proto::GraphQL),
            "json-rpc" | "jsonrpc" => Some(Proto::JsonRpc),
            "capnp" | "capnproto" => Some(Proto::Capnp),
            "mcp-streamable-http" | "mcp" => Some(Proto::McpStreamableHttp),
            _ => None,
        }
    }

    /// Get default content type for protocol
    pub fn default_content_type(&self) -> &'static str {
        match self {
            Proto::GraphQL => "application/json",
            Proto::JsonRpc => "application/json",
            Proto::Capnp => "application/octet-stream",
            Proto::McpStreamableHttp => "application/json",
        }
    }

    /// Check if protocol requires special handling
    pub fn requires_special_handling(&self) -> bool {
        matches!(self, Proto::GraphQL | Proto::Capnp)
    }
}

/// Context for tracking protocol conversion
#[derive(Debug, Clone)]
pub struct ProtocolContext {
    /// The original protocol type
    pub protocol: Proto,
    /// Original query for GraphQL response shaping
    pub original_query: Option<String>,
    /// Unique request identifier
    pub request_id: String,
    /// Additional metadata for conversion
    pub metadata: ProtocolMetadata,
    /// GraphQL-specific context for fragment resolution
    pub graphql_context: Option<GraphQLContext>,
}

impl ProtocolContext {
    /// Create new protocol context
    pub fn new(protocol: Proto, request_id: String) -> Self {
        Self {
            protocol,
            original_query: None,
            request_id,
            metadata: ProtocolMetadata::default(),
            graphql_context: None,
        }
    }

    /// Create context with original query
    pub fn with_query(protocol: Proto, request_id: String, query: String) -> Self {
        Self {
            protocol,
            original_query: Some(query),
            request_id,
            metadata: ProtocolMetadata::default(),
            graphql_context: None,
        }
    }

    /// Create context with metadata
    pub fn with_metadata(protocol: Proto, request_id: String, metadata: ProtocolMetadata) -> Self {
        Self {
            protocol,
            original_query: None,
            request_id,
            metadata,
            graphql_context: None,
        }
    }

    /// Create context with GraphQL context
    pub fn with_graphql_context(
        protocol: Proto,
        request_id: String,
        graphql_context: GraphQLContext,
    ) -> Self {
        Self {
            protocol,
            original_query: None,
            request_id,
            metadata: ProtocolMetadata::default(),
            graphql_context: Some(graphql_context),
        }
    }

    /// Check if context has original query
    pub fn has_original_query(&self) -> bool {
        self.original_query.is_some()
    }

    /// Get original query reference
    pub fn original_query(&self) -> Option<&str> {
        self.original_query.as_deref()
    }

    /// Set original query
    pub fn set_original_query(&mut self, query: String) {
        self.original_query = Some(query);
    }

    /// Get protocol type
    pub fn protocol(&self) -> &Proto {
        &self.protocol
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Get metadata
    pub fn metadata(&self) -> &ProtocolMetadata {
        &self.metadata
    }

    /// Update metadata
    pub fn set_metadata(&mut self, metadata: ProtocolMetadata) {
        self.metadata = metadata;
    }

    /// Set GraphQL context
    pub fn set_graphql_context(&mut self, graphql_context: GraphQLContext) {
        self.graphql_context = Some(graphql_context);
    }

    /// Get GraphQL context
    pub fn get_graphql_context(&self) -> Option<&GraphQLContext> {
        self.graphql_context.as_ref()
    }

    /// Get mutable GraphQL context
    pub fn get_graphql_context_mut(&mut self) -> Option<&mut GraphQLContext> {
        self.graphql_context.as_mut()
    }

    /// Check if context is valid
    pub fn is_valid(&self) -> bool {
        !self.request_id.is_empty()
    }

    /// Create error context
    pub fn create_error_context(error_msg: &str) -> Self {
        Self {
            protocol: Proto::JsonRpc,
            original_query: None,
            request_id: uuid::Uuid::new_v4().to_string(),
            metadata: ProtocolMetadata {
                error_message: Some(error_msg.to_string()),
                ..Default::default()
            },
            graphql_context: None,
        }
    }
}

/// Additional metadata for protocol conversion
#[derive(Debug, Clone, Default)]
pub struct ProtocolMetadata {
    /// Content type from original request
    pub content_type: Option<String>,
    /// User agent from original request
    pub user_agent: Option<String>,
    /// Custom headers that may affect conversion
    pub custom_headers: std::collections::HashMap<String, String>,
    /// Timestamp when conversion started
    pub conversion_start: Option<std::time::Instant>,
    /// Error message if conversion failed
    pub error_message: Option<String>,
    /// Additional conversion options
    pub options: ConversionOptions,
}

impl ProtocolMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: String) -> Self {
        self.content_type = Some(content_type);
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = Some(user_agent);
        self
    }

    /// Add custom header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.custom_headers.insert(key, value);
        self
    }

    /// Set conversion start time
    pub fn mark_conversion_start(&mut self) {
        self.conversion_start = Some(std::time::Instant::now());
    }

    /// Get conversion duration
    pub fn conversion_duration(&self) -> Option<std::time::Duration> {
        self.conversion_start.map(|start| start.elapsed())
    }

    /// Check if metadata has error
    pub fn has_error(&self) -> bool {
        self.error_message.is_some()
    }

    /// Set error message
    pub fn set_error(&mut self, error: String) {
        self.error_message = Some(error);
    }
}

/// Options for protocol conversion
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Whether to preserve original field names
    pub preserve_field_names: bool,
    /// Whether to validate converted JSON-RPC
    pub validate_jsonrpc: bool,
    /// Maximum depth for nested objects
    pub max_depth: usize,
    /// Whether to include debug information
    pub include_debug_info: bool,
    /// Custom timeout for conversion
    pub timeout_ms: Option<u64>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            preserve_field_names: true,
            validate_jsonrpc: true,
            max_depth: 10,
            include_debug_info: false,
            timeout_ms: Some(5000), // 5 seconds
        }
    }
}

impl ConversionOptions {
    /// Create options for development
    pub fn development() -> Self {
        Self {
            include_debug_info: true,
            timeout_ms: Some(30000), // 30 seconds for debugging
            ..Default::default()
        }
    }

    /// Create options for production
    pub fn production() -> Self {
        Self {
            include_debug_info: false,
            timeout_ms: Some(1000), // 1 second for production
            ..Default::default()
        }
    }

    /// Create options for testing
    pub fn testing() -> Self {
        Self {
            validate_jsonrpc: false, // Skip validation for faster tests
            timeout_ms: Some(10000), // 10 seconds for tests
            ..Default::default()
        }
    }
}

/// Result type for protocol conversion
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Errors that can occur during protocol conversion
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Invalid protocol format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("GraphQL parsing error: {0}")]
    GraphQLError(String),

    #[error("Cap'n Proto error: {0}")]
    CapnProtoError(String),

    #[error("Conversion timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Fragment error: {0}")]
    FragmentError(String),

    #[error("Fragment '{name}' not found")]
    FragmentNotFound { name: String },

    #[error("Circular fragment dependency: {cycle}")]
    CircularFragmentDependency { cycle: String },

    #[error("Fragment type condition error: {0}")]
    TypeConditionError(String),

    #[error("Fragment validation failed: {0}")]
    FragmentValidationError(String),
}

impl ConversionError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            ConversionError::InvalidFormat(_) => false,
            ConversionError::UnsupportedProtocol(_) => false,
            ConversionError::JsonError(_) => false,
            ConversionError::GraphQLError(_) => true,
            ConversionError::CapnProtoError(_) => true,
            ConversionError::Timeout { .. } => true,
            ConversionError::ValidationError(_) => true,
            ConversionError::InternalError(_) => false,
            ConversionError::FragmentError(_) => true,
            ConversionError::FragmentNotFound { .. } => false,
            ConversionError::CircularFragmentDependency { .. } => false,
            ConversionError::TypeConditionError(_) => true,
            ConversionError::FragmentValidationError(_) => true,
        }
    }

    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ConversionError::InvalidFormat(_) => ErrorSeverity::Error,
            ConversionError::UnsupportedProtocol(_) => ErrorSeverity::Warning,
            ConversionError::JsonError(_) => ErrorSeverity::Error,
            ConversionError::GraphQLError(_) => ErrorSeverity::Warning,
            ConversionError::CapnProtoError(_) => ErrorSeverity::Warning,
            ConversionError::Timeout { .. } => ErrorSeverity::Warning,
            ConversionError::ValidationError(_) => ErrorSeverity::Info,
            ConversionError::InternalError(_) => ErrorSeverity::Critical,
            ConversionError::FragmentError(_) => ErrorSeverity::Warning,
            ConversionError::FragmentNotFound { .. } => ErrorSeverity::Error,
            ConversionError::CircularFragmentDependency { .. } => ErrorSeverity::Error,
            ConversionError::TypeConditionError(_) => ErrorSeverity::Warning,
            ConversionError::FragmentValidationError(_) => ErrorSeverity::Warning,
        }
    }

    /// Create JSON-RPC error response
    pub fn to_jsonrpc_error(&self, id: Option<serde_json::Value>) -> serde_json::Value {
        let (code, message) = match self {
            ConversionError::InvalidFormat(msg) => (-32700, format!("Parse error: {}", msg)),
            ConversionError::UnsupportedProtocol(proto) => (
                -32601,
                format!("Method not found: unsupported protocol {}", proto),
            ),
            ConversionError::JsonError(e) => (-32700, format!("Parse error: {}", e)),
            ConversionError::GraphQLError(msg) => (-32602, format!("Invalid params: {}", msg)),
            ConversionError::CapnProtoError(msg) => (-32602, format!("Invalid params: {}", msg)),
            ConversionError::Timeout { timeout_ms } => (
                -32603,
                format!("Internal error: timeout after {}ms", timeout_ms),
            ),
            ConversionError::ValidationError(msg) => (-32602, format!("Invalid params: {}", msg)),
            ConversionError::InternalError(msg) => (-32603, format!("Internal error: {}", msg)),
            ConversionError::FragmentError(msg) => (-32602, format!("Fragment error: {}", msg)),
            ConversionError::FragmentNotFound { name } => {
                (-32602, format!("Fragment '{}' not found", name))
            }
            ConversionError::CircularFragmentDependency { cycle } => {
                (-32602, format!("Circular fragment dependency: {}", cycle))
            }
            ConversionError::TypeConditionError(msg) => {
                (-32602, format!("Fragment type condition error: {}", msg))
            }
            ConversionError::FragmentValidationError(msg) => {
                (-32602, format!("Fragment validation failed: {}", msg))
            }
        };

        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id.unwrap_or(serde_json::Value::Null),
            "error": {
                "code": code,
                "message": message
            }
        })
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Protocol detection result
#[derive(Debug, Clone)]
pub struct ProtocolDetection {
    /// Detected protocol
    pub protocol: Proto,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Detection method used
    pub method: DetectionMethod,
}

/// Methods used for protocol detection
#[derive(Debug, Clone, PartialEq)]
pub enum DetectionMethod {
    /// Detected by content type header
    ContentType,
    /// Detected by request structure
    Structure,
    /// Detected by user agent
    UserAgent,
    /// Detected by URL path
    UrlPath,
    /// Default fallback detection
    Fallback,
}

impl ProtocolDetection {
    /// Create new detection result
    pub fn new(protocol: Proto, confidence: f64, method: DetectionMethod) -> Self {
        Self {
            protocol,
            confidence,
            method,
        }
    }

    /// Check if detection is confident
    pub fn is_confident(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Check if detection is uncertain
    pub fn is_uncertain(&self) -> bool {
        self.confidence < 0.5
    }
}

/// Fragment registry for storing and resolving GraphQL fragment definitions
#[derive(Debug, Clone)]
pub struct FragmentRegistry {
    fragments: HashMap<String, FragmentDefinition>,
    resolution_stack: Vec<String>, // For circular dependency detection
}

impl FragmentRegistry {
    /// Create a new fragment registry
    pub fn new() -> Self {
        Self {
            fragments: HashMap::new(),
            resolution_stack: Vec::new(),
        }
    }

    /// Register a fragment definition
    pub fn register_fragment(
        &mut self,
        name: String,
        definition: FragmentDefinition,
    ) -> Result<(), ConversionError> {
        if self.fragments.contains_key(&name) {
            return Err(ConversionError::FragmentValidationError(format!(
                "Fragment '{}' is already defined",
                name
            )));
        }
        self.fragments.insert(name, definition);
        Ok(())
    }

    /// Get a fragment definition by name
    pub fn get_fragment(&self, name: &str) -> Option<&FragmentDefinition> {
        self.fragments.get(name)
    }

    /// Validate that no circular dependencies exist
    pub fn validate_no_cycles(&self, fragment_name: &str) -> Result<(), ConversionError> {
        if self.resolution_stack.contains(&fragment_name.to_string()) {
            let cycle = format!(
                "{} -> {}",
                self.resolution_stack.join(" -> "),
                fragment_name
            );
            return Err(ConversionError::CircularFragmentDependency { cycle });
        }
        Ok(())
    }

    /// Push fragment name onto resolution stack for cycle detection
    pub fn push_resolution(&mut self, name: String) {
        self.resolution_stack.push(name);
    }

    /// Pop fragment name from resolution stack
    pub fn pop_resolution(&mut self) {
        self.resolution_stack.pop();
    }

    /// Get all registered fragment names
    pub fn get_fragment_names(&self) -> Vec<String> {
        self.fragments.keys().cloned().collect()
    }

    /// Check if a fragment is registered
    pub fn has_fragment(&self, name: &str) -> bool {
        self.fragments.contains_key(name)
    }

    /// Clear all fragments and reset state
    pub fn clear(&mut self) {
        self.fragments.clear();
        self.resolution_stack.clear();
    }
}

/// Fragment cache for performance optimization
#[derive(Debug, Clone)]
pub struct FragmentCache {
    resolved_fields: HashMap<String, Vec<String>>,
    cache_hits: u64,
    cache_misses: u64,
}

impl FragmentCache {
    /// Create a new fragment cache
    pub fn new() -> Self {
        Self {
            resolved_fields: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get cached fields for a fragment
    pub fn get(&mut self, fragment_name: &str) -> Option<&Vec<String>> {
        if let Some(fields) = self.resolved_fields.get(fragment_name) {
            self.cache_hits += 1;
            Some(fields)
        } else {
            self.cache_misses += 1;
            None
        }
    }

    /// Cache resolved fields for a fragment
    pub fn insert(&mut self, fragment_name: String, fields: Vec<String>) {
        self.resolved_fields.insert(fragment_name, fields);
    }

    /// Get cache hit rate for performance monitoring
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> (u64, u64, f64) {
        (self.cache_hits, self.cache_misses, self.cache_hit_rate())
    }

    /// Clear cache and reset statistics
    pub fn clear(&mut self) {
        self.resolved_fields.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

/// GraphQL-specific context for fragment resolution
#[derive(Debug, Clone)]
pub struct GraphQLContext {
    pub fragment_registry: FragmentRegistry,
    pub fragment_cache: FragmentCache,
    pub type_info: Option<String>, // Current type context for validation
    pub schema_types: std::collections::HashMap<String, GraphQLTypeInfo>, // Schema type information
}

/// GraphQL type information for validation
#[derive(Debug, Clone)]
pub struct GraphQLTypeInfo {
    pub name: String,
    pub kind: GraphQLTypeKind,
    pub fields: Vec<String>,
    pub interfaces: Vec<String>,
    pub possible_types: Vec<String>,
}

/// GraphQL type kinds
#[derive(Debug, Clone, PartialEq)]
pub enum GraphQLTypeKind {
    Object,
    Interface,
    Union,
    Scalar,
    Enum,
    InputObject,
}

impl GraphQLContext {
    /// Create new GraphQL context
    pub fn new() -> Self {
        Self {
            fragment_registry: FragmentRegistry::new(),
            fragment_cache: FragmentCache::new(),
            type_info: None,
            schema_types: std::collections::HashMap::new(),
        }
    }

    /// Create context with type information
    pub fn with_type_info(type_info: String) -> Self {
        Self {
            fragment_registry: FragmentRegistry::new(),
            fragment_cache: FragmentCache::new(),
            type_info: Some(type_info),
            schema_types: std::collections::HashMap::new(),
        }
    }

    /// Create context with schema information
    pub fn with_schema(schema_types: std::collections::HashMap<String, GraphQLTypeInfo>) -> Self {
        Self {
            fragment_registry: FragmentRegistry::new(),
            fragment_cache: FragmentCache::new(),
            type_info: None,
            schema_types,
        }
    }

    /// Set current type context
    pub fn set_type_info(&mut self, type_info: String) {
        self.type_info = Some(type_info);
    }

    /// Get current type context
    pub fn get_type_info(&self) -> Option<&str> {
        self.type_info.as_deref()
    }

    /// Add type information to schema
    pub fn add_type(&mut self, type_info: GraphQLTypeInfo) {
        self.schema_types.insert(type_info.name.clone(), type_info);
    }

    /// Get type information from schema
    pub fn get_type(&self, type_name: &str) -> Option<&GraphQLTypeInfo> {
        self.schema_types.get(type_name)
    }

    /// Check if type exists in schema
    pub fn has_type(&self, type_name: &str) -> bool {
        self.schema_types.contains_key(type_name)
    }

    /// Validate type compatibility for fragments
    pub fn is_type_compatible(&self, fragment_type: &str, target_type: &str) -> bool {
        // Same type is always compatible
        if fragment_type == target_type {
            return true;
        }

        // Check if fragment type is an interface/union that target type implements
        if let Some(fragment_info) = self.get_type(fragment_type) {
            if let Some(target_info) = self.get_type(target_type) {
                match fragment_info.kind {
                    GraphQLTypeKind::Interface => {
                        // Target type must implement this interface
                        target_info.interfaces.contains(&fragment_type.to_string())
                    }
                    GraphQLTypeKind::Union => {
                        // Target type must be a possible type of this union
                        fragment_info
                            .possible_types
                            .contains(&target_type.to_string())
                    }
                    GraphQLTypeKind::Object => {
                        // Object types are only compatible with themselves
                        false
                    }
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }
}
