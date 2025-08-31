//! Protocol normalization module
//!
//! This module provides comprehensive protocol normalization functionality
//! for converting GraphQL, JSON-RPC, Cap'n Proto, and MCP Streamable HTTP
//! protocols with zero allocation patterns and blazing-fast performance.

pub mod conversion;
pub mod parsers;
pub mod schema_introspection;
pub mod types;

#[cfg(test)]
mod schema_introspection_test;

#[cfg(test)]
mod integration_test;

// Re-export key types and functions for ergonomic usage
pub use conversion::{
    create_error_response, detect_protocol, from_json_rpc, to_json_rpc, to_json_rpc_with_headers,
    validate_json_rpc,
};
pub use parsers::{
    capnp_from_json_rpc, capnp_to_json_rpc, create_graphql_error, extract_operation_name,
    extract_operation_type, graphql_from_json_rpc, graphql_to_json_rpc, parse_capnp_message,
    parse_graphql_variables, validate_capnp_format, validate_graphql_query,
};
pub use types::{
    ConversionError, ConversionOptions, ConversionResult, DetectionMethod, ErrorSeverity, Proto,
    ProtocolContext, ProtocolDetection, ProtocolMetadata,
};

/// Convenience function to normalize any protocol to JSON-RPC
pub fn normalize_to_jsonrpc(
    user: &str,
    body: &[u8],
    headers: Option<&pingora::http::RequestHeader>,
) -> anyhow::Result<(ProtocolContext, serde_json::Value)> {
    to_json_rpc_with_headers(user, body, headers)
}

/// Convenience function to convert JSON-RPC response back to original protocol
pub fn denormalize_from_jsonrpc(
    ctx: &ProtocolContext,
    response: &serde_json::Value,
) -> ConversionResult<Vec<u8>> {
    from_json_rpc(ctx, response)
}

/// Quick protocol detection without full conversion
pub fn quick_detect_protocol(
    body: &[u8],
    headers: Option<&pingora::http::RequestHeader>,
) -> ConversionResult<Proto> {
    let detection = detect_protocol(body, headers)?;
    Ok(detection.protocol)
}

/// Create a protocol context for testing
pub fn test_context(protocol: Proto) -> ProtocolContext {
    ProtocolContext::new(protocol, uuid::Uuid::new_v4().to_string())
}
