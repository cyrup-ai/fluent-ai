//! Protocol detection and normalization to JSON-RPC
//!
//! This module provides the main conversion logic to normalize various
//! protocols to JSON-RPC with zero allocation patterns and blazing-fast
//! performance.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use sweetmcp_axum::JSONRPC_VERSION;
use tracing::{debug, warn};
use uuid::Uuid;

use super::types::{
    ConversionError, ConversionResult, DetectionMethod, Proto, ProtocolContext, ProtocolDetection,
    ProtocolMetadata,
};

/// Normalize incoming protocol to JSON-RPC for cyrup-mcp-api
pub fn to_json_rpc(_user: &str, body: &[u8]) -> Result<(ProtocolContext, Value)> {
    to_json_rpc_with_headers(_user, body, None)
}

/// Normalize incoming protocol to JSON-RPC with optional header context
pub fn to_json_rpc_with_headers(
    _user: &str,
    body: &[u8],
    req_header: Option<&pingora::http::RequestHeader>,
) -> Result<(ProtocolContext, Value)> {
    let detection = detect_protocol(body, req_header)?;
    let request_id = generate_request_id();

    debug!(
        "Detected protocol: {:?} with confidence: {}",
        detection.protocol, detection.confidence
    );

    match detection.protocol {
        Proto::JsonRpc => handle_json_rpc(body, request_id),
        Proto::McpStreamableHttp => handle_mcp_streamable_http(body, request_id),
        Proto::GraphQL => handle_graphql(body, request_id),
        Proto::Capnp => handle_capnp(body, request_id),
    }
}

/// Detect protocol from request body and headers
pub fn detect_protocol(
    body: &[u8],
    req_header: Option<&pingora::http::RequestHeader>,
) -> ConversionResult<ProtocolDetection> {
    // Try JSON-RPC first (most specific)
    if let Ok(v) = serde_json::from_slice::<Value>(body) {
        if v.get("jsonrpc").is_some() {
            return Ok(ProtocolDetection::new(
                Proto::JsonRpc,
                1.0,
                DetectionMethod::Structure,
            ));
        }

        // Check for MCP Streamable HTTP
        if is_mcp_streamable_http(&v) {
            return Ok(ProtocolDetection::new(
                Proto::McpStreamableHttp,
                0.9,
                DetectionMethod::Structure,
            ));
        }

        // Check for GraphQL
        if is_graphql_query(&v) {
            return Ok(ProtocolDetection::new(
                Proto::GraphQL,
                0.8,
                DetectionMethod::Structure,
            ));
        }
    }

    // Check headers for protocol hints
    if let Some(header) = req_header {
        if let Some(detection) = detect_from_headers(header) {
            return Ok(detection);
        }
    }

    // Try Cap'n Proto (binary format)
    if is_capnp_binary(body) {
        return Ok(ProtocolDetection::new(
            Proto::Capnp,
            0.7,
            DetectionMethod::Structure,
        ));
    }

    // Default fallback to JSON-RPC
    Ok(ProtocolDetection::new(
        Proto::JsonRpc,
        0.3,
        DetectionMethod::Fallback,
    ))
}

/// Detect protocol from HTTP headers
fn detect_from_headers(header: &pingora::http::RequestHeader) -> Option<ProtocolDetection> {
    // Check Content-Type header
    if let Some(content_type) = header.headers.get("content-type") {
        if let Ok(ct_str) = content_type.to_str() {
            if ct_str.contains("application/graphql") {
                return Some(ProtocolDetection::new(
                    Proto::GraphQL,
                    0.9,
                    DetectionMethod::ContentType,
                ));
            }
            if ct_str.contains("application/capnp") {
                return Some(ProtocolDetection::new(
                    Proto::Capnp,
                    0.9,
                    DetectionMethod::ContentType,
                ));
            }
        }
    }

    // Check User-Agent
    if let Some(user_agent) = header.headers.get("user-agent") {
        if let Ok(ua_str) = user_agent.to_str() {
            if ua_str.contains("GraphQL") {
                return Some(ProtocolDetection::new(
                    Proto::GraphQL,
                    0.6,
                    DetectionMethod::UserAgent,
                ));
            }
            if ua_str.contains("MCP") {
                return Some(ProtocolDetection::new(
                    Proto::McpStreamableHttp,
                    0.7,
                    DetectionMethod::UserAgent,
                ));
            }
        }
    }

    // Check URL path
    let path = header.uri.path();
    if path.contains("/graphql") {
        return Some(ProtocolDetection::new(
            Proto::GraphQL,
            0.8,
            DetectionMethod::UrlPath,
        ));
    }
    if path.contains("/mcp") || path.contains("/rpc") {
        return Some(ProtocolDetection::new(
            Proto::JsonRpc,
            0.7,
            DetectionMethod::UrlPath,
        ));
    }

    None
}

/// Handle JSON-RPC protocol
fn handle_json_rpc(body: &[u8], request_id: String) -> Result<(ProtocolContext, Value)> {
    let v = serde_json::from_slice::<Value>(body).context("Failed to parse JSON-RPC body")?;

    // Validate it's proper JSON-RPC
    let _method = v
        .get("method")
        .and_then(|m| m.as_str())
        .ok_or_else(|| anyhow::anyhow!("JSON-RPC missing method"))?;

    let id = v
        .get("id")
        .cloned()
        .unwrap_or_else(|| json!(request_id.clone()));

    let ctx = ProtocolContext::new(Proto::JsonRpc, request_id);

    // Pass through valid JSON-RPC unchanged
    Ok((ctx, v))
}

/// Handle MCP Streamable HTTP protocol
fn handle_mcp_streamable_http(body: &[u8], request_id: String) -> Result<(ProtocolContext, Value)> {
    let v = serde_json::from_slice::<Value>(body)
        .context("Failed to parse MCP Streamable HTTP body")?;

    let ctx = ProtocolContext::new(Proto::McpStreamableHttp, request_id.clone());

    // Convert to standard JSON-RPC format
    let json_rpc = json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": v.get("method").unwrap_or(&json!("unknown")),
        "params": v.get("params").unwrap_or(&json!({})),
        "id": v.get("id").unwrap_or(&json!(request_id))
    });

    Ok((ctx, json_rpc))
}

/// Handle GraphQL protocol
fn handle_graphql(body: &[u8], request_id: String) -> Result<(ProtocolContext, Value)> {
    let v = serde_json::from_slice::<Value>(body).context("Failed to parse GraphQL body")?;

    let query = v
        .get("query")
        .and_then(|q| q.as_str())
        .ok_or_else(|| anyhow::anyhow!("GraphQL missing query"))?;

    let variables = v.get("variables").cloned().unwrap_or(json!({}));
    let operation_name = v.get("operationName").cloned();

    let mut ctx = ProtocolContext::new(Proto::GraphQL, request_id.clone());
    ctx.set_original_query(query.to_string());

    // Convert GraphQL to JSON-RPC
    let json_rpc =
        super::parsers::graphql_to_json_rpc(query, variables, operation_name, &request_id)?;

    Ok((ctx, json_rpc))
}

/// Handle Cap'n Proto protocol
fn handle_capnp(body: &[u8], request_id: String) -> Result<(ProtocolContext, Value)> {
    let ctx = ProtocolContext::new(Proto::Capnp, request_id.clone());

    // Convert Cap'n Proto to JSON-RPC
    let json_rpc = super::parsers::capnp_to_json_rpc(body, &request_id)?;

    Ok((ctx, json_rpc))
}

/// Check if JSON value represents MCP Streamable HTTP
fn is_mcp_streamable_http(v: &Value) -> bool {
    // MCP Streamable HTTP has specific structure
    v.get("method").is_some() && 
    v.get("jsonrpc").is_none() && // Not standard JSON-RPC
    (v.get("params").is_some() || v.get("arguments").is_some())
}

/// Check if JSON value represents GraphQL query
fn is_graphql_query(v: &Value) -> bool {
    v.get("query").is_some() || (v.get("operationName").is_some() && v.get("variables").is_some())
}

/// Check if binary data is Cap'n Proto format
fn is_capnp_binary(body: &[u8]) -> bool {
    // Cap'n Proto segment table validation based on official specification
    // Reference: capnproto-rust/capnp/src/serialize.rs

    // Minimum message size: 8 bytes (segment count + first segment length)
    if body.len() < 8 {
        return false;
    }

    // Parse segment count from first 4 bytes (little-endian u32)
    let segment_count_minus_one = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);

    // Convert to actual segment count (adding 1 back)
    let segment_count = segment_count_minus_one.wrapping_add(1) as usize;

    // Validate segment count bounds (from SEGMENTS_COUNT_LIMIT = 512)
    if segment_count == 0 || segment_count >= 512 {
        return false;
    }

    // Parse first segment length from next 4 bytes (little-endian u32)
    let first_segment_length = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;

    // Validate segment length is reasonable (prevent overflow attacks)
    if first_segment_length >= (1 << 29) {
        return false;
    }

    // Calculate minimum required message size
    // Segment table size: (segment_count + 1) / 2 * 8 bytes (rounded up to word boundary)
    let segment_table_words = (segment_count + 1) / 2;
    let segment_table_bytes = segment_table_words * 8;

    // Add first segment content size (in bytes, segments are measured in words)
    let first_segment_bytes = first_segment_length * 8;
    let minimum_message_size = segment_table_bytes + first_segment_bytes;

    // Verify message is at least as large as required by segment table
    if body.len() < minimum_message_size {
        return false;
    }

    // If we have more than one segment, validate additional segment lengths
    if segment_count > 1 && segment_count <= 4 {
        // For small segment counts, validate the remaining segment lengths
        let mut total_segment_bytes = first_segment_bytes;
        let mut byte_offset = 8; // Start after first segment length

        for i in 1..segment_count {
            if byte_offset + 4 > body.len() {
                return false;
            }

            let segment_length = u32::from_le_bytes([
                body[byte_offset],
                body[byte_offset + 1],
                body[byte_offset + 2],
                body[byte_offset + 3],
            ]) as usize;

            // Validate segment length
            if segment_length >= (1 << 29) {
                return false;
            }

            total_segment_bytes = match total_segment_bytes.checked_add(segment_length * 8) {
                Some(total) => total,
                None => return false, // Overflow protection
            };

            byte_offset += 4;

            // Skip padding byte if needed (segment table is word-aligned)
            if i % 2 == 1 && i < segment_count - 1 {
                byte_offset += 4; // Skip padding word
            }
        }

        // Verify total message size matches expectations
        if body.len() < segment_table_bytes + total_segment_bytes {
            return false;
        }
    }

    // Additional heuristic: Check for packed format characteristics
    // Packed format typically has fewer zero bytes due to compression
    let sample_size = body.len().min(64);
    let zero_count = body.iter().take(sample_size).filter(|&&b| b == 0).count();
    let zero_ratio = zero_count as f32 / sample_size as f32;

    // Pure zero data is unlikely to be valid Cap'n Proto
    if zero_ratio > 0.9 {
        return false;
    }

    // All validations passed - likely Cap'n Proto format
    true
}

/// Generate unique request ID
fn generate_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// Convert JSON-RPC response back to original protocol format
pub fn from_json_rpc(
    ctx: &ProtocolContext,
    json_rpc_response: &Value,
) -> ConversionResult<Vec<u8>> {
    debug!("Converting JSON-RPC response back to {:?}", ctx.protocol);

    match ctx.protocol {
        Proto::JsonRpc => {
            // Pass through unchanged
            serde_json::to_vec(json_rpc_response).map_err(|e| ConversionError::JsonError(e))
        }
        Proto::McpStreamableHttp => {
            // MCP Streamable HTTP uses standard JSON-RPC format
            serde_json::to_vec(json_rpc_response).map_err(|e| ConversionError::JsonError(e))
        }
        Proto::GraphQL => super::parsers::graphql_from_json_rpc(ctx, json_rpc_response),
        Proto::Capnp => super::parsers::capnp_from_json_rpc(ctx, json_rpc_response),
    }
}

/// Validate JSON-RPC structure
pub fn validate_json_rpc(value: &Value) -> ConversionResult<()> {
    // Check required fields
    if !value.is_object() {
        return Err(ConversionError::ValidationError(
            "JSON-RPC must be an object".to_string(),
        ));
    }

    let obj = value.as_object().ok_or_else(|| {
        ConversionError::ValidationError("Failed to access JSON-RPC object".to_string())
    })?;

    // Check jsonrpc version
    match obj.get("jsonrpc") {
        Some(Value::String(version)) if version == "2.0" => {}
        Some(_) => {
            return Err(ConversionError::ValidationError(
                "Invalid JSON-RPC version".to_string(),
            ))
        }
        None => {
            return Err(ConversionError::ValidationError(
                "Missing jsonrpc field".to_string(),
            ))
        }
    }

    // Check method field
    if let Some(method) = obj.get("method") {
        if !method.is_string() {
            return Err(ConversionError::ValidationError(
                "Method must be a string".to_string(),
            ));
        }
    }

    // Check id field (optional but must be string, number, or null if present)
    if let Some(id) = obj.get("id") {
        match id {
            Value::String(_) | Value::Number(_) | Value::Null => {}
            _ => {
                return Err(ConversionError::ValidationError(
                    "ID must be string, number, or null".to_string(),
                ))
            }
        }
    }

    Ok(())
}

/// Create error response in JSON-RPC format
pub fn create_error_response(
    id: Option<Value>,
    code: i32,
    message: &str,
    data: Option<Value>,
) -> Value {
    let mut error = json!({
        "code": code,
        "message": message
    });

    if let Some(data_value) = data {
        if let Some(error_obj) = error.as_object_mut() {
            error_obj.insert("data".to_string(), data_value);
        }
    }

    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id.unwrap_or(Value::Null),
        "error": error
    })
}
