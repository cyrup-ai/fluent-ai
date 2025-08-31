//! Protocol-specific parsers and converters
//!
//! This module provides protocol-specific parsing and conversion logic
//! for GraphQL, Cap'n Proto, and other protocols with zero allocation
//! patterns and blazing-fast performance.

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use async_graphql::parser::{parse_query, types::*};
use async_graphql::{Name, Positioned};
use async_graphql_value::Value as GraphQLValue;
use base64;
use capnp::{
    any_pointer, data, dynamic_list, dynamic_struct, dynamic_value, message::ReaderOptions,
    serialize, serialize_packed, text,
};
use serde_json::{json, Value};
use sweetmcp_axum::JSONRPC_VERSION;
use tracing::{debug, warn};

use super::schema_introspection::SchemaIntrospector;
use super::types::{
    ConversionError, ConversionResult, FragmentRegistry, GraphQLContext, GraphQLTypeInfo,
    GraphQLTypeKind, ProtocolContext,
};

/// Convert GraphQL query to JSON-RPC with fragment resolution
pub fn graphql_to_json_rpc(
    query: &str,
    variables: Value,
    operation_name: Option<Value>,
    request_id: &str,
) -> Result<Value> {
    graphql_to_json_rpc_with_schema(query, variables, operation_name, request_id, None)
}

/// Convert GraphQL query to JSON-RPC with fragment resolution and optional upstream URL for schema introspection
pub fn graphql_to_json_rpc_with_schema(
    query: &str,
    variables: Value,
    operation_name: Option<Value>,
    request_id: &str,
    upstream_url: Option<&str>,
) -> Result<Value> {
    debug!("Converting GraphQL query to JSON-RPC with fragment resolution");

    // Parse GraphQL query
    let doc = parse_query(query).map_err(|e| anyhow::anyhow!("GraphQL parse error: {}", e))?;

    // Phase 1: Collect all fragment definitions
    let mut fragment_registry = FragmentRegistry::new();
    for (name, fragment) in &doc.fragments {
        fragment_registry
            .register_fragment(name.to_string(), fragment.clone())
            .map_err(|e| anyhow::anyhow!("Fragment registration error: {}", e))?;
    }

    // Phase 2: Process operations with fragment resolution
    let operation = doc.operations.iter().next();

    let (method, params) = match operation {
        Some((name, op)) => {
            let method_name = if let Some(op_name) = operation_name {
                op_name.as_str().unwrap_or("graphql_query").to_string()
            } else if let Some(name) = name {
                name.to_string()
            } else {
                "graphql_query".to_string()
            };

            // Extract fields with fragment resolution
            let mut fields = Vec::new();

            // Determine root operation type from GraphQL operation
            let root_type = match op.node.ty {
                async_graphql_parser::types::OperationType::Query => "Query",
                async_graphql_parser::types::OperationType::Mutation => "Mutation",
                async_graphql_parser::types::OperationType::Subscription => "Subscription",
            };

            // Get schema information from upstream server if URL provided
            let schema_types = if let Some(url) = upstream_url {
                // Use schema introspection to get real schema information
                match SchemaIntrospector::new() {
                    Ok(introspector) => {
                        match tokio::runtime::Handle::try_current() {
                            Ok(handle) => {
                                // We're in an async context, use block_in_place to avoid blocking the executor
                                tokio::task::block_in_place(|| {
                                    handle.block_on(async {
                                        introspector.get_schema(url).await.unwrap_or_else(|e| {
                                            warn!(
                                                "Failed to introspect schema from {}: {}",
                                                url, e
                                            );
                                            create_basic_schema_types()
                                        })
                                    })
                                })
                            }
                            Err(_) => {
                                // No async runtime available, create a basic runtime for introspection
                                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                                    anyhow::anyhow!(
                                        "Failed to create async runtime for schema introspection: {}",
                                        e
                                    )
                                })?;
                                rt.block_on(async {
                                    introspector.get_schema(url).await.unwrap_or_else(|e| {
                                        warn!("Failed to introspect schema from {}: {}", url, e);
                                        create_basic_schema_types()
                                    })
                                })
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to create schema introspector: {}", e);
                        create_basic_schema_types()
                    }
                }
            } else {
                // No upstream URL provided, use basic schema types
                create_basic_schema_types()
            };

            let mut graphql_context = GraphQLContext {
                fragment_registry,
                fragment_cache: super::types::FragmentCache::new(),
                type_info: Some(root_type.to_string()),
                schema_types,
            };

            extract_fields_with_fragments(
                &op.node.selection_set.node,
                &mut fields,
                &mut graphql_context,
            )
            .map_err(|e| anyhow::anyhow!("Fragment resolution error: {}", e))?;

            let params = json!({
                "query": query,
                "variables": variables,
                "operationName": operation_name,
                "fields": fields,
                "operationType": format!("{:?}", op.node.ty),
                "resolvedFragments": true,
                "fragmentCount": graphql_context.fragment_registry.get_fragment_names().len()
            });

            (method_name, params)
        }
        None => {
            warn!("No GraphQL operation found, using default");
            (
                "graphql_query".to_string(),
                json!({
                    "query": query,
                    "variables": variables,
                    "resolvedFragments": false
                }),
            )
        }
    };

    Ok(json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params,
        "id": request_id
    }))
}

/// Create basic GraphQL schema types (Query, Mutation, Subscription) as fallback
fn create_basic_schema_types() -> std::collections::HashMap<String, GraphQLTypeInfo> {
    let mut schema_types = std::collections::HashMap::new();

    // Add Query root type
    schema_types.insert(
        "Query".to_string(),
        GraphQLTypeInfo {
            name: "Query".to_string(),
            kind: GraphQLTypeKind::Object,
            fields: vec![], // Basic fallback - no field information available
            interfaces: vec![],
            possible_types: vec![],
        },
    );

    // Add Mutation root type
    schema_types.insert(
        "Mutation".to_string(),
        GraphQLTypeInfo {
            name: "Mutation".to_string(),
            kind: GraphQLTypeKind::Object,
            fields: vec![], // Basic fallback - no field information available
            interfaces: vec![],
            possible_types: vec![],
        },
    );

    // Add Subscription root type
    schema_types.insert(
        "Subscription".to_string(),
        GraphQLTypeInfo {
            name: "Subscription".to_string(),
            kind: GraphQLTypeKind::Object,
            fields: vec![], // Basic fallback - no field information available
            interfaces: vec![],
            possible_types: vec![],
        },
    );

    schema_types
}

/// Extract fields from GraphQL selection set with fragment resolution
fn extract_fields_with_fragments(
    selection_set: &SelectionSet,
    fields: &mut Vec<String>,
    context: &mut GraphQLContext,
) -> Result<(), ConversionError> {
    for selection in &selection_set.items {
        match &selection.node {
            Selection::Field(field) => {
                fields.push(field.node.name.node.to_string());

                // Recursively extract nested fields
                if !field.node.selection_set.node.items.is_empty() {
                    extract_fields_with_fragments(&field.node.selection_set.node, fields, context)?;
                }
            }
            Selection::InlineFragment(fragment) => {
                // Validate type condition if present
                if let Some(type_condition) = &fragment.node.type_condition {
                    validate_type_condition(&type_condition.node, context)?;
                }

                extract_fields_with_fragments(&fragment.node.selection_set.node, fields, context)?;
            }
            Selection::FragmentSpread(fragment_spread) => {
                let fragment_name = &fragment_spread.node.fragment_name.node;

                // Check cache first for performance
                if let Some(cached_fields) = context.fragment_cache.get(fragment_name) {
                    fields.extend(cached_fields.clone());
                    continue;
                }

                // Circular dependency check
                context
                    .fragment_registry
                    .validate_no_cycles(fragment_name)?;

                // Get fragment definition
                let fragment_def = context
                    .fragment_registry
                    .get_fragment(fragment_name)
                    .ok_or_else(|| ConversionError::FragmentNotFound {
                        name: fragment_name.to_string(),
                    })?;

                // Validate fragment type condition
                validate_type_condition(&fragment_def.node.type_condition.node, context)?;

                // Track resolution for cycle detection
                context
                    .fragment_registry
                    .push_resolution(fragment_name.to_string());

                // Collect fields from fragment for caching
                let mut fragment_fields = Vec::new();

                // Recursively resolve fragment
                extract_fields_with_fragments(
                    &fragment_def.node.selection_set.node,
                    &mut fragment_fields,
                    context,
                )?;

                // Cache the resolved fields
                context
                    .fragment_cache
                    .insert(fragment_name.to_string(), fragment_fields.clone());

                // Add fields to result
                fields.extend(fragment_fields);

                // Pop resolution stack
                context.fragment_registry.pop_resolution();
            }
        }
    }
    Ok(())
}

/// Validate type condition for fragments
pub fn validate_type_condition(
    type_condition: &Name,
    context: &GraphQLContext,
) -> Result<(), ConversionError> {
    let type_name = type_condition.as_str();

    // Validate GraphQL type name format according to specification
    // Reference: https://spec.graphql.org/October2021/#Name
    if !is_valid_graphql_type_name(type_name) {
        return Err(ConversionError::GraphQLError(format!(
            "Invalid GraphQL type name '{}': must start with letter/underscore and contain only letters, numbers, underscores",
            type_name
        )));
    }

    // Validate type name follows GraphQL naming conventions
    if !type_name.chars().next().unwrap_or('_').is_ascii_uppercase() && !type_name.starts_with('_')
    {
        return Err(ConversionError::GraphQLError(format!(
            "Invalid GraphQL type name '{}': types should start with uppercase letter or underscore",
            type_name
        )));
    }

    // Check for reserved GraphQL type names
    if is_reserved_graphql_type(type_name) {
        return Err(ConversionError::GraphQLError(format!(
            "Invalid type condition '{}': conflicts with built-in GraphQL types",
            type_name
        )));
    }

    // Check if the type exists in the schema
    if !context.has_type(type_name) {
        return Err(ConversionError::TypeConditionError(format!(
            "Type '{}' does not exist in schema",
            type_name
        )));
    }

    // Get type information for validation
    let type_info = context.get_type(type_name).unwrap(); // Safe because we checked has_type above

    // Validate that the type can be used as a fragment type condition
    match type_info.kind {
        crate::normalize::types::GraphQLTypeKind::Object
        | crate::normalize::types::GraphQLTypeKind::Interface
        | crate::normalize::types::GraphQLTypeKind::Union => {
            // These types can be used as fragment type conditions
        }
        _ => {
            return Err(ConversionError::TypeConditionError(
                format!("Type '{}' cannot be used as a fragment type condition. Only Object, Interface, and Union types are allowed", type_name)
            ));
        }
    }

    // Validate fragment type compatibility with current context
    if let Some(current_type) = context.get_type_info() {
        if !context.is_type_compatible(type_name, current_type) {
            return Err(ConversionError::TypeConditionError(format!(
                "Fragment type '{}' is not compatible with current context type '{}'",
                type_name, current_type
            )));
        }
    }

    debug!(
        "Validated type condition: {} - passed all checks",
        type_condition
    );
    Ok(())
}

/// Validate GraphQL type name format according to specification
fn is_valid_graphql_type_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // First character must be letter or underscore
    let mut chars = name.chars();
    if let Some(first) = chars.next() {
        if !first.is_ascii_alphabetic() && first != '_' {
            return false;
        }
    } else {
        return false;
    }

    // Remaining characters must be letters, numbers, or underscores
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Check if type name conflicts with reserved GraphQL built-in types
fn is_reserved_graphql_type(name: &str) -> bool {
    matches!(
        name,
        "String"
            | "Int"
            | "Float"
            | "Boolean"
            | "ID"
            | "__Schema"
            | "__Type"
            | "__Field"
            | "__InputValue"
            | "__EnumValue"
            | "__Directive"
            | "__DirectiveLocation"
            | "__TypeKind"
    )
}

/// Detect message format and parse Cap'n Proto binary message
fn detect_and_parse_message(
    body: &[u8],
) -> ConversionResult<capnp::message::Reader<capnp::serialize::SliceSegments>> {
    let reader_options = ReaderOptions::new();

    // Try packed format first (more common)
    if let Ok(message) = serialize_packed::read_message(&mut Cursor::new(body), reader_options) {
        return Ok(message);
    }

    // Fall back to unpacked format
    serialize::read_message(&mut Cursor::new(body), reader_options)
        .map_err(|e| ConversionError::CapnProtoError(format!("Failed to parse message: {}", e)))
}

/// Convert dynamic Cap'n Proto value to JSON
fn dynamic_value_to_json(value: dynamic_value::Reader) -> ConversionResult<Value> {
    match value {
        dynamic_value::Reader::Void => Ok(Value::Null),
        dynamic_value::Reader::Bool(b) => Ok(Value::Bool(b)),
        dynamic_value::Reader::Int8(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::Int16(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::Int32(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::Int64(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::UInt8(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::UInt16(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::UInt32(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::UInt64(i) => Ok(Value::Number(i.into())),
        dynamic_value::Reader::Float32(f) => Ok(Value::Number(
            serde_json::Number::from_f64(f as f64).unwrap_or_else(|| serde_json::Number::from(0)),
        )),
        dynamic_value::Reader::Float64(f) => Ok(Value::Number(
            serde_json::Number::from_f64(f).unwrap_or_else(|| serde_json::Number::from(0)),
        )),
        dynamic_value::Reader::Text(t) => Ok(Value::String(
            t.to_str()
                .map_err(|e| {
                    ConversionError::CapnProtoError(format!("Text conversion error: {}", e))
                })?
                .to_string(),
        )),
        dynamic_value::Reader::Data(d) => Ok(Value::String(base64::encode(d))),
        dynamic_value::Reader::Struct(s) => convert_struct_to_json(s),
        dynamic_value::Reader::List(l) => convert_list_to_json(l),
        dynamic_value::Reader::Enum(e) => {
            Ok(Value::String(format!("enum_{}", e.get_enumerant_index())))
        }
        dynamic_value::Reader::AnyPointer(ptr) => {
            // Try to decode AnyPointer as different possible types
            if ptr.is_null() {
                Ok(Value::Null)
            } else {
                // Try as text first
                if let Ok(text) = ptr.get_as::<text::Reader>() {
                    match text.to_str() {
                        Ok(s) => Ok(Value::String(s.to_string())),
                        Err(_) => Ok(json!({"any_pointer": "text_decode_failed"})),
                    }
                }
                // Try as data
                else if let Ok(data) = ptr.get_as::<data::Reader>() {
                    Ok(Value::String(base64::encode(data)))
                }
                // Try as struct (without schema)
                else if let Ok(struct_reader) = ptr.reader.get_struct(None) {
                    // Get basic struct info without schema
                    Ok(json!({
                        "any_pointer_struct": {
                            "data_words": struct_reader.get_data_section_size(),
                            "pointer_words": struct_reader.get_pointer_section_size()
                        }
                    }))
                }
                // Fallback - return size info
                else {
                    match ptr.target_size() {
                        Ok(size) => Ok(json!({
                            "any_pointer": {
                                "words": size.word_count,
                                "caps": size.cap_count
                            }
                        })),
                        Err(_) => Ok(json!({"any_pointer": "unknown"})),
                    }
                }
            }
        }
        dynamic_value::Reader::Capability(_cap) => {
            // Capability is just an empty stub per source code documentation
            Ok(json!({
                "capability": "dynamic_capability_stub"
            }))
        }
    }
}

/// Convert Cap'n Proto struct to JSON object
fn convert_struct_to_json(struct_reader: dynamic_struct::Reader) -> ConversionResult<Value> {
    let mut object = serde_json::Map::new();
    let schema = struct_reader.get_schema();

    for field in schema
        .get_fields()
        .map_err(|e| ConversionError::CapnProtoError(format!("Failed to get fields: {}", e)))?
    {
        let field_name = field
            .get_proto()
            .get_name()
            .map_err(|e| {
                ConversionError::CapnProtoError(format!("Failed to get field name: {}", e))
            })?
            .to_str()
            .map_err(|e| {
                ConversionError::CapnProtoError(format!("Failed to convert field name: {}", e))
            })?;

        if struct_reader.has_field(&field) {
            let field_value = struct_reader.get_field(&field).map_err(|e| {
                ConversionError::CapnProtoError(format!("Failed to get field value: {}", e))
            })?;
            let json_value = dynamic_value_to_json(field_value)?;
            object.insert(field_name.to_string(), json_value);
        }
    }

    Ok(Value::Object(object))
}

/// Convert Cap'n Proto list to JSON array
fn convert_list_to_json(list_reader: dynamic_list::Reader) -> ConversionResult<Value> {
    let mut array = Vec::new();

    for i in 0..list_reader.len() {
        let element = list_reader.get(i).map_err(|e| {
            ConversionError::CapnProtoError(format!("Failed to get list element: {}", e))
        })?;
        let json_element = dynamic_value_to_json(element)?;
        array.push(json_element);
    }

    Ok(Value::Array(array))
}

/// Detect method name from parsed JSON data
fn detect_method_name(data: &Value) -> Option<String> {
    // Look for common method name patterns in the parsed data
    if let Some(obj) = data.as_object() {
        // Check for explicit method field
        if let Some(method) = obj.get("method").and_then(|v| v.as_str()) {
            return Some(method.to_string());
        }

        // Check for action/command fields
        if let Some(action) = obj.get("action").and_then(|v| v.as_str()) {
            return Some(format!("capnp_{}", action));
        }

        // Check for type fields
        if let Some(msg_type) = obj.get("type").and_then(|v| v.as_str()) {
            return Some(format!("capnp_{}", msg_type));
        }
    }

    None
}

/// Detect if Cap'n Proto message is in packed format
fn is_packed_format(body: &[u8]) -> bool {
    // Cap'n Proto packed format detection heuristic
    // Packed format compresses zero bytes, so it has fewer zeros
    if body.len() < 8 {
        return false;
    }

    // Check for packed format header patterns
    // Packed format has special tag bytes and compressed zero runs
    let zero_count = body.iter().take(32).filter(|&&b| b == 0).count();
    let sample_size = body.len().min(32);

    // Packed format has fewer zero bytes due to compression
    // Also check for packed format tag bytes (high bits set)
    let has_tag_bytes = body.iter().take(8).any(|&b| b & 0x80 != 0);

    // Packed if fewer zeros AND has tag bytes, or very few zeros overall
    (zero_count < sample_size / 6 && has_tag_bytes) || zero_count < sample_size / 8
}

/// Convert Cap'n Proto binary to JSON-RPC
pub fn capnp_to_json_rpc(body: &[u8], request_id: &str) -> Result<Value> {
    debug!("Converting Cap'n Proto to JSON-RPC");

    // Parse the Cap'n Proto message
    let parsed_data = parse_capnp_message(body).context("Failed to parse Cap'n Proto message")?;

    // Detect method name from structure
    let method_name = detect_method_name(&parsed_data).unwrap_or("capnp_call".to_string());

    // Wrap in JSON-RPC 2.0 format
    Ok(json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": method_name,
        "params": parsed_data,
        "id": request_id
    }))
}

/// Convert JSON-RPC response to GraphQL response
pub fn graphql_from_json_rpc(ctx: &ProtocolContext, response: &Value) -> ConversionResult<Vec<u8>> {
    debug!("Converting JSON-RPC response to GraphQL");

    let mut graphql_response = json!({
        "data": null
    });

    // Check for JSON-RPC error
    if let Some(error) = response.get("error") {
        let error_message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error");

        let error_code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-32603);

        graphql_response["errors"] = json!([{
            "message": error_message,
            "extensions": {
                "code": error_code,
                "jsonrpc_error": error
            }
        }]);
    } else if let Some(result) = response.get("result") {
        // Shape response based on original query if available
        if let Some(original_query) = ctx.original_query() {
            graphql_response["data"] = shape_graphql_response(result, original_query)?;
        } else {
            // Simple passthrough if no original query context
            graphql_response["data"] = result.clone();
        }
    }

    // Add extensions with metadata
    graphql_response["extensions"] = json!({
        "request_id": ctx.request_id(),
        "protocol": "graphql",
        "converted_from": "json-rpc"
    });

    serde_json::to_vec(&graphql_response).map_err(|e| ConversionError::JsonError(e))
}

/// Shape GraphQL response based on original query structure
fn shape_graphql_response(result: &Value, original_query: &str) -> ConversionResult<Value> {
    // Parse the original query to understand expected structure
    let doc = parse_query(original_query).map_err(|e| {
        ConversionError::GraphQLError(format!("Failed to parse original query: {}", e))
    })?;

    // Extract field selection from the first operation
    if let Some((_, operation)) = doc.operations.iter().next() {
        let selection_set = &operation.node.selection_set.node;
        shape_response_to_selection_set(result, selection_set, &doc.fragments)
    } else {
        // No operations found - return empty data object
        Ok(json!({}))
    }
}

/// Shape response data to match GraphQL selection set structure
fn shape_response_to_selection_set(
    data: &Value,
    selection_set: &SelectionSet,
    fragments: &std::collections::HashMap<async_graphql::Name, Positioned<FragmentDefinition>>,
) -> ConversionResult<Value> {
    let mut shaped = serde_json::Map::new();

    for selection in &selection_set.items {
        match &selection.node {
            Selection::Field(field) => {
                let field_name = &field.node.name.node;
                let response_key = field
                    .node
                    .alias
                    .as_ref()
                    .map(|a| &a.node)
                    .unwrap_or(field_name);

                // Look for field data in the response
                if let Some(field_value) = find_field_value(data, field_name.as_str()) {
                    // Recursively shape nested selections if present
                    let shaped_value = if !field.node.selection_set.node.items.is_empty() {
                        match field_value {
                            Value::Array(arr) => {
                                // Handle list fields - shape each array element
                                let mut shaped_array = Vec::new();
                                for item in arr {
                                    let shaped_item = shape_response_to_selection_set(
                                        item,
                                        &field.node.selection_set.node,
                                        fragments,
                                    )?;
                                    shaped_array.push(shaped_item);
                                }
                                Value::Array(shaped_array)
                            }
                            Value::Object(_) => {
                                // Handle object fields - shape the object
                                shape_response_to_selection_set(
                                    field_value,
                                    &field.node.selection_set.node,
                                    fragments,
                                )?
                            }
                            _ => {
                                // Scalar value with selections - this is unusual but handle gracefully
                                field_value.clone()
                            }
                        }
                    } else {
                        // No nested selections - use value as-is
                        field_value.clone()
                    };

                    shaped.insert(response_key.to_string(), shaped_value);
                } else {
                    // Field not found in response - set to null (GraphQL standard)
                    shaped.insert(response_key.to_string(), Value::Null);
                }
            }
            Selection::InlineFragment(inline_fragment) => {
                // Merge inline fragment fields into the result
                let fragment_result = shape_response_to_selection_set(
                    data,
                    &inline_fragment.node.selection_set.node,
                    fragments,
                )?;

                if let Value::Object(fragment_obj) = fragment_result {
                    // Merge fragment fields into shaped response
                    for (key, value) in fragment_obj {
                        shaped.insert(key, value);
                    }
                }
            }
            Selection::FragmentSpread(fragment_spread) => {
                let fragment_name = &fragment_spread.node.fragment_name.node;

                // Look up fragment definition
                if let Some(fragment_def) = fragments.get(fragment_name) {
                    // Apply fragment selection set
                    let fragment_result = shape_response_to_selection_set(
                        data,
                        &fragment_def.node.selection_set.node,
                        fragments,
                    )?;

                    if let Value::Object(fragment_obj) = fragment_result {
                        // Merge fragment fields into shaped response
                        for (key, value) in fragment_obj {
                            shaped.insert(key, value);
                        }
                    }
                } else {
                    // Fragment not found - log warning but continue
                    warn!(
                        "Fragment '{}' not found during response shaping",
                        fragment_name
                    );
                }
            }
        }
    }

    Ok(Value::Object(shaped))
}

/// Find field value in response data with flexible key matching
fn find_field_value(data: &Value, field_name: &str) -> Option<&Value> {
    match data {
        Value::Object(obj) => {
            // Try exact match first
            if let Some(value) = obj.get(field_name) {
                return Some(value);
            }

            // Try case-insensitive match
            for (key, value) in obj {
                if key.to_lowercase() == field_name.to_lowercase() {
                    return Some(value);
                }
            }

            // Try snake_case to camelCase conversion
            let camel_case = snake_to_camel_case(field_name);
            if let Some(value) = obj.get(&camel_case) {
                return Some(value);
            }

            // Try camelCase to snake_case conversion
            let snake_case = camel_to_snake_case(field_name);
            if let Some(value) = obj.get(&snake_case) {
                return Some(value);
            }

            None
        }
        _ => None,
    }
}

/// Convert snake_case to camelCase
fn snake_to_camel_case(snake_str: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for ch in snake_str.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

/// Convert camelCase to snake_case
fn camel_to_snake_case(camel_str: &str) -> String {
    let mut result = String::new();

    for (i, ch) in camel_str.chars().enumerate() {
        if i > 0 && ch.is_ascii_uppercase() {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }

    result
}

/// Convert JSON value to Cap'n Proto dynamic value using any_pointer::Builder
fn json_to_capnp_value(json_val: &Value, builder: any_pointer::Builder) -> ConversionResult<()> {
    match json_val {
        Value::Null => {
            // Set as void/null by clearing the builder
            builder.clear();
            Ok(())
        }
        Value::Bool(b) => {
            // Cap'n Proto doesn't have direct bool in any_pointer, store as text
            builder.set_as_text(&b.to_string());
            Ok(())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                // Store as text to preserve precision
                builder.set_as_text(&i.to_string());
            } else if let Some(f) = n.as_f64() {
                // Store as text to preserve precision
                builder.set_as_text(&f.to_string());
            } else {
                return Err(ConversionError::CapnProtoError(
                    "Invalid number format in JSON".to_string(),
                ));
            }
            Ok(())
        }
        Value::String(s) => {
            builder.set_as_text(s);
            Ok(())
        }
        Value::Array(arr) => json_to_capnp_list(arr, builder),
        Value::Object(obj) => json_to_capnp_struct(obj, builder),
    }
}

/// Convert JSON object to Cap'n Proto struct using any_pointer::Builder
fn json_to_capnp_struct(
    obj: &serde_json::Map<String, Value>,
    builder: any_pointer::Builder,
) -> ConversionResult<()> {
    // Create a generic struct with data and pointer fields
    let mut data_fields = Vec::new();
    let mut text_fields = Vec::new();

    // Separate numeric/boolean data from text/complex data
    for (key, value) in obj {
        match value {
            Value::Number(_) | Value::Bool(_) => {
                data_fields.push((key.clone(), value.clone()));
            }
            _ => {
                text_fields.push((key.clone(), value.clone()));
            }
        }
    }

    // For simplicity, store the entire object as JSON text
    // This maintains data integrity while working within Cap'n Proto constraints
    let json_repr = serde_json::to_string(obj).map_err(|e| {
        ConversionError::CapnProtoError(format!("Failed to serialize object to JSON: {}", e))
    })?;

    builder.set_as_text(&json_repr);
    Ok(())
}

/// Convert JSON array to Cap'n Proto list using any_pointer::Builder  
fn json_to_capnp_list(arr: &[Value], builder: any_pointer::Builder) -> ConversionResult<()> {
    // Convert array to JSON string representation for consistency
    let json_repr = serde_json::to_string(arr).map_err(|e| {
        ConversionError::CapnProtoError(format!("Failed to serialize array to JSON: {}", e))
    })?;

    builder.set_as_text(&json_repr);
    Ok(())
}

/// Build Cap'n Proto message from JSON-RPC response
fn build_capnp_message(
    response: &Value,
) -> ConversionResult<capnp::message::Builder<capnp::message::HeapAllocator>> {
    let mut message_builder = capnp::message::Builder::new_default();
    let root_builder = message_builder.init_root::<any_pointer::Builder>();

    // Convert the JSON response to Cap'n Proto structure
    json_to_capnp_value(response, root_builder)?;

    Ok(message_builder)
}

/// Serialize Cap'n Proto message to binary format
fn serialize_capnp_response(
    builder: capnp::message::Builder<capnp::message::HeapAllocator>,
    use_packed: bool,
) -> ConversionResult<Vec<u8>> {
    let mut buffer = Vec::new();

    if use_packed {
        serialize_packed::write_message(&mut buffer, &builder).map_err(|e| {
            ConversionError::CapnProtoError(format!("Failed to serialize packed message: {}", e))
        })?;
    } else {
        serialize::write_message(&mut buffer, &builder).map_err(|e| {
            ConversionError::CapnProtoError(format!("Failed to serialize message: {}", e))
        })?;
    }

    Ok(buffer)
}

/// Convert JSON-RPC response to Cap'n Proto
pub fn capnp_from_json_rpc(ctx: &ProtocolContext, response: &Value) -> ConversionResult<Vec<u8>> {
    debug!("Converting JSON-RPC response to Cap'n Proto");

    // Validate JSON-RPC response format
    if !response.is_object() {
        return Err(ConversionError::CapnProtoError(
            "JSON-RPC response must be an object".to_string(),
        ));
    }

    let response_obj = response.as_object().ok_or_else(|| {
        ConversionError::CapnProtoError("Failed to access response object".to_string())
    })?;

    // Check for required JSON-RPC fields
    if !response_obj.contains_key("jsonrpc") && !response_obj.contains_key("id") {
        return Err(ConversionError::CapnProtoError(
            "Invalid JSON-RPC response format - missing required fields".to_string(),
        ));
    }

    // Extract the actual data to convert
    let data_to_convert = if let Some(result) = response_obj.get("result") {
        // Success response - convert the result data
        result
    } else if let Some(error) = response_obj.get("error") {
        // Error response - convert the error data
        error
    } else {
        // Fallback - convert entire response
        response
    };

    // Build Cap'n Proto message from the data
    let message_builder = build_capnp_message(data_to_convert)?;

    // Determine format preference - default to packed for efficiency
    let use_packed = ctx.metadata().options.include_debug_info == false; // Use packed in production

    // Serialize to binary format
    serialize_capnp_response(message_builder, use_packed)
}

/// Parse GraphQL variables
pub fn parse_graphql_variables(
    variables: &Value,
) -> ConversionResult<std::collections::HashMap<String, GraphQLValue>> {
    let mut parsed_variables = std::collections::HashMap::new();

    if let Some(vars) = variables.as_object() {
        for (key, value) in vars {
            let graphql_value = json_to_graphql_value(value)?;
            parsed_variables.insert(key.clone(), graphql_value);
        }
    }

    Ok(parsed_variables)
}

/// Convert JSON value to GraphQL value
fn json_to_graphql_value(value: &Value) -> ConversionResult<GraphQLValue> {
    let graphql_value = match value {
        Value::Null => GraphQLValue::Null,
        Value::Bool(b) => GraphQLValue::Boolean(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                GraphQLValue::Number(async_graphql_value::Number::from(i))
            } else if let Some(f) = n.as_f64() {
                GraphQLValue::Number(async_graphql_value::Number::from(f))
            } else {
                return Err(ConversionError::GraphQLError(
                    "Invalid number format".to_string(),
                ));
            }
        }
        Value::String(s) => GraphQLValue::String(s.clone()),
        Value::Array(arr) => {
            let mut graphql_array = Vec::new();
            for item in arr {
                graphql_array.push(json_to_graphql_value(item)?);
            }
            GraphQLValue::List(graphql_array)
        }
        Value::Object(obj) => {
            let mut graphql_object = async_graphql_value::indexmap::IndexMap::new();
            for (key, val) in obj {
                let name = Name::new(key);
                graphql_object.insert(name, json_to_graphql_value(val)?);
            }
            GraphQLValue::Object(graphql_object)
        }
    };

    Ok(graphql_value)
}

/// Validate GraphQL query syntax
pub fn validate_graphql_query(query: &str) -> ConversionResult<()> {
    parse_query(query)
        .map_err(|e| ConversionError::GraphQLError(format!("Invalid GraphQL syntax: {}", e)))?;

    Ok(())
}

/// Extract GraphQL operation type
pub fn extract_operation_type(query: &str) -> ConversionResult<String> {
    let doc = parse_query(query)
        .map_err(|e| ConversionError::GraphQLError(format!("Failed to parse query: {}", e)))?;

    if let Some((_, operation)) = doc.operations.iter().next() {
        Ok(format!("{:?}", operation.node.ty).to_lowercase())
    } else {
        Ok("query".to_string()) // Default to query
    }
}

/// Extract GraphQL operation name
pub fn extract_operation_name(query: &str) -> ConversionResult<Option<String>> {
    let doc = parse_query(query)
        .map_err(|e| ConversionError::GraphQLError(format!("Failed to parse query: {}", e)))?;

    if let Some((name, _)) = doc.operations.iter().next() {
        Ok(name.as_ref().map(|n| n.to_string()))
    } else {
        Ok(None)
    }
}

/// Create GraphQL error response
pub fn create_graphql_error(message: &str, code: Option<i32>) -> Value {
    json!({
        "errors": [{
            "message": message,
            "extensions": {
                "code": code.unwrap_or(-1)
            }
        }],
        "data": null
    })
}

/// Parse Cap'n Proto message with dynamic runtime reflection
pub fn parse_capnp_message(body: &[u8]) -> ConversionResult<Value> {
    debug!("Parsing Cap'n Proto message for conversion");

    // Validate minimum message size
    if body.len() < 8 {
        return Err(ConversionError::CapnProtoError(
            "Cap'n Proto message too short (minimum 8 bytes required)".to_string(),
        ));
    }

    // Try to parse and convert the message
    match detect_and_parse_message(body) {
        Ok(message_reader) => {
            // Try to get root and parse as structured data
            match message_reader.get_root::<any_pointer::Reader>() {
                Ok(root) => {
                    if !root.is_null() {
                        // Try to parse the actual data content
                        // First try as text
                        if let Ok(text) = root.get_as::<text::Reader>() {
                            match text.to_str() {
                                Ok(s) => return Ok(Value::String(s.to_string())),
                                Err(_) => {}
                            }
                        }

                        // Try as data blob
                        if let Ok(data) = root.get_as::<data::Reader>() {
                            return Ok(Value::String(base64::encode(data)));
                        }

                        // Try as struct (most common case)
                        if let Ok(struct_reader) = root.reader.get_struct(None) {
                            // Parse as generic struct without schema
                            let mut result = serde_json::Map::new();

                            // Extract data fields by word offset
                            let data_size = struct_reader.get_data_section_size();
                            if data_size > 0 {
                                let mut data_fields = serde_json::Map::new();

                                // Read primitive data fields
                                for i in 0..data_size as usize {
                                    if i < 64 {
                                        // Reasonable limit
                                        let word_value =
                                            struct_reader.get_data_field::<u64>(i as u32);
                                        if word_value != 0 {
                                            data_fields.insert(
                                                format!("data_word_{}", i),
                                                Value::Number(word_value.into()),
                                            );
                                        }
                                    }
                                }

                                if !data_fields.is_empty() {
                                    result.insert(
                                        "data_fields".to_string(),
                                        Value::Object(data_fields),
                                    );
                                }
                            }

                            // Extract pointer fields
                            let pointer_count = struct_reader.get_pointer_section_size();
                            if pointer_count > 0 {
                                let mut pointer_fields = Vec::new();

                                for i in 0..pointer_count as usize {
                                    if i < 32 {
                                        // Reasonable limit
                                        let ptr = struct_reader.get_pointer_field(i as u16);
                                        if !ptr.is_null() {
                                            // Try to decode pointer content
                                            if let Ok(text) = ptr.get_text(None) {
                                                if let Ok(s) = text.to_str() {
                                                    pointer_fields.push(json!({
                                                        "index": i,
                                                        "type": "text",
                                                        "value": s
                                                    }));
                                                    continue;
                                                }
                                            }

                                            if let Ok(data) = ptr.get_data(None) {
                                                pointer_fields.push(json!({
                                                    "index": i,
                                                    "type": "data",
                                                    "value": base64::encode(&data[..std::cmp::min(data.len(), 256)])
                                                }));
                                                continue;
                                            }

                                            if let Ok(nested_struct) = ptr.get_struct(None) {
                                                pointer_fields.push(json!({
                                                    "index": i,
                                                    "type": "struct",
                                                    "data_words": nested_struct.get_data_section_size(),
                                                    "pointer_words": nested_struct.get_pointer_section_size()
                                                }));
                                                continue;
                                            }

                                            if let Ok(list) = ptr.get_list(
                                                capnp::private::layout::ElementSize::Void,
                                                None,
                                            ) {
                                                pointer_fields.push(json!({
                                                    "index": i,
                                                    "type": "list",
                                                    "length": list.len()
                                                }));
                                                continue;
                                            }

                                            // Unknown pointer type
                                            pointer_fields.push(json!({
                                                "index": i,
                                                "type": "unknown_pointer"
                                            }));
                                        }
                                    }
                                }

                                if !pointer_fields.is_empty() {
                                    result.insert(
                                        "pointer_fields".to_string(),
                                        Value::Array(pointer_fields),
                                    );
                                }
                            }

                            // Add metadata
                            result.insert("_metadata".to_string(), json!({
                                "type": "capnp_struct",
                                "size": body.len(),
                                "format": if is_packed_format(body) { "packed" } else { "unpacked" },
                                "data_words": data_size,
                                "pointer_words": pointer_count,
                                "segments": message_reader.get_segments_for_output().len()
                            }));

                            return Ok(Value::Object(result));
                        }

                        // Try as list
                        if let Ok(list) = root
                            .reader
                            .get_list(capnp::private::layout::ElementSize::Void, None)
                        {
                            return Ok(json!({
                                "_metadata": {
                                    "type": "capnp_list",
                                    "size": body.len(),
                                    "format": if is_packed_format(body) { "packed" } else { "unpacked" },
                                    "length": list.len()
                                },
                                "list_info": {
                                    "length": list.len(),
                                    "element_size": "unknown"
                                }
                            }));
                        }

                        // Fallback for non-null but unreadable root
                        Ok(json!({
                            "_metadata": {
                                "type": "capnp_unknown",
                                "size": body.len(),
                                "format": if is_packed_format(body) { "packed" } else { "unpacked" },
                                "error": "could_not_decode_root_type"
                            }
                        }))
                    } else {
                        // Null root
                        Ok(json!({
                            "_metadata": {
                                "type": "capnp_null",
                                "size": body.len(),
                                "format": if is_packed_format(body) { "packed" } else { "unpacked" }
                            }
                        }))
                    }
                }
                Err(e) => {
                    // Failed to get root
                    Ok(json!({
                        "_metadata": {
                            "type": "capnp_error",
                            "size": body.len(),
                            "format": if is_packed_format(body) { "packed" } else { "unpacked" },
                            "error": format!("Failed to get root: {}", e)
                        },
                        "debug_data": base64::encode(&body[..std::cmp::min(body.len(), 256)])
                    }))
                }
            }
        }
        Err(_) => {
            // Failed to parse message - fall back to raw data representation
            Ok(json!({
                "type": "capnp_binary",
                "size": body.len(),
                "format": if is_packed_format(body) { "packed" } else { "unpacked" },
                "structured": false,
                "data": base64::encode(&body[..std::cmp::min(body.len(), 256)]) // First 256 bytes for debugging
            }))
        }
    }
}

/// Validate Cap'n Proto binary format
pub fn validate_capnp_format(body: &[u8]) -> ConversionResult<()> {
    // Minimum message size check
    if body.len() < 8 {
        return Err(ConversionError::CapnProtoError(
            "Cap'n Proto message too short - minimum 8 bytes required".to_string(),
        ));
    }

    // Check 8-byte alignment for optimal performance
    if body.as_ptr() as usize % 8 != 0 {
        // Note: This is a performance warning, not a hard requirement with "unaligned" feature
        tracing::warn!("Cap'n Proto message buffer is not 8-byte aligned - may impact performance");
    }

    // Parse segment table header
    let segment_count =
        u32::from_le_bytes(body[0..4].try_into().map_err(|_| {
            ConversionError::CapnProtoError("Invalid segment count bytes".to_string())
        })?)
        .wrapping_add(1) as usize;

    // Validate segment count
    const SEGMENTS_COUNT_LIMIT: usize = 512;
    if segment_count == 0 || segment_count >= SEGMENTS_COUNT_LIMIT {
        return Err(ConversionError::CapnProtoError(format!(
            "Invalid segment count: {} (must be 1-{})",
            segment_count,
            SEGMENTS_COUNT_LIMIT - 1
        )));
    }

    // Calculate segment table size (each segment length is 4 bytes, padded to 8-byte boundary)
    let segment_table_words = (segment_count + 1) / 2; // Round up to word boundary
    let segment_table_bytes = segment_table_words * 8;

    if body.len() < segment_table_bytes {
        return Err(ConversionError::CapnProtoError(format!(
            "Message too short for segment table: {} bytes needed, {} available",
            segment_table_bytes,
            body.len()
        )));
    }

    // Validate segment lengths and calculate total body size
    let mut total_body_words: usize = 0;
    for i in 0..segment_count {
        let segment_length_offset = 4 + (i * 4);
        if segment_length_offset + 4 > body.len() {
            return Err(ConversionError::CapnProtoError(
                "Segment table extends beyond message boundary".to_string(),
            ));
        }

        let segment_length = u32::from_le_bytes(
            body[segment_length_offset..segment_length_offset + 4]
                .try_into()
                .map_err(|_| {
                    ConversionError::CapnProtoError("Invalid segment length bytes".to_string())
                })?,
        ) as usize;

        total_body_words = total_body_words
            .checked_add(segment_length)
            .ok_or_else(|| {
                ConversionError::CapnProtoError(
                    "Message size overflow - segments too large".to_string(),
                )
            })?;
    }

    // Validate total message size
    let expected_total_bytes = segment_table_bytes + (total_body_words * 8);
    if body.len() < expected_total_bytes {
        return Err(ConversionError::CapnProtoError(format!(
            "Message body too short: {} bytes needed, {} available",
            expected_total_bytes,
            body.len()
        )));
    }

    // Validate message doesn't exceed reasonable limits (1GB)
    const MAX_MESSAGE_SIZE: usize = 1024 * 1024 * 1024;
    if expected_total_bytes > MAX_MESSAGE_SIZE {
        return Err(ConversionError::CapnProtoError(format!(
            "Message too large: {} bytes exceeds {} byte limit",
            expected_total_bytes, MAX_MESSAGE_SIZE
        )));
    }

    Ok(())
}

/// Helper function to create method name from GraphQL operation
pub fn create_method_name(operation_name: Option<&str>, operation_type: &str) -> String {
    match operation_name {
        Some(name) => format!("graphql_{}", name),
        None => format!("graphql_{}", operation_type),
    }
}

/// Extract arguments from GraphQL field with variable resolution
pub fn extract_field_arguments(
    field: &Field,
    variables: &Value,
) -> std::collections::HashMap<String, Value> {
    let mut args = std::collections::HashMap::new();

    for (name, value) in &field.node.arguments {
        // Convert GraphQL value to JSON value with variable resolution
        if let Ok(json_value) = graphql_value_to_json_with_variables(&value.node, variables) {
            args.insert(name.node.to_string(), json_value);
        }
    }

    args
}

/// Convert GraphQL value to JSON value (legacy function for backward compatibility)
fn graphql_value_to_json(value: &async_graphql::parser::types::Value) -> ConversionResult<Value> {
    // Use empty variables context for backward compatibility
    graphql_value_to_json_with_variables(value, &Value::Object(serde_json::Map::new()))
}

/// Convert GraphQL value to JSON value with variable context
fn graphql_value_to_json_with_variables(
    value: &async_graphql::parser::types::Value,
    variables: &Value,
) -> ConversionResult<Value> {
    use async_graphql::parser::types::Value as GQLValue;

    let json_value = match value {
        GQLValue::Variable(var_name) => {
            // Resolve variable from provided context
            variables
                .get(var_name.as_str())
                .cloned()
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "GraphQL variable '{}' not found in context, using null",
                        var_name
                    );
                    Value::Null
                })
        }
        GQLValue::Number(n) => Value::Number(
            serde_json::Number::from_f64(n.as_f64().unwrap_or(0.0))
                .unwrap_or_else(|| serde_json::Number::from(0)),
        ),
        GQLValue::String(s) => Value::String(s.clone()),
        GQLValue::Boolean(b) => Value::Bool(*b),
        GQLValue::Null => Value::Null,
        GQLValue::Enum(e) => Value::String(e.to_string()),
        GQLValue::List(list) => {
            let mut json_array = Vec::new();
            for item in list {
                json_array.push(graphql_value_to_json_with_variables(item, variables)?);
            }
            Value::Array(json_array)
        }
        GQLValue::Object(obj) => {
            let mut json_object = serde_json::Map::new();
            for (key, val) in obj {
                json_object.insert(
                    key.to_string(),
                    graphql_value_to_json_with_variables(val, variables)?,
                );
            }
            Value::Object(json_object)
        }
    };

    Ok(json_value)
}
