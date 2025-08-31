use serde_json::{json, Value};

use super::parsers::{create_basic_schema_types, graphql_to_json_rpc_with_schema};
use super::schema_introspection::SchemaIntrospector;
use super::types::{GraphQLTypeInfo, GraphQLTypeKind};

#[tokio::test]
async fn test_graphql_to_json_rpc_with_fallback_schema() {
    let query = r#"
        query GetUser {
            user {
                id
                name
                ...UserDetails
            }
        }
        
        fragment UserDetails on User {
            email
            profile {
                avatar
            }
        }
    "#;

    let variables = json!({});
    let operation_name = None;
    let request_id = "test-123";

    // Test with no upstream URL (should use fallback schema)
    let result = graphql_to_json_rpc_with_schema(
        query,
        variables.clone(),
        operation_name.clone(),
        request_id,
        None,
    );

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    // Verify JSON-RPC structure
    assert_eq!(json_rpc["jsonrpc"], "2.0");
    assert_eq!(json_rpc["id"], request_id);
    assert_eq!(json_rpc["method"], "GetUser");

    // Verify params contain resolved fragments
    let params = &json_rpc["params"];
    assert!(params["resolvedFragments"].as_bool().unwrap_or(false));
    assert!(params["query"].as_str().unwrap().contains("UserDetails"));
}

#[tokio::test]
async fn test_fragment_resolution_with_schema_validation() {
    let query = r#"
        query GetData {
            data {
                ...DataFragment
            }
        }
        
        fragment DataFragment on Query {
            additionalField
        }
    "#;

    let variables = json!({});
    let request_id = "test-fragment-validation";

    // Test fragment resolution with basic schema types
    let result = graphql_to_json_rpc_with_schema(query, variables, None, request_id, None);

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    // Verify the fragment was processed (even with basic schema)
    let params = &json_rpc["params"];
    assert!(params.get("resolvedFragments").is_some());
}

#[test]
fn test_schema_introspector_with_custom_settings() {
    use std::time::Duration;

    let introspector =
        SchemaIntrospector::with_settings(Duration::from_secs(10), Duration::from_secs(1800))
            .expect("Failed to create introspector with custom settings");

    // Just verify it was created successfully - we can't easily test the internal settings
    // without exposing them, but the creation itself validates the settings work
    assert!(true); // Placeholder assertion - creation success is the real test
}

#[test]
fn test_basic_schema_type_creation() {
    use super::parsers::create_basic_schema_types;

    let schema_types = create_basic_schema_types();

    // Verify Query type
    assert!(schema_types.contains_key("Query"));
    let query_type = &schema_types["Query"];
    assert_eq!(query_type.kind, GraphQLTypeKind::Object);
    assert_eq!(query_type.name, "Query");

    // Verify Mutation type
    assert!(schema_types.contains_key("Mutation"));
    let mutation_type = &schema_types["Mutation"];
    assert_eq!(mutation_type.kind, GraphQLTypeKind::Object);
    assert_eq!(mutation_type.name, "Mutation");

    // Verify Subscription type
    assert!(schema_types.contains_key("Subscription"));
    let subscription_type = &schema_types["Subscription"];
    assert_eq!(subscription_type.kind, GraphQLTypeKind::Object);
    assert_eq!(subscription_type.name, "Subscription");
}

#[test]
fn test_fallback_schema_completeness() {
    let schema_types = create_basic_schema_types();

    // Verify all required root types are present
    let required_types = ["Query", "Mutation", "Subscription"];
    for type_name in &required_types {
        assert!(
            schema_types.contains_key(*type_name),
            "Missing required type: {}",
            type_name
        );

        let type_info = &schema_types[*type_name];
        assert_eq!(type_info.name, *type_name);
        assert_eq!(type_info.kind, GraphQLTypeKind::Object);
        assert!(type_info.interfaces.is_empty());
        assert!(type_info.possible_types.is_empty());
        // Fields are empty in fallback schema (no introspection data available)
        assert!(type_info.fields.is_empty());
    }
}
