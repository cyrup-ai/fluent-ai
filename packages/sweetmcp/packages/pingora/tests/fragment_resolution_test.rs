use async_graphql::parser::{parse_query, types::*};
use serde_json::{json, Value};
use sweetmcp_pingora::normalize::{
    parsers::graphql_to_json_rpc,
    types::{ConversionError, FragmentCache, FragmentRegistry, GraphQLContext},
};

#[tokio::test]
async fn test_fragment_spread_resolution() {
    let query = r#"
        fragment UserFields on User {
            id
            name
            email
        }
        
        query GetUser {
            user {
                ...UserFields
                createdAt
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-1");

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    // Verify the JSON-RPC structure
    assert_eq!(json_rpc["jsonrpc"], "2.0");
    assert_eq!(json_rpc["method"], "graphql_query");
    assert_eq!(json_rpc["id"], "test-request-1");

    // Verify fragment resolution metadata
    let params = &json_rpc["params"];
    assert_eq!(params["resolvedFragments"], true);
    assert_eq!(params["fragmentCount"], 1);

    // Verify fields include both fragment fields and direct fields
    let fields = params["fields"].as_array().unwrap();
    let field_names: Vec<&str> = fields.iter().filter_map(|f| f.as_str()).collect();

    assert!(field_names.contains(&"id"));
    assert!(field_names.contains(&"name"));
    assert!(field_names.contains(&"email"));
    assert!(field_names.contains(&"createdAt"));
}

#[tokio::test]
async fn test_nested_fragment_spreads() {
    let query = r#"
        fragment ContactInfo on User {
            email
            phone
        }
        
        fragment UserProfile on User {
            ...ContactInfo
            avatar
            bio
        }
        
        query GetUserProfile {
            user {
                id
                ...UserProfile
                lastLogin
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-2");

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    let params = &json_rpc["params"];
    assert_eq!(params["resolvedFragments"], true);
    assert_eq!(params["fragmentCount"], 2);

    // Verify all nested fields are resolved
    let fields = params["fields"].as_array().unwrap();
    let field_names: Vec<&str> = fields.iter().filter_map(|f| f.as_str()).collect();

    // Direct fields
    assert!(field_names.contains(&"id"));
    assert!(field_names.contains(&"lastLogin"));

    // From UserProfile fragment
    assert!(field_names.contains(&"avatar"));
    assert!(field_names.contains(&"bio"));

    // From nested ContactInfo fragment
    assert!(field_names.contains(&"email"));
    assert!(field_names.contains(&"phone"));
}

#[tokio::test]
async fn test_circular_fragment_dependency_detection() {
    let query = r#"
        fragment FragmentA on User {
            id
            ...FragmentB
        }
        
        fragment FragmentB on User {
            name
            ...FragmentA
        }
        
        query GetUser {
            user {
                ...FragmentA
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-3");

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Circular fragment dependency"));
}

#[tokio::test]
async fn test_missing_fragment_error() {
    let query = r#"
        query GetUser {
            user {
                id
                ...NonExistentFragment
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-4");

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Fragment 'NonExistentFragment' not found"));
}

#[tokio::test]
async fn test_inline_fragments() {
    let query = r#"
        query GetNode {
            node {
                id
                ... on User {
                    name
                    email
                }
                ... on Post {
                    title
                    content
                }
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-5");

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    let params = &json_rpc["params"];
    let fields = params["fields"].as_array().unwrap();
    let field_names: Vec<&str> = fields.iter().filter_map(|f| f.as_str()).collect();

    // Should include fields from both inline fragments
    assert!(field_names.contains(&"id"));
    assert!(field_names.contains(&"name"));
    assert!(field_names.contains(&"email"));
    assert!(field_names.contains(&"title"));
    assert!(field_names.contains(&"content"));
}

#[tokio::test]
async fn test_fragment_cache_performance() {
    let mut context = GraphQLContext::new();

    // Register a fragment
    let doc = parse_query(
        r#"
        fragment TestFragment on User {
            id
            name
            email
        }
    "#,
    )
    .unwrap();

    for (name, fragment) in &doc.fragments {
        context
            .fragment_registry
            .register_fragment(name.to_string(), fragment.clone())
            .unwrap();
    }

    // First access should be a cache miss
    let cached = context.fragment_cache.get("TestFragment");
    assert!(cached.is_none());

    // Simulate caching resolved fields
    context.fragment_cache.insert(
        "TestFragment".to_string(),
        vec!["id".to_string(), "name".to_string(), "email".to_string()],
    );

    // Second access should be a cache hit
    let cached = context.fragment_cache.get("TestFragment");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().len(), 3);

    // Verify cache statistics
    let (hits, misses, hit_rate) = context.fragment_cache.get_stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
    assert_eq!(hit_rate, 0.5);
}

#[tokio::test]
async fn test_fragment_registry_operations() {
    let mut registry = FragmentRegistry::new();

    // Parse a fragment definition
    let doc = parse_query(
        r#"
        fragment UserInfo on User {
            id
            name
        }
    "#,
    )
    .unwrap();

    // Register the fragment
    for (name, fragment) in &doc.fragments {
        let result = registry.register_fragment(name.to_string(), fragment.clone());
        assert!(result.is_ok());
    }

    // Verify fragment exists
    assert!(registry.has_fragment("UserInfo"));
    assert!(!registry.has_fragment("NonExistent"));

    // Get fragment names
    let names = registry.get_fragment_names();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"UserInfo".to_string()));

    // Test duplicate registration
    for (name, fragment) in &doc.fragments {
        let result = registry.register_fragment(name.to_string(), fragment.clone());
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_complex_fragment_scenario() {
    let query = r#"
        fragment BaseUser on User {
            id
            name
        }
        
        fragment UserWithProfile on User {
            ...BaseUser
            email
            avatar
        }
        
        fragment UserWithPosts on User {
            ...BaseUser
            posts {
                id
                title
            }
        }
        
        query GetCompleteUser {
            user {
                ...UserWithProfile
                ...UserWithPosts
                createdAt
                lastLogin
            }
        }
    "#;

    let result = graphql_to_json_rpc(query, json!({}), None, "test-request-6");

    assert!(result.is_ok());
    let json_rpc = result.unwrap();

    let params = &json_rpc["params"];
    assert_eq!(params["resolvedFragments"], true);
    assert_eq!(params["fragmentCount"], 3);

    // Verify all fields are included (including duplicates from BaseUser)
    let fields = params["fields"].as_array().unwrap();
    let field_names: Vec<&str> = fields.iter().filter_map(|f| f.as_str()).collect();

    // Base fields (may appear multiple times due to fragment spreading)
    assert!(field_names.contains(&"id"));
    assert!(field_names.contains(&"name"));

    // Profile-specific fields
    assert!(field_names.contains(&"email"));
    assert!(field_names.contains(&"avatar"));

    // Posts-specific fields
    assert!(field_names.contains(&"posts"));
    assert!(field_names.contains(&"title"));

    // Direct query fields
    assert!(field_names.contains(&"createdAt"));
    assert!(field_names.contains(&"lastLogin"));
}
