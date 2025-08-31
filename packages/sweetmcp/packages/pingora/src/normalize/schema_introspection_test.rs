use super::schema_introspection::SchemaIntrospector;
use super::types::{GraphQLTypeInfo, GraphQLTypeKind};

#[tokio::test]
async fn test_schema_introspector_creation() {
    let introspector = SchemaIntrospector::new().expect("Failed to create introspector");

    // Test cache statistics on new instance
    let (total, valid) = introspector.cache_stats().await;
    assert_eq!(total, 0);
    assert_eq!(valid, 0);
}

#[tokio::test]
async fn test_cache_cleanup() {
    let introspector = SchemaIntrospector::new().expect("Failed to create introspector");

    // Test cache cleanup on empty cache
    introspector.cleanup_cache().await;
    let (total, valid) = introspector.cache_stats().await;
    assert_eq!(total, 0);
    assert_eq!(valid, 0);
}

#[test]
fn test_create_basic_schema_types() {
    use super::parsers::create_basic_schema_types;

    let schema_types = create_basic_schema_types();

    // Verify all basic types are present
    assert!(schema_types.contains_key("Query"));
    assert!(schema_types.contains_key("Mutation"));
    assert!(schema_types.contains_key("Subscription"));

    // Verify they are all Object types
    for (name, type_info) in &schema_types {
        assert_eq!(type_info.name, *name);
        assert_eq!(type_info.kind, GraphQLTypeKind::Object);
        assert!(type_info.interfaces.is_empty());
        assert!(type_info.possible_types.is_empty());
    }
}
