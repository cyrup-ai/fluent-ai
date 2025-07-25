use std::sync::Arc;

use fluent_ai::engine::{
    NoOpEngine, get_default_engine, get_engine, register_engine, set_default_engine};

#[tokio::test]
async fn test_engine_registry_basic_operations() {
    // Test registering an engine
    let no_op_engine = Arc::new(NoOpEngine);
    assert!(register_engine("test_engine", no_op_engine.clone()));

    // Test retrieving a registered engine
    let retrieved = get_engine("test_engine").unwrap();
    // Test that we can call methods on the retrieved engine
    let _tools = retrieved.available_tools().await;

    // Test setting default engine (requires Arc<dyn Engine>, not string)
    assert!(set_default_engine(no_op_engine.clone()));

    // Test getting default engine
    let default = get_default_engine().unwrap();
    // Test that we can call methods on the default engine
    let _tools = default.available_tools().await;
}

#[tokio::test]
async fn test_engine_registry_error_cases() {
    // Test getting non-existent engine
    let result = get_engine("non_existent");
    assert!(result.is_none());

    // Note: There's no way to set a "non-existent" default since it requires Arc<dyn Engine>
    // So we'll test that getting default works when no custom default is set
    let default = get_default_engine();
    assert!(default.is_some()); // Should have NoOpEngine as default
}
