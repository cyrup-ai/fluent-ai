use fluent_ai::engine::{
    get_default_engine, get_engine, register_engine, set_default_engine, NoOpEngine,
};
use std::sync::Arc;

#[tokio::test]
async fn test_engine_registry_basic_operations() {
    // Test registering an engine
    let no_op_engine = Arc::new(NoOpEngine);
    register_engine("test_engine", no_op_engine.clone()).unwrap();

    // Test retrieving a registered engine
    let retrieved = get_engine("test_engine").unwrap();
    // Test that we can call methods on the retrieved engine
    let tools = retrieved.available_tools().await;
    assert!(tools.is_ok());

    // Test setting default engine (requires Arc<dyn Engine>, not string)
    set_default_engine(no_op_engine.clone()).unwrap();

    // Test getting default engine
    let default = get_default_engine().unwrap();
    // Test that we can call methods on the default engine
    let tools = default.available_tools().await;
    assert!(tools.is_ok());
}

#[tokio::test]
async fn test_engine_registry_error_cases() {
    // Test getting non-existent engine
    let result = get_engine("non_existent");
    assert!(result.is_err());

    // Note: There's no way to set a "non-existent" default since it requires Arc<dyn Engine>
    // So we'll test that getting default works when no custom default is set
    let default = get_default_engine();
    assert!(default.is_ok()); // Should have NoOpEngine as default
}
