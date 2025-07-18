use std::sync::Arc;

use fluent_ai::engine::builder::{EngineBuilder, engine_builder};
use fluent_ai::engine::{NoOpEngine, get_default_engine, get_engine};

#[tokio::test]
async fn test_builder_basic_flow() {
    let engine = Arc::new(NoOpEngine);

    let config = engine_builder()
        .engine(engine.clone())
        .name("test_engine")
        .build()
        .expect("Builder should create valid config");

    assert_eq!(config.name, "test_engine");
    assert!(!config.is_default);

    // Test that the engine works by calling a method
    let _tools = config.engine.available_tools().await;
}

#[tokio::test]
async fn test_builder_with_default() {
    let engine = Arc::new(NoOpEngine);

    let config = engine_builder()
        .engine(engine.clone())
        .name("test_engine")
        .as_default()
        .build()
        .expect("Builder should create valid config with default");

    assert_eq!(config.name, "test_engine");
    assert!(config.is_default);

    // Test that the engine works by calling a method
    let _tools = config.engine.available_tools().await;
}

#[tokio::test]
async fn test_builder_register() {
    let engine = Arc::new(NoOpEngine);

    // Test basic registration
    engine_builder()
        .engine(engine.clone())
        .name("test_register_engine")
        .build_and_register()
        .expect("Builder should register engine successfully");

    // Verify it was registered
    let retrieved = get_engine("test_register_engine")
        .expect("Engine should be retrievable after registration");
    let _tools = retrieved.available_tools().await;
}

#[tokio::test]
async fn test_builder_register_as_default() {
    let engine = Arc::new(NoOpEngine);

    // Test registration as default
    engine_builder()
        .engine(engine.clone())
        .name("test_default_engine")
        .as_default()
        .build_and_register()
        .expect("Builder should register default engine successfully");

    // Verify it was set as default
    let default =
        get_default_engine().expect("Default engine should be retrievable after registration");
    let _tools = default.available_tools().await;
}
