use domain::model_info::{ModelInfoData, ModelRegistry};

#[test]
fn test_model_info_creation() {
    let model = ModelInfoData {
        provider_name: "test".to_string(),
        name: "test-model".to_string(),
        max_input_tokens: Some(1000),
        max_output_tokens: Some(1000),
        input_price: Some(1.0),
        output_price: Some(2.0),
        supports_vision: Some(false),
        supports_function_calling: Some(true),
        require_max_tokens: Some(false),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(1024),
    };

    assert_eq!(model.provider_name, "test");
    assert_eq!(model.name, "test-model");
    assert_eq!(model.max_input_tokens, Some(1000));
    assert_eq!(model.supports_function_calling, Some(true));
}

#[test]
fn test_model_registry() {
    let mut registry = ModelRegistry::new();
    let model = ModelInfoData::new("test", "test-model")
        .with_max_input_tokens(1000)
        .with_max_output_tokens(1000)
        .with_pricing(1.0, 2.0)
        .with_function_calling(true);

    registry.register(model);
    let found = registry.get("test", "test-model");
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "test-model");
}
