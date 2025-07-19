use fluent_ai_domain::models::Models;

#[test]
fn test_models_enum_integration() {
    // Test that we can use the Models enum from the domain package
    let model = Models::Gpt41;
    let info = model.info();

    // Verify basic properties
    assert_eq!(info.provider_name, "openai");
    assert_eq!(info.name, "gpt-4.1");

    // Test another model
    let claude = Models::ClaudeOpus420250514;
    let claude_info = claude.info();
    assert_eq!(claude_info.provider_name, "anthropic");
    assert_eq!(claude_info.name, "claude-opus-4-20250514");

    // Test serialization/deserialization
    let serialized = serde_json::to_string(&model).unwrap();
    let deserialized: Models = serde_json::from_str(&serialized).unwrap();
    assert_eq!(model, deserialized);
}
