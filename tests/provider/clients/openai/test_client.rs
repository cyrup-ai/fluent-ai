use fluent_ai_provider::clients::openai::client::{OpenAIClient, CircuitBreaker, CircuitBreakerState};
use fluent_ai_provider::clients::openai::{models, endpoints};
use fluent_ai_provider::traits::{CompletionClient, ProviderClient};
use fluent_ai_provider::types::EndpointType;
use std::time::Duration;

#[test]
fn test_client_creation() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string());
    assert!(client.is_ok());
}

#[test]
fn test_client_with_organization() {
    let client = OpenAIClient::new_with_organization(
        "sk-test123456789012345678901234567890123456789012345678901234567890".to_string(),
        Some("org-test123456789012345678901234567890".to_string()),
    );
    assert!(client.is_ok());
}

#[test]
fn test_invalid_api_key() {
    let client = OpenAIClient::new("".to_string());
    assert!(client.is_err());
    
    let client = OpenAIClient::new("short".to_string());
    assert!(client.is_err());
}

#[test]
fn test_model_validation() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    assert!(client.validate_model(models::GPT_4O, EndpointType::ChatCompletions).is_ok());
    assert!(client.validate_model(models::GPT_4O_MINI, EndpointType::ChatCompletions).is_ok());
    assert!(client.validate_model(models::TEXT_EMBEDDING_3_LARGE, EndpointType::Embeddings).is_ok());
    assert!(client.validate_model(models::WHISPER_1, EndpointType::AudioTranscription).is_ok());
    assert!(client.validate_model("invalid-model", EndpointType::ChatCompletions).is_err());
}

#[test]
fn test_model_info() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    let info = client.model_info(models::GPT_4O).expect("Failed to get model info for GPT-4O");
    assert_eq!(info.name, models::GPT_4O);
    assert_eq!(info.family, "gpt-4");
    assert_eq!(info.generation, "gpt-4");
    assert!(info.supports_streaming);
    assert!(info.supports_tools);
    assert!(info.supports_vision);
}

#[test]
fn test_api_key_update() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    let result = client.update_api_key("sk-new1234567890123456789012345678901234567890123456789012345678".to_string());
    assert!(result.is_ok());
    
    let result = client.update_api_key("".to_string());
    assert!(result.is_err());
}

#[test]
fn test_circuit_breaker() {
    let cb = CircuitBreaker::new(3, Duration::from_secs(60));
    assert_eq!(cb.get_state(), CircuitBreakerState::Closed);
    assert!(cb.is_request_allowed());
    
    // Record failures
    cb.record_failure();
    cb.record_failure();
    cb.record_failure();
    
    assert_eq!(cb.get_state(), CircuitBreakerState::Open);
    assert!(!cb.is_request_allowed());
    
    // Record success should reset
    cb.record_success();
    assert_eq!(cb.get_state(), CircuitBreakerState::Closed);
}

#[test]
fn test_endpoint_routing() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    assert_eq!(client.route_endpoint(EndpointType::ChatCompletions), endpoints::CHAT_COMPLETIONS);
    assert_eq!(client.route_endpoint(EndpointType::Embeddings), endpoints::EMBEDDINGS);
    assert_eq!(client.route_endpoint(EndpointType::AudioTranscription), endpoints::AUDIO_TRANSCRIPTIONS);
    assert_eq!(client.route_endpoint(EndpointType::TextToSpeech), endpoints::AUDIO_SPEECH);
    assert_eq!(client.route_endpoint(EndpointType::VisionAnalysis), endpoints::CHAT_COMPLETIONS);
}

#[test]
fn test_completion_client_trait() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    let result = client.completion_model(models::GPT_4O);
    assert!(result.is_ok());
    
    let result = client.completion_model("invalid-model");
    assert!(result.is_err());
}

#[test]
fn test_provider_client_trait() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    assert_eq!(client.provider_name(), "openai");
}

#[test]
fn test_performance_metrics() {
    let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
        .expect("Failed to create OpenAI client for test");
    
    let metrics = client.get_metrics();
    assert_eq!(metrics.total_requests, 0);
    assert_eq!(metrics.successful_requests, 0);
    assert_eq!(metrics.failed_requests, 0);
    assert_eq!(metrics.concurrent_requests, 0);
    
    // Test reset
    client.reset_metrics();
    let metrics = client.get_metrics();
    assert_eq!(metrics.total_requests, 0);
}