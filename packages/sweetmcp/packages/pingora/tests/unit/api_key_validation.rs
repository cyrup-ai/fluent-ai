//! Unit tests for API key validation functionality

use std::sync::Arc;
use std::time::Duration;

use sweetmcp_pingora::config::Config;
use sweetmcp_pingora::edge::auth::validation::AuthHandler;
use sweetmcp_pingora::edge::core::service::EdgeService;
use tokio::sync::mpsc;

/// Create a test EdgeService with a known JWT secret
fn create_test_service() -> EdgeService {
    let mut config = Config::from_env().expect("Failed to create test config");
    
    // Use a known secret for testing
    let test_secret = [42u8; 32];
    config.jwt_secret = Arc::new(test_secret);
    
    let (bridge_tx, _) = mpsc::channel(100);
    let peer_registry = sweetmcp_pingora::peer_discovery::PeerRegistry::new();
    
    EdgeService::new(Arc::new(config), bridge_tx, peer_registry)
}

#[tokio::test]
async fn test_generate_and_validate_api_key() {
    let service = create_test_service();
    
    // Generate an API key
    let api_key = AuthHandler::generate_api_key(&service, "test_client", 3600)
        .expect("Failed to generate API key");
    
    // Validate the generated key
    assert!(AuthHandler::validate_api_key(&service, &api_key));
}

#[tokio::test]
async fn test_api_key_expiration() {
    let service = create_test_service();
    
    // Generate an API key that expires in 1 second
    let api_key = AuthHandler::generate_api_key(&service, "test_client", 1)
        .expect("Failed to generate API key");
    
    // Should be valid initially
    assert!(AuthHandler::validate_api_key(&service, &api_key));
    
    // Wait for expiration
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Should be invalid after expiration
    assert!(!AuthHandler::validate_api_key(&service, &api_key));
}

#[tokio::test]
async fn test_invalid_api_key_format() {
    let service = create_test_service();
    
    // Test various invalid formats
    assert!(!AuthHandler::validate_api_key(&service, ""));
    assert!(!AuthHandler::validate_api_key(&service, "invalid"));
    assert!(!AuthHandler::validate_api_key(&service, "not_base64_!@#"));
    assert!(!AuthHandler::validate_api_key(&service, "dGVzdA==")); // "test" in base64, but wrong format
}

#[tokio::test]
async fn test_tampered_api_key() {
    let service = create_test_service();
    
    // Generate a valid API key
    let api_key = AuthHandler::generate_api_key(&service, "test_client", 3600)
        .expect("Failed to generate API key");
    
    // Tamper with the key by changing one character
    let mut tampered_key = api_key.clone();
    tampered_key.pop();
    tampered_key.push('X');
    
    // Tampered key should be invalid
    assert!(!AuthHandler::validate_api_key(&service, &tampered_key));
}

#[tokio::test]
async fn test_different_client_ids() {
    let service = create_test_service();
    
    // Generate keys for different clients
    let key1 = AuthHandler::generate_api_key(&service, "client1", 3600)
        .expect("Failed to generate API key for client1");
    let key2 = AuthHandler::generate_api_key(&service, "client2", 3600)
        .expect("Failed to generate API key for client2");
    
    // Both should be valid
    assert!(AuthHandler::validate_api_key(&service, &key1));
    assert!(AuthHandler::validate_api_key(&service, &key2));
    
    // Keys should be different
    assert_ne!(key1, key2);
}

#[tokio::test]
async fn test_api_key_with_different_secrets() {
    let service1 = create_test_service();
    
    // Create another service with different secret
    let mut config = Config::from_env().expect("Failed to create test config");
    let different_secret = [84u8; 32]; // Different from test secret
    config.jwt_secret = Arc::new(different_secret);
    
    let (bridge_tx, _) = mpsc::channel(100);
    let peer_registry = sweetmcp_pingora::peer_discovery::PeerRegistry::new();
    let service2 = EdgeService::new(Arc::new(config), bridge_tx, peer_registry);
    
    // Generate key with service1
    let api_key = AuthHandler::generate_api_key(&service1, "test_client", 3600)
        .expect("Failed to generate API key");
    
    // Should be valid with service1
    assert!(AuthHandler::validate_api_key(&service1, &api_key));
    
    // Should be invalid with service2 (different secret)
    assert!(!AuthHandler::validate_api_key(&service2, &api_key));
}

#[tokio::test]
async fn test_extract_api_key_expiration() {
    let service = create_test_service();
    
    // Generate an API key with known expiration
    let expiry_seconds = 7200; // 2 hours
    let api_key = AuthHandler::generate_api_key(&service, "test_client", expiry_seconds)
        .expect("Failed to generate API key");
    
    // Extract expiration
    let extracted_expiration = AuthHandler::extract_api_key_expiration(&api_key)
        .expect("Failed to extract expiration");
    
    // Should be approximately now + expiry_seconds
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let expected_expiration = now + expiry_seconds;
    
    // Allow 5 second tolerance for test execution time
    assert!((extracted_expiration as i64 - expected_expiration as i64).abs() <= 5);
}

#[tokio::test]
async fn test_api_key_signature_verification() {
    let service = create_test_service();
    
    // Generate a valid API key
    let api_key = AuthHandler::generate_api_key(&service, "test_client", 3600)
        .expect("Failed to generate API key");
    
    // Verify signature should return true
    let is_valid = AuthHandler::verify_api_key_signature(&service, &api_key)
        .expect("Failed to verify signature");
    assert!(is_valid);
    
    // Test with invalid base64
    let invalid_result = AuthHandler::verify_api_key_signature(&service, "invalid_base64_!@#");
    assert!(invalid_result.is_err());
}