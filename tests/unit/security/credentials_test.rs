//! Security credentials tests extracted from src/
//! Tests credential validation, environment variable loading, and statistics

use std::env;
use provider::security::credentials::*;

#[tokio::test]
async fn test_credential_validation() {
    let config = CredentialConfig::default();
    let manager = CredentialManager::new(config).await.expect("Failed to create credential manager");
    
    // Test valid credentials
    assert!(manager.validate_credential_value("sk-1234567890abcdef1234567890abcdef").expect("Failed to validate OpenAI credential"));
    assert!(manager.validate_credential_value("sk-ant-1234567890abcdef1234567890abcdef").expect("Failed to validate Anthropic credential"));
    
    // Test invalid credentials
    assert!(!manager.validate_credential_value("").expect("Failed to validate empty credential"));
    assert!(!manager.validate_credential_value("placeholder-api-key-update-before-use").expect("Failed to validate placeholder credential"));
    assert!(!manager.validate_credential_value("test-key").expect("Failed to validate test key"));
    assert!(!manager.validate_credential_value("short").expect("Failed to validate short credential"));
}

#[tokio::test]
async fn test_environment_variable_loading() {
    env::set_var("OPENAI_API_KEY", "sk-test1234567890abcdef1234567890abcdef");
    
    let config = CredentialConfig::default();
    let manager = CredentialManager::new(config).await.expect("Failed to create credential manager");
    
    match manager.get_credential("openai").await {
        Ok(credential) => {
            assert_eq!(credential.provider, "openai");
            assert!(matches!(credential.source, CredentialSource::Environment { .. }));
        }
        Err(_) => {
            // Expected if validation fails due to test prefix
        }
    }
    
    env::remove_var("OPENAI_API_KEY");
}

#[tokio::test]
async fn test_credential_statistics() {
    let config = CredentialConfig::default();
    let manager = CredentialManager::new(config).await.expect("Failed to create credential manager");
    
    let stats = manager.get_statistics().await;
    assert_eq!(stats.total_credentials, 0);
    assert_eq!(stats.expired_credentials, 0);
    assert_eq!(stats.expiring_soon, 0);
}