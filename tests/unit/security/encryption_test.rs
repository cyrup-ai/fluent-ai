//! Security encryption tests extracted from src/
//! Tests encryption roundtrip, key rotation, and key age tracking

use std::env;
use provider::security::encryption::*;

#[test]
fn test_encryption_roundtrip() {
    let temp_dir = env::temp_dir();
    let key_path = temp_dir.join("test_encryption_key");
    
    // Remove test key if it exists
    let _ = std::fs::remove_file(&key_path);
    
    let engine = EncryptionEngine::new(key_path.to_str().expect("Failed to convert path to string")).expect("Failed to create encryption engine");
    
    let test_credential = "sk-1234567890abcdef1234567890abcdef";
    
    // Test encryption
    let encrypted = engine.encrypt_credential(test_credential).expect("Failed to encrypt credential");
    assert_ne!(encrypted, test_credential);
    
    // Test decryption
    let decrypted = engine.decrypt_credential(&encrypted).expect("Failed to decrypt credential");
    assert_eq!(decrypted, test_credential);
    
    // Cleanup
    let _ = std::fs::remove_file(&key_path);
}

#[test]
fn test_key_rotation() {
    let temp_dir = env::temp_dir();
    let key_path = temp_dir.join("test_rotation_key");
    
    // Remove test key if it exists
    let _ = std::fs::remove_file(&key_path);
    
    let mut engine = EncryptionEngine::new(key_path.to_str().expect("Failed to convert path to string")).expect("Failed to create encryption engine");
    
    let test_data = b"test data for encryption";
    let encrypted_before = engine.encrypt(test_data).expect("Failed to encrypt data before rotation");
    
    // Rotate key
    engine.rotate_key().expect("Failed to rotate key");
    
    // Old encrypted data should still decrypt if we kept the old key
    // New encryption should use the new key
    let encrypted_after = engine.encrypt(test_data).expect("Failed to encrypt data after rotation");
    
    // The ciphertexts should be different (different keys)
    assert_ne!(encrypted_before.ciphertext, encrypted_after.ciphertext);
    
    // Both should decrypt correctly with current key
    let decrypted_after = engine.decrypt(&encrypted_after).expect("Failed to decrypt after rotation");
    assert_eq!(decrypted_after, test_data);
    
    // Cleanup
    let _ = std::fs::remove_file(&key_path);
}

#[test]
fn test_key_age_tracking() {
    let temp_dir = env::temp_dir();
    let key_path = temp_dir.join("test_age_key");
    
    // Remove test key if it exists
    let _ = std::fs::remove_file(&key_path);
    
    let engine = EncryptionEngine::new(key_path.to_str().expect("Failed to convert path to string")).expect("Failed to create encryption engine");
    
    let age = engine.key_age();
    assert!(age.as_secs() < 1); // Should be very recent
    
    let needs_rotation = engine.needs_key_rotation(std::time::Duration::from_secs(3600));
    assert!(!needs_rotation); // Shouldn't need rotation yet
    
    // Cleanup
    let _ = std::fs::remove_file(&key_path);
}