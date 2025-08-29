# Task 2.1: Implement Keychain CA Loading

## Description
Implement platform-specific keychain/certificate store integration for loading certificate authorities from system stores.

## File Location
`src/tls/builder/authority.rs:374`

## Violation
```rust
pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
    // TODO: Implement keychain CA loading
    super::responses::CertificateAuthorityResponse {
        success: false,
        authority: None,
        operation: super::responses::CaOperation::LoadFailed,
        issues: vec!["Keychain loading not yet implemented".to_string()],
        files_created: vec![],
    }
}
```

## Issue Analysis
- **Risk Level**: HIGH - Platform integration missing
- **Impact**: Cannot load certificates from system certificate stores
- **Frequency**: Medium - Enterprise environments often use system certificate stores
- **Business Impact**: Limits enterprise deployment options

## Success Criteria
- [ ] Remove TODO comment and stub implementation
- [ ] Implement macOS Security Framework integration
- [ ] Implement Windows Certificate Store integration
- [ ] Provide fallback for unsupported platforms
- [ ] Extract certificate and private key data
- [ ] Create proper `CertificateAuthority` objects
- [ ] Handle platform-specific errors gracefully

## Dependencies
- **Prerequisite**: Milestone 1 (certificate parsing infrastructure)
- **Platform**: Requires platform-specific dependencies
- **Architecture**: Must maintain fluent_ai_async patterns

## Implementation Plan

### Step 1: Add platform-specific dependencies
```toml
# In Cargo.toml
[target.'cfg(target_os = "macos")'.dependencies]
security-framework = "2.9"

[target.'cfg(target_os = "windows")'.dependencies]
windows = { version = "0.52", features = ["Win32_Security_Cryptography"] }
```

### Step 2: Implement macOS Security Framework integration
```rust
pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
    #[cfg(target_os = "macos")]
    {
        use security_framework::certificate::SecCertificate;
        use security_framework::keychain::SecKeychain;
        use security_framework::identity::SecIdentity;
        use security_framework::os::macos::keychain::SecKeychainExt;
        
        match SecKeychain::default() {
            Ok(keychain) => {
                // Search for certificate by name
                let search_query = format!("kSecAttrLabel = '{}'", self.name);
                
                match keychain.find_certificate(&self.name) {
                    Ok(sec_cert) => {
                        // Extract certificate data
                        let cert_data = sec_cert.to_der();
                        let cert_pem = pem::encode(&pem::Pem {
                            tag: "CERTIFICATE".to_string(),
                            contents: cert_data,
                        });
                        
                        // Find corresponding private key
                        match keychain.find_identity(&self.name) {
                            Ok(identity) => {
                                let private_key = identity.private_key()?;
                                let key_data = private_key.external_representation()?;
                                let key_pem = pem::encode(&pem::Pem {
                                    tag: "PRIVATE KEY".to_string(),
                                    contents: key_data,
                                });
                                
                                // Parse certificate for metadata
                                match parse_certificate_from_pem(&cert_pem) {
                                    Ok(parsed_cert) => {
                                        let authority = CertificateAuthority {
                                            name: self.name.clone(),
                                            certificate_pem: cert_pem,
                                            private_key_pem: key_pem,
                                            metadata: CaMetadata {
                                                subject: parsed_cert.subject.clone(),
                                                issuer: parsed_cert.issuer.clone(),
                                                serial_number: hex::encode(&parsed_cert.serial_number),
                                                valid_from: parsed_cert.not_before,
                                                valid_until: parsed_cert.not_after,
                                                key_algorithm: parsed_cert.key_algorithm.clone(),
                                                key_size: parsed_cert.key_size,
                                                created_at: SystemTime::now(),
                                                source: CaSource::Keychain,
                                            },
                                        };
                                        
                                        return super::responses::CertificateAuthorityResponse {
                                            success: true,
                                            authority: Some(authority),
                                            operation: super::responses::CaOperation::Loaded,
                                            issues: vec![],
                                            files_created: vec![],
                                        };
                                    }
                                    Err(e) => {
                                        return super::responses::CertificateAuthorityResponse {
                                            success: false,
                                            authority: None,
                                            operation: super::responses::CaOperation::LoadFailed,
                                            issues: vec![format!("Failed to parse keychain certificate: {}", e)],
                                            files_created: vec![],
                                        };
                                    }
                                }
                            }
                            Err(e) => {
                                return super::responses::CertificateAuthorityResponse {
                                    success: false,
                                    authority: None,
                                    operation: super::responses::CaOperation::LoadFailed,
                                    issues: vec![format!("Failed to find private key in keychain: {}", e)],
                                    files_created: vec![],
                                };
                            }
                        }
                    }
                    Err(e) => {
                        return super::responses::CertificateAuthorityResponse {
                            success: false,
                            authority: None,
                            operation: super::responses::CaOperation::LoadFailed,
                            issues: vec![format!("Failed to find certificate in keychain: {}", e)],
                            files_created: vec![],
                        };
                    }
                }
            }
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to access keychain: {}", e)],
                    files_created: vec![],
                };
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows Certificate Store implementation
        use windows::Win32::Security::Cryptography::*;
        
        // Implementation for Windows Certificate Store
        // This is a simplified outline - full implementation needed
        super::responses::CertificateAuthorityResponse {
            success: false,
            authority: None,
            operation: super::responses::CaOperation::LoadFailed,
            issues: vec!["Windows certificate store not yet implemented".to_string()],
            files_created: vec![],
        }
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        super::responses::CertificateAuthorityResponse {
            success: false,
            authority: None,
            operation: super::responses::CaOperation::LoadFailed,
            issues: vec!["Keychain loading not supported on this platform".to_string()],
            files_created: vec![],
        }
    }
}
```

### Step 3: Implement Windows Certificate Store support
- Use Windows API for certificate store access
- Handle both user and machine certificate stores
- Extract certificate and private key data
- Convert to PEM format for compatibility

### Step 4: Add comprehensive error handling
- Platform-specific error handling
- Graceful degradation for unsupported platforms
- Clear error messages for troubleshooting

## Platform Considerations

### macOS Security Framework
- Requires user authorization for keychain access
- Supports both user and system keychains
- Integrated with macOS security model
- May require application signing for access

### Windows Certificate Store
- Supports personal, machine, and enterprise stores
- Requires appropriate permissions
- Integration with Windows security model
- May require elevated privileges

### Linux/Other Platforms
- No standard system certificate store
- Could integrate with specific solutions (e.g., PKCS#11)
- Graceful fallback to file-based loading

## Validation Steps
1. Compile code for each platform
2. Test certificate discovery on macOS keychain
3. Test private key extraction and access
4. Verify certificate parsing and metadata extraction
5. Test error handling for missing certificates
6. Test permission handling and user prompts
7. Verify PEM format compatibility

## Security Implications
- Keychain access requires proper user authorization
- Private key extraction must be secure
- Platform security model integration critical
- Audit trail for certificate access important

## Related Issues
- Depends on Milestone 1 for certificate parsing
- Enables enterprise keychain integration
- Foundation for platform-native certificate management