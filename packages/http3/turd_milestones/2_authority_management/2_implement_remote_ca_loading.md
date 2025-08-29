# Task 2.2: Implement Remote CA Loading

## Description
Implement HTTP-based certificate authority loading using the Http3 builder API to fetch certificates from remote URLs.

## File Location
`src/tls/builder/authority.rs:401`

## Violation
```rust
pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
    // TODO: Implement remote CA loading
    super::responses::CertificateAuthorityResponse {
        success: false,
        authority: None,
        operation: super::responses::CaOperation::LoadFailed,
        issues: vec!["Remote loading not yet implemented".to_string()],
        files_created: vec![],
    }
}
```

## Issue Analysis
- **Risk Level**: HIGH - Missing HTTP-based CA loading capability
- **Impact**: Cannot load certificates from remote URLs or certificate authorities
- **Frequency**: Medium - Used for federated certificate management
- **Business Impact**: Limits distributed and cloud certificate authority deployment

## Success Criteria
- [ ] Remove TODO comment and stub implementation
- [ ] Implement Http3 client integration for certificate fetching
- [ ] Support PEM and DER certificate formats
- [ ] Handle HTTP errors and timeouts gracefully
- [ ] Parse and validate downloaded certificates
- [ ] Create proper `CertificateAuthority` objects
- [ ] Maintain fluent_ai_async streaming patterns

## Dependencies
- **Prerequisite**: Milestone 1 (certificate parsing infrastructure)
- **Uses**: Http3 builder API from the codebase
- **Architecture**: Must use fluent_ai_async patterns (NO tokio)

## Implementation Plan

### Step 1: Implement Http3-based certificate fetching
```rust
pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
    use crate::Http3;
    use fluent_ai_async::AsyncStream;
    use crate::tls::certificate::parsing::parse_certificate_from_pem;
    
    let client = Http3::default();
    
    // Fetch certificate from remote URL
    let mut response_stream = match client
        .get(&self.url)
        .with_timeout(self.timeout)
        .send_stream()
        .await {
            Ok(stream) => stream,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to initiate request to {}: {}", self.url, e)],
                    files_created: vec![],
                };
            }
        };
    
    // Collect response data
    let mut cert_data = Vec::new();
    while let Some(chunk_result) = response_stream.next().await {
        match chunk_result {
            Ok(chunk) => cert_data.extend_from_slice(&chunk),
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to download certificate data: {}", e)],
                    files_created: vec![],
                };
            }
        }
    }
    
    // Convert to UTF-8 string
    let cert_content = match String::from_utf8(cert_data) {
        Ok(content) => content,
        Err(e) => {
            return super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::LoadFailed,
                issues: vec![format!("Invalid UTF-8 in certificate data: {}", e)],
                files_created: vec![],
            };
        }
    };
    
    // Determine if content is PEM or DER based on format
    let cert_pem = if cert_content.starts_with("-----BEGIN CERTIFICATE-----") {
        cert_content
    } else {
        // Assume DER format, convert to PEM
        match base64::decode(&cert_content.trim()) {
            Ok(der_data) => {
                format!(
                    "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----",
                    base64::encode(&der_data)
                        .chars()
                        .collect::<Vec<_>>()
                        .chunks(64)
                        .map(|chunk| chunk.iter().collect::<String>())
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            }
            Err(_) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec!["Certificate data is neither valid PEM nor base64-encoded DER".to_string()],
                    files_created: vec![],
                };
            }
        }
    };
    
    // Parse and validate certificate
    match parse_certificate_from_pem(&cert_pem) {
        Ok(parsed_cert) => {
            // Verify this is a CA certificate
            if !parsed_cert.is_ca {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec!["Downloaded certificate is not a valid CA certificate".to_string()],
                    files_created: vec![],
                };
            }
            
            // Create CertificateAuthority object
            let authority = CertificateAuthority {
                name: self.name.clone(),
                certificate_pem: cert_pem,
                private_key_pem: String::new(), // Remote CAs typically don't include private keys
                metadata: CaMetadata {
                    subject: parsed_cert.subject.clone(),
                    issuer: parsed_cert.issuer.clone(),
                    serial_number: hex::encode(&parsed_cert.serial_number),
                    valid_from: parsed_cert.not_before,
                    valid_until: parsed_cert.not_after,
                    key_algorithm: parsed_cert.key_algorithm.clone(),
                    key_size: parsed_cert.key_size,
                    created_at: SystemTime::now(),
                    source: CaSource::Remote { url: self.url.clone() },
                },
            };
            
            super::responses::CertificateAuthorityResponse {
                success: true,
                authority: Some(authority),
                operation: super::responses::CaOperation::Loaded,
                issues: vec![],
                files_created: vec![],
            }
        }
        Err(e) => {
            super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::LoadFailed,
                issues: vec![format!("Failed to parse downloaded certificate: {}", e)],
                files_created: vec![],
            }
        }
    }
}
```

### Step 2: Add enhanced format detection
- Support common certificate URL patterns
- Handle content-type headers for format detection
- Support certificate chains (multiple certificates)
- Validate certificate authority constraints

### Step 3: Implement robust error handling
- HTTP status code handling
- Network timeout handling
- Certificate validation errors
- Malformed data handling

## Enhanced Implementation with Chain Support
```rust
// Support for certificate chains
let cert_chain = if cert_content.contains("-----BEGIN CERTIFICATE-----") {
    // Parse PEM chain
    parse_certificate_chain_from_pem(&cert_content)?
} else {
    // Single DER certificate
    vec![parse_single_der_certificate(&cert_content)?]
};

// Find the CA certificate in the chain
let ca_cert = cert_chain.iter()
    .find(|cert| cert.is_ca && cert.key_usage.contains(&"keyCertSign".to_string()))
    .ok_or_else(|| "No valid CA certificate found in chain")?;
```

## URL Patterns Support
```rust
// Common CA certificate URL patterns
match self.url.as_str() {
    url if url.ends_with(".pem") => "PEM",
    url if url.ends_with(".crt") => "PEM", 
    url if url.ends_with(".cer") => "DER",
    url if url.ends_with(".der") => "DER",
    _ => "UNKNOWN", // Auto-detect
}
```

## Validation Steps
1. Compile code and verify Http3 integration
2. Test with public CA certificate URLs
3. Test with PEM format certificates
4. Test with DER format certificates
5. Test timeout handling
6. Test HTTP error handling (404, 500, etc.)
7. Test certificate chain handling
8. Verify CA validation works correctly
9. Test with invalid/malformed certificates

## Security Implications
- Remote certificate fetching must validate certificate authenticity
- HTTPS should be preferred for certificate URLs
- Downloaded certificates must be validated as proper CAs
- Timeout and retry logic prevents DoS attacks
- Certificate chain validation is critical

## Common Remote CA Sources
- Public certificate authorities (Let's Encrypt, etc.)
- Enterprise internal CAs
- Cloud provider certificate services
- Federal PKI and government CAs
- Industry-specific certificate authorities

## Related Issues
- Depends on Milestone 1 for certificate parsing
- Uses fluent_ai_async HTTP client architecture
- Enables federated certificate authority management
- Foundation for distributed certificate deployment