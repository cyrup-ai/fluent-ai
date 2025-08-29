# Task 1.1: Implement Root Certificate Conversion

## Description
Replace stub implementation of custom root certificate conversion with proper PEM to rustls certificate conversion.

## File Location
`src/tls/tls_manager.rs:240`

## Violation
```rust
// Add custom root certificates
for cert_pem in &self.config.custom_root_certs {
    if let Ok(cert) = parse_certificate_from_der(cert_pem.as_bytes()) {
        // Convert parsed certificate back to rustls certificate
        // This is a simplified implementation - in production, proper conversion would be needed
        tracing::debug!("Added custom root certificate: {}", cert.subject);
    }
}
```

## Issue Analysis
- **Risk Level**: CRITICAL - Core security functionality missing
- **Impact**: Custom root certificates are parsed but not actually added to trust store
- **Frequency**: Medium - When custom root certificates are configured
- **Security Impact**: High - Custom CAs not trusted, certificate validation may fail

## Success Criteria
- [ ] Remove stub comment completely
- [ ] Implement proper PEM to rustls certificate conversion
- [ ] Actually add certificates to the rustls root store
- [ ] Handle conversion errors gracefully
- [ ] Maintain existing configuration interface
- [ ] Verify custom root certificates are trusted

## Dependencies
- **Prerequisite**: Task 1.0 (certificate parsing stub replacement)
- **Uses**: Real certificate parsing from Task 1.0
- **Architecture**: Must maintain fluent_ai_async patterns

## Implementation Plan

### Step 1: Implement proper PEM parsing and conversion
```rust
// BEFORE:
for cert_pem in &self.config.custom_root_certs {
    if let Ok(cert) = parse_certificate_from_der(cert_pem.as_bytes()) {
        // This is a simplified implementation - in production, proper conversion would be needed
        tracing::debug!("Added custom root certificate: {}", cert.subject);
    }
}

// AFTER:
use rustls_pemfile;
use std::io::Cursor;

for cert_pem in &self.config.custom_root_certs {
    let mut cursor = Cursor::new(cert_pem.as_bytes());
    match rustls_pemfile::certs(&mut cursor).next() {
        Some(Ok(cert_der)) => {
            match rustls::Certificate(cert_der) {
                rustls_cert => {
                    root_store.add(&rustls_cert)
                        .map_err(|e| TlsError::CertificateValidation(format!("Failed to add root cert: {}", e)))?;
                    
                    // Parse for logging (using real parser now)
                    if let Ok(parsed) = parse_certificate_from_der(&cert_der) {
                        tracing::debug!("Added custom root certificate: {}", 
                            parsed.subject.get("CN").unwrap_or(&"Unknown".to_string()));
                    }
                }
            }
        }
        Some(Err(e)) => {
            return Err(TlsError::CertificateParsing(format!("Failed to parse PEM: {}", e)));
        }
        None => {
            return Err(TlsError::CertificateParsing("No certificate found in PEM data".to_string()));
        }
    }
}
```

### Step 2: Handle multiple certificates in PEM
- Iterate through all certificates in PEM data
- Add each valid certificate to root store
- Log successful additions with certificate details

### Step 3: Improve error handling
- Distinguish between PEM parsing errors and trust store errors
- Provide actionable error messages
- Continue processing other certificates on individual failures

## Technical Notes
- PEM files may contain multiple certificates
- `rustls_pemfile::certs()` returns iterator over certificate DER data
- `root_store.add()` integrates certificates into rustls trust validation
- Real certificate parsing from Task 1.0 enables proper logging

## Alternative Implementation (More Robust)
```rust
for cert_pem in &self.config.custom_root_certs {
    let mut cursor = Cursor::new(cert_pem.as_bytes());
    let mut cert_count = 0;
    
    for cert_result in rustls_pemfile::certs(&mut cursor) {
        match cert_result {
            Ok(cert_der) => {
                let rustls_cert = rustls::Certificate(cert_der.clone());
                match root_store.add(&rustls_cert) {
                    Ok(()) => {
                        cert_count += 1;
                        if let Ok(parsed) = parse_certificate_from_der(&cert_der) {
                            tracing::debug!("Added custom root certificate {}: {}", 
                                cert_count,
                                parsed.subject.get("CN").unwrap_or(&"Unknown".to_string()));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to add root certificate {}: {}", cert_count + 1, e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to parse certificate from PEM: {}", e);
            }
        }
    }
    
    if cert_count == 0 {
        return Err(TlsError::CertificateParsing("No valid certificates found in PEM data".to_string()));
    }
}
```

## Validation Steps
1. Compile code and verify integration
2. Test with single certificate PEM files
3. Test with multi-certificate PEM files
4. Verify certificates are actually trusted in TLS connections
5. Test error handling with malformed PEM data
6. Confirm custom root CAs work for certificate validation

## Security Implications
- Custom root certificates enable private CA trust
- Proper integration is critical for enterprise deployments
- Error handling must prevent trust store corruption
- Certificate validation must respect custom root authorities

## Related Issues
- Depends on Task 1.0 for real certificate parsing
- Enables proper enterprise CA trust
- Foundation for certificate authority management