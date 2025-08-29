# Task 1.0: Replace Certificate Parsing Stub

## Description
Replace stub implementation of certificate parsing with proper x509-cert integration to enable real certificate analysis.

## File Location
`src/tls/tls_manager.rs:25`

## Violation
```rust
fn parse_certificate_from_der(der_bytes: &[u8]) -> Result<TypesParsedCertificate, TlsError> {
    // Simple implementation - in production this would be more comprehensive
    Ok(TypesParsedCertificate {
        subject: "Unknown".to_string(),
        serial_number: der_bytes.get(..16).unwrap_or(&[]).to_vec(),
        ocsp_urls: Vec::new(),
        // ... hardcoded fields
    })
}
```

## Issue Analysis
- **Risk Level**: CRITICAL - Core functionality missing
- **Impact**: Certificate parsing returns hardcoded values instead of actual data
- **Frequency**: High - Used throughout TLS certificate validation
- **Business Impact**: Certificate validation completely broken

## Success Criteria
- [ ] Remove hardcoded return values completely
- [ ] Integrate with existing `crate::tls::certificate::parser` module
- [ ] Parse actual certificate fields from DER data
- [ ] Maintain function signature compatibility
- [ ] Return proper error for invalid DER data
- [ ] All certificate-dependent functionality works correctly

## Dependencies
- **Prerequisite**: Milestone 0 must be completed (unwrap fixes)
- **Uses**: Existing `parse_x509_certificate_from_der_internal` function
- **Architecture**: Must maintain fluent_ai_async patterns

## Implementation Plan

### Step 1: Replace stub with real implementation
```rust
// BEFORE:
fn parse_certificate_from_der(der_bytes: &[u8]) -> Result<TypesParsedCertificate, TlsError> {
    // Simple implementation - in production this would be more comprehensive
    Ok(TypesParsedCertificate {
        subject: "Unknown".to_string(),
        serial_number: der_bytes.get(..16).unwrap_or(&[]).to_vec(),
        ocsp_urls: Vec::new(),
        // ... hardcoded fields
    })
}

// AFTER:
fn parse_certificate_from_der(der_bytes: &[u8]) -> Result<TypesParsedCertificate, TlsError> {
    use x509_cert::{der::Decode, Certificate as X509CertCert};
    
    let cert = X509CertCert::from_der(der_bytes)
        .map_err(|e| TlsError::CertificateParsing(format!("X.509 parsing failed: {}", e)))?;
    
    // Use existing parser module
    crate::tls::certificate::parser::parse_x509_certificate_from_der_internal(&cert)
}
```

### Step 2: Update imports and dependencies
- Import necessary x509-cert types
- Ensure parser module is properly referenced
- Verify error type compatibility

### Step 3: Test integration
- Test with real certificate DER data
- Verify parsed fields match expected values
- Test error handling with malformed DER

## Technical Notes
- This function is used by TLS manager for certificate validation
- Must integrate with existing certificate parsing infrastructure
- Parser module already has comprehensive certificate parsing logic
- Function should delegate to existing tested implementation

## Validation Steps
1. Compile code and verify integration
2. Run certificate parsing unit tests
3. Test with various certificate types (RSA, ECDSA, etc.)
4. Verify OCSP URL extraction works
5. Test error handling with invalid DER data
6. Confirm TLS validation functionality restored

## Integration Impact
- Enables proper certificate subject/issuer validation
- Restores OCSP URL extraction for revocation checking
- Fixes certificate serial number handling
- Enables proper certificate expiration checking

## Related Issues
- Depends on Milestone 0 unwrap fixes
- Enables proper TLS certificate validation
- Foundation for authority management features