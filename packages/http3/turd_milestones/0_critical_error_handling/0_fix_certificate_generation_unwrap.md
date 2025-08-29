# Task 0.0: Fix Certificate Generation Unwrap

## Description
Replace dangerous `.unwrap()` call in certificate generation with proper error handling to prevent production crashes.

## File Location
`src/tls/builder/certificate.rs:777`

## Violation
```rust
Ok(signed_cert) => rcgen::Certificate::from_params(params).unwrap(), /* Use signed cert */
```

## Issue Analysis
- **Risk Level**: CRITICAL - Immediate crash risk
- **Impact**: Certificate generation failure causes panic instead of graceful error handling
- **Frequency**: High - Certificate generation is a core operation

## Success Criteria
- [ ] Remove `.unwrap()` call completely
- [ ] Implement proper error handling with `CertificateGenerationResponse`
- [ ] Return detailed error information on failure
- [ ] Maintain existing function signature and behavior for success cases
- [ ] Code compiles without warnings
- [ ] All existing tests continue to pass

## Dependencies
- None (independent task)

## Implementation Plan

### Step 1: Replace unwrap with match expression
```rust
// BEFORE:
Ok(signed_cert) => rcgen::Certificate::from_params(params).unwrap(), /* Use signed cert */

// AFTER:
Ok(signed_cert) => {
    match rcgen::Certificate::from_params(params) {
        Ok(cert) => cert,
        Err(e) => {
            return CertificateGenerationResponse {
                success: false,
                certificate_info: None,
                files_created: vec![],
                certificate_pem: None,
                private_key_pem: None,
                issues: vec![TlsIssue {
                    issue_type: TlsIssueType::CertificateGeneration,
                    message: format!("Failed to create certificate from parameters: {}", e),
                    suggestion: Some("Check certificate generation parameters".to_string()),
                }],
            };
        }
    }
}
```

### Step 2: Verify error path handling
- Ensure early return doesn't break cleanup logic
- Verify files_created tracking is accurate
- Confirm proper error propagation

### Step 3: Test error scenarios
- Test with invalid certificate parameters
- Verify error response structure
- Confirm no panics in failure cases

## Technical Notes
- This is in the CA-signed certificate generation branch
- The function returns `CertificateGenerationResponse` which has comprehensive error handling fields
- Error should be descriptive enough for debugging certificate parameter issues
- The `rcgen::Certificate::from_params()` failure is typically due to invalid certificate parameters

## Validation Steps
1. Compile code and verify no unwrap warnings
2. Run existing certificate generation tests
3. Add negative test case for parameter validation failure
4. Verify error response contains useful debugging information

## Related Issues
Part of comprehensive unwrap elimination effort. This task is independent but should be completed before other certificate-related enhancements.