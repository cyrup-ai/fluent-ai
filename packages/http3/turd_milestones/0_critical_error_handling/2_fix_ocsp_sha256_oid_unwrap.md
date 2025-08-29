# Task 0.2: Fix OCSP SHA-256 OID Creation Unwrap

## Description
Replace dangerous `new_unwrap()` call in OCSP request generation with proper error handling to prevent crashes on OID creation failure.

## File Location
`src/tls/ocsp.rs:253`

## Violation
```rust
oid: der::asn1::ObjectIdentifier::new_unwrap("2.16.840.1.101.3.4.2.1"), // SHA-256
```

## Issue Analysis
- **Risk Level**: CRITICAL - Immediate crash risk
- **Impact**: OCSP request generation failure causes panic instead of error handling
- **Frequency**: Low - SHA-256 OID is well-known, but defensive programming required
- **Security Impact**: High - OCSP validation bypass through crash

## Success Criteria
- [ ] Remove `new_unwrap()` call completely
- [ ] Use proper error handling with `TlsError::OcspValidation`
- [ ] Maintain OCSP request generation functionality
- [ ] Provide clear error message for OID creation failure
- [ ] Code compiles without warnings
- [ ] All existing OCSP tests continue to pass

## Dependencies
- None (independent task)

## Implementation Plan

### Step 1: Replace new_unwrap with proper error handling
```rust
// BEFORE:
oid: der::asn1::ObjectIdentifier::new_unwrap("2.16.840.1.101.3.4.2.1"), // SHA-256

// AFTER:
oid: der::asn1::ObjectIdentifier::new("2.16.840.1.101.3.4.2.1")
    .map_err(|e| TlsError::OcspValidation(format!("Invalid SHA-256 OID: {}", e)))?,
```

### Step 2: Verify error context integration
- Ensure error handling integrates with surrounding OCSP code
- Confirm proper error propagation through call stack
- Verify function signature supports `Result` return type

### Step 3: Consider OID constants
- Evaluate using const OID definitions for well-known values
- Consider `const_oid` crate for compile-time OID validation
- Maintain compatibility with existing OCSP implementation

## Technical Notes
- SHA-256 OID "2.16.840.1.101.3.4.2.1" is RFC-defined and stable
- This is part of OCSP certificate ID generation
- Failure here prevents OCSP revocation checking
- Error should be descriptive for OCSP debugging

## Alternative Approach: Const OID
Consider using compile-time constants:
```rust
use const_oid::db::rfc5912::SHA256;

// Then use:
oid: SHA256,
```

## Validation Steps
1. Compile code and verify no unwrap warnings
2. Run existing OCSP validation tests
3. Test OCSP request generation end-to-end
4. Verify error handling doesn't break OCSP flow
5. Consider adding negative test for malformed OID (if applicable)

## Security Implications
- OCSP validation is critical for certificate revocation checking
- Crashes in OCSP handling can bypass revocation checks
- Error handling must maintain OCSP security guarantees

## Related Issues
Part of comprehensive unwrap elimination effort. Critical for TLS security integrity and OCSP functionality.