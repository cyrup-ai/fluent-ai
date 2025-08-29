# Task 0.3: Fix OCSP Nonce Extension OID Creation Unwrap

## Description
Replace dangerous `new_unwrap()` call in OCSP nonce validation with proper error handling to prevent crashes on OID creation failure.

## File Location
`src/tls/ocsp.rs:323`

## Violation
```rust
ext.extn_id == der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.5.5.7.48.1.2")
```

## Issue Analysis
- **Risk Level**: CRITICAL - Immediate crash risk
- **Impact**: OCSP nonce validation failure causes panic instead of error handling
- **Frequency**: Low - Nonce extension OID is RFC-defined, but defensive programming required
- **Security Impact**: High - OCSP replay attack protection bypass through crash

## Success Criteria
- [ ] Remove `new_unwrap()` call completely
- [ ] Use proper error handling with `TlsError::OcspValidation`
- [ ] Maintain OCSP nonce validation functionality
- [ ] Provide clear error message for OID creation failure
- [ ] Code compiles without warnings
- [ ] All existing OCSP tests continue to pass

## Dependencies
- None (independent task)

## Implementation Plan

### Step 1: Replace new_unwrap with proper error handling
```rust
// BEFORE:
ext.extn_id == der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.5.5.7.48.1.2")

// AFTER:
let nonce_oid = der::asn1::ObjectIdentifier::new("1.3.6.1.5.5.7.48.1.2")
    .map_err(|e| TlsError::OcspValidation(format!("Invalid nonce extension OID: {}", e)))?;
ext.extn_id == nonce_oid
```

### Step 2: Verify control flow integration
- Ensure error handling integrates with OCSP response validation
- Confirm proper error propagation in nonce checking logic
- Verify function signature supports `Result` return type

### Step 3: Consider OID constants optimization
- Evaluate using const OID definitions for well-known extensions
- Consider `const_oid` crate for compile-time validation
- Maintain nonce validation security properties

## Technical Notes
- OCSP Nonce extension OID "1.3.6.1.5.5.7.48.1.2" is RFC 6960 defined
- This is part of OCSP replay attack prevention
- Nonce validation ensures OCSP responses are fresh
- Failure here compromises OCSP security guarantees

## Alternative Approach: Const OID
Consider using compile-time constants:
```rust
use const_oid::db::rfc6960::ID_PKCS9_AT_EXTENSION_REQ; // or appropriate constant

// Then use:
let nonce_oid = ID_PKCS9_AT_EXTENSION_REQ; // if available
```

## Validation Steps
1. Compile code and verify no unwrap warnings
2. Run existing OCSP validation tests
3. Test OCSP nonce validation end-to-end
4. Verify error handling preserves OCSP security
5. Test with OCSP responses containing nonce extensions

## Security Implications
- OCSP nonce validation prevents replay attacks
- Crashes in nonce handling can bypass replay protection
- Error handling must preserve OCSP security properties
- Nonce validation is critical for fresh OCSP responses

## Related Issues
Part of comprehensive unwrap elimination effort. Critical for OCSP security and replay attack protection.