# Task 2.0: Implement Domain Validation Logic

## Description
Implement proper domain validation logic in `can_sign_for_domain` to ensure certificate authorities can only sign certificates for authorized domains.

## File Location
`src/tls/builder/authority.rs:58`

## Violation
```rust
pub fn can_sign_for_domain(&self, _domain: &str) -> bool {
    // TODO: Implement domain validation logic
    self.is_valid()
}
```

## Issue Analysis
- **Risk Level**: HIGH - Critical security validation missing
- **Impact**: Certificate authorities can sign for any domain without validation
- **Frequency**: High - Used for all certificate signing operations
- **Security Impact**: Critical - Enables unauthorized certificate generation

## Success Criteria
- [ ] Remove TODO comment completely
- [ ] Implement proper domain matching against CA certificate
- [ ] Validate domain against CA subject and Subject Alternative Names
- [ ] Support wildcard domain matching where appropriate
- [ ] Maintain CA validity checking
- [ ] Prevent unauthorized domain signing

## Dependencies
- **Prerequisite**: Milestone 1 (certificate parsing must work)
- **Uses**: `parse_certificate_from_pem` for CA certificate analysis
- **Security**: Critical for certificate authority constraints

## Implementation Plan

### Step 1: Parse CA certificate for domain constraints
```rust
// BEFORE:
pub fn can_sign_for_domain(&self, _domain: &str) -> bool {
    // TODO: Implement domain validation logic
    self.is_valid()
}

// AFTER:
pub fn can_sign_for_domain(&self, domain: &str) -> bool {
    use crate::tls::certificate::parsing::parse_certificate_from_pem;
    
    if !self.is_valid() {
        return false;
    }
    
    // Parse CA certificate to check constraints
    if let Ok(ca_cert) = parse_certificate_from_pem(&self.certificate_pem) {
        // Check if this is a proper CA
        if !ca_cert.is_ca {
            return false;
        }
        
        // Check domain against CA subject and SANs
        if let Some(cn) = ca_cert.subject.get("CN") {
            if domain == cn || domain.ends_with(&format!(".{}", cn)) {
                return true;
            }
        }
        
        // Check against SANs
        for san in &ca_cert.san_dns_names {
            if domain == san || (san.starts_with("*.") && domain.ends_with(&san[2..])) {
                return true;
            }
        }
    }
    
    false
}
```

### Step 2: Implement robust domain matching
- Exact domain matches
- Subdomain validation for CA domains
- Wildcard SAN support (*.example.com)
- Prevent unauthorized domain signing

### Step 3: Add comprehensive validation
- Verify CA certificate constraints
- Check basic constraints extension
- Validate key usage for certificate signing
- Ensure CA validity period

## Enhanced Implementation with Constraints
```rust
pub fn can_sign_for_domain(&self, domain: &str) -> bool {
    use crate::tls::certificate::parsing::parse_certificate_from_pem;
    
    // Basic validity check
    if !self.is_valid() {
        tracing::warn!("CA certificate is not valid (expired or not yet valid)");
        return false;
    }
    
    // Parse CA certificate for detailed validation
    let ca_cert = match parse_certificate_from_pem(&self.certificate_pem) {
        Ok(cert) => cert,
        Err(e) => {
            tracing::error!("Failed to parse CA certificate: {}", e);
            return false;
        }
    };
    
    // Verify this is actually a CA certificate
    if !ca_cert.is_ca {
        tracing::error!("Certificate is not marked as CA in BasicConstraints");
        return false;
    }
    
    // Check key usage for certificate signing
    if !ca_cert.key_usage.contains(&"keyCertSign".to_string()) {
        tracing::error!("CA certificate missing required keyCertSign usage");
        return false;
    }
    
    // Domain validation against subject CN
    if let Some(cn) = ca_cert.subject.get("CN") {
        if validate_domain_match(domain, cn) {
            tracing::debug!("Domain {} validated against CA CN: {}", domain, cn);
            return true;
        }
    }
    
    // Domain validation against Subject Alternative Names
    for san in &ca_cert.san_dns_names {
        if validate_domain_match(domain, san) {
            tracing::debug!("Domain {} validated against CA SAN: {}", domain, san);
            return true;
        }
    }
    
    tracing::warn!("Domain {} not authorized for CA: no matching CN or SAN", domain);
    false
}

fn validate_domain_match(domain: &str, pattern: &str) -> bool {
    // Exact match
    if domain == pattern {
        return true;
    }
    
    // Wildcard pattern matching
    if pattern.starts_with("*.") {
        let pattern_suffix = &pattern[2..];
        return domain.ends_with(pattern_suffix) && !domain[..domain.len() - pattern_suffix.len()].contains('.');
    }
    
    // Subdomain matching for CA domains
    if domain.ends_with(&format!(".{}", pattern)) {
        return true;
    }
    
    false
}
```

## Validation Steps
1. Compile code and verify integration
2. Test with valid domain/CA combinations
3. Test with invalid domain requests
4. Test wildcard SAN matching
5. Test subdomain validation
6. Verify CA constraint enforcement
7. Test with non-CA certificates (should reject)

## Security Test Cases
- Valid exact domain match
- Valid subdomain of CA domain
- Valid wildcard SAN match
- Invalid domain (not in CA scope)
- Expired CA certificate
- Non-CA certificate
- Missing keyCertSign usage

## Security Implications
- Prevents unauthorized certificate generation
- Enforces CA domain constraints
- Critical for preventing certificate mis-issuance
- Must be tamper-resistant and comprehensive

## Related Issues
- Depends on Milestone 1 for certificate parsing
- Foundation for secure certificate authority operations
- Critical for certificate signing authorization