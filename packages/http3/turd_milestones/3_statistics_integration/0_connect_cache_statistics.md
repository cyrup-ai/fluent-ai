# Task 3.0: Connect Cache Statistics

## Description
Replace hardcoded cache statistics with real data from TlsManager cache implementation to provide accurate monitoring metrics.

## File Location
`src/tls/builder/certificate.rs:522-523`

## Violation
```rust
cache_hits: 0,              // TODO: Get from TlsManager
cache_misses: 0,            // TODO: Get from TlsManager
```

## Issue Analysis
- **Risk Level**: MEDIUM - Monitoring functionality degraded
- **Impact**: Cache statistics always show zero, preventing performance monitoring
- **Frequency**: High - Used in all certificate generation responses
- **Business Impact**: Limits observability and performance optimization

## Success Criteria
- [ ] Remove TODO comments completely
- [ ] Connect to actual TlsManager cache statistics
- [ ] Provide accurate cache hit/miss metrics
- [ ] Maintain existing response structure
- [ ] Support both OCSP and CRL cache statistics
- [ ] Enable proper cache performance monitoring

## Dependencies
- **Uses**: Existing TlsManager cache infrastructure
- **Architecture**: Must maintain fluent_ai_async patterns
- **Parallel**: Can be developed independently of other milestones

## Implementation Plan

### Step 1: Identify TlsManager cache interface
```rust
// Analyze existing TlsManager for cache statistics methods
// Look for OCSP and CRL cache implementations
// Identify statistics collection mechanisms
```

### Step 2: Implement cache statistics integration
```rust
// BEFORE:
cache_hits: 0,              // TODO: Get from TlsManager
cache_misses: 0,            // TODO: Get from TlsManager

// AFTER:
cache_hits: tls_manager.get_cache_stats().ocsp_hits + tls_manager.get_cache_stats().crl_hits,
cache_misses: tls_manager.get_cache_stats().ocsp_misses + tls_manager.get_cache_stats().crl_misses,
```

### Step 3: Handle TlsManager availability
```rust
// More robust implementation with fallback
let cache_stats = if let Some(manager) = &self.tls_manager {
    manager.get_cache_stats()
} else {
    CacheStats::default()
};

cache_hits: cache_stats.total_hits(),
cache_misses: cache_stats.total_misses(),
```

## Enhanced Implementation with Detailed Metrics
```rust
// If detailed statistics are available
let cache_stats = tls_manager.get_cache_stats();

// In the response structure:
cache_hits: cache_stats.ocsp_hits + cache_stats.crl_hits,
cache_misses: cache_stats.ocsp_misses + cache_stats.crl_misses,

// Potentially add more detailed metrics:
ocsp_cache_hits: cache_stats.ocsp_hits,
ocsp_cache_misses: cache_stats.ocsp_misses,
crl_cache_hits: cache_stats.crl_hits,
crl_cache_misses: cache_stats.crl_misses,
```

## Investigation Steps

### Step 1: Analyze TlsManager cache architecture
- Examine `src/tls/tls_manager.rs` for cache statistics methods
- Review OCSP cache implementation for metrics collection
- Review CRL cache implementation for metrics collection
- Identify statistics aggregation patterns

### Step 2: Examine cache statistics structures
```rust
// Look for existing structures like:
pub struct CacheStats {
    pub ocsp_hits: u64,
    pub ocsp_misses: u64,
    pub crl_hits: u64,
    pub crl_misses: u64,
    // Other metrics...
}
```

### Step 3: Verify integration points
- Ensure certificate generation has access to TlsManager
- Verify statistics are thread-safe and accurate
- Confirm metrics are updated in real-time

## Technical Implementation Details

### Cache Statistics Collection
```rust
impl TlsManager {
    pub fn get_cache_stats(&self) -> CacheStats {
        CacheStats {
            ocsp_hits: self.ocsp_cache.hits(),
            ocsp_misses: self.ocsp_cache.misses(),
            crl_hits: self.crl_cache.hits(),
            crl_misses: self.crl_cache.misses(),
        }
    }
}

impl CacheStats {
    pub fn total_hits(&self) -> u64 {
        self.ocsp_hits + self.crl_hits
    }
    
    pub fn total_misses(&self) -> u64 {
        self.ocsp_misses + self.crl_misses
    }
    
    pub fn hit_ratio(&self) -> f64 {
        let total = self.total_hits() + self.total_misses();
        if total == 0 {
            0.0
        } else {
            self.total_hits() as f64 / total as f64
        }
    }
}
```

### Integration in Certificate Builder
```rust
// In certificate generation response
let tls_stats = self.tls_manager
    .as_ref()
    .map(|m| m.get_cache_stats())
    .unwrap_or_default();

CertificateGenerationResponse {
    // ... other fields
    cache_hits: tls_stats.total_hits(),
    cache_misses: tls_stats.total_misses(),
    // ... other fields
}
```

## Validation Steps
1. Compile code and verify TlsManager integration
2. Generate certificates and verify statistics update
3. Test cache hit scenarios (repeated operations)
4. Test cache miss scenarios (new operations)
5. Verify statistics accuracy across multiple operations
6. Test with both OCSP and CRL operations
7. Confirm thread safety of statistics collection

## Monitoring Benefits
- Accurate cache performance metrics
- Ability to optimize cache sizing
- Identification of cache effectiveness
- Performance monitoring and alerting
- Capacity planning data

## Alternative Implementation: Real-time Statistics
```rust
// If statistics need to be captured per-operation
let operation_start_stats = tls_manager.get_cache_stats();

// ... perform certificate operations ...

let operation_end_stats = tls_manager.get_cache_stats();

let operation_cache_hits = operation_end_stats.total_hits() - operation_start_stats.total_hits();
let operation_cache_misses = operation_end_stats.total_misses() - operation_start_stats.total_misses();
```

## Related Issues
- Independent of other TURD milestones
- Improves observability and monitoring
- Enables cache performance optimization
- Foundation for advanced telemetry