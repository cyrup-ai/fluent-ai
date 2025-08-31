# Parser Statistics - Fake Implementation

## Description
Replace hardcoded statistics with actual tracking in `src/normalize/parsers.rs` lines 302-309. Current implementation returns fake statistics with all values set to 0.

## Current Problem
```rust
pub fn get_parser_stats() -> ParserStats {
    // In a real implementation, this would track actual statistics
    ParserStats {
        graphql_queries_parsed: 0,
        capnp_messages_parsed: 0,
        parsing_errors: 0,
        average_parse_time_ms: 0.0,
    }
}
```

## Success Criteria
- [ ] Implement thread-safe atomic counters using `std::sync::atomic::AtomicU64`
- [ ] Add timing measurements with `std::time::Instant`
- [ ] Create global statistics registry with proper synchronization
- [ ] Implement exponential moving average for parse times
- [ ] Add metrics export for monitoring systems
- [ ] Track success/failure rates by protocol
- [ ] Include memory usage statistics

## Technical Resolution
- Implement thread-safe atomic counters using `std::sync::atomic::AtomicU64`
- Add timing measurements with `std::time::Instant`
- Create global statistics registry with proper synchronization
- Implement exponential moving average for parse times
- Add metrics export for monitoring systems

## Dependencies
- Milestone 0 must be completed (foundation safety fixes)
- Milestone 2 must be completed (protocol processing)

## Priority
MEDIUM - Observability

## Files Affected
- `src/normalize/parsers.rs` (lines 302-309)

## Testing Requirements
- Test counter accuracy
- Test timing measurements
- Test thread safety
- Test metrics export
- Performance test statistics overhead