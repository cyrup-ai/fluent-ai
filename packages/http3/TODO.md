# HTTP3 Error and Warning Fixes

## STATUS SUMMARY
- TOTAL ERRORS: 91
- TOTAL WARNINGS: 227
- ERRORS FIXED: 0
- WARNINGS FIXED: 0

## CRITICAL ERRORS (Must be fixed first)

### Import Resolution Errors
- [ ] Fix unresolved import `crate::jsonpath::buffer::Buffer` in `byte_processor/trait_impl.rs:6`
- [ ] Fix cannot find value `h3_error` in `protocols/transport.rs:146`

### Missing Method/Field Errors  
- [ ] Fix `keyring::Error::Unknown` not found in `tls/builder/authority.rs:650,681`
- [ ] Fix `JsonPathError` missing methods: `BufferUnderflow`, `UnexpectedByte`, `InvalidUtf8`, `UnexpectedEndOfInput`, `InvalidNumber`
- [ ] Fix `HttpResponse` missing `url` and `clone` methods in `middleware/cache.rs:113,118`
- [ ] Fix h2 adapter function call with wrong argument count in `protocols/h2/adapter.rs:53`
- [ ] Fix `quiche::Config` missing `set_verify_depth` method in `protocols/h3/strategy.rs:101`
- [ ] Fix `quiche::h3::Header` field access (`name`, `value`) in `protocols/h3/strategy.rs:384-395`
- [ ] Fix `Extra::default()` not found in `proxy/matcher/public_interface.rs:21,31`
- [ ] Fix `builder_core::Http3Builder` missing `bytes` method in `tls/ocsp.rs:242`
- [ ] Fix `ClientStats` missing `snapshot` method in `lib.rs:140`

### Type System Errors
- [ ] Fix type annotations needed for `h2::client::Connection` in `protocols/h2/streaming.rs:79`
- [ ] Fix trait bound issues with `ConnectionTrait` in `protocols/h2/strategy.rs:123`
- [ ] Fix `H2Config`/`H3Config` type mismatches in `protocols/strategy.rs` and `auto_strategy.rs`
- [ ] Fix `enable_0rtt` field not found on `QuicheConfig` in `protocols/strategy.rs:483`
- [ ] Fix wire protocol frame type mismatches (`H2Frame`/`H3Frame` vs `FrameChunk`)
- [ ] Fix `HashMap` vs `Vec` type mismatches in wire protocol serialization
- [ ] Fix `usize` vs `u64` arithmetic in wire protocol parsing
- [ ] Fix `H3Frame::PushPromise` missing `header_block` field in `protocols/wire.rs:397`
- [ ] Fix Result type mismatches in `proxy/mod.rs:22,32,40`
- [ ] Fix ambiguous associated types in `retry/policy.rs:130-145`
- [ ] Fix `client_stats::ClientStats` type mismatch in `lib.rs:212`

### Borrowing and Lifetime Errors
- [ ] Fix lifetime/borrowing issues in `http/response.rs:377,384`
- [ ] Fix borrow of moved value `data` in `protocols/h2/connection.rs:219`
- [ ] Fix cannot borrow `ready_h2` as mutable in `protocols/h2/h2.rs:91`
- [ ] Fix lifetime issues with `json_str`/`form_str` in `protocols/h3/adapter.rs:160,164`
- [ ] Fix borrowing conflicts with `frame_buffer` in `protocols/h3/connection.rs:133,163,190`
- [ ] Fix borrowed data escapes in `protocols/h3/connection.rs:100`

## WARNINGS (227 total - organized by type)

### Unused Import Warnings (122 warnings)
- [ ] Remove unused imports in `client/core.rs:12,13` (`H3Config`, `ProtocolConfigs`, `HttpVersion`)
- [ ] Remove unused imports in `connect/builder/tls.rs:5` (`Duration`)
- [ ] Remove unused imports in `connect/proxy/intercepted.rs:6,7,9` (multiple)
- [ ] Remove unused imports in `connect/service/core.rs:8` (`Uri`)
- [ ] Remove unused imports in `connect/service/direct.rs:6` (`SocketAddr`)
- [ ] Remove unused imports in `connect/types/connection.rs:8` (`AsyncRead`, `AsyncWrite`)
- [ ] Remove unused imports in `http/conversions.rs:7` (`crate::prelude::*`)
- [ ] Remove unused imports in `http/into_url.rs:3` (`crate::prelude::*`)
- [ ] Remove unused imports in JSONPath function evaluator integration tests (30+ files)
- [ ] Remove unused imports in middleware, protocols, proxy, telemetry, TLS modules (60+ files)

### Unused Variable Warnings (85 warnings)  
- [ ] Fix unused variables with underscore prefix or implement functionality
- [ ] Variables in builder methods, cache operations, connection management
- [ ] Variables in JSONPath processing, protocol handling, proxy matching
- [ ] Variables in TLS certificate handling, QUIC connection management

### Mutable Variable Warnings (15 warnings)
- [ ] Remove unnecessary `mut` declarations across builder, cache, connect, protocols modules

### Deprecated Method Warnings (5 warnings)
- [ ] Replace deprecated `cookie::CookieBuilder::finish()` with `build()` 
- [ ] Replace deprecated `Connection::new_h2()` with `new_h2_with_addr()` 
- [ ] Replace deprecated `Connection::new_h3()` with `new_h3_with_addr()`

## QUALITY ASSURANCE TASKS
*These will be inserted after each fix to ensure quality*

## SUCCESS CRITERIA
- [ ] `cargo check` shows 0 errors
- [ ] `cargo check` shows 0 warnings  
- [ ] All code compiles successfully
- [ ] All functionality is production-ready (no stubs)
- [ ] All error handling is proper (no unwrap/expect)