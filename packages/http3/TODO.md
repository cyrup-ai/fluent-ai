# HTTP3 Package Production Readiness TODO

## Critical Non-Production Code Violations

### 1. PLACEHOLDER IMPLEMENTATIONS (3 violations)

**File: `/src/hyper/wasm/response.rs:181`**
- **Violation**: "WASM stream operations placeholder - not fully supported in streams-first"
- **Issue**: Incomplete WASM streaming implementation with futures dependency
- **Solution**: Replace with proper AsyncStream<BytesWrapper> implementation using fluent_ai_async patterns. Remove futures_util dependency and implement native streaming body collection.

**File: `/src/hyper/async_impl/h3_client/connect.rs:301-302`**
- **Violation**: "Placeholder - real implementation needed" for H3 connection and send_request
- **Issue**: Mock H3 connection with None values instead of real implementation
- **Solution**: Implement real H3 connection establishment using quinn QUIC client with proper TLS configuration, certificate validation, and h3::client::new() integration.

### 2. NON-PRODUCTION COMMENTS (10 violations)

**File: `/src/hyper/error.rs:519`**
- **Violation**: "In a real scenario, this would come from hyper operations"
- **Issue**: Mock timeout error instead of real hyper integration
- **Solution**: Implement proper hyper error mapping with real timeout detection from underlying HTTP operations.

**File: `/src/hyper/async_impl/upgrade.rs:80,280`**
- **Violation**: "In production: write to underlying TCP/TLS stream"
- **Issue**: Missing actual I/O implementation for protocol upgrades
- **Solution**: Implement real bidirectional stream I/O using TcpStream/TlsStream with proper AsyncStream wrapping for WebSocket/HTTP2 upgrades.

**File: `/src/hyper/async_impl/body.rs:354,548`**
- **Violation**: "For now, emit a data frame and complete (real implementation would stream actual body data)"
- **Issue**: Mock body streaming instead of real HTTP body processing
- **Solution**: Implement proper HTTP body streaming with chunked transfer encoding, content-length handling, and compression support.

**File: `/src/json_path/core_evaluator.rs:976`**
- **Violation**: "for now" temporary implementation
- **Solution**: Complete JSONPath evaluation with full RFC 9535 compliance.

**File: `/src/json_path/functions.rs:431`**
- **Violation**: "For now" incomplete function implementation
- **Solution**: Implement complete JSONPath function library with proper type coercion and error handling.

**File: `/src/hyper/async_impl/h3_client/connect.rs:267`**
- **Violation**: "For now, return an error indicating this needs proper async integration"
- **Solution**: Implement real QUIC connection waiting using quinn's connection establishment with proper timeout handling.

**File: `/src/json_path/filter_parser.rs:244,468`**
- **Violation**: "For now" incomplete filter parsing
- **Solution**: Complete filter expression parsing with full operator support and proper precedence handling.

### 3. TODO COMMENTS (15 violations)

**File: `/src/json_path/safe_parsing.rs:48`**
- **Violation**: "TODO: Implement proper error recovery"
- **Solution**: Add comprehensive error recovery with position tracking, syntax error reporting, and partial parse tree reconstruction.

**File: `/src/json_path/state_machine.rs:91,95,99,612`**
- **Violation**: Multiple "TODO: Handle edge cases" comments
- **Solution**: Implement complete state machine with all JSONPath syntax edge cases, proper state transitions, and error handling.

**File: `/src/json_path/deserializer/processor.rs:32,35,60`**
- **Violation**: "TODO: Optimize performance" and incomplete implementations
- **Solution**: Implement zero-allocation deserializer with streaming JSON processing, proper memory management, and performance optimizations.

**File: `/src/json_path/deserializer/core.rs:47,51,55,59`**
- **Violation**: Multiple "TODO: Add validation" comments
- **Solution**: Add comprehensive JSON schema validation, type checking, and constraint enforcement with proper error reporting.

### 4. DANGEROUS ERROR HANDLING (200+ violations)

**Critical expect() Usage (High Priority)**:
- `/src/hyper/async_impl/request.rs`: 50+ expect() calls (lines 621-1114)
- `/src/json_path/core_evaluator.rs`: 20+ expect() calls (lines 1250-1483)
- `/src/hyper/wasm/client.rs`: 12+ expect() calls (lines 384-476)
- `/src/hyper/proxy.rs`: 16+ expect() calls (lines 543-1198)

**Solution**: Replace ALL expect() calls with proper Result handling using:
```rust
// Replace: value.expect("message")
// With: value.map_err(|e| ErrorType::new(format!("Context: {}", e)))?
```

**Critical unwrap() Usage**:
- `/src/hyper/connect.rs:36`: Address parsing unwrap
- `/src/hyper/proxy.rs:123`: URI parsing unwrap  
- `/src/hyper/async_impl/h3_client/mod.rs:86`: Authority parsing unwrap

**Solution**: Replace with proper error propagation using Result<T,E> and ? operator.

### 5. DEBUG LOGGING VIOLATIONS (150+ violations)

**Files with println!/eprintln! usage**:
- `/src/json_path/core_evaluator.rs`: 50+ debug prints
- `/src/json_path/mod.rs`: 25+ debug prints
- `/src/json_path/test_parser_debug.rs`: 15+ debug prints
- `/src/builder/fluent.rs`: 5+ debug prints
- `/src/hyper/async_impl/multipart.rs`: 5+ debug prints

**Solution**: Replace ALL println!/eprintln! with proper structured logging:
```rust
// Replace: println!("Debug: {}", value);
// With: log::debug!("Context description: {}", value);
```

### 6. LEGACY/BACKWARD COMPATIBILITY (30+ violations)

**Files requiring modernization**:
- `/src/common/auth_method.rs:1,11`: Legacy authentication patterns
- `/src/common/content_types.rs:1,8`: Legacy content type handling
- `/src/hyper/async_impl/client.rs:15,58,342,572-574`: Legacy hyper client patterns

**Solution**: Remove backward compatibility shims and implement modern patterns using fluent_ai_async exclusively.

## FILE DECOMPOSITION (42 files >300 lines)

### CRITICAL - Files >1000 lines requiring immediate decomposition:

**1. `/src/hyper/async_impl/client.rs` (2534 lines)**
- **Decompose into**:
  - `client/core.rs` - Core client implementation (400 lines)
  - `client/builder.rs` - ClientBuilder implementation (300 lines)
  - `client/config.rs` - Configuration management (200 lines)
  - `client/tls.rs` - TLS/certificate handling (300 lines)
  - `client/proxy.rs` - Proxy configuration (250 lines)
  - `client/middleware.rs` - Middleware integration (200 lines)
  - `client/execution.rs` - Request execution logic (400 lines)
  - `client/streaming.rs` - Streaming request/response (300 lines)
  - `client/error_handling.rs` - Error conversion and handling (184 lines)

**2. `/src/hyper/connect.rs` (1640 lines)**
- **Decompose into**:
  - `connect/core.rs` - Core connection logic (300 lines)
  - `connect/tcp.rs` - TCP connection handling (250 lines)
  - `connect/tls.rs` - TLS handshake and configuration (300 lines)
  - `connect/proxy.rs` - Proxy connection logic (200 lines)
  - `connect/dns.rs` - DNS resolution (150 lines)
  - `connect/timeout.rs` - Connection timeout handling (100 lines)
  - `connect/pool.rs` - Connection pooling (200 lines)
  - `connect/happy_eyeballs.rs` - IPv4/IPv6 dual-stack (140 lines)

**3. `/src/json_path/core_evaluator.rs` (1490 lines)**
- **Decompose into**:
  - `json_path/evaluator/core.rs` - Core evaluation engine (300 lines)
  - `json_path/evaluator/expressions.rs` - Expression evaluation (250 lines)
  - `json_path/evaluator/filters.rs` - Filter processing (200 lines)
  - `json_path/evaluator/functions.rs` - Function implementations (200 lines)
  - `json_path/evaluator/operators.rs` - Operator handling (150 lines)
  - `json_path/evaluator/selectors.rs` - Selector processing (200 lines)
  - `json_path/evaluator/context.rs` - Evaluation context (90 lines)
  - `json_path/evaluator/optimization.rs` - Performance optimizations (100 lines)

### HIGH PRIORITY - Files 800-1200 lines:

**4. `/src/hyper/proxy.rs` (1205 lines)**
- **Decompose into**:
  - `proxy/core.rs` - Core proxy logic (200 lines)
  - `proxy/http.rs` - HTTP proxy handling (200 lines)
  - `proxy/socks.rs` - SOCKS proxy implementation (250 lines)
  - `proxy/auth.rs` - Proxy authentication (150 lines)
  - `proxy/tunnel.rs` - CONNECT tunnel handling (200 lines)
  - `proxy/config.rs` - Proxy configuration (105 lines)
  - `proxy/detection.rs` - Proxy auto-detection (100 lines)

**5. `/src/hyper/async_impl/request.rs` (1173 lines)**
- **Decompose into**:
  - `request/core.rs` - Core request building (200 lines)
  - `request/body.rs` - Request body handling (200 lines)
  - `request/headers.rs` - Header management (150 lines)
  - `request/multipart.rs` - Multipart form handling (200 lines)
  - `request/streaming.rs` - Streaming request body (150 lines)
  - `request/validation.rs` - Request validation (123 lines)
  - `request/serialization.rs` - Request serialization (150 lines)

### MEDIUM PRIORITY - Files 600-800 lines:

Continue decomposition for remaining 35 files following similar patterns with logical separation of concerns.

## TESTING EXTRACTION

**Files with embedded tests requiring extraction**:
- `/src/json_path/debug_at_test.rs` - Extract to `/tests/json_path/debug_at_test.rs`
- `/src/json_path/debug_execution_test.rs` - Extract to `/tests/json_path/debug_execution_test.rs`
- `/src/json_path/debug_error_test.rs` - Extract to `/tests/json_path/debug_error_test.rs`
- `/src/json_path/debug_infinite_loop.rs` - Extract to `/tests/json_path/debug_infinite_loop.rs`
- `/src/json_path/test_parser_debug.rs` - Extract to `/tests/json_path/test_parser_debug.rs`

**Nextest Bootstrap Required**:
1. Add nextest configuration in `.config/nextest.toml`
2. Verify all extracted tests pass with `cargo nextest run`
3. Remove debug test files from src/ after extraction

## IMPLEMENTATION CONSTRAINTS

**Zero-Allocation Requirements**:
- Use `ArrayVec`/`SmallVec` for bounded collections
- Implement streaming with `AsyncStream<T, CAP>` exclusively
- Use `Arc<str>` for shared string data
- Avoid `Vec::push()` in hot paths

**No Unsafe Code**:
- Replace all `unsafe { std::mem::zeroed() }` with proper initialization
- Use safe alternatives for all pointer operations
- Implement proper bounds checking

**No Locking**:
- Use `crossbeam` lock-free data structures
- Implement atomic operations for shared state
- Use `Arc` for immutable shared data

**Elegant Ergonomic Code**:
- Implement builder patterns for complex configurations
- Use method chaining for fluent APIs
- Provide comprehensive error context
- Use `Result<T, E>` for all fallible operations

## QUALITY ASSURANCE STEPS

1. **Compilation**: Achieve zero errors and warnings
2. **Testing**: All tests pass with `cargo nextest run`
3. **Performance**: Benchmark critical paths for zero-allocation compliance
4. **Documentation**: All public APIs have comprehensive docs
5. **Integration**: Verify compatibility with fluent_ai_async patterns