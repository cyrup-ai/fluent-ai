# HTTP3 Package Production Readiness TODO

## Non-Production Code Patterns Found

### "for now" Implementations
1. **File**: `src/hyper/proxy.rs:732`
   - **Issue**: `// Return None for now since we can't convert without allocation`
   - **Resolution**: Implement proper HeaderValue conversion using zero-allocation approach with pre-allocated buffer or static conversion table

2. **File**: `src/hyper/async_impl/body.rs:143`
   - **Issue**: `// Return Body with reusable data for now - simplified approach`
   - **Resolution**: Implement proper streaming body with actual data flow using AsyncStream pattern

3. **File**: `src/hyper/async_impl/body.rs:531`
   - **Issue**: `// In real async context, would yield here For now break;`
   - **Resolution**: Replace with proper async yielding using fluent_ai_async::yield_now() or channel-based coordination

4. **File**: `src/hyper/async_impl/h3_client/pool.rs:322`
   - **Issue**: `// For now - The async h3 operations require significant reworking`
   - **Resolution**: Implement complete HTTP/3 connection pool with proper async operations using quinn and h3 crates

5. **File**: `src/hyper/async_impl/h3_client/mod.rs:100`
   - **Issue**: `// for now - Err("HTTP/3 connection establishment requires async context")`
   - **Resolution**: Implement real HTTP/3 connection establishment using quinn QUIC transport

6. **File**: `src/hyper/async_impl/h3_client/connect.rs:237`
   - **Issue**: `// For now - Err(std::io::Error::new(std::io::ErrorKind::Other,...))`
   - **Resolution**: Implement actual HTTP/3 connection logic with proper error handling

7. **File**: `src/hyper/async_impl/client.rs:506`
   - **Issue**: `// Skip certificate loading for now to avoid type mismatches`
   - **Resolution**: Implement proper certificate loading with type-safe conversion

8. **File**: `src/hyper/async_impl/client.rs:2091`
   - **Issue**: `// For now - let bytes_request = http::Request::from_parts(parts, bytes_body);`
   - **Resolution**: Implement proper Request<Body> to Request<Bytes> conversion with streaming support

9. **File**: `src/json_path/filter_parser.rs:244`
   - **Issue**: `// For now - i += 1;`
   - **Resolution**: Implement proper filter expression parsing logic

10. **File**: `src/json_path/filter_parser.rs:468`
    - **Issue**: `// Unknown function - let it pass for now (could be user-defined)`
    - **Resolution**: Implement comprehensive function validation with proper error reporting

11. **File**: `src/json_path/core_evaluator.rs:976`
    - **Issue**: `// Handle simple property access for now`
    - **Resolution**: Implement full property path evaluation with nested object traversal

12. **File**: `src/json_path/functions.rs:431`
    - **Issue**: `// Filter evaluation would require the full filter evaluator For now`
    - **Resolution**: Implement complete filter evaluation system

### "in production" Mock Code
1. **File**: `src/hyper/async_impl/upgrade.rs:80`
   - **Issue**: `// Simulate actual network write operation In production`
   - **Resolution**: Implement real network write operations with proper error handling and flow control

2. **File**: `src/hyper/async_impl/upgrade.rs:280`
   - **Issue**: `// Integration point for hyper's upgraded connection In production`
   - **Resolution**: Implement complete hyper connection upgrade handling with bidirectional streaming

3. **File**: `src/hyper/async_impl/h3_client/connect.rs:272`
   - **Issue**: `// In production - using polling mechanisms compatible with streams-only architecture`
   - **Resolution**: Implement production-ready polling mechanisms for HTTP/3 connections

4. **File**: `src/hyper/error.rs:519`
   - **Issue**: `// Create a mock hyper timeout error In a real`
   - **Resolution**: Implement proper hyper timeout error creation with real error context

### "placeholder" Implementations
1. **File**: `src/hyper/wasm/response.rs:181`
   - **Issue**: `// WASM stream operations placeholder - not fully supported in streams-first`
   - **Resolution**: Implement complete WASM stream operations or remove WASM support if not needed

### TODO Comments
1. **File**: `src/json_path/state_machine.rs:91`
   - **Issue**: `/// Type of JSON structure at this depth TODO`
   - **Resolution**: Complete documentation and implementation for JSON structure type tracking

2. **File**: `src/json_path/state_machine.rs:95`
   - **Issue**: `/// Key name (for objects) or index (for arrays) TODO`
   - **Resolution**: Complete frame identifier implementation for object keys and array indices

3. **File**: `src/json_path/state_machine.rs:99`
   - **Issue**: `/// JSONPath selector that matched this frame TODO`
   - **Resolution**: Implement selector tracking for matched frames

4. **File**: `src/json_path/state_machine.rs:612`
   - **Issue**: `/// Need more data to continue processing TODO`
   - **Resolution**: Implement proper streaming data continuation logic

5. **File**: `src/json_path/deserializer/core.rs:47`
   - **Issue**: `/// Current selector index being evaluated in the JSONPath expression TODO`
   - **Resolution**: Implement selector index tracking for JSONPath evaluation

6. **File**: `src/json_path/deserializer/core.rs:51`
   - **Issue**: `/// Whether we're currently in recursive descent mode TODO`
   - **Resolution**: Implement recursive descent mode tracking

7. **File**: `src/json_path/deserializer/core.rs:55`
   - **Issue**: `/// Stack of depth levels where recursive descent should continue searching TODO`
   - **Resolution**: Implement recursive descent stack management

8. **File**: `src/json_path/deserializer/core.rs:59`
   - **Issue**: `/// Path breadcrumbs for backtracking during recursive descent TODO`
   - **Resolution**: Implement path breadcrumb system for backtracking

9. **File**: `src/json_path/deserializer/processor.rs:32`
   - **Issue**: `TODO // This suggests different architectural approaches were tried`
   - **Resolution**: Consolidate to single canonical architectural approach

10. **File**: `src/json_path/deserializer/processor.rs:35`
    - **Issue**: `/// Read next byte from streaming buffer using persistent position tracking TODO`
    - **Resolution**: Implement persistent position tracking for streaming buffer

11. **File**: `src/json_path/deserializer/processor.rs:60`
    - **Issue**: `/// Process single JSON byte and update parsing state TODO`
    - **Resolution**: Implement single-byte JSON processing with state updates

12. **File**: `src/json_path/safe_parsing.rs:48`
    - **Issue**: `/// Whether strict UTF-8 validation is enabled TODO`
    - **Resolution**: Implement strict UTF-8 validation option

### println! Debug Statements (Replace with env_logger)
**Files with println! statements requiring replacement:**
- `src/hyper/mod.rs:47`
- `src/hyper/error.rs:54`
- `src/hyper/connect.rs:350,1167,1176`
- `src/hyper/redirect.rs:126`
- `src/middleware.rs:145,152`
- `src/builder/fluent.rs:46,51,164,192,213`
- `src/builder/methods.rs:276`
- `src/client/execution.rs:33,35,37,40`
- `src/client/core.rs:83,106`
- `src/hyper/async_impl/multipart.rs:99,757,764,807,814`
- `src/hyper/async_impl/response.rs:197,283,350`
- `src/lib.rs:44,45,60,62,66,72`
- `src/hyper/async_impl/client.rs:625,665`
- `src/json_path/debug_*.rs` (multiple debug files)
- `src/json_path/filter.rs:42,51,110`
- `src/json_path/core_evaluator.rs:46,285,310,341,349,353,378,400,870,918,1289,1290,1293,1338,1348,1356,1362,1367,1374,1381,1387,1392,1398,1405,1418,1424,1432,1442,1448,1453,1459,1462,1468,1475,1484`
- `src/json_path/mod.rs:296,300,308,314,318,323,326,333,351,358,368,377,385,391,398,407,415,423,430,433,437,443`
- `src/common/cache/response_cache.rs:100,109`

**Resolution**: Replace all println!/eprintln! with proper structured logging using env_logger crate

## Files Requiring Modular Decomposition (>300 lines)

### 1. `src/hyper/async_impl/client.rs` (2,483 lines)
**Decomposition Plan:**
- `src/hyper/async_impl/client/core.rs` - Core client implementation
- `src/hyper/async_impl/client/builder.rs` - Client builder pattern
- `src/hyper/async_impl/client/configuration.rs` - Client configuration
- `src/hyper/async_impl/client/tls.rs` - TLS handling
- `src/hyper/async_impl/client/proxy.rs` - Proxy support
- `src/hyper/async_impl/client/cookies.rs` - Cookie management
- `src/hyper/async_impl/client/headers.rs` - Header processing
- `src/hyper/async_impl/client/mod.rs` - Module exports

### 2. `src/hyper/connect.rs` (1,605 lines)
**Decomposition Plan:**
- `src/hyper/connect/core.rs` - Core connection logic
- `src/hyper/connect/builder.rs` - Connection builder
- `src/hyper/connect/tls.rs` - TLS connection handling
- `src/hyper/connect/proxy.rs` - Proxy connection logic
- `src/hyper/connect/pool.rs` - Connection pooling
- `src/hyper/connect/dns.rs` - DNS resolution
- `src/hyper/connect/mod.rs` - Module exports

### 3. `src/json_path/core_evaluator.rs` (1,490 lines)
**Decomposition Plan:**
- `src/json_path/evaluator/core.rs` - Core evaluation logic
- `src/json_path/evaluator/selectors.rs` - Selector evaluation
- `src/json_path/evaluator/filters.rs` - Filter evaluation
- `src/json_path/evaluator/functions.rs` - Function evaluation
- `src/json_path/evaluator/recursion.rs` - Recursive descent
- `src/json_path/evaluator/properties.rs` - Property access
- `src/json_path/evaluator/mod.rs` - Module exports

### 4. `src/hyper/async_impl/request.rs` (1,173 lines)
**Decomposition Plan:**
- `src/hyper/async_impl/request/core.rs` - Core request implementation
- `src/hyper/async_impl/request/builder.rs` - Request builder
- `src/hyper/async_impl/request/body.rs` - Request body handling
- `src/hyper/async_impl/request/headers.rs` - Header management
- `src/hyper/async_impl/request/auth.rs` - Authentication
- `src/hyper/async_impl/request/mod.rs` - Module exports

### 5. `src/hyper/proxy.rs` (1,082 lines)
**Decomposition Plan:**
- `src/hyper/proxy/core.rs` - Core proxy logic
- `src/hyper/proxy/config.rs` - Proxy configuration
- `src/hyper/proxy/auth.rs` - Proxy authentication
- `src/hyper/proxy/tunnel.rs` - HTTP CONNECT tunneling
- `src/hyper/proxy/interceptor.rs` - Request interception
- `src/hyper/proxy/mod.rs` - Module exports

### 6. `src/json_path/parser_broken_decomp.rs` (1,069 lines)
**Action**: Remove this file as it appears to be a broken decomposition attempt

### 7. `src/hyper/async_impl/multipart.rs` (879 lines)
**Decomposition Plan:**
- `src/hyper/async_impl/multipart/core.rs` - Core multipart logic
- `src/hyper/async_impl/multipart/form.rs` - Form data handling
- `src/hyper/async_impl/multipart/file.rs` - File upload handling
- `src/hyper/async_impl/multipart/boundary.rs` - Boundary generation
- `src/hyper/async_impl/multipart/mod.rs` - Module exports

### 8. `src/hyper/tls.rs` (859 lines)
**Decomposition Plan:**
- `src/hyper/tls/core.rs` - Core TLS implementation
- `src/hyper/tls/config.rs` - TLS configuration
- `src/hyper/tls/certificates.rs` - Certificate handling
- `src/hyper/tls/rustls.rs` - Rustls integration
- `src/hyper/tls/native.rs` - Native TLS integration
- `src/hyper/tls/mod.rs` - Module exports

### 9. `src/wrappers.rs` (803 lines)
**Decomposition Plan:**
- `src/wrappers/bytes.rs` - Bytes wrapper
- `src/wrappers/response.rs` - Response wrappers
- `src/wrappers/connection.rs` - Connection wrappers
- `src/wrappers/stream.rs` - Stream wrappers
- `src/wrappers/mod.rs` - Module exports

### 10. `src/json_path/deserializer_old.rs` (740 lines)
**Action**: Remove this file as it's marked as old

## Embedded Tests to Extract

**Files containing embedded tests:**
- `src/json_path/debug_*.rs` - Multiple debug test files should be moved to `tests/json_path/`
- `src/json_path/core_evaluator.rs` - Contains #[test] functions
- `src/hyper/async_impl/multipart.rs` - Contains test code with println! statements

**Test Extraction Plan:**
1. Create `tests/json_path/` directory structure
2. Extract all #[test] functions to appropriate test files
3. Remove test code from src files
4. Bootstrap nextest configuration
5. Verify all tests pass after extraction

## Implementation Constraints

All implementations must follow these constraints:
- Zero allocation where possible
- Blazing-fast performance
- No unsafe code
- No unchecked operations
- No locking mechanisms
- Elegant ergonomic code
- Use fluent_ai_async::AsyncStream exclusively
- No async_trait or Box<dyn Future> patterns
- Proper error handling via error-as-data pattern
- No unwrap() or expect() in src code
- Use env_logger for all logging instead of println!

## Priority Order

1. **High Priority**: Fix "for now" and "in production" implementations
2. **High Priority**: Replace println! with proper logging
3. **Medium Priority**: Complete TODO implementations
4. **Medium Priority**: Extract embedded tests
5. **Low Priority**: Modular decomposition of large files