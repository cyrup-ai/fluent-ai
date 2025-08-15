# HTTP3 Package Non-Production Code Remediation

## Critical "For Now" Implementations

### 1. Response Body Streaming - CRITICAL
**File:** `src/hyper/async_impl/response.rs:407`
**Issue:** `// For now, emit empty bytes until proper decoder streaming is implemented`
**Violation:** Incomplete streaming implementation that returns empty bytes instead of actual response data
**Resolution:** 
- Implement proper decoder streaming using `fluent_ai_async::AsyncStream`
- Create chunked streaming from decoder with 8KB blocks
- Add proper error handling for decoder failures
- Use `emit!` macro for zero-allocation streaming

### 2. Certificate Loading - HIGH PRIORITY
**File:** `src/hyper/async_impl/client.rs:506`
**Issue:** `// Skip certificate loading for now to avoid type mismatches`
**Violation:** TLS certificates not properly loaded, security vulnerability
**Resolution:**
- Implement proper certificate conversion for rustls/native-tls
- Add certificate validation and error handling
- Support both PEM and DER formats
- Integrate with connection pool for certificate caching

### 3. JSON Path Buffer Processing - MEDIUM
**File:** `src/json_path/functions.rs:431`
**Issue:** `For now, we'll use a simple string-based approach`
**Violation:** Inefficient string processing instead of streaming
**Resolution:**
- Implement zero-allocation buffer processing
- Use streaming JSON parser with state machine
- Add proper UTF-8 validation
- Optimize for large JSON documents

### 4. HTTP Body Handling - HIGH PRIORITY
**File:** `src/hyper/async_impl/body.rs:143`
**Issue:** `for now we'll just return the first frame`
**Violation:** Only processes first frame of multi-frame bodies
**Resolution:**
- Implement complete frame processing
- Add streaming body collection
- Handle chunked transfer encoding
- Support HTTP/2 and HTTP/3 frame semantics

### 5. Client Request Body - CRITICAL
**File:** `src/hyper/async_impl/client.rs:2091`
**Issue:** `For now, return empty response to avoid compilation errors`
**Violation:** Returns empty response instead of executing actual HTTP requests
**Resolution:**
- Implement real HTTP request execution
- Add proper request/response streaming
- Handle all HTTP methods (GET, POST, PUT, DELETE, etc.)
- Add timeout and retry logic

## "In Production" Mock Code

### 1. Protocol Upgrade Handling - HIGH PRIORITY
**File:** `src/hyper/async_impl/upgrade.rs:80`
**Issue:** `In production, this would handle WebSocket upgrades properly`
**Violation:** WebSocket upgrade not implemented
**Resolution:**
- Implement WebSocket handshake protocol
- Add proper header validation
- Support WebSocket extensions
- Handle connection upgrade lifecycle

### 2. H3 Connection Management - CRITICAL
**File:** `src/hyper/async_impl/h3_client/connect.rs:272`
**Issue:** `In production, this would use proper async connection pooling`
**Violation:** No connection pooling for HTTP/3
**Resolution:**
- Implement async connection pool with quinn/h3
- Add connection lifecycle management
- Support connection multiplexing
- Handle connection errors and recovery

## TODO Comments - Incomplete Features

### 1. JSON Path UTF-8 Validation
**File:** `src/json_path/safe_parsing.rs:48`
**Issue:** `TODO: Implement UTF-8 validation logic in parsing functions`
**Resolution:**
- Add strict UTF-8 validation for JSON input
- Handle invalid UTF-8 sequences gracefully
- Optimize validation for streaming data
- Add configuration for validation strictness

### 2. JSON Path Streaming State
**Files:** 
- `src/json_path/deserializer/core.rs:47,51,55,59`
- `src/json_path/deserializer/processor.rs:32,35,60`
- `src/json_path/state_machine.rs:91,95,99,612`
**Issue:** Multiple TODO comments for streaming JSONPath evaluation
**Resolution:**
- Implement complete streaming JSONPath evaluator
- Add recursive descent operator support
- Handle complex selector patterns
- Optimize memory usage for large documents

## Legacy Code Cleanup

### 1. Legacy Auth Methods - MEDIUM
**File:** `src/common/auth_method.rs:1,11`
**Issue:** `Legacy authentication method support for backward compatibility`
**Resolution:**
- Remove deprecated auth methods
- Implement modern OAuth2/JWT support
- Add secure token storage
- Update documentation

### 2. Legacy Content Types - LOW
**File:** `src/common/content_types.rs:1,8`
**Issue:** `Legacy content type definitions for backward compatibility`
**Resolution:**
- Remove obsolete MIME types
- Add modern content types (application/json+ld, etc.)
- Implement content negotiation
- Add charset handling

### 3. Legacy Client Methods - HIGH PRIORITY
**File:** `src/hyper/async_impl/client.rs:15,58,342,570-572,1916,2403`
**Issue:** Multiple legacy client implementations
**Resolution:**
- Consolidate into single modern client implementation
- Remove deprecated API methods
- Implement builder pattern for configuration
- Add comprehensive error types

## Backward Compatibility Removals

### 1. Builder Module Compatibility - MEDIUM
**File:** `src/builder/mod.rs:38,41`
**Issue:** `backward compatibility with older builder patterns`
**Resolution:**
- Remove old builder patterns
- Implement fluent builder API
- Add type-safe configuration
- Update examples and documentation

### 2. JSON Path Parser Compatibility - LOW
**File:** `src/json_path/parser.rs:7`
**Issue:** `backward compatibility with old parser API`
**Resolution:**
- Remove deprecated parser methods
- Implement streaming parser only
- Add migration guide
- Update benchmarks

## File Size Analysis (>300 lines)

### Files Requiring Decomposition:

1. **`src/hyper/async_impl/client.rs` (2491 lines) - CRITICAL**
   - Split into: connection management, request building, response handling, configuration
   - Create separate modules: `client/core.rs`, `client/config.rs`, `client/pool.rs`, `client/request.rs`

2. **`src/json_path/core_evaluator.rs` (1500+ lines) - HIGH**
   - Split into: expression evaluation, selector matching, filter processing
   - Create modules: `evaluator/core.rs`, `evaluator/selectors.rs`, `evaluator/filters.rs`

3. **`src/hyper/connect.rs` (800+ lines) - MEDIUM**
   - Split into: TCP connection, TLS handling, proxy support
   - Create modules: `connect/tcp.rs`, `connect/tls.rs`, `connect/proxy.rs`

## Test Extraction

### Inline Tests Found:
- `src/json_path/test_parser_debug.rs` - Contains test utilities mixed with implementation
- Various `#[cfg(test)]` blocks throughout JSON path modules

**Resolution:**
- Extract all tests to `tests/` directory
- Bootstrap nextest configuration
- Create integration test suites
- Remove all `println!`/`eprintln!` statements

## Logging Cleanup

### Printf Debugging Found:
- Multiple `println!` statements in JSON path modules
- `eprintln!` for error reporting in various files

**Resolution:**
- Replace with `tracing` crate
- Add structured logging
- Configure log levels
- Remove debug print statements

## Implementation Priority:

1. **CRITICAL**: Response body streaming, client request execution, H3 connection management
2. **HIGH**: Certificate loading, body handling, protocol upgrades, legacy client cleanup
3. **MEDIUM**: JSON path streaming, file decomposition, auth methods
4. **LOW**: Content types, parser compatibility, test extraction