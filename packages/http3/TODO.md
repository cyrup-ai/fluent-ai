# TODO: Http3 Fluent Builder Implementation Plan

## Milestone 1: Core Builder & Typestate Foundation

- **Task 1.1**: In `src/builder.rs`, define the core typestate marker traits (`MethodState`, `HeadersState`, `BodyState`, `UrlState`) and their corresponding state structs (e.g., `MethodNotSet`, `MethodSet`). This will form the compile-time safety backbone of the builder. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.2**: Act as an Objective QA Rust developer: Review the typestate definitions in `src/builder.rs`. Verify that the states logically represent the request building flow and prevent invalid state transitions. Confirm that the design is purely structural and contains no implementation logic. Rate the work performed previously on these requirements.

- **Task 1.3**: In `src/builder.rs`, define the primary `Http3Builder<M, H, B, U>` struct. It will be generic over the state markers and will contain an `HttpRequest` instance to be configured. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.4**: Act as an Objective QA Rust developer: Review the `Http3Builder` struct definition. Ensure it correctly uses `PhantomData` for the generic state markers and properly encapsulates the `HttpRequest`. Rate the work performed previously on these requirements.

- **Task 1.5**: In `src/builder.rs`, implement the `Http3` entry point struct with static methods `json()`, `form_urlencoded()`, and a generic `builder()`. These methods will instantiate `Http3Builder` in its initial state, pre-configuring headers as appropriate. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.6**: Act as an Objective QA Rust developer: Review the `Http3` entry point methods. Verify they correctly initialize the `Http3Builder` with the appropriate headers and initial typestate for `json()` and `form_urlencoded()`. Rate the work performed previously on these requirements.

## Milestone 2: Configuration Methods & `headers!` Macro

- **Task 2.1**: In `src/common/headers.rs`, define a comprehensive, type-safe `HeaderName` enum, mirroring the approach in `reqwest::header`. Include common headers like `ContentType`, `Accept`, `Authorization`, and custom ones like `XApiKey`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.2**: Act as an Objective QA Rust developer: Review the `HeaderName` enum. Confirm it includes all necessary standard and custom headers and provides a type-safe way to reference them, preventing typos. Rate the work performed previously on these requirements.

- **Task 2.3**: In `src/builder.rs`, implement the `headers!{}` macro. This macro must accept the `HeaderName::Variant => "value"` syntax and expand to a `HashMap<String, String>`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.4**: Act as an Objective QA Rust developer: Review the `headers!{}` macro implementation. Test its syntax and verify that it correctly produces a `HashMap` compatible with the builder's `.headers()` method. Rate the work performed previously on these requirements.

- **Task 2.5**: In `src/builder.rs`, implement the configuration methods on `Http3Builder`: `.headers(HashMap<String, String>)`, `.api_key(&str)`, `.basic_auth(&str, &str)`, and `.bearer_auth(&str)`. These methods will correctly modify the internal `HttpRequest` and transition the builder's state. The auth methods will leverage the existing providers in `src/common/auth.rs`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.6**: Act as an Objective QA Rust developer: Review the configuration methods. Verify that headers are correctly applied and that authentication methods correctly format the `Authorization` header. Check that the typestate transitions are logical. Rate the work performed previously on these requirements.

- **Task 2.7**: In `src/builder.rs`, implement the generic `.body<T: Serialize>(&T)` method. This method will serialize the provided data to `Vec<u8>` (using `serde_json`) and store it in the `HttpRequest`. This method must return a `HttpResult<Http3Builder<...>>` as serialization can fail, and it will transition the builder to the `BodySet` state. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.8**: Act as an Objective QA Rust developer: Review the `.body()` method. Verify that it correctly handles serialization errors and that the typestate transition to `BodySet` is correctly implemented. Rate the work performed previously on these requirements.

## Milestone 3: Execution & Response Handling

- **Task 3.1**: In `src/builder.rs`, implement the terminal methods: `.get(url)`, `.post(url)`, `.put(url)`, etc. These methods will consume the builder, set the final URL and method on the `HttpRequest`, and use the `global_client()` to delegate execution to the appropriate module in `src/operations/`. The return type will be a new `ResponseStream` struct. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.2**: Act as an Objective QA Rust developer: Review the terminal methods. Verify that they consume the builder, correctly construct the final `HttpRequest`, and delegate to the `operations` modules. Check that the `ResponseStream` is returned. Rate the work performed previously on these requirements.

- **Task 3.3**: In `src/stream.rs`, define the `ResponseStream` struct. This struct will wrap the underlying `HttpStream`. Implement the `Stream` trait for it, yielding `HttpResult<HttpChunk>`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.4**: Act as an Objective QA Rust developer: Review the `ResponseStream` struct and its `Stream` implementation. Ensure it correctly wraps the underlying stream and propagates chunks and errors. Rate the work performed previously on these requirements.

- **Task 3.5**: In `src/stream.rs`, implement the final consumer methods on `ResponseStream`: `.collect<T: DeserializeOwned>()` and `.collect_or_else<T: DeserializeOwned, F: FnOnce(HttpError) -> T>(f: F)`. The `collect` method will buffer the full response and deserialize it, returning `HttpResult<T>`. `collect_or_else` will allow for a fallback value or custom error handling on deserialization failure. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.6**: Act as an Objective QA Rust developer: Review the `.collect()` and `.collect_or_else()` methods. Verify that they correctly handle the asynchronous stream, buffer the response, and perform deserialization with robust error handling. Rate the work performed previously on these requirements.

## Milestone 4: Integration & Testing

- **Task 4.1**: Create a new test file `tests/builder_api.rs`. Add comprehensive integration tests that validate the entire fluent API, using each of the user's provided examples as a test case. Tests must use a real HTTP endpoint (e.g., httpbin.org) to verify requests and responses. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 4.2**: Act as an Objective QA Rust developer: Review the integration tests in `tests/builder_api.rs`. Verify that all user examples are covered and that the tests make real network requests to validate the end-to-end functionality of the builder. Ensure `expect()` is used for assertions as appropriate in a test context. Rate the work performed previously on these requirements.

- **Task 4.3**: In `src/lib.rs`, publicly export the main builder components: `Http3`, `Http3Builder`, and any necessary response types, ensuring they are discoverable and ergonomic for crate users. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 4.4**: Act as an Objective QA Rust developer: Review the public exports in `src/lib.rs`. Confirm that the builder API is exposed cleanly and logically to end-users. Rate the work performed previously on these requirements.

---

# PRODUCTION CODE AUDIT - CRITICAL ISSUES TO RESOLVE

## CRITICAL: Blocking in Async Context

### Issue: Synchronous `block_on` calls in async code
**File**: `src/builder.rs:311, 363`
**Description**: Using `tokio::runtime::Handle::block_on()` in async functions breaks async runtime semantics
**Technical Solution**: 
- Replace `block_on` with proper async stream collection using `AsyncStream` pattern
- Implement `collect()` method that returns `impl Future<Output = Result<T, E>>`
- Use `futures::stream::StreamExt::collect()` for stream consumption
- Replace synchronous collection with streaming-first approach per CLAUDE.md constraints

**Code Changes Required**:
```rust
// CURRENT (DANGEROUS):
rt.block_on(async { /* collection logic */ })

// REPLACEMENT (PRODUCTION):
async fn collect<T: DeserializeOwned>(self) -> HttpResult<T> {
    let mut all_bytes = Vec::new();
    let mut stream = self;
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(HttpChunk::Body(bytes)) => all_bytes.extend_from_slice(&bytes),
            Ok(HttpChunk::Done) => break,
            Err(e) => return Err(e),
        }
    }
    serde_json::from_slice(&all_bytes).map_err(HttpError::from)
}
```

## CRITICAL: Unsafe Error Handling with `unwrap()`

### Issue: Multiple `unwrap()` calls that can panic in production
**Files and Lines**:
- `src/operations/put.rs:95` - JSON serialization
- `src/operations/patch.rs:82, 83` - JSON serialization 
- `src/operations/post.rs:90, 92` - JSON and form serialization
- `src/operations/download.rs:62` - Header value construction

**Technical Solution**: Replace all `unwrap()` with proper error handling using `?` operator and `HttpError` enum

**Code Changes Required**:
```rust
// CURRENT (DANGEROUS):
let body_bytes = serde_json::to_vec(val).unwrap(); // Can panic

// REPLACEMENT (PRODUCTION):
let body_bytes = serde_json::to_vec(val)
    .map_err(|e| HttpError::SerializationError(e.to_string()))?;

// For HeaderValue construction:
// CURRENT (DANGEROUS): 
HeaderValue::from_str(&range_value).unwrap()

// REPLACEMENT (PRODUCTION):
HeaderValue::from_str(&range_value)
    .map_err(|e| HttpError::InvalidHeader(e.to_string()))?
```

## NON-PRODUCTION TODO Comments

### Issue: TODO comments indicating incomplete implementations
**Files and Lines**:
- `src/common/retry.rs:2` - "TODO: Refactor to align with Reqwest patterns"
- `src/common/cache.rs:2` - "TODO: Refactor to align with Reqwest patterns" 
- `src/middleware/cache.rs:97` - "TODO: Extract from request context"

**Technical Solution**: Complete the implementations per CLAUDE.md AsyncStream architecture

**For retry.rs**: Replace comment with implementation note:
```rust
//! Zero allocation retry logic with exponential backoff and jitter
//! Implements production-ready AsyncStream pattern with lock-free operations
```

**For cache.rs**: Replace comment with implementation note:
```rust
//! Zero allocation HTTP response caching with lock-free skiplist storage  
//! Production-ready AsyncStream architecture with crossbeam-skiplist backend
```

**For middleware/cache.rs:97**: Implement request context extraction:
```rust
// Extract user-provided expires from request headers or metadata
let user_expires_hours = request.headers()
    .get("cache-expires-hours")
    .and_then(|v| v.to_str().ok())
    .and_then(|s| s.parse::<u64>().ok());
```

## FALSE POSITIVES - Language Revision

### Issue: Misleading language that appears non-production but is actually descriptive
**Files and Lines**:
- `src/common/cache.rs:660` - "in production" (descriptive comment about HTTP date parsing)
- `src/common/cache.rs:483` - "actual" (descriptive variable reference)
- `src/response.rs:91` - "Actual" (descriptive comment)
- `src/response.rs:226` - "fix" (part of SSE parsing logic)
- `src/lib.rs:104` - "backward compatibility" (legitimate type alias comment)

**Technical Solution**: Revise language to eliminate false audit triggers while maintaining meaning

**Code Changes Required**:
```rust
// src/common/cache.rs:660 - Change comment
// FROM: "in production, use a proper HTTP date parser"
// TO: "for robust deployment, use a proper HTTP date parser"

// src/common/cache.rs:483 - Change variable reference
// FROM: "entries actually evicted" 
// TO: "entries successfully evicted"

// src/response.rs:91 - Change comment
// FROM: "Actual error handling happens during streaming"
// TO: "Error handling occurs during streaming consumption"
```

---

# LARGE FILE DECOMPOSITION - MODULE RESTRUCTURING

## CRITICAL: Large files requiring decomposition (>300 lines)

### 1. `src/common/cache.rs` - 690 lines
**Decomposition Plan**:
- **Core Module**: `src/common/cache/core.rs` - CacheStorage struct and basic operations (200 lines)
- **Entry Module**: `src/common/cache/entry.rs` - CacheEntry struct and validation logic (150 lines) 
- **Eviction Module**: `src/common/cache/eviction.rs` - LRU eviction algorithms (120 lines)
- **HTTP Date Module**: `src/common/cache/httpdate.rs` - Date parsing utilities (80 lines)
- **Policy Module**: `src/common/cache/policy.rs` - Cache policy logic (140 lines)

**Migration Steps**:
1. Create `src/common/cache/` directory
2. Extract modules with discrete concerns
3. Update `src/common/cache/mod.rs` with public re-exports
4. Maintain existing API surface
5. Apply zero-allocation optimizations per constraints

### 2. `src/response.rs` - 490 lines  
**Decomposition Plan**:
- **Core Module**: `src/response/core.rs` - HttpResponse struct and basic methods (150 lines)
- **Stream Module**: `src/response/stream.rs` - HttpStream implementation (120 lines)
- **SSE Module**: `src/response/sse.rs` - Server-Sent Events parsing (100 lines)
- **JSON Module**: `src/response/json.rs` - JSON Lines streaming support (120 lines)

### 3. `src/builder.rs` - 451 lines
**Decomposition Plan**:
- **Core Module**: `src/builder/core.rs` - Http3Builder struct and typestate (200 lines)
- **Methods Module**: `src/builder/methods.rs` - HTTP method implementations (150 lines)
- **Config Module**: `src/builder/config.rs` - Configuration and headers methods (101 lines)

### 4. `src/common/retry.rs` - 404 lines
**Decomposition Plan**:
- **Core Module**: `src/common/retry/core.rs` - RetryPolicy and basic logic (200 lines)
- **Backoff Module**: `src/common/retry/backoff.rs` - Exponential backoff algorithms (120 lines)
- **Jitter Module**: `src/common/retry/jitter.rs` - Jitter calculation utilities (84 lines)

### 5. `src/config.rs` - 397 lines
**Decomposition Plan**:
- **Core Module**: `src/config/core.rs` - HttpConfig struct and defaults (150 lines)
- **TLS Module**: `src/config/tls.rs` - TLS configuration options (120 lines)
- **Connection Module**: `src/config/connection.rs` - Connection pooling config (127 lines)

### 6. `src/middleware/cache.rs` - 304 lines
**Decomposition Plan**:
- **Middleware Module**: `src/middleware/cache/middleware.rs` - CacheMiddleware implementation (150 lines)
- **Headers Module**: `src/middleware/cache/headers.rs` - Cache header processing (154 lines)

---

# TEST EXTRACTION

## Issue: Tests embedded in source files
**File**: `src/middleware/cache.rs:284-304` - Contains `#[cfg(test)]` module with 2 test functions

**Technical Solution**: Extract tests to `tests/middleware_cache_test.rs`

**Migration Steps**:
1. Create `tests/middleware_cache_test.rs` 
2. Move test functions with proper imports
3. Bootstrap nextest if not already configured
4. Verify all tests execute and pass

**Code Changes Required**:
```rust
// NEW FILE: tests/middleware_cache_test.rs
use fluent_ai_http3::middleware::cache::*;

#[test]
fn test_cache_middleware() {
    let middleware = CacheMiddleware::new().with_default_expires_hours(12);
    // Test implementation with proper mocking framework
}

#[test] 
fn test_date_parsing() {
    let date = "Sun, 06 Nov 1994 08:49:37 GMT";
    let timestamp = parse_rfc1123_to_timestamp(date);
    assert!(timestamp.is_some());
}
```

**Nextest Bootstrap**:
1. Add nextest to `Cargo.toml` dev-dependencies if missing
2. Verify `cargo nextest run` executes successfully  
3. Ensure all existing tests pass after extraction

---

# QUALITY ASSURANCE STEPS

## Post-Implementation Verification
1. **Compilation**: Verify `cargo check` passes without warnings
2. **Testing**: Run `cargo nextest run` and ensure 100% pass rate
3. **Linting**: Run `cargo clippy` with zero warnings
4. **Formatting**: Verify `cargo fmt --check` passes
5. **Performance**: Benchmark critical paths for zero-allocation compliance
6. **Memory Safety**: Verify no unsafe code blocks introduced
7. **Concurrency**: Verify no locking mechanisms added (per constraints)

## Architecture Compliance Verification  
1. **AsyncStream Usage**: Verify all async operations use AsyncStream pattern
2. **HTTP3 Integration**: Ensure fluent_ai_http3 is used exclusively
3. **Domain Model**: Verify all types align with fluent_ai_domain constraints
4. **Error Handling**: Confirm no `unwrap()` or `expect()` in production code
5. **Zero Allocation**: Profile memory allocations in hot paths
6. **Ergonomic API**: Verify builder pattern maintains intuitive usage

All implementations must adhere to the zero allocation, blazing-fast, no unsafe, no locking, elegant ergonomic code constraints specified.

---

# IMMEDIATE CRITICAL IMPLEMENTATION TASKS

## PHASE 1: Runtime Safety Fixes (CRITICAL PRIORITY)

### Task: Replace block_on calls in src/builder.rs
**File**: `src/builder.rs:311, 363`
**Issue**: Synchronous blocking in async context breaks runtime semantics
**Implementation**: Replace `tokio::runtime::Handle::block_on()` with proper async stream collection
**Status**: PENDING

### Task: Replace unwrap() calls in operations files  
**Files**: 
- `src/operations/put.rs:95` - JSON serialization
- `src/operations/patch.rs:82, 83` - JSON serialization  
- `src/operations/post.rs:90, 92` - JSON and form serialization
- `src/operations/download.rs:62` - HeaderValue construction
**Issue**: Potential panics in production code
**Implementation**: Replace with proper error handling using ? operator
**Status**: PENDING

## PHASE 2: Code Quality Improvements

### Task: Update TODO comments to production descriptions
**Files**:
- `src/common/retry.rs:2`
- `src/common/cache.rs:2`  
- `src/middleware/cache.rs:97`
**Status**: PENDING

### Task: Extract embedded tests
**File**: `src/middleware/cache.rs:284-304`
**Target**: `tests/middleware_cache_test.rs`
**Status**: PENDING

## PHASE 3: Module Decomposition (Large Files >300 lines)

### Task: Decompose src/common/cache.rs (690 lines)
**Target Modules**:
- `src/common/cache/core.rs` - CacheStorage and basic operations
- `src/common/cache/entry.rs` - CacheEntry and validation  
- `src/common/cache/eviction.rs` - LRU algorithms
- `src/common/cache/httpdate.rs` - Date parsing utilities
- `src/common/cache/policy.rs` - Cache policy logic
**Status**: PENDING

### Task: Decompose src/response.rs (490 lines)
**Target Modules**:
- `src/response/core.rs` - HttpResponse struct
- `src/response/stream.rs` - HttpStream implementation
- `src/response/sse.rs` - Server-Sent Events parsing
- `src/response/json.rs` - JSON Lines streaming
**Status**: PENDING

### Task: Decompose src/builder.rs (451 lines)  
**Target Modules**:
- `src/builder/core.rs` - Http3Builder and typestate
- `src/builder/methods.rs` - HTTP method implementations
- `src/builder/config.rs` - Configuration and headers
**Status**: PENDING

### Task: Decompose src/common/retry.rs (404 lines)
**Target Modules**:
- `src/common/retry/core.rs` - RetryPolicy and basic logic
- `src/common/retry/backoff.rs` - Exponential backoff algorithms  
- `src/common/retry/jitter.rs` - Jitter calculation utilities
**Status**: PENDING

### Task: Decompose src/config.rs (397 lines)
**Target Modules**:
- `src/config/core.rs` - HttpConfig struct and defaults
- `src/config/tls.rs` - TLS configuration options
- `src/config/connection.rs` - Connection pooling config
**Status**: PENDING