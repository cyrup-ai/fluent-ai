# Tokio Removal Plan - HTTP3 Library

## Phase 1: H2Connection Tokio Elimination

### 1. Replace TokioIo with Standard TCP Streams
- **File**: `/Volumes/samsung_t9/fluent-ai/packages/http3/src/protocols/h2/connection.rs` (lines 13, 19)
- **Current**: `h2::client::Connection<hyper_util::rt::TokioIo<tokio::net::TcpStream>>`
- **Replace with**: `h2::client::Connection<std::net::TcpStream>` or hyper-compatible alternative
- **Architecture**: Use standard library TCP streams instead of tokio-specific wrappers
- **Rationale**: Eliminate tokio runtime dependency while maintaining HTTP/2 functionality

### 2. Update H2Connection Constructor
- **File**: `/Volumes/samsung_t9/fluent-ai/packages/http3/src/protocols/h2/connection.rs` (line 19)
- **Action**: Update constructor parameter type to match new connection type
- **Architecture**: Ensure fluent_ai_async compatibility without tokio dependency

### 3. Verify H2 Chunks Implementation
- **File**: `/Volumes/samsung_t9/fluent-ai/packages/http3/src/protocols/h2/chunks.rs` (line 6)
- **Action**: Confirm comment accuracy - ensure implementation is truly tokio-free
- **Architecture**: Validate that H2ConnectionChunk enum works without tokio

## Phase 2: Dependency Cleanup

### 4. Remove Tokio from Cargo.toml
- **File**: `/Volumes/samsung_t9/fluent-ai/packages/http3/Cargo.toml` (line ~62)
- **Action**: Remove `tokio = { version = "1.0", features = ["rt", "net", "time"] }`
- **Method**: Use `cargo remove tokio` (per user memory about cargo commands)
- **Architecture**: Eliminate tokio runtime dependency entirely

### 5. Verify No Remaining Tokio References
- **Action**: Search entire codebase for any remaining tokio imports or usage
- **Command**: `grep -r "tokio\|TokioIo" src/`
- **Architecture**: Ensure complete tokio elimination

## Phase 3: Alternative Implementation Strategy

### Option A: Standard Library TCP
- Replace `TokioIo<tokio::net::TcpStream>` with `std::net::TcpStream`
- May require adapter layer for hyper compatibility
- Pros: Zero external async runtime dependency
- Cons: May need custom async handling

### Option B: Hyper-util Alternative
- Use `hyper_util::rt::TokioExecutor` alternatives if available
- Research hyper-util runtime-agnostic options
- Pros: Maintains hyper ecosystem compatibility
- Cons: May still have indirect tokio dependency

### Option C: Custom Stream Wrapper
- Create custom stream wrapper implementing required traits
- Wrap `std::net::TcpStream` with necessary async traits
- Pros: Full control over implementation
- Cons: More complex implementation

## Recommended Approach: Option A + Custom Adapter

1. Replace tokio types with standard library equivalents
2. Create minimal adapter layer if needed for hyper compatibility
3. Ensure fluent_ai_async patterns work with new implementation
4. Maintain zero-allocation streaming architecture

## Validation Criteria

- [ ] Zero tokio references in source code
- [ ] Zero tokio dependencies in Cargo.toml
- [ ] HTTP/2 functionality still works
- [ ] All tests pass
- [ ] fluent_ai_async compatibility maintained
- [ ] Zero compilation errors