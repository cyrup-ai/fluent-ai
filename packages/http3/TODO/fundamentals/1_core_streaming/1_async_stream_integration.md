### 5. Files to Create/Modify

#### 5.1 New Files to Create

- `src/types/chunks.rs` - HttpResponseChunk and MessageChunk implementations
- `src/async_impl/response/core/streaming.rs` - Core streaming response types
- `src/async_impl/client/connection.rs` - Connection management infrastructure

#### 5.2 Existing Files to Modify

- `src/lib.rs` - Export new public types
- `src/types/mod.rs` - Include chunks module
- `src/async_impl/mod.rs` - Include new modules

#### 5.3 Integration Points for Protocol Tasks

```rust
// Protocol tasks will extend these base patterns:

// h2 integration will use:
// - HttpResponseChunk as base chunk type
// - HttpConnection for connection management
// - create_error_aware_stream for h2-specific error handling

// h3 integration will use:
// - HttpResponseChunk as base chunk type  
// - HttpConnectionPool for QUIC connection management
// - create_timeout_aware_stream for QUIC timeout handling

// Quinn integration will use:
// - HttpResponseChunk as base chunk type
// - Custom connection types extending HttpConnection
// - All streaming patterns for QUIC stream management

// WASM integration will use:
// - HttpResponseChunk as base chunk type
// - Browser-specific connection handling
// - create_basic_response_stream for fetch API integration

// Type Consolidation Requirements
// Remove conflicting `HttpResponseChunk` struct implementations from:
//   - `src/types/response.rs`
//   - `src/wasm/body/body_impl.rs`
//   - `src/async_impl/response/core/static_constructors.rs`
// Replace with canonical enum-based `HttpResponseChunk` from this specification
// Add `MessageChunk` implementation for `bytes::Bytes` type:

impl MessageChunk for bytes::Bytes {
    fn is_error(&self) -> bool { false }
    
    fn bad_chunk(error: String) -> Self {
        bytes::Bytes::from(error)
    }
    
    fn error(&self) -> Option<&str> { None }
}
