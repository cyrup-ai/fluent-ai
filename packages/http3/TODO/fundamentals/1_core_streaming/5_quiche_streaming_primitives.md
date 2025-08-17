# Quiche Streaming Primitives Task

## Objective

Integrate Quiche's synchronous streaming primitives with AsyncStream architecture, using direct synchronous APIs that fit perfectly inside AsyncStream producers.

## Research Summary

### Quiche QUIC Implementation Analysis

Quiche is a low-level QUIC implementation by Cloudflare that provides synchronous APIs perfect for integration with fluent_ai_async architecture. Based on comprehensive source code analysis from [./tmp/quiche/quiche/src/lib.rs](./tmp/quiche/quiche/src/lib.rs) (336,121 bytes), Quiche provides direct synchronous primitives that eliminate the Future-based constraints of Quinn.

**Research Citations:**
- Connection creation: Lines 1748-1751 in ./tmp/quiche/quiche/src/lib.rs
- Configuration methods: Lines 1134-1199 in ./tmp/quiche/quiche/src/lib.rs  
- Stream operations: Lines 5277-5279, 5411-5413 in ./tmp/quiche/quiche/src/lib.rs
- Packet processing examples: Lines 122-142 in ./tmp/quiche/quiche/src/lib.rs
- Dependencies: ./tmp/quiche/Cargo.toml

**Key Quiche Components:**

- `Connection`: QUIC connection management with direct packet processing
- `Config`: Configuration for QUIC connections and transport parameters
- `RecvInfo`/`SendInfo`: Packet metadata for network operations
- Stream operations via connection methods (no separate stream objects)

### Real Quiche Synchronous APIs

Based on comprehensive source code analysis from [./tmp/quiche/quiche/src/lib.rs](./tmp/quiche/quiche/src/lib.rs) (336,121 bytes total), Quiche provides these **direct synchronous** methods with exact line citations:

#### Connection Management (Lines 1748-1751)

```rust
// Create connections - Source: Lines 1748-1751
pub fn connect(
    server_name: Option<&str>, scid: &ConnectionId, local: SocketAddr,
    peer: SocketAddr, config: &mut Config,
) -> Result<Connection>

pub fn accept(
    scid: &ConnectionId, odcid: Option<&ConnectionId>, local: SocketAddr,
    peer: SocketAddr, config: &mut Config,
) -> Result<Connection>

// Connection state methods - Source: Various lines in lib.rs
pub fn is_established(&self) -> bool
pub fn is_closed(&self) -> bool  
pub fn timeout(&self) -> Option<Duration>
pub fn on_timeout(&mut self)
```

#### Packet Processing (Lines 122-142 example pattern)

```rust
// Receive packets from network - Synchronous operation
pub fn recv(&mut self, buf: &mut [u8], info: RecvInfo) -> Result<usize>

// Send packets to network - Synchronous operation
pub fn send(&mut self, out: &mut [u8]) -> Result<(usize, SendInfo)>

// Example usage pattern from Lines 122-142:
loop {
    let (read, from) = socket.recv_from(&mut buf).unwrap();
    let recv_info = quiche::RecvInfo { from, to };
    
    let read = match conn.recv(&mut buf[..read], recv_info) {
        Ok(v) => v,
        Err(quiche::Error::Done) => break, // Done reading
        Err(e) => break, // Handle error
    };
}
```

#### Stream Operations (Lines 5277-5279, 5411-5413)

```rust
// Read from streams - Source: Lines 5277-5279
pub fn stream_recv(
    &mut self, stream_id: u64, out: &mut [u8],
) -> Result<(usize, bool)>

// Write to streams - Source: Lines 5411-5413  
pub fn stream_send(
    &mut self, stream_id: u64, buf: &[u8], fin: bool,
) -> Result<usize>

// Zero-copy write - Source: Lines 5435-5437
pub fn stream_send_zc(
    &mut self, stream_id: u64, buf: F::Buf, len: Option<usize>, fin: bool,
) -> Result<(usize, Option<F::Buf>)>

// Stream state queries - Synchronous iterators
pub fn readable(&self) -> StreamIter
pub fn writable(&self) -> StreamIter

// Example usage pattern:
for stream_id in conn.readable() {
    let mut buf = [0u8; 8192];
    match conn.stream_recv(stream_id, &mut buf) {
        Ok((len, fin)) => println!("Got {} bytes, fin={}", len, fin),
        Err(quiche::Error::Done) => continue,
        Err(e) => break,
    }
}
```

### fluent_ai_async Integration Architecture

Based on analysis of [../async-stream/src/stream/core.rs](../async-stream/src/stream/core.rs), the integration uses:

**Core Primitives:**

- `AsyncStream::with_channel` for all streaming operations
- `ArrayQueue<T>` for zero-allocation hot paths with const-generic capacity
- `SegQueue<T>` for dynamic capacity scenarios
- Error-as-data pattern with `MessageChunk` trait
- `emit!` macro for sending chunks to stream consumers

**Integration Pattern:**

1. Quiche synchronous methods return `Result<T, E>` immediately
2. Loop inside `AsyncStream::with_channel` for continuous processing
3. NO polling, NO context parameters, NO wakers needed
4. Results are converted to data chunks via `MessageChunk::bad_chunk()` for errors
5. Break from loop on errors or completion

## Required Actions

### 1. Corrected Quiche Synchronous Streaming Pattern (Source: ./tmp/quiche/quiche/src/lib.rs)

```rust
use fluent_ai_async::{AsyncStream, emit};
use quiche::{Connection, Config, RecvInfo, SendInfo};
use std::net::UdpSocket;

// ARCHITECTURE: Quiche Integration with AsyncStream - Loop-based synchronous pattern
// NO polling, NO context parameters, NO wakers needed

// CORRECT: Packet receiving with Quiche synchronous APIs
pub fn receive_packets(mut conn: Connection, socket: UdpSocket) -> AsyncStream<HttpChunk, 1024> {
    let to = socket.local_addr().expect("Failed to get local address");
    
    AsyncStream::with_channel(move |sender| {
        let mut buf = [0u8; 65535];
        
        loop {
            // Single operation attempt
            let (read, from) = match socket.recv_from(&mut buf) {
                Ok(result) => result,
                Err(e) => {
                    emit!(sender, QuicheReadChunk::bad_chunk(format!("Read error: {}", e)));
                    break;
                }
            };
            
            let recv_info = quiche::RecvInfo { from, to };
            
            match conn.recv(&mut buf[..read], recv_info) {
                Ok(v) => emit!(sender, v.into_http_chunk()),
                Err(e) => {
                    emit!(sender, QuicheReadChunk::bad_chunk(format!("Read error: {}", e)));
                    break;
                }
            }
        }
    })
}

// CORRECT: Stream reading with Quiche synchronous APIs
pub fn read_stream_data(mut conn: Connection, stream_id: u64) -> AsyncStream<QuicheStreamChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut buf = [0u8; 8192];
        
        loop {
            match conn.stream_recv(stream_id, &mut buf) {
                Ok((len, fin)) => {
                    if len > 0 {
                        emit!(sender, QuicheStreamChunk::data_chunk(buf[..len].to_vec()));
                    }
                    if fin {
                        emit!(sender, QuicheStreamChunk::stream_complete());
                        break;
                    }
                }
                Err(quiche::Error::Done) => {
                    // No data available - continue loop
                    continue;
                }
                Err(e) => {
                    emit!(sender, QuicheStreamChunk::bad_chunk(format!("Stream read error: {}", e)));
                    break;
                }
            }
        }
    })
}

// CORRECT: Stream writing with Quiche synchronous APIs
pub fn write_stream_data(mut conn: Connection, stream_id: u64, data: Vec<u8>, fin: bool) -> AsyncStream<QuicheWriteResult, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut offset = 0;
        
        loop {
            match conn.stream_send(stream_id, &data[offset..], fin && offset + 8192 >= data.len()) {
                Ok(written) => {
                    offset += written;
                    emit!(sender, QuicheWriteResult::bytes_written(written));
                    
                    if offset >= data.len() {
                        emit!(sender, QuicheWriteResult::write_complete());
                        break;
                    }
                }
                Err(quiche::Error::Done) => {
                    // Stream not writable - continue loop
                    continue;
                }
                Err(e) => {
                    emit!(sender, QuicheWriteResult::bad_chunk(format!("Stream write error: {}", e)));
                    break;
                }
            }
        }
    })
}

// CORRECT: Readable streams iteration with Quiche synchronous APIs
pub fn process_readable_streams(conn: Connection) -> AsyncStream<QuicheReadableChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        loop {
            for stream_id in conn.readable() {
                emit!(sender, QuicheReadableChunk::readable_stream(stream_id));
            }
            
            // Check if connection is still active
            if conn.is_closed() {
                emit!(sender, QuicheReadableChunk::connection_closed());
                break;
            }
        }
    })
}
```

## ULTRATHINK Architecture Planning

### Core Integration Strategy

**Problem Analysis:**
Quinn's poll-based APIs require a `Context` with a `Waker` for notification when `Poll::Pending` is returned. The fluent_ai_async architecture handles this through AsyncStream's elite polling loop, identical to H2/H3 integration patterns.

**Key Insight:** Quinn integration follows EXACTLY the same pattern as H2/H3 - Context parameter passed from caller, AsyncStream handles all polling internally, no custom threading or waker creation.

### Data Flow Architecture

```
Quinn Poll API → Poll::Pending → AsyncStream elite polling handles wait → Continue Loop
                ↓
            Poll::Ready(result) → emit!(sender, chunk) → AsyncStream consumer
```

### MessageChunk Implementations Required

Based on [../async-stream/src/stream/core.rs](../async-stream/src/stream/core.rs), all stream types need `MessageChunk` trait:

```rust
// Required chunk types for Quiche integration
pub struct QuichePacketChunk {
    pub bytes_processed: usize,
    pub error: Option<String>,
}

impl MessageChunk for QuichePacketChunk {
    fn bad_chunk(error: String) -> Self {
        Self { bytes_processed: 0, error: Some(error) }
    }
    
    fn is_error(&self) -> bool {
        self.error.is_some()
    }
    
    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

pub struct QuicheStreamChunk {
    pub data: Vec<u8>,
    pub stream_id: u64,
    pub fin: bool,
    pub error: Option<String>,
}

impl MessageChunk for QuicheStreamChunk {
    fn bad_chunk(error: String) -> Self {
        Self { 
            data: Vec::new(), 
            stream_id: 0, 
            fin: false, 
            error: Some(error) 
        }
    }
    
    fn is_error(&self) -> bool {
        self.error.is_some()
    }
    
    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

pub struct QuicheWriteResult {
    pub bytes_written: usize,
    pub stream_id: u64,
    pub is_complete: bool,
    pub error: Option<String>,
}

impl MessageChunk for QuicheWriteResult {
    fn bad_chunk(error: String) -> Self {
        Self { 
            bytes_written: 0, 
            stream_id: 0, 
            is_complete: false, 
            error: Some(error) 
        }
    }
    
    fn is_error(&self) -> bool {
        self.error.is_some()
    }
    
    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}
```

### 2. Implementation Files Required

**New Files to Create:**

- `src/async_impl/client/quiche_client.rs` - Quiche HTTP/3 client implementation
- `src/async_impl/connection/quiche_connection.rs` - QUIC connection management
- `src/types/quiche_chunks.rs` - MessageChunk implementations for Quiche types
- `src/quiche/packet_processor.rs` - Packet processing loop
- `src/quiche/timer_manager.rs` - Connection timeout handling
- `src/quiche/stream_manager.rs` - Stream ID management

**Existing Files to Modify:**

- `src/lib.rs` - Export Quiche integration types
- `src/client/core.rs` - Replace Quinn with Quiche backend
- `Cargo.toml` - Remove Quinn dependencies, add Quiche dependencies

### 3. Dependencies Analysis

Based on Quiche source analysis from [./tmp/quiche/Cargo.toml](./tmp/quiche/Cargo.toml) and [./tmp/quiche/quiche/Cargo.toml](./tmp/quiche/quiche/Cargo.toml):

```toml
[dependencies]
quiche = { version = "0.22", features = ["boringssl-vendored"] }
# Note: No tokio or async runtime dependencies needed
# Quiche uses synchronous APIs that integrate directly with AsyncStream

# Quiche internal dependencies (from ./tmp/quiche/quiche/Cargo.toml):
# - ring or boringssl for crypto
# - libc for system calls
# - log for logging
# - octets for buffer management
```

### 4. Architecture Requirements

**✅ CORRECT Patterns:**

- Use Quiche's synchronous APIs: `conn.recv()`, `conn.stream_recv()`, `conn.stream_send()`
- Loop inside `AsyncStream::with_channel` for continuous processing
- Error-as-data pattern with `MessageChunk::bad_chunk()`
- `emit!` macro for sending chunks to stream consumers
- Break from loop on errors or completion
- Direct socket operations with `socket.recv_from()` and `socket.send_to()`

**❌ FORBIDDEN Patterns:**

- NO Future-based APIs or async/await in streaming producers
- NO external async runtimes (tokio runtime usage)
- NO manual `Future` implementations or polling
- NO `Result<T,E>` inside `AsyncStream<T>`
- NO Parker/Unparker or custom Waker creation
- NO manual thread spawning or `std::thread::spawn`
- NO `unwrap()` calls - use proper error handling

## Complete 15-File Quinn to Quiche Conversion Plan

**Research Source:** Complete analysis of [./tmp/quiche/quiche/src/lib.rs](./tmp/quiche/quiche/src/lib.rs) (336,121 bytes)
**Documentation:** [./docs/complete_15_file_quinn_to_quiche_mappings.md](./docs/complete_15_file_quinn_to_quiche_mappings.md)

### Files Requiring Changes:

#### Simple Import Removals (4 files)
1. `src/hyper/async_impl/client/mod.rs` - Remove `use quinn::VarInt;` (Line 39)
2. `src/hyper/async_impl/client/config/types.rs` - Remove `use quinn::VarInt;` (Line 13)
3. `src/hyper/async_impl/client/builder/protocols.rs` - Remove `use quinn::VarInt;` (Line 9)
4. `src/hyper/async_impl/client/builder/types.rs` - Remove `use quinn::VarInt;` (Line 16)

#### Major Rewrites (6 files)
5. `src/hyper/async_impl/h3_client/connect/connector_core.rs` - Replace Quinn config with Quiche config
6. `src/hyper/async_impl/h3_client/connect/types.rs` - Replace quinn_connection with quiche_connection  
7. `src/hyper/async_impl/h3_client/connect/h3_establishment.rs` - Replace endpoint with socket/config
8. `src/async_impl/client/quinn_client.rs` → `quiche_client.rs` - Complete rewrite
9. `src/async_impl/connection/quinn_connection.rs` → `quiche_connection.rs` - Complete rewrite
10. `src/types/quinn_chunks.rs` → `quiche_chunks.rs` - Complete replacement

#### Streaming Updates (2 files)
11. `src/streaming/transport.rs` - Replace Quinn imports with Quiche
12. `src/streaming/pipeline.rs` - Replace Quinn imports with Quiche

#### Module Re-exports (2 files)
13. `src/async_impl/client/mod.rs` - Update quinn_client to quiche_client
14. `src/async_impl/connection/mod.rs` - Update quinn_connection to quiche_connection

#### Dependencies (1 file)
15. `Cargo.toml` - Remove Quinn deps, add Quiche deps, update features

## Simple Implementation Pattern (Source: Lines 122-142)

The Quiche integration follows the exact pattern from ./tmp/quiche/quiche/src/lib.rs:

```rust
AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
    loop {
        // 1. Receive UDP packet (synchronous)
        let (read, from) = match socket.recv_from(&mut buf) {
            Ok(result) => result,
            Err(e) => {
                emit!(sender, HttpChunk::bad_chunk(format!("Socket error: {}", e)));
                break;
            }
        };
        
        // 2. Process QUIC packet (synchronous - Lines 122-142 pattern)
        let recv_info = quiche::RecvInfo { from, to };
        match conn.recv(&mut buf[..read], recv_info) {
            Ok(_bytes_processed) => {
                // 3. Read HTTP3 data from streams (synchronous - Lines 5277-5279)
                for stream_id in conn.readable() {
                    let mut stream_buf = [0u8; 8192];
                    match conn.stream_recv(stream_id, &mut stream_buf) {
                        Ok((len, _fin)) => {
                            emit!(sender, HttpChunk::data_chunk(stream_buf[..len].to_vec()));
                        }
                        Err(quiche::Error::Done) => continue,
                        Err(e) => {
                            emit!(sender, HttpChunk::bad_chunk(format!("Stream error: {}", e)));
                        }
                    }
                }
            }
            Err(e) => {
                emit!(sender, HttpChunk::bad_chunk(format!("Packet error: {}", e)));
                break;
            }
        }
    }
})
```

**Key Benefits:**
- No async/await - Pure synchronous APIs from Quiche source
- No polling/wakers - Direct Result<T,E> return values
- No Future complexity - Simple loop-based processing  
- Zero allocation - Direct buffer operations
- Error-as-data - All errors converted to MessageChunk::bad_chunk()

**All implementations maintain fluent_ai_async architecture compliance with proper loop-based streaming patterns.**
