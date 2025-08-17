# Quinn to Quiche Conversion Plan

This document outlines all required changes to migrate the fluent-ai http3 package from Quinn QUIC implementation to Quiche QUIC implementation.

## Overview

The migration involves replacing Quinn's Future-based async APIs with Quiche's synchronous APIs while maintaining compatibility with the `fluent_ai_async::AsyncStream` architecture.

## 1. Dependency Changes

### Cargo.toml Updates

**Remove Quinn dependencies:**
```toml
# Remove these lines:
quinn = { version = "0.11.8", features = ["rustls"], optional = true }
quinn = { version = "0.11", default-features = false, features = ["rustls"], optional = true }
quinn-proto = "0.11.12"
h3-quinn = { version = "0.0.10", optional = true }
```

**Add Quiche dependencies:**
```toml
# Add these lines:
quiche = { version = "0.22", features = ["boringssl-vendored"], optional = true }
```

**Update feature flags:**
```toml
# Change from:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:h3-quinn", "dep:quinn"]

# To:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:quiche"]
```

## 2. Core Module Changes

### 2.1 Connection Module (`src/client/core.rs`)

**Current Quinn-based structure:**
```rust
use quinn::{Connection, Endpoint, ClientConfig};

pub struct QuicConnection {
    connection: Connection,
    endpoint: Endpoint,
}
```

**New Quiche-based structure:**
```rust
use quiche::{Connection, Config};
use std::net::UdpSocket;

pub struct QuicConnection<F = quiche::DefaultBufFactory> 
where 
    F: quiche::BufFactory,
{
    connection: Connection<F>,
    socket: UdpSocket,
    config: Config,
    local_addr: SocketAddr,
    peer_addr: SocketAddr,
}
```

### 2.2 Stream Management (`src/async_impl/connection/`)

**Current Quinn approach:**
```rust
// Future-based stream operations
let (send_stream, recv_stream) = connection.open_bi().await?;
let data = recv_stream.read_to_end(usize::MAX).await?;
```

**New Quiche approach with AsyncStream:**
```rust
use fluent_ai_async::AsyncStream;

// Synchronous stream operations in AsyncStream context
AsyncStream::with_channel(move |tx| {
    for stream_id in conn.readable() {
        let mut buf = [0u8; 8192];
        match conn.stream_recv(stream_id, &mut buf) {
            Ok((len, fin)) => {
                emit!(tx, MessageChunk::data_chunk(buf[..len].to_vec()));
                if fin {
                    emit!(tx, MessageChunk::end_chunk());
                }
            },
            Err(quiche::Error::Done) => {
                // No data available - will be polled again
            },
            Err(e) => {
                emit!(tx, MessageChunk::bad_chunk(format!("Stream error: {}", e)));
            }
        }
    }
})
```

## 3. File-by-File Changes

### 3.1 `src/client/core.rs`

**Changes required:**
1. Replace Quinn Connection with Quiche Connection
2. Add UDP socket management
3. Implement packet send/receive loop
4. Add timer management for connection timeouts
5. Update connection state checking methods

**Key method updates:**
```rust
// Before (Quinn):
async fn connect(&self) -> Result<Connection> {
    let connection = self.endpoint.connect(self.addr, &self.server_name)?.await?;
    Ok(connection)
}

// After (Quiche):
fn connect(&mut self) -> Result<()> {
    let scid = quiche::ConnectionId::from_ref(&[/* generate random ID */]);
    self.connection = quiche::connect(
        Some(&self.server_name),
        &scid,
        self.local_addr,
        self.peer_addr,
        &mut self.config
    )?;
    Ok(())
}
```

### 3.2 `src/async_impl/client/mod.rs`

**Changes required:**
1. Replace Quinn stream opening with Quiche stream ID management
2. Update stream reading/writing to use synchronous Quiche APIs
3. Integrate with AsyncStream for async compatibility

### 3.3 `src/async_impl/connection/mod.rs`

**Changes required:**
1. Remove Future-based connection handling
2. Add packet processing loop integration
3. Implement connection state management
4. Add timeout handling

### 3.4 `src/async_impl/request/mod.rs`

**Changes required:**
1. Update HTTP/3 request sending to use Quiche streams
2. Replace async stream operations with AsyncStream patterns
3. Update response reading logic

### 3.5 `src/builder/methods/` (All files)

**Changes required:**
1. Update connection builder to use Quiche configuration
2. Remove Quinn-specific configuration options
3. Add Quiche-specific configuration methods

## 4. New Components Required

### 4.1 Packet Processing Loop (`src/quiche/packet_loop.rs`)

**New file required:**
```rust
use quiche::{Connection, RecvInfo, SendInfo};
use std::net::UdpSocket;

pub struct PacketProcessor {
    socket: UdpSocket,
    connection: Connection,
}

impl PacketProcessor {
    pub fn process_packets(&mut self) -> Result<()> {
        // Receive packets
        let mut buf = [0u8; 65535];
        while let Ok((len, from)) = self.socket.recv_from(&mut buf) {
            let recv_info = RecvInfo {
                from,
                to: self.socket.local_addr()?,
            };
            
            match self.connection.recv(&mut buf[..len], recv_info) {
                Ok(_) => {},
                Err(quiche::Error::Done) => break,
                Err(e) => return Err(e.into()),
            }
        }
        
        // Send packets
        let mut out = [0u8; 1500];
        while let Ok((len, send_info)) = self.connection.send(&mut out) {
            self.socket.send_to(&out[..len], &send_info.to)?;
        }
        
        Ok(())
    }
}
```

### 4.2 Timer Management (`src/quiche/timer.rs`)

**New file required:**
```rust
use std::time::{Duration, Instant};
use quiche::Connection;

pub struct ConnectionTimer {
    connection: Connection,
    next_timeout: Option<Instant>,
}

impl ConnectionTimer {
    pub fn update_timeout(&mut self) {
        if let Some(timeout) = self.connection.timeout() {
            self.next_timeout = Some(Instant::now() + timeout);
        } else {
            self.next_timeout = None;
        }
    }
    
    pub fn handle_timeout(&mut self) -> bool {
        if let Some(timeout) = self.next_timeout {
            if Instant::now() >= timeout {
                self.connection.on_timeout();
                self.update_timeout();
                return true;
            }
        }
        false
    }
}
```

### 4.3 Stream ID Management (`src/quiche/stream_manager.rs`)

**New file required:**
```rust
pub struct StreamIdManager {
    next_bidi_stream_id: u64,
    next_uni_stream_id: u64,
    is_client: bool,
}

impl StreamIdManager {
    pub fn new(is_client: bool) -> Self {
        let (bidi_start, uni_start) = if is_client {
            (0, 2) // Client-initiated streams
        } else {
            (1, 3) // Server-initiated streams
        };
        
        Self {
            next_bidi_stream_id: bidi_start,
            next_uni_stream_id: uni_start,
            is_client,
        }
    }
    
    pub fn next_bidi_stream(&mut self) -> u64 {
        let id = self.next_bidi_stream_id;
        self.next_bidi_stream_id += 4;
        id
    }
    
    pub fn next_uni_stream(&mut self) -> u64 {
        let id = self.next_uni_stream_id;
        self.next_uni_stream_id += 4;
        id
    }
}
```

## 5. Integration Points

### 5.1 AsyncStream Integration

**Pattern for all streaming operations:**
```rust
use fluent_ai_async::AsyncStream;

// Reading from Quiche streams
AsyncStream::with_channel(move |tx| {
    let mut buf = [0u8; 8192];
    
    // Single poll attempt - no manual loops
    match conn.stream_recv(stream_id, &mut buf) {
        Ok((len, fin)) => {
            if len > 0 {
                emit!(tx, MessageChunk::data_chunk(buf[..len].to_vec()));
            }
            if fin {
                emit!(tx, MessageChunk::end_chunk());
            }
        },
        Err(quiche::Error::Done) => {
            // No data available - AsyncStream will poll again
        },
        Err(e) => {
            emit!(tx, MessageChunk::bad_chunk(format!("Stream error: {}", e)));
        }
    }
})

// Writing to Quiche streams
AsyncStream::with_channel(move |tx| {
    match conn.stream_send(stream_id, &data, fin) {
        Ok(written) => {
            emit!(tx, MessageChunk::data_chunk(written.to_string().into_bytes()));
            if written < data.len() {
                // Partial write - will be retried
            }
        },
        Err(quiche::Error::Done) => {
            // Stream not writable - will be retried
        },
        Err(e) => {
            emit!(tx, MessageChunk::bad_chunk(format!("Write error: {}", e)));
        }
    }
})
```

### 5.2 Error Handling Updates

**Quinn error mapping to Quiche:**
```rust
// Before (Quinn):
match quinn_result {
    Err(quinn::ConnectionError::ApplicationClosed(_)) => { /* handle */ },
    Err(quinn::ReadError::Reset(_)) => { /* handle */ },
    // ...
}

// After (Quiche):
match quiche_result {
    Err(quiche::Error::Done) => { /* no data/space available */ },
    Err(quiche::Error::BufferTooShort) => { /* buffer too small */ },
    Err(quiche::Error::InvalidStreamState(_)) => { /* stream error */ },
    // ...
}
```

## 6. Configuration Updates

### 6.1 QUIC Configuration

**Replace Quinn ClientConfig with Quiche Config:**
```rust
// Before (Quinn):
let mut transport_config = quinn::TransportConfig::default();
transport_config.max_concurrent_bidi_streams(100u32.into());
let mut client_config = quinn::ClientConfig::new(Arc::new(crypto_config));
client_config.transport_config(Arc::new(transport_config));

// After (Quiche):
let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)?;
config.set_application_protos(&[b"h3"])?;
config.set_initial_max_data(10_000_000);
config.set_initial_max_stream_data_bidi_local(1_000_000);
config.set_initial_max_stream_data_bidi_remote(1_000_000);
config.set_initial_max_streams_bidi(100);
config.set_initial_max_streams_uni(100);
config.verify_peer(false); // For development
```

### 6.2 TLS Configuration

**Update TLS setup for Quiche:**
```rust
// Quiche TLS configuration
config.load_cert_chain_from_pem_file("cert.pem")?;
config.load_priv_key_from_pem_file("key.pem")?;
config.set_application_protos(&[b"h3", b"h3-29", b"h3-28", b"h3-27"])?;
```

## 7. Testing Updates

### 7.1 Unit Tests

**Update all unit tests to use Quiche APIs:**
- Replace Quinn mock connections with Quiche connections
- Update stream operation tests
- Add packet processing tests
- Add timer management tests

### 7.2 Integration Tests

**Update integration tests:**
- Test full HTTP/3 request/response cycle
- Test connection establishment and teardown
- Test stream multiplexing
- Test error handling scenarios

## 8. Performance Considerations

### 8.1 Zero-Allocation Patterns

**Maintain zero-allocation streaming:**
```rust
// Use pre-allocated buffers
let mut buf = [0u8; 8192];
let (len, fin) = conn.stream_recv(stream_id, &mut buf)?;

// Use zero-copy operations where possible
let (written, remaining) = conn.stream_send_zc(stream_id, buf, None, fin)?;
```

### 8.2 Buffer Management

**Efficient buffer reuse:**
```rust
// Reuse buffers across operations
struct BufferPool {
    buffers: Vec<Vec<u8>>,
}

impl BufferPool {
    fn get_buffer(&mut self) -> Vec<u8> {
        self.buffers.pop().unwrap_or_else(|| vec![0u8; 8192])
    }
    
    fn return_buffer(&mut self, mut buf: Vec<u8>) {
        buf.clear();
        self.buffers.push(buf);
    }
}
```

## 9. Migration Steps

### Phase 1: Core Infrastructure
1. Update Cargo.toml dependencies
2. Create new Quiche wrapper modules
3. Implement packet processing loop
4. Add timer management

### Phase 2: Connection Management
1. Replace Quinn Connection with Quiche Connection
2. Update connection establishment logic
3. Implement connection state management
4. Add error handling

### Phase 3: Stream Operations
1. Replace Future-based stream operations
2. Integrate with AsyncStream patterns
3. Update HTTP/3 request/response handling
4. Test stream multiplexing

### Phase 4: Testing and Optimization
1. Update all tests to use Quiche
2. Performance testing and optimization
3. Memory usage validation
4. Integration testing

## 10. Validation Criteria

### Functional Requirements
- [ ] HTTP/3 requests work correctly
- [ ] Stream multiplexing functions properly
- [ ] Connection establishment and teardown work
- [ ] Error handling is comprehensive
- [ ] All existing tests pass

### Performance Requirements
- [ ] Zero-allocation streaming maintained
- [ ] No async_trait usage
- [ ] No boxed futures in streaming producers
- [ ] Memory usage within acceptable limits
- [ ] Latency comparable to Quinn implementation

### Architectural Requirements
- [ ] Full compatibility with fluent_ai_async::AsyncStream
- [ ] No manual polling loops in streaming producers
- [ ] Proper error conversion to MessageChunk::bad_chunk()
- [ ] Single poll attempts per AsyncStream invocation
- [ ] Synchronous APIs used inside async contexts

This migration plan ensures a systematic transition from Quinn to Quiche while maintaining the architectural purity and performance characteristics required by the fluent-ai http3 package.