# Quinn to Quiche API Mapping

This document provides comprehensive mappings between Quinn QUIC APIs and Quiche QUIC APIs for the fluent-ai http3 package migration.

## Core Connection APIs

### Connection Creation

**Quinn:**
```rust
// Client connection
let connection = quinn::Endpoint::connect(addr, server_name)?;
let connection = connection.await?;

// Server connection  
let incoming = endpoint.accept().await.unwrap();
let connection = incoming.await?;
```

**Quiche:**
```rust
// Client connection
let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)?;
let conn = quiche::connect(Some(&server_name), &scid, local, peer, &mut config)?;

// Server connection
let conn = quiche::accept(&scid, None, local, peer, &mut config)?;
```

### Connection State

**Quinn:**
```rust
// Check if connection is established
connection.close_reason() // Returns None if still connected

// Connection stats
connection.stats()
```

**Quiche:**
```rust
// Check if connection is established
conn.is_established() -> bool

// Connection stats  
conn.stats() -> Stats
```

## Stream Management APIs

### Opening Streams

**Quinn (Future-based):**
```rust
// Open bidirectional stream
let (send, recv) = connection.open_bi().await?;

// Open unidirectional stream
let send = connection.open_uni().await?;
```

**Quiche (Synchronous):**
```rust
// Quiche doesn't have explicit "open stream" methods
// Streams are created implicitly when first used with stream_send()
// Stream IDs follow QUIC specification:
// - Client-initiated bidi: 0, 4, 8, 12, ...
// - Server-initiated bidi: 1, 5, 9, 13, ...
// - Client-initiated uni: 2, 6, 10, 14, ...
// - Server-initiated uni: 3, 7, 11, 15, ...

let stream_id = if is_client { 0 } else { 1 }; // First bidi stream
conn.stream_send(stream_id, data, fin)?; // Creates stream implicitly
```

### Stream Reading

**Quinn (Future-based):**
```rust
// Read from stream
let mut buf = [0u8; 1024];
let bytes_read = recv_stream.read(&mut buf).await?;

// Read to end
let data = recv_stream.read_to_end(usize::MAX).await?;
```

**Quiche (Synchronous):**
```rust
// Read from stream
let mut buf = [0u8; 1024];
let (bytes_read, fin) = conn.stream_recv(stream_id, &mut buf)?;

// Check for readable streams
for stream_id in conn.readable() {
    while let Ok((read, fin)) = conn.stream_recv(stream_id, &mut buf) {
        // Process data
        if fin { break; }
    }
}
```

### Stream Writing

**Quinn (Future-based):**
```rust
// Write to stream
send_stream.write_all(data).await?;

// Finish stream
send_stream.finish().await?;
```

**Quiche (Synchronous):**
```rust
// Write to stream
let bytes_written = conn.stream_send(stream_id, data, fin)?;

// Write with zero-copy
let (bytes_written, remaining_buf) = conn.stream_send_zc(stream_id, buf, None, fin)?;
```

### Stream State Queries

**Quinn:**
```rust
// Check if stream is finished
recv_stream.stop_reason() // Returns Some(error) if stopped

// Stream priority (not directly available)
```

**Quiche:**
```rust
// Check readable streams
conn.readable() -> StreamIter

// Check writable streams  
conn.writable() -> StreamIter

// Stream priority
conn.stream_priority(stream_id, urgency, incremental)?;
```

## Packet Processing APIs

### Receiving Packets

**Quinn:**
```rust
// Quinn handles packet processing internally through the endpoint
// No direct packet processing API exposed
```

**Quiche:**
```rust
// Receive and process packet
let recv_info = quiche::RecvInfo { from: peer_addr, to: local_addr };
let bytes_processed = conn.recv(&mut packet_buf[..packet_len], recv_info)?;
```

### Sending Packets

**Quinn:**
```rust
// Quinn handles packet sending internally
// No direct packet generation API exposed
```

**Quiche:**
```rust
// Generate outgoing packet
let mut out_buf = [0u8; 1500];
let (packet_len, send_info) = conn.send(&mut out_buf)?;
// Send packet_buf[..packet_len] to send_info.to address
```

## Timer Management

**Quinn:**
```rust
// Timer management is internal to Quinn
// Applications don't directly handle timeouts
```

**Quiche:**
```rust
// Get next timeout
let timeout_duration = conn.timeout();

// Handle timeout expiration
conn.on_timeout();
```

## Configuration APIs

### Basic Configuration

**Quinn:**
```rust
let mut config = quinn::ClientConfig::new(Arc::new(crypto_config));
config.transport_config(Arc::new(transport_config));
```

**Quiche:**
```rust
let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)?;
config.set_application_protos(&[b"h3"])?;
config.set_initial_max_data(10_000_000);
config.set_initial_max_stream_data_bidi_local(1_000_000);
config.set_initial_max_stream_data_bidi_remote(1_000_000);
config.set_initial_max_streams_bidi(100);
config.set_initial_max_streams_uni(100);
```

### TLS Configuration

**Quinn:**
```rust
let crypto = rustls::ClientConfig::builder()
    .with_safe_defaults()
    .with_root_certificates(roots)
    .with_no_client_auth();
```

**Quiche:**
```rust
config.load_cert_chain_from_pem_file("cert.pem")?;
config.load_priv_key_from_pem_file("key.pem")?;
config.verify_peer(false); // For testing
```

## Error Handling

**Quinn:**
```rust
// Quinn uses standard Rust Result<T, E> patterns
// Errors are typically quinn::ConnectionError, quinn::ReadError, etc.
match connection.open_bi().await {
    Ok((send, recv)) => { /* handle streams */ },
    Err(quinn::ConnectionError::ApplicationClosed(_)) => { /* handle close */ },
    Err(e) => { /* handle other errors */ }
}
```

**Quiche:**
```rust
// Quiche uses quiche::Error enum
match conn.stream_recv(stream_id, &mut buf) {
    Ok((len, fin)) => { /* handle data */ },
    Err(quiche::Error::Done) => { /* no more data available */ },
    Err(quiche::Error::InvalidStreamState(_)) => { /* stream error */ },
    Err(e) => { /* handle other errors */ }
}
```

## Key Architectural Differences

### 1. Async vs Sync APIs

**Quinn:** Uses async/await throughout, returns Futures that must be awaited
**Quiche:** Uses synchronous APIs that return immediately with success/failure

### 2. Stream Management

**Quinn:** Explicit stream opening with Future-based operations
**Quiche:** Implicit stream creation, synchronous read/write operations

### 3. Packet Handling

**Quinn:** Internal packet processing, no direct access
**Quiche:** Explicit packet receive/send cycle, application manages I/O

### 4. Event Loop Integration

**Quinn:** Built-in async runtime integration
**Quiche:** Application provides event loop, timers, and I/O handling

## Integration with fluent_ai_async

For integration with `fluent_ai_async::AsyncStream`, Quiche's synchronous APIs fit perfectly:

```rust
use fluent_ai_async::AsyncStream;

// Quiche streaming pattern
AsyncStream::with_channel(move |tx| {
    // Single poll attempt - no manual loops
    match conn.stream_recv(stream_id, &mut buf) {
        Ok((len, fin)) => {
            emit!(tx, MessageChunk::data_chunk(buf[..len].to_vec()));
            if fin {
                emit!(tx, MessageChunk::end_chunk());
            }
        },
        Err(quiche::Error::Done) => {
            // No data available, will be polled again later
        },
        Err(e) => {
            emit!(tx, MessageChunk::bad_chunk(format!("Stream error: {}", e)));
        }
    }
})
```

This pattern allows async streaming while using Quiche's synchronous APIs internally, avoiding the Future-based constraints that made Quinn incompatible with the fluent_ai_async architecture.