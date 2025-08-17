# Line-by-Line Quinn to Quiche Conversion Plan

This document provides exact line-by-line mappings for converting every Quinn usage to Quiche equivalents.

## Files with Quinn Dependencies

### 1. `/src/hyper/async_impl/client/mod.rs`

**Line 39:**
```rust
// BEFORE:
use quinn::VarInt;

// AFTER:
// Remove - VarInt not needed with Quiche's simpler API
```

### 2. `/src/hyper/async_impl/client/builder/types.rs`

**Line 16:**
```rust
// BEFORE:
use quinn::VarInt;

// AFTER:
// Remove - VarInt not needed with Quiche's simpler API
```

### 3. `/src/hyper/async_impl/client/builder/protocols.rs`

**Line 9:**
```rust
// BEFORE:
use quinn::VarInt;

// AFTER:
// Remove - VarInt not needed with Quiche's simpler API
```

### 4. `/src/hyper/async_impl/client/config/types.rs`

**Line 13:**
```rust
// BEFORE:
use quinn::VarInt;

// AFTER:
// Remove - VarInt not needed with Quiche's simpler API
```

### 5. `/src/hyper/async_impl/h3_client/connect/connector_core.rs`

**Lines 10-11:**
```rust
// BEFORE:
use quinn::crypto::rustls::QuicClientConfig;
use quinn::{ClientConfig, Endpoint, TransportConfig};

// AFTER:
use quiche::{Config, Connection};
use std::net::UdpSocket;
```

**Lines 55-56:**
```rust
// BEFORE:
let mut quinn_config = ClientConfig::new(quic_client_config);

// AFTER:
let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)?;
config.set_application_protos(&[b"h3"])?;
```

**Lines 58-70:**
```rust
// BEFORE:
let mut transport_config = TransportConfig::default();
// ... transport config setup
transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
quinn_config.transport_config(Arc::new(transport_config));

// AFTER:
config.set_initial_max_data(10_000_000);
config.set_initial_max_stream_data_bidi_local(1_000_000);
config.set_initial_max_stream_data_bidi_remote(1_000_000);
config.set_initial_max_streams_bidi(100);
config.set_initial_max_streams_uni(100);
config.set_max_idle_timeout(30_000); // 30 seconds
```

**Lines 73-89:**
```rust
// BEFORE:
let socket_addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0));
let endpoint = match Endpoint::client(socket_addr) {
    Ok(ep) => ep,
    Err(e) => {
        return None;
    }
};
endpoint.set_default_client_config(quinn_config);

// AFTER:
let socket = UdpSocket::bind("0.0.0.0:0")?;
socket.set_nonblocking(true)?;
// Connection will be created when needed with quiche::connect()
```

### 6. `/src/hyper/async_impl/h3_client/client/stream_execution_internal.rs`

**Lines 56-57:**
```rust
// BEFORE:
let response_stream = Self::execute_h3_request_with_quinn_streaming(

// AFTER:
let response_stream = Self::execute_h3_request_with_quiche_streaming(
```

**Lines 88-89:**
```rust
// BEFORE:
fn execute_h3_request_with_quinn_streaming(

// AFTER:
fn execute_h3_request_with_quiche_streaming(
```

**Line 143:**
```rust
// BEFORE:
mut send_stream: quinn::SendStream,

// AFTER:
conn: &mut quiche::Connection,
stream_id: u64,
```

**Line 223:**
```rust
// BEFORE:
send_stream: quinn::SendStream,

// AFTER:
conn: &mut quiche::Connection,
stream_id: u64,
```

**Line 265:**
```rust
// BEFORE:
recv_stream: quinn::RecvStream,

// AFTER:
conn: &mut quiche::Connection,
stream_id: u64,
```

### 7. `/src/hyper/async_impl/h3_client/connect/types.rs`

**Line 12:**
```rust
// BEFORE:
pub(crate) quinn_connection: Option<quinn::Connection>,

// AFTER:
pub(crate) quiche_connection: Option<quiche::Connection>,
pub(crate) socket: Option<UdpSocket>,
```

**Line 19:**
```rust
// BEFORE:
quinn_connection: None,

// AFTER:
quiche_connection: None,
socket: None,
```

**Line 36:**
```rust
// BEFORE:
quinn_connection: None,

// AFTER:
quiche_connection: None,
socket: None,
```

**Line 44:**
```rust
// BEFORE:
pub fn open_bi(&self) -> Result<(quinn::SendStream, quinn::RecvStream), String> {

// AFTER:
pub fn open_bi(&self) -> Result<u64, String> { // Returns stream_id
```

**Line 45:**
```rust
// BEFORE:
match &self.quinn_connection {

// AFTER:
match &self.quiche_connection {
```

**Line 56:**
```rust
// BEFORE:
pub fn open_bi_streaming(&self) -> AsyncStream<(quinn::SendStream, quinn::RecvStream)> {

// AFTER:
pub fn open_bi_streaming(&self) -> AsyncStream<u64> { // Returns stream_id
```

**Line 57:**
```rust
// BEFORE:
let quinn_conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Lines 65-67:**
```rust
// BEFORE:
quinn::SendStream::bad_chunk("No connection available".to_string()),
quinn::RecvStream::bad_chunk("No connection available".to_string())

// AFTER:
emit!(sender, HttpChunk::bad_chunk("No connection available".to_string()));
return;
```

**Lines 80-82:**
```rust
// BEFORE:
quinn::SendStream::bad_chunk(format!("Failed to open send stream: {}", e)),
quinn::RecvStream::bad_chunk(format!("Failed to open recv stream: {}", e)),

// AFTER:
emit!(sender, HttpChunk::bad_chunk(format!("Failed to open stream: {}", e)));
return;
```

**Line 90:**
```rust
// BEFORE:
pub fn open_uni_streaming(&self) -> AsyncStream<quinn::SendStream> {

// AFTER:
pub fn open_uni_streaming(&self) -> AsyncStream<u64> { // Returns stream_id
```

**Line 91:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Line 117:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Line 137:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Line 162:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Line 183:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

**Line 198:**
```rust
// BEFORE:
let conn = self.quinn_connection.clone();

// AFTER:
let conn = self.quiche_connection.clone();
```

### 8. `/src/hyper/async_impl/h3_client/connect/h3_establishment.rs`

**Line 78:**
```rust
// BEFORE:
let endpoint = match Self::create_quinn_endpoint() {

// AFTER:
let (socket, config) = match Self::create_quiche_config() {
```

### 9. `/src/types/quinn_chunks.rs`

**ENTIRE FILE TO BE REPLACED:**
```rust
// BEFORE: Entire file with Quinn types

// AFTER: Replace with quiche_chunks.rs
use fluent_ai_async::prelude::MessageChunk;

#[derive(Debug, Clone)]
pub struct QuicheConnectionChunk {
    pub bytes_processed: usize,
    pub error: Option<String>,
}

impl MessageChunk for QuicheConnectionChunk {
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
```

### 10. `/src/streaming/pipeline.rs`

**Line 10:**
```rust
// BEFORE:
use quinn::{RecvStream, SendStream};

// AFTER:
use quiche::Connection;
```

### 11. `/src/streaming/transport.rs`

**Line 7:**
```rust
// BEFORE:
use quinn::{Connection as QuinnConnection, Endpoint, RecvStream, SendStream};

// AFTER:
use quiche::Connection;
use std::net::UdpSocket;
```

### 12. `/src/async_impl/client/mod.rs`

**Lines 15-17:**
```rust
// BEFORE:
pub use quinn_client::{
    QuinnConnectionPool, QuinnHttp3Client, QuinnRequestBuilder, QuinnResponseHandler,
};

// AFTER:
pub use quiche_client::{
    QuicheConnectionPool, QuicheHttp3Client, QuicheRequestBuilder, QuicheResponseHandler,
};
```

### 13. `/src/async_impl/client/quinn_client.rs`

**ENTIRE FILE TO BE RENAMED AND REWRITTEN:**
```rust
// BEFORE: quinn_client.rs with Quinn imports

// AFTER: quiche_client.rs
use quiche::{Config, Connection};
use std::net::UdpSocket;
use fluent_ai_async::{AsyncStream, emit};

// All Quinn-specific code replaced with Quiche equivalents
```

### 14. `/src/async_impl/connection/mod.rs`

**Lines 5-7:**
```rust
// BEFORE:
pub use quinn_connection::{
    QuinnConnection, QuinnRecvStream, QuinnSendStream, establish_connection,
};

// AFTER:
pub use quiche_connection::{
    QuicheConnection, establish_connection,
};
```

### 15. `/src/async_impl/connection/quinn_connection.rs`

**ENTIRE FILE TO BE RENAMED AND REWRITTEN:**
```rust
// BEFORE: quinn_connection.rs with Quinn imports

// AFTER: quiche_connection.rs
use quiche::{Config, Connection, RecvInfo, SendInfo};
use std::net::UdpSocket;
use fluent_ai_async::{AsyncStream, emit};

// Complete rewrite using Quiche synchronous APIs
```

## Core Streaming Pattern Replacements

### Quinn Stream Operations → Quiche Stream Operations

**Before (Quinn Future-based):**
```rust
let (send_stream, recv_stream) = connection.open_bi().await?;
let data = recv_stream.read_to_end(usize::MAX).await?;
send_stream.write_all(&data).await?;
```

**After (Quiche Synchronous in AsyncStream):**
```rust
AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
    loop {
        // Process packets
        let (read, from) = match socket.recv_from(&mut buf) {
            Ok(result) => result,
            Err(e) => {
                emit!(sender, HttpChunk::bad_chunk(format!("Socket error: {}", e)));
                break;
            }
        };
        
        let recv_info = quiche::RecvInfo { from, to };
        match conn.recv(&mut buf[..read], recv_info) {
            Ok(_) => {
                // Read from streams
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

## File Renames Required

1. `quinn_client.rs` → `quiche_client.rs`
2. `quinn_connection.rs` → `quiche_connection.rs`  
3. `quinn_chunks.rs` → `quiche_chunks.rs`

## Cargo.toml Changes

**Remove:**
```toml
quinn = { version = "0.11.8", features = ["rustls"], optional = true }
quinn = { version = "0.11", default-features = false, features = ["rustls"], optional = true }
quinn-proto = "0.11.12"
h3-quinn = { version = "0.0.10", optional = true }
```

**Add:**
```toml
quiche = { version = "0.22", features = ["boringssl-vendored"], optional = true }
```

**Update feature flags:**
```toml
# Change from:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:h3-quinn", "dep:quinn"]

# To:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:quiche"]
```

## Success Criteria

- [ ] No `quinn` in Cargo.toml
- [ ] No references to `quinn` whatsoever in any source file
- [ ] All Quinn imports replaced with Quiche equivalents
- [ ] All Quinn API calls replaced with Quiche synchronous APIs
- [ ] All streaming patterns use AsyncStream with Quiche APIs
- [ ] All files compile without Quinn dependencies
- [ ] All tests pass with Quiche implementation