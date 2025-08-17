# Complete Quinn to Quiche Conversion Plan (src/ + Cargo.toml)

## Cargo.toml Changes Required

### Dependencies to Remove
```toml
# Line 51 - REMOVE
quinn = { version = "0.11.8", features = ["rustls"], optional = true }

# Line 62 - REMOVE  
quinn-proto = "0.11.12"

# Line 85 - REMOVE
h3-quinn = { version = "0.0.10", optional = true }

# Line 86 - REMOVE
quinn = { version = "0.11", default-features = false, features = ["rustls"], optional = true }
```

### Dependencies to Add
```toml
# Add after line 84
quiche = { version = "0.22", features = ["boringssl-vendored"], optional = true }
```

### Feature Flags to Update
```toml
# Line 164 - CHANGE FROM:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:h3-quinn", "dep:quinn"]

# TO:
http3 = ["rustls-tls-manual-roots", "dep:h3", "dep:quiche"]

# Line 198 - CHANGE FROM:
__rustls-ring = ["hyper-rustls?/ring", "rustls?/ring", "quinn?/ring"]

# TO:
__rustls-ring = ["hyper-rustls?/ring", "rustls?/ring"]
```

## src/ Files Requiring Changes

### Simple Import Removals (4 files)

#### 1. src/hyper/async_impl/client/mod.rs
```rust
# Line 39 - REMOVE:
use quinn::VarInt;
```

#### 2. src/hyper/async_impl/client/config/types.rs  
```rust
# Line 13 - REMOVE:
use quinn::VarInt;
```

#### 3. src/hyper/async_impl/client/builder/protocols.rs
```rust
# Line 9 - REMOVE:
use quinn::VarInt;
```

#### 4. src/hyper/async_impl/client/builder/types.rs
```rust
# Line 16 - REMOVE:
use quinn::VarInt;
```

### Major File Rewrites Required (6 files)

#### 1. src/hyper/async_impl/h3_client/connect/connector_core.rs

**Lines 10-11 - Replace imports:**
```rust
# BEFORE:
use quinn::crypto::rustls::QuicClientConfig;
use quinn::{ClientConfig, Endpoint, TransportConfig};

# AFTER:
use quiche::{Config, Connection};
use std::net::UdpSocket;
```

**Lines 55-70 - Replace Quinn config with Quiche config:**
```rust
# BEFORE:
let mut quinn_config = ClientConfig::new(quic_client_config);
let mut transport_config = TransportConfig::default();
// ... transport config setup
transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
quinn_config.transport_config(Arc::new(transport_config));

# AFTER:
let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)?;
config.set_application_protos(&[b"h3"])?;
config.set_initial_max_data(10_000_000);
config.set_initial_max_stream_data_bidi_local(1_000_000);
config.set_initial_max_stream_data_bidi_remote(1_000_000);
config.set_initial_max_streams_bidi(100);
config.set_initial_max_streams_uni(100);
config.set_max_idle_timeout(30_000);
```

**Lines 73-89 - Replace endpoint creation:**
```rust
# BEFORE:
let socket_addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0));
let endpoint = match Endpoint::client(socket_addr) {
    Ok(ep) => ep,
    Err(e) => return None,
};
endpoint.set_default_client_config(quinn_config);

# AFTER:
let socket = UdpSocket::bind("0.0.0.0:0")?;
socket.set_nonblocking(true)?;
// Connection created when needed with quiche::connect()
```

#### 2. src/hyper/async_impl/h3_client/connect/types.rs

**Lines 12, 19, 36 - Replace quinn_connection field:**
```rust
# BEFORE:
pub(crate) quinn_connection: Option<quinn::Connection>,

# AFTER:
pub(crate) quiche_connection: Option<quiche::Connection>,
pub(crate) socket: Option<UdpSocket>,
```

**Line 44 - Change method signature:**
```rust
# BEFORE:
pub fn open_bi(&self) -> Result<(quinn::SendStream, quinn::RecvStream), String> {

# AFTER:
pub fn open_bi(&self) -> Result<u64, String> { // Returns stream_id
```

**Line 56 - Change streaming return type:**
```rust
# BEFORE:
pub fn open_bi_streaming(&self) -> AsyncStream<(quinn::SendStream, quinn::RecvStream)> {

# AFTER:
pub fn open_bi_streaming(&self) -> AsyncStream<u64> { // Returns stream_id
```

**Lines 65-67, 80-82 - Replace Quinn stream error handling:**
```rust
# BEFORE:
quinn::SendStream::bad_chunk("No connection available".to_string()),
quinn::RecvStream::bad_chunk("No connection available".to_string())

# AFTER:
emit!(sender, HttpChunk::bad_chunk("No connection available".to_string()));
return;
```

#### 3. src/hyper/async_impl/h3_client/connect/h3_establishment.rs

**Line 78 - Replace Quinn endpoint creation:**
```rust
# BEFORE:
let endpoint = match Self::create_quinn_endpoint() {

# AFTER:
let (socket, config) = match Self::create_quiche_config() {
```

**Line 128 - Change function parameter:**
```rust
# BEFORE:
endpoint: quinn::Endpoint,

# AFTER:
socket: UdpSocket,
config: quiche::Config,
```

**Lines 152, 184-228 - Replace all Quinn API calls with Quiche equivalents**

#### 4. src/async_impl/client/quinn_client.rs → quiche_client.rs

**ENTIRE FILE REWRITE:**
```rust
# BEFORE: All Quinn imports and APIs

# AFTER:
use quiche::{Config, Connection};
use std::net::UdpSocket;
use fluent_ai_async::{AsyncStream, emit};

// Complete rewrite using Quiche synchronous APIs in AsyncStream loops
```

#### 5. src/async_impl/connection/quinn_connection.rs → quiche_connection.rs

**ENTIRE FILE REWRITE:**
```rust
# BEFORE: All Quinn imports and APIs

# AFTER:
use quiche::{Config, Connection, RecvInfo, SendInfo};
use std::net::UdpSocket;
use fluent_ai_async::{AsyncStream, emit};

// Complete rewrite using Quiche synchronous APIs
```

#### 6. src/types/quinn_chunks.rs → quiche_chunks.rs

**ENTIRE FILE REPLACEMENT:**
```rust
# BEFORE: Quinn-specific chunk types

# AFTER:
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

### Streaming Module Updates (2 files)

#### 1. src/streaming/transport.rs
```rust
# Line 7 - CHANGE FROM:
use quinn::{Connection as QuinnConnection, Endpoint, RecvStream, SendStream};

# TO:
use quiche::Connection;
use std::net::UdpSocket;
```

#### 2. src/streaming/pipeline.rs
```rust
# Line 10 - CHANGE FROM:
use quinn::{RecvStream, SendStream};

# TO:
use quiche::Connection;
```

### Module Re-exports Updates

#### src/async_impl/client/mod.rs
```rust
# Lines 15-17 - CHANGE FROM:
pub use quinn_client::{
    QuinnConnectionPool, QuinnHttp3Client, QuinnRequestBuilder, QuinnResponseHandler,
};

# TO:
pub use quiche_client::{
    QuicheConnectionPool, QuicheHttp3Client, QuicheRequestBuilder, QuicheResponseHandler,
};
```

#### src/async_impl/connection/mod.rs
```rust
# Lines 5-7 - CHANGE FROM:
pub use quinn_connection::{
    QuinnConnection, QuinnRecvStream, QuinnSendStream, establish_connection,
};

# TO:
pub use quiche_connection::{
    QuicheConnection, establish_connection,
};
```

## Core Streaming Pattern Conversion

### Quinn Future-based → Quiche Synchronous in AsyncStream

**BEFORE (Quinn async/await):**
```rust
let (send_stream, recv_stream) = connection.open_bi().await?;
let data = recv_stream.read_to_end(usize::MAX).await?;
send_stream.write_all(&data).await?;
```

**AFTER (Quiche synchronous in AsyncStream loop):**
```rust
AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
    loop {
        // Process QUIC packets
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

1. `src/async_impl/client/quinn_client.rs` → `src/async_impl/client/quiche_client.rs`
2. `src/async_impl/connection/quinn_connection.rs` → `src/async_impl/connection/quiche_connection.rs`
3. `src/types/quinn_chunks.rs` → `src/types/quiche_chunks.rs`

## Success Criteria

- [ ] Zero `quinn` references in Cargo.toml
- [ ] Zero `quinn` imports in any src/ file  
- [ ] All Quinn API calls replaced with Quiche synchronous APIs
- [ ] All streaming patterns use AsyncStream with proper loops
- [ ] All files compile without Quinn dependencies
- [ ] Maintains fluent_ai_async architecture compliance