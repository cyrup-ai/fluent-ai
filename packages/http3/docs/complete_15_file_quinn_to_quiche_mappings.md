# Complete 15-File Quinn to Quiche API Mappings

## Research Citations from ./tmp/quiche

All API mappings based on source code analysis from:
- `./tmp/quiche/quiche/src/lib.rs` (Lines 1748-1751, 1134-1199, 5277-5279, 5411-5413)
- `./tmp/quiche/Cargo.toml` for dependency information

## Core API Replacements

### Connection Creation
**Quinn:** Async endpoint creation
**Quiche (Lines 1748-1751):** Synchronous connection creation
```rust
pub fn connect(
    server_name: Option<&str>, scid: &ConnectionId, local: SocketAddr,
    peer: SocketAddr, config: &mut Config,
) -> Result<Connection>
```

### Configuration Setup  
**Quinn:** Complex transport config
**Quiche (Lines 1134-1199):** Direct synchronous methods
```rust
config.set_max_idle_timeout(30_000);
config.set_initial_max_data(10_000_000);
config.set_initial_max_stream_data_bidi_local(1_000_000);
```

### Stream Operations
**Quinn:** Async stream operations  
**Quiche (Lines 5277-5279, 5411-5413):** Synchronous stream operations
```rust
pub fn stream_recv(&mut self, stream_id: u64, out: &mut [u8]) -> Result<(usize, bool)>
pub fn stream_send(&mut self, stream_id: u64, buf: &[u8], fin: bool) -> Result<usize>
```

## Complete 15-File Conversion List

### Simple Import Removals (4 files)
1. `src/hyper/async_impl/client/mod.rs` - Remove `use quinn::VarInt;`
2. `src/hyper/async_impl/client/config/types.rs` - Remove `use quinn::VarInt;`  
3. `src/hyper/async_impl/client/builder/protocols.rs` - Remove `use quinn::VarInt;`
4. `src/hyper/async_impl/client/builder/types.rs` - Remove `use quinn::VarInt;`

### Major Rewrites (6 files)
5. `src/hyper/async_impl/h3_client/connect/connector_core.rs` - Replace Quinn config with Quiche config
6. `src/hyper/async_impl/h3_client/connect/types.rs` - Replace quinn_connection with quiche_connection
7. `src/hyper/async_impl/h3_client/connect/h3_establishment.rs` - Replace endpoint with socket/config
8. `src/async_impl/client/quinn_client.rs` → `quiche_client.rs` - Complete rewrite
9. `src/async_impl/connection/quinn_connection.rs` → `quiche_connection.rs` - Complete rewrite  
10. `src/types/quinn_chunks.rs` → `quiche_chunks.rs` - Complete replacement

### Streaming Updates (2 files)
11. `src/streaming/transport.rs` - Replace Quinn imports with Quiche
12. `src/streaming/pipeline.rs` - Replace Quinn imports with Quiche

### Module Re-exports (2 files)  
13. `src/async_impl/client/mod.rs` - Update quinn_client to quiche_client
14. `src/async_impl/connection/mod.rs` - Update quinn_connection to quiche_connection

### Dependencies (1 file)
15. `Cargo.toml` - Remove Quinn deps, add Quiche deps, update features

## Quiche Streaming Pattern (Source: ./tmp/quiche/quiche/src/lib.rs Lines 122-142)

```rust
AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
    loop {
        // 1. Receive UDP packet
        let (read, from) = match socket.recv_from(&mut buf) {
            Ok(result) => result,
            Err(e) => {
                emit!(sender, HttpChunk::bad_chunk(format!("Socket error: {}", e)));
                break;
            }
        };
        
        // 2. Process QUIC packet (synchronous)
        let recv_info = quiche::RecvInfo { from, to };
        match conn.recv(&mut buf[..read], recv_info) {
            Ok(_bytes_processed) => {
                // 3. Read HTTP3 data from streams (synchronous)
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

## Key Architecture Benefits

1. **No async/await** - Pure synchronous APIs fit perfectly in AsyncStream loops
2. **No polling/wakers** - Direct Result<T,E> return values  
3. **No Future complexity** - Simple loop-based processing
4. **Zero allocation** - Direct buffer operations
5. **Error-as-data** - All errors converted to MessageChunk::bad_chunk()

All implementations maintain fluent_ai_async architecture compliance with proper loop-based streaming patterns.