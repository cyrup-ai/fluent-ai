# H3 Streaming Primitives Task

## Objective

Integrate h3 crate's lowest-level streaming primitives with AsyncStream architecture, using ONLY poll-based APIs and avoiding all Future-based interfaces.

## Required Actions

### 1. H3 Direct Poll-Based Streaming Pattern

```rust
use fluent_ai_async::{AsyncStream, emit};
use h3::client::{Connection, SendRequest};
use h3::quic::{RecvStream, SendStream, BufRecvStream};
use std::task::{Context, Poll};

// CORRECT: H3 connection accept using direct poll-based primitives
let h3_accept_stream = AsyncStream::<H3ConnectionChunk, 1024>::with_channel(move |sender| {
    let mut connection = h3_connection;
    let mut context = polling_context;
    
    // Use h3's direct poll_accept_recv primitive - NO Futures
    loop {
        match connection.poll_accept_recv(&mut context) {
            Poll::Ready(Ok(recv_stream)) => {
                emit!(sender, H3ConnectionChunk::new_recv_stream(recv_stream.stream_id()));
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H3ConnectionChunk::bad_chunk(format!("Accept error: {}", e)));
                break;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
});

// CORRECT: H3 bidirectional stream accept using direct poll-based primitives
let h3_accept_bidi_stream = AsyncStream::<H3BiStreamChunk, 1024>::with_channel(move |sender| {
    let mut connection = h3_connection;
    let mut context = polling_context;
    
    // Use h3's direct poll_accept_bidi primitive - NO Futures
    loop {
        match connection.poll_accept_bidi(&mut context) {
            Poll::Ready(Ok((send_stream, recv_stream))) => {
                emit!(sender, H3BiStreamChunk::new_bidi_stream(send_stream.stream_id(), recv_stream.stream_id()));
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H3BiStreamChunk::bad_chunk(format!("Accept bidi error: {}", e)));
                break;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
});

// CORRECT: H3 BufRecvStream using direct poll-based primitives
let h3_recv_stream = AsyncStream::<H3DataChunk, 1024>::with_channel(move |sender| {
    let mut buf_recv_stream = h3_buf_recv_stream;
    let mut context = polling_context;
    
    // Use h3's direct poll_data primitive - NO Futures
    loop {
        match buf_recv_stream.poll_data(&mut context) {
            Poll::Ready(Ok(Some(data))) => {
                emit!(sender, H3DataChunk::from_bytes(data));
            }
            Poll::Ready(Ok(None)) => {
                emit!(sender, H3DataChunk::stream_complete());
                break;
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H3DataChunk::bad_chunk(format!("Data error: {}", e)));
                break;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
});

// CORRECT: H3 SendStream using direct poll-based primitives
let h3_send_stream = AsyncStream::<H3SendResult, 1024>::with_channel(move |sender| {
    let mut send_stream = h3_send_stream;
    let mut context = polling_context;
    
    // Use h3's direct poll_ready and send_data primitives - NO Futures
    for data_chunk in data_chunks {
        match send_stream.poll_ready(&mut context) {
            Poll::Ready(Ok(())) => {
                match send_stream.send_data(data_chunk) {
                    Ok(()) => emit!(sender, H3SendResult::data_sent()),
                    Err(e) => {
                        emit!(sender, H3SendResult::bad_chunk(format!("Send error: {}", e)));
                        return;
                    }
                }
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H3SendResult::bad_chunk(format!("Ready error: {}", e)));
                return;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
    
    // Finish stream using direct poll_finish primitive
    match send_stream.poll_finish(&mut context) {
        Poll::Ready(Ok(())) => emit!(sender, H3SendResult::send_complete()),
        Poll::Ready(Err(e)) => emit!(sender, H3SendResult::bad_chunk(format!("Finish error: {}", e))),
        Poll::Pending => {} // AsyncStream elite polling loop handles this
    }
});

// CORRECT: H3 SendRequest using direct poll-based primitives
let h3_send_request_stream = AsyncStream::<H3RequestChunk, 1024>::with_channel(move |sender| {
    let mut send_request = h3_send_request;
    let mut context = polling_context;
    
    // Use h3's direct poll_ready primitive - NO Futures
    match send_request.poll_ready(&mut context) {
        Poll::Ready(Ok(())) => {
            match send_request.send_request(request) {
                Ok(send_stream) => {
                    emit!(sender, H3RequestChunk::request_sent(send_stream.stream_id()));
                }
                Err(e) => {
                    emit!(sender, H3RequestChunk::bad_chunk(format!("Send request error: {}", e)));
                }
            }
        }
        Poll::Ready(Err(e)) => {
            emit!(sender, H3RequestChunk::bad_chunk(format!("Ready error: {}", e)));
        }
        Poll::Pending => {} // AsyncStream elite polling loop handles this
    }
});
```

### 2. Files to Modify

- `src/async_impl/client/h3_client.rs`
- `src/async_impl/connection/h3_connection.rs`
- `src/types/h3_chunks.rs`

### 3. Architecture Requirements

- ✅ Use h3's direct poll-based primitives: `poll_accept_recv()`, `poll_accept_bidi()`, `poll_data()`, `poll_ready()`, `poll_finish()`
- ✅ All connection handling via `AsyncStream::with_channel` producers only
- ✅ Pure emit! patterns for all data streaming
- ✅ Direct `Context`/`Poll` handling ONLY within AsyncStream producers
- ❌ NO Future-based h3 APIs (`.accept().await`, `.send_request().await`)
- ❌ NO async/await in streaming producers
- ❌ NO external async runtimes
- ❌ NO manual waker creation or thread spawning

### 4. Validation

- Manual verification of h3 streaming integration
- End-to-end data flow testing
- Error handling pathway validation
