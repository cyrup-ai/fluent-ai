# H2 Streaming Primitives Task

## Objective

Integrate h2 crate's lowest-level streaming primitives with AsyncStream architecture, using ONLY poll-based APIs and avoiding all Future-based interfaces.

## Required Actions

### 1. H2 Direct Poll-Based Streaming Pattern

```rust
use fluent_ai_async::{AsyncStream, emit};
use h2::{client, server, RecvStream, SendStream};
use std::task::{Context, Poll};

// CORRECT: H2 connection using direct poll-based primitives
let h2_connection_stream = AsyncStream::<H2ConnectionChunk, 1024>::with_channel(move |sender| {
    let mut connection = h2_connection;
    let mut context = polling_context;
    
    // Use h2's direct poll_ready primitive - NO Futures
    loop {
        match connection.poll_ready(&mut context) {
            Poll::Ready(Ok(())) => {
                emit!(sender, H2ConnectionChunk::ready());
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H2ConnectionChunk::bad_chunk(format!("Connection error: {}", e)));
                break;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
});

// CORRECT: H2 SendRequest using direct poll-based primitives
let h2_send_request_stream = AsyncStream::<H2RequestChunk, 1024>::with_channel(move |sender| {
    let mut send_request = h2_send_request;
    let mut context = polling_context;
    
    // Use h2's direct poll_ready primitive - NO Futures
    match send_request.poll_ready(&mut context) {
        Poll::Ready(Ok(())) => {
            match send_request.send_request(request, false) {
                Ok((response, send_stream)) => {
                    emit!(sender, H2RequestChunk::sent(response, send_stream));
                }
                Err(e) => {
                    emit!(sender, H2RequestChunk::bad_chunk(format!("Send error: {}", e)));
                }
            }
        }
        Poll::Ready(Err(e)) => {
            emit!(sender, H2RequestChunk::bad_chunk(format!("Ready error: {}", e)));
        }
        Poll::Pending => {} // AsyncStream elite polling loop handles this
    }
});

// CORRECT: H2 RecvStream using direct poll-based primitives
let h2_recv_stream = AsyncStream::<H2DataChunk, 1024>::with_channel(move |sender| {
    let mut recv_stream = h2_recv_stream;
    let mut context = polling_context;
    
    // Use h2's direct poll_data primitive - NO Futures
    loop {
        match recv_stream.poll_data(&mut context) {
            Poll::Ready(Some(Ok(data))) => {
                emit!(sender, H2DataChunk::from_bytes(data));
            }
            Poll::Ready(Some(Err(e))) => {
                emit!(sender, H2DataChunk::bad_chunk(format!("Data error: {}", e)));
                break;
            }
            Poll::Ready(None) => {
                emit!(sender, H2DataChunk::stream_complete());
                break;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
});

// CORRECT: H2 SendStream using direct poll-based primitives
let h2_send_stream = AsyncStream::<H2SendResult, 1024>::with_channel(move |sender| {
    let mut send_stream = h2_send_stream;
    let mut context = polling_context;
    
    // Use h2's direct poll_ready and send_data primitives - NO Futures
    for data_chunk in data_chunks {
        match send_stream.poll_ready(&mut context) {
            Poll::Ready(Ok(())) => {
                match send_stream.send_data(data_chunk, false) {
                    Ok(()) => emit!(sender, H2SendResult::data_sent()),
                    Err(e) => {
                        emit!(sender, H2SendResult::bad_chunk(format!("Send error: {}", e)));
                        return;
                    }
                }
            }
            Poll::Ready(Err(e)) => {
                emit!(sender, H2SendResult::bad_chunk(format!("Ready error: {}", e)));
                return;
            }
            Poll::Pending => break, // AsyncStream elite polling loop handles this
        }
    }
    
    // Send end of stream
    match send_stream.send_data(bytes::Bytes::new(), true) {
        Ok(()) => emit!(sender, H2SendResult::send_complete()),
        Err(e) => emit!(sender, H2SendResult::bad_chunk(format!("End stream error: {}", e))),
    }
});
```

### 2. H2 Chunk Types with MessageChunk Implementation

```rust
use cyrup_sugars::MessageChunk;

// H2 Connection Chunk
#[derive(Debug, Clone)]
pub enum H2ConnectionChunk {
    Ready,
    ConnectionError { message: String },
}

impl MessageChunk for H2ConnectionChunk {
    fn bad_chunk(error: String) -> Self {
        H2ConnectionChunk::ConnectionError { message: error }
    }
    
    fn error(&self) -> Option<&str> {
        match self {
            H2ConnectionChunk::ConnectionError { message } => Some(message.as_str()),
            _ => None,
        }
    }
    
    fn is_error(&self) -> bool {
        matches!(self, H2ConnectionChunk::ConnectionError { .. })
    }
}

// H2 Request Chunk
#[derive(Debug, Clone)]
pub enum H2RequestChunk {
    Sent { response: h2::RecvResponse, send_stream: h2::SendStream },
    SendError { message: String },
}

impl MessageChunk for H2RequestChunk {
    fn bad_chunk(error: String) -> Self {
        H2RequestChunk::SendError { message: error }
    }
    
    fn error(&self) -> Option<&str> {
        match self {
            H2RequestChunk::SendError { message } => Some(message.as_str()),
            _ => None,
        }
    }
    
    fn is_error(&self) -> bool {
        matches!(self, H2RequestChunk::SendError { .. })
    }
}

// H2 Data Chunk
#[derive(Debug, Clone)]
pub enum H2DataChunk {
    Data { bytes: bytes::Bytes },
    StreamComplete,
    DataError { message: String },
}

impl MessageChunk for H2DataChunk {
    fn bad_chunk(error: String) -> Self {
        H2DataChunk::DataError { message: error }
    }
    
    fn error(&self) -> Option<&str> {
        match self {
            H2DataChunk::DataError { message } => Some(message.as_str()),
            _ => None,
        }
    }
    
    fn is_error(&self) -> bool {
        matches!(self, H2DataChunk::DataError { .. })
    }
}

// H2 Send Result
#[derive(Debug, Clone)]
pub enum H2SendResult {
    DataSent,
    SendComplete,
    SendError { message: String },
}

impl MessageChunk for H2SendResult {
    fn bad_chunk(error: String) -> Self {
        H2SendResult::SendError { message: error }
    }
    
    fn error(&self) -> Option<&str> {
        match self {
            H2SendResult::SendError { message } => Some(message.as_str()),
            _ => None,
        }
    }
    
    fn is_error(&self) -> bool {
        matches!(self, H2SendResult::SendError { .. })
    }
}
```

### 2. Files to Modify

- `src/async_impl/client/h2_client.rs`
- `src/async_impl/connection/h2_connection.rs`
- `src/types/h2_chunks.rs`

### 3. Architecture Requirements

- Use h2's direct poll-based primitives: `poll_ready()`, `poll_data()`, `send_data()`
- All connection handling via `AsyncStream::with_channel` producers only
- Pure emit! patterns for all data streaming
- Direct `Context`/`Poll` handling ONLY within AsyncStream producers
- NO Future-based h2 APIs (`.ready().await`, `.send_request().await`)
- NO async/await in streaming producers
- NO external async runtimes
- NO manual waker creation or thread spawning

### 4. Validation

- Manual verification of h2 streaming integration
- End-to-end data flow testing
- Error handling pathway validation
