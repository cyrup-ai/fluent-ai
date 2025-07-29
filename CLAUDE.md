# Domain Model Architecture

## CRITICAL RULE: PACKAGE ISOLATION AND STANDALONE REQUIREMENTS

### fluent-ai-candle: STANDALONE PACKAGE

**MANDATORY**: `fluent-ai-candle` is a STANDALONE package that:

- **CANNOT depend on** `./packages/domain` (fluent_ai_domain)
- **CANNOT depend on** `./packages/fluent-ai` 
- **MUST define its own domain types** with "Candle" prefixes
- **MUST use ONLY** `./packag../async-stream` for streaming
- **MUST use ONLY** `./packages/http3` for HTTP operations
- **NO FUTURES architecture** - AsyncStream only
- **Zero allocation, lock-free patterns** throughout

### Required Dependencies ONLY

```toml
[dependencies]
fluent_ai_async = { path = "../async-stream" }
fluent-ai-http3 = { path = "../http3" }
# NO OTHER fluent-ai-* dependencies allowed
```

### Standalone Domain Types (Candle-Prefixed)

`fluent-ai-candle` MUST define its own versions of ALL domain types:

```rust
// fluent-ai-candle defines these (NOT imported from domain)
pub struct CandleAgent { /* ... */ }
pub trait CandleAgentRole { /* ... */ }
pub struct CandleMessage { /* ... */ }
pub enum CandleMessageRole { /* ... */ }
pub trait CandleCompletionProvider { /* ... */ }
// ... ALL domain types with "Candle" prefix
```

### Forbidden Imports

**NEVER IMPORT**:
```rust
// FORBIDDEN - causes dependency on other packages
use fluent_ai_domain::*;
use fluent_ai::*;
use fluent_ai_provider::*;
use fluent_ai_memory::*;
```

### Required Import Pattern

**ONLY ALLOWED**:
```rust
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use fluent_ai_http3::{Http3, HttpClient, HttpConfig};
// Local candle types only
use crate::domain::{CandleMessage, CandleMessageRole, /* ... */};
```

### Legacy Package Rules (for reference)

For NON-candle packages, the original domain centralization rules apply:

- **fluent_ai_domain**: Domain models, types, traits (ONLY) - NO dependencies
- **fluent_ai_provider**: Completion clients using domain types
- **fluent_ai_memory**: Memory implementations using domain + provider types
- **fluent_ai**: Top-level orchestration package

But `fluent-ai-candle` is COMPLETELY ISOLATED from this hierarchy.

# HTTP Client Configuration

## MANDATORY HTTP3 LIBRARY USAGE

**CRITICAL**: ALL HTTP requests in this project MUST use `fluent_ai_http3` exclusively.

### Required HTTP Client

```rust
use fluent_ai_http3::{HttpClient, HttpConfig};

// For AI provider clients
let client = HttpClient::with_config(HttpConfig::ai_optimized())
    .expect("Failed to create HTTP3 client");

// For streaming operations
let client = HttpClient::with_config(HttpConfig::streaming_optimized())
    .expect("Failed to create HTTP3 client");
```

### HTTP3 Library Features

- **HTTP/3 (QUIC) with HTTP/2 fallback**
- **Zero-allocation streaming-first architecture**
- **Built-in Server-Sent Events (SSE) support**
- **JSON Lines streaming for providers like Ollama**
- **Exponential backoff retry with jitter**
- **Connection pooling and intelligent reuse**
- **Lock-free caching with crossbeam-skiplist**

### Streaming-First Approach

ALL HTTP operations should use streaming-first patterns:

```rust
// Stream response and collect if needed
let response = client.send(request).await?;
let mut stream = response.stream();

// For SSE streams
let mut sse_stream = response.sse();
while let Some(event) = sse_stream.next().await {
    // Process SSE event
}

// For JSON Lines streams
let mut json_stream = response.json_lines::<serde_json::Value>();
while let Some(chunk) = json_stream.next().await {
    // Process JSON chunk
}

// Only collect if you need the full response
let full_response = stream.collect().await?;
```

### Forbidden HTTP Libraries

**DO NOT USE**:
- `reqwest` - replaced with fluent_ai_http3
- `hyper` - replaced with fluent_ai_http3
- `tokio-tungstenite` - use HTTP3 SSE instead
- `reqwest_eventsource` - use HTTP3 SSE instead
- Any other HTTP client library

### Error Handling

```rust
use fluent_ai_http3::HttpError;

match client.send(request).await {
    Ok(response) => {
        if response.status().is_success() {
            // Process success response
        } else {
            // Handle HTTP error status
        }
    }
    Err(HttpError::ConnectionFailed(msg)) => {
        // Handle connection failure
    }
    Err(HttpError::Timeout) => {
        // Handle timeout
    }
    Err(e) => {
        // Handle other HTTP3 errors
    }
}
```

### Provider Implementation Pattern

```rust
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};

pub struct ProviderClient {
    client: HttpClient,
    // ... other fields
}

impl ProviderClient {
    pub fn new() -> Self {
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .expect("Failed to create HTTP3 client");
        
        Self { client }
    }
    
    pub fn make_request_stream(&self, url: &str, body: Vec<u8>) -> AsyncStream<Response> {
        let (sender, receiver) = channel::<Response>();
        
        let request = HttpRequest::post(url, body)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key));
            
        let response_stream = self.client.send_stream(request);
        response_stream.on_chunk(|chunk| {
            match chunk {
                Ok(response) => emit!(sender, response),
                Err(err) => handle_error!(err, "HTTP request failed"),
            }
        });
        
        receiver
    }
}
```

### JSON POST with Serde Request/Response Marshaling

**RECOMMENDED PATTERN**: Use the elegant Http3 builder with Serde types for type-safe JSON communication:

```rust
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::{Deserialize, Serialize};

// Define your request/response types
#[derive(Serialize, Debug)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    temperature: f64,
}

#[derive(Deserialize, Debug)]
struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,
    usage: Usage,
}

// Pattern 1: Collect to typed response (most common)
async fn send_completion_request(request: &CompletionRequest) -> Result<CompletionResponse, HttpError> {
    let response = Http3::json()
        .debug()                              // Enable debug logging
        .api_key(&api_key)                   // Authorization: Bearer
        .body(request)                       // Serialize request to JSON
        .post("https://api.openai.com/v1/chat/completions")
        .collect::<CompletionResponse>();    // Deserialize response from JSON
    
    Ok(response)
}

// Pattern 2: Stream processing for real-time responses
async fn stream_completion_request(request: &CompletionRequest) -> AsyncStream<CompletionChunk> {
    let mut stream = Http3::json()
        .debug()
        .api_key(&api_key)
        .body(request)
        .post("https://api.openai.com/v1/chat/completions");
    
    // Process chunks as they arrive
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(HttpChunk::Body(bytes)) => {
                // Parse SSE or JSON Lines format
                if let Ok(chunk_data) = parse_completion_chunk(&bytes) {
                    yield chunk_data;
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }
}

// Pattern 3: Error handling with fallback
async fn robust_completion_request(request: &CompletionRequest) -> CompletionResponse {
    Http3::json()
        .api_key(&api_key)
        .body(request)
        .post("https://api.openai.com/v1/chat/completions")
        .collect_or_else(|error| {
            eprintln!("Request failed: {}", error);
            CompletionResponse::default() // Fallback response
        })
}

// Pattern 4: Custom headers with Serde marshaling
async fn completion_with_custom_headers(request: &CompletionRequest) -> CompletionResponse {
    Http3::json()
        .headers(|| {
            use std::collections::HashMap;
            let mut map = HashMap::new();
            map.insert("X-Request-ID", "req-123");
            map.insert("X-Client-Version", "1.0.0");
            map
        })
        .bearer_token(&api_key)              // Alternative to .api_key()
        .body(request)
        .post("https://api.openai.com/v1/chat/completions")
        .collect::<CompletionResponse>()
}
```

**KEY ADVANTAGES**:
- **Type Safety**: Compile-time verification of request/response types
- **Automatic Serialization**: Serde handles JSON marshaling transparently
- **Streaming Support**: Real-time processing for AI streaming responses
- **Error Handling**: Built-in error handling with collect_or_else patterns
- **Debug Support**: Easy debugging with .debug() method
- **Zero Allocation**: Efficient memory usage with streaming-first design

### Streaming Implementation Pattern

```rust
use fluent_ai_http3::{HttpClient, HttpConfig};

pub fn stream_completion(&self, request: CompletionRequest) -> AsyncStream<CompletionChunk> {
    let (sender, receiver) = channel::<CompletionChunk>();
    
    let client = HttpClient::with_config(HttpConfig::streaming_optimized());
    
    let http_request = HttpRequest::post(&self.url, request_body)
        .header("Content-Type", "application/json")
        .header("Authorization", &format!("Bearer {}", self.api_key));
    
    let response_stream = client.send_stream(http_request);
    
    // Use SSE for most AI providers - NO FUTURES!
    response_stream.sse().on_chunk(|sse_event| {
        match parse_completion_chunk(sse_event.data) {
            Ok(chunk) => emit!(sender, chunk),
            Err(err) => handle_error!(err, "SSE parsing failed"),
        }
    });
    
    receiver
}
```

### Code Quality Standards

- **Zero allocation**: Use streaming patterns, avoid unnecessary cloning
- **No unsafe code**: All operations must be memory-safe
- **No locking**: Use lock-free data structures and async patterns
- **Never use `unwrap()` or `expect()`** in production code
- **Elegant ergonomics**: API should be intuitive and composable

### Migration Checklist

When updating any HTTP usage:

1. ✅ Replace `reqwest::Client` with `fluent_ai_http3::HttpClient`
2. ✅ Replace `reqwest::Request` with `fluent_ai_http3::HttpRequest`
3. ✅ Replace `reqwest_eventsource::EventSource` with `response.sse()`
4. ✅ Use `HttpConfig::ai_optimized()` or `HttpConfig::streaming_optimized()`
5. ✅ Update error handling to use `fluent_ai_http3::HttpError`
6. ✅ Implement streaming-first patterns with `.collect()` fallback
7. ✅ Test compilation and functionality

### Performance Optimization

- Use connection pooling for repeated requests
- Implement exponential backoff for retries
- Stream responses instead of buffering when possible
- Use zero-copy patterns for large payloads
- Leverage HTTP/3's multiplexing capabilities

### Example Provider Integration

```rust
// Complete example for a new AI provider
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest, HttpError};

pub struct NewProviderClient {
    client: HttpClient,
    api_key: String,
    base_url: String,
}

impl NewProviderClient {
    pub fn new(api_key: String) -> Result<Self, HttpError> {
        let client = HttpClient::with_config(HttpConfig::ai_optimized())?;
        
        Ok(Self {
            client,
            api_key,
            base_url: "https://api.newprovider.com/v1".to_string(),
        })
    }
    
    pub async fn stream_completion(&self, prompt: &str) -> Result<StreamingResponse, HttpError> {
        let request_body = serde_json::json!({
            "prompt": prompt,
            "stream": true
        });
        
        let request = HttpRequest::post(&format!("{}/completions", self.base_url), 
                                     serde_json::to_vec(&request_body)?)?
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key));
        
        let response = self.client.send(request).await?;
        let mut sse_stream = response.sse();
        
        // Process streaming response...
        Ok(StreamingResponse::new(receiver))
    }
}
```

This HTTP3 library provides the foundation for all network operations in fluent-ai with optimal performance, reliability, and modern HTTP/3 capabilities.

# ASYNCSTREAM ARCHITECTURE - CORRECT USAGE

## CRITICAL PRINCIPLE: AsyncStream with .collect() PATTERN

fluent-ai uses `AsyncStream<T>` for all async operations with proper error handling:

### CORE ASYNCSTREAM PATTERNS

**REQUIRED USAGE**:
- `AsyncStream<T>` - All async operations return streams
- `AsyncStream::with_channel()` - Primary constructor for creating streams
- `.collect().await?` - Convert stream to single result when needed
- Proper `Send + 'static` bounds for thread safety
- Standard Rust error handling with `Result<T, E>`

### CORRECT ASYNCSTREAM CONSTRUCTION

```rust
use fluent_ai_async::{AsyncStream, AsyncStreamSender};

// CORRECT - AsyncStream with proper channel setup
pub fn stream_completions(prompt: String) -> AsyncStream<CompletionChunk> {
    AsyncStream::with_channel(move |sender: AsyncStreamSender<CompletionChunk>| {
        Box::pin(async move {
            // Async work inside the closure
            let response = http_client.post(&url).await?;
            let mut stream = response.stream();
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(data) => {
                        if let Ok(completion) = parse_completion(&data) {
                            let _ = sender.send(completion).await;
                        }
                    }
                    Err(e) => {
                        eprintln!("Stream error: {}", e);
                        break;
                    }
                }
            }
            
            Ok(())
        })
    })
}
```

### USAGE PATTERNS

```rust
// Pattern 1: Streaming consumption (real-time processing)
let stream = stream_completions("Hello world".to_string());
pin_mut!(stream);
while let Some(chunk) = stream.next().await {
    println!("Received: {}", chunk.text);
}

// Pattern 2: Collect to single result when needed
let stream = stream_completions("Hello world".to_string());
let all_chunks: Vec<CompletionChunk> = stream.collect().await?;

// Pattern 3: Process first result only
let stream = stream_completions("Hello world".to_string());
if let Some(first_chunk) = stream.next().await {
    println!("First chunk: {}", first_chunk.text);
}
```

### ERROR HANDLING PATTERN

```rust
use fluent_ai_async::{AsyncStream, AsyncStreamSender, StreamError};

pub fn robust_stream_operation(input: String) -> AsyncStream<ProcessedResult> {
    AsyncStream::with_channel(move |sender: AsyncStreamSender<ProcessedResult>| {
        Box::pin(async move {
            // All error handling is standard Rust Result<T, E>
            let processed = match process_input(&input).await {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Processing failed: {}", e);
                    return Err(StreamError::ProcessingFailed(e.to_string()));
                }
            };
            
            // Send successful results
            let _ = sender.send(processed).await;
            Ok(())
        })
    })
}
```

### THREAD SAFETY REQUIREMENTS

```rust
// REQUIRED: All AsyncStream closures must be Send + 'static
pub fn thread_safe_stream(data: Vec<String>) -> AsyncStream<String> {
    AsyncStream::with_channel(move |sender: AsyncStreamSender<String>| {
        Box::pin(async move {
            for item in data {  // data is moved into closure
                let _ = sender.send(item).await;
            }
            Ok(())
        })
    })
}

// REQUIRED: Sender must have proper bounds
impl<T> AsyncStreamSender<T> 
where
    T: Send + 'static
{
    // Implementation ensures thread safety
}
```

### PERFORMANCE BENEFITS

- **Zero allocation**: Efficient streaming with minimal overhead
- **Thread safety**: Proper Send bounds ensure safe concurrent usage  
- **Flexible consumption**: Stream or collect as needed
- **Standard Rust patterns**: Uses familiar Result<T, E> error handling
- **Real-time processing**: Immediate value emission as available

### MIGRATION FROM LEGACY PATTERNS

```rust
// OLD PATTERN (deprecated)
async fn old_pattern() -> Result<Vec<String>, Error> {
    // async/await everywhere
}

// NEW PATTERN (correct)
fn new_pattern() -> AsyncStream<String> {
    AsyncStream::with_channel(|sender| {
        Box::pin(async move {
            // async work in closure
            // send individual results as available
            Ok(())
        })
    })
}
```

This AsyncStream architecture provides the foundation for all async operations in fluent-ai with optimal performance and ergonomics.

# JSON POST WITH SERDE MARSHALING - USAGE PATTERNS

## CORE PATTERN: Fluent Builder with Type-Safe Serde Integration

The fluent_ai_http3 library provides elegant JSON POST operations with automatic Serde serialization/deserialization:

### BASIC JSON POST PATTERN

```rust
use fluent_ai_http3::Http3;
use serde::{Deserialize, Serialize};

// Define request/response types
#[derive(Serialize, Debug)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    choices: Vec<Choice>,
}

// CORRECT: Streaming-first approach
async fn send_completion_request(request: CompletionRequest) -> Result<CompletionResponse, HttpError> {
    let response = Http3::json()                          // Sets Content-Type: application/json
        .api_key("sk-...")                               // Authorization: Bearer sk-...
        .body(&request)                                  // Automatic serde_json serialization
        .post("https://api.openai.com/v1/chat/completions")
        .collect::<CompletionResponse>()                 // Automatic serde_json deserialization
        .await?;
    
    Ok(response)
}
```

### ADVANCED PATTERN: Headers + Error Handling

```rust
use fluent_ai_http3::{Http3, HttpError, header};
use std::collections::HashMap;

async fn advanced_json_post<Req, Resp>(
    url: &str,
    request: &Req,
    api_key: &str,
) -> Result<Resp, HttpError>
where
    Req: Serialize,
    Resp: for<'de> Deserialize<'de>,
{
    let response = Http3::json()
        .debug()                                         // Enable request/response logging
        .headers(|| {
            let mut map = HashMap::new();
            map.insert(header::X_API_KEY, api_key);
            map.insert(header::USER_AGENT, "fluent-ai/1.0");
            map.insert(header::ACCEPT, "application/json");
            map
        })
        .body(request)                                   // Generic serde serialization
        .post(url)
        .collect::<Resp>()                              // Generic serde deserialization
        .await?;
    
    Ok(response)
}
```

### STREAMING PATTERN: Real-time JSON Processing

```rust
use fluent_ai_http3::{Http3, HttpChunk};
use futures_util::StreamExt;
use serde_json::Value;

async fn stream_json_chunks(request: &CompletionRequest) -> Result<(), HttpError> {
    let mut stream = Http3::json()
        .body(request)
        .post("https://api.openai.com/v1/chat/completions")
        .stream();                                       // Get raw HttpChunk stream
    
    while let Some(chunk_result) = stream.next().await {
        match chunk_result? {
            HttpChunk::Head(status, headers) => {
                println!("Response status: {}", status);
                println!("Content-Type: {:?}", headers.get("content-type"));
            }
            HttpChunk::Body(bytes) => {
                // Parse JSON chunks for streaming responses
                if let Ok(json_str) = std::str::from_utf8(&bytes) {
                    for line in json_str.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if let Ok(chunk_data) = serde_json::from_str::<Value>(data) {
                                println!("Received chunk: {}", chunk_data);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}
```

### ERROR HANDLING PATTERN: Collect with Fallback

```rust
use fluent_ai_http3::{Http3, HttpError};

async fn robust_json_post(request: &CompletionRequest) -> CompletionResponse {
    Http3::json()
        .body(request)
        .post("https://api.openai.com/v1/chat/completions")
        .collect_or_else(|error: HttpError| {
            eprintln!("API call failed: {}", error);
            
            // Return fallback response on error
            CompletionResponse {
                id: "error".to_string(),
                object: "error".to_string(),
                created: 0,
                choices: vec![],
            }
        })
        .await
}
```

### PROVIDER INTEGRATION PATTERN: Reusable Client

```rust
use fluent_ai_http3::{HttpClient, HttpConfig, Http3};

pub struct AIProviderClient {
    client: HttpClient,
    api_key: String,
    base_url: String,
}

impl AIProviderClient {
    pub fn new(api_key: String) -> Result<Self, HttpError> {
        let config = HttpConfig::ai_optimized()
            .with_timeout(Duration::from_secs(120))
            .with_max_retries(3)
            .with_http3_enabled(true);
            
        let client = HttpClient::with_config(config)?;
        
        Ok(Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }
    
    pub async fn completion<Req, Resp>(&self, request: &Req) -> Result<Resp, HttpError>
    where
        Req: Serialize,
        Resp: for<'de> Deserialize<'de>,
    {
        Http3::new(&self.client)                         // Use existing client instance
            .json()
            .bearer_auth(&self.api_key)
            .body(request)
            .post(&format!("{}/chat/completions", self.base_url))
            .collect::<Resp>()
            .await
    }
}
```

### FORM DATA PATTERN: Alternative Content Types

```rust
use std::collections::HashMap;

// For application/x-www-form-urlencoded content
async fn send_form_data() -> Result<FormResponse, HttpError> {
    let form_data = HashMap::from([
        ("grant_type".to_string(), "client_credentials".to_string()),
        ("client_id".to_string(), "your_client_id".to_string()),
        ("client_secret".to_string(), "your_secret".to_string()),
    ]);
    
    let response = Http3::form_urlencoded()              // Sets Content-Type: application/x-www-form-urlencoded
        .body(&form_data)                               // HashMap -> URL encoded serialization
        .post("https://oauth.provider.com/token")
        .collect::<FormResponse>()                      // Serde deserialization
        .await?;
    
    Ok(response)
}
```

### KEY BENEFITS OF THIS PATTERN

1. **Type Safety**: Request/response types checked at compile time
2. **Zero Allocation**: Streaming-first design with .collect() when needed  
3. **Automatic Marshaling**: Serde handles JSON serialization/deserialization
4. **Flexible Headers**: Type-safe header management with closures
5. **Error Recovery**: Built-in error handling with fallback options
6. **HTTP/3 Performance**: QUIC protocol with HTTP/2 fallback
7. **Connection Reuse**: Efficient client pooling for multiple requests

This pattern integrates seamlessly with the fluent-ai ecosystem while providing the performance and ergonomics required for production AI applications.