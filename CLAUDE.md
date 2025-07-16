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
    
    pub async fn make_request(&self, url: &str, body: Vec<u8>) -> Result<Response, HttpError> {
        let request = HttpRequest::post(url, body)?
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key));
            
        self.client.send(request).await
    }
}
```

### Streaming Implementation Pattern

```rust
use fluent_ai_http3::{HttpClient, HttpConfig};

pub async fn stream_completion(&self, request: CompletionRequest) -> Result<StreamingResponse, HttpError> {
    let client = HttpClient::with_config(HttpConfig::streaming_optimized())?;
    
    let http_request = HttpRequest::post(&self.url, request_body)?
        .header("Content-Type", "application/json")
        .header("Authorization", &format!("Bearer {}", self.api_key));
    
    let response = client.send(http_request).await?;
    
    // Use SSE for most AI providers
    let mut sse_stream = response.sse();
    
    while let Some(event) = sse_stream.next().await {
        match event {
            Ok(sse_event) => {
                // Process SSE event data
                if let Some(data) = sse_event.data {
                    // Parse and emit completion chunk
                }
            }
            Err(e) => {
                // Handle SSE parsing error
            }
        }
    }
    
    Ok(StreamingResponse::new(receiver))
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