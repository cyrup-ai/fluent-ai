# Domain Model Architecture

## CRITICAL RULE: ALL DOMAIN MODELS MUST LIVE IN fluent_ai_domain

**MANDATORY**: ALL domain models, types, traits, and abstractions MUST be defined in the `fluent_ai_domain` package. NO other packages should define domain models.

### Domain Model Centralization

- **CompletionProvider trait**: Lives in `fluent_ai_domain::completion`
- **ModelConfig struct**: Lives in `fluent_ai_domain::model` 
- **All completion types**: Live in `fluent_ai_domain::completion`
- **All model types**: Live in `fluent_ai_domain::model`
- **Message types**: Live in `fluent_ai_domain::chat`
- **Error types**: Live in `fluent_ai_domain::error`

### Import Pattern for REAL LLM Calls

```rust
// CORRECT: Import domain types and provider clients
use fluent_ai_domain::{
    completion::CompletionProvider,  // Trait definition
    model::ModelConfig,
    chat::Message,
    error::CompletionError,
};

// CORRECT: Import actual completion clients for REAL calls
use fluent_ai_provider::{
    openai::OpenAIClient,           // Real OpenAI client
    anthropic::AnthropicClient,     // Real Anthropic client
};

// WRONG: Defining domain models in other packages
use fluent_ai_memory::ModelConfig; // WRONG PACKAGE
```

### Package Dependency Chain

**CRITICAL DEPENDENCY ORDER**: `fluent_ai` -> `fluent_ai_memory` -> `fluent_ai_provider` -> `fluent_ai_domain`

- **fluent_ai_domain**: Domain models, types, traits (ONLY) - NO dependencies
- **fluent_ai_provider**: Completion clients (OpenAI, Anthropic, etc.) using domain types
- **fluent_ai_memory**: Memory implementations using BOTH domain types AND provider clients for REAL LLM calls
- **fluent_ai**: Top-level package that orchestrates everything

### Zero Duplication Rule

NO package except `fluent_ai_domain` should define:
- Completion traits or types
- Model configuration types
- Message or chat types
- Provider abstractions
- Any domain model concepts

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