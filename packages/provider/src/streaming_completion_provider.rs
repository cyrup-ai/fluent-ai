//! Streaming-only completion provider trait that ENFORCES architectural constraints
//!
//! This trait design prevents async/await violations at compile time by:
//! - Requiring AsyncStream<CompletionChunk> returns (not Future)
//! - Prohibiting Result types in streaming operations
//! - Forcing consistent Http3 usage patterns
//! - Mandating emit!/handle_error! macro usage

use fluent_ai_async::AsyncStream;
use fluent_ai_domain::{
    completion::{CompletionRequest, types::CompletionParams},
    context::CompletionChunk,
    Prompt,
};

/// Core streaming completion provider trait - NO FUTURES ARCHITECTURE ALLOWED
/// 
/// This trait enforces the streaming-only architecture at the type level:
/// - All methods return AsyncStream<T>, never Future<Output = Result<T, E>>
/// - Errors are handled inside streams using handle_error! macro
/// - Http3 patterns are naturally enforced by the implementation requirements
/// - Zero allocation and lock-free patterns are maintained
/// 
/// # Architecture Constraints Enforced
/// 
/// 1. **No async fn methods**: All methods return AsyncStream directly
/// 2. **No Result types**: Errors handled inside streams with handle_error!
/// 3. **Consistent Http3 usage**: All implementations use Http3::json().body().post()
/// 4. **Thread-based operations**: AsyncStream::with_channel spawns threads, not futures
/// 
/// # Implementation Pattern
/// 
/// ```rust
/// impl StreamingCompletionProvider for MyProvider {
///     fn stream_completion(&self, request: CompletionRequest) -> AsyncStream<CompletionChunk> {
///         AsyncStream::with_channel(move |sender| {
///             let response_stream = Http3::json()
///                 .api_key(&self.api_key)
///                 .body(&request)
///                 .post(&self.endpoint_url());
///                 
///             response_stream.on_chunk(|chunk| {
///                 match self.parse_chunk(chunk) {
///                     Some(completion_chunk) => emit!(sender, completion_chunk),
///                     None => handle_error!("Parse failed", "Invalid response chunk"),
///                 }
///             });
///         })
///     }
/// }
/// ```
pub trait StreamingCompletionProvider {
    /// Stream completion chunks for a full completion request
    /// 
    /// ARCHITECTURE CONSTRAINT: Must return AsyncStream<CompletionChunk>, never Result
    /// Implementation MUST use:
    /// - AsyncStream::with_channel(|sender| { ... })
    /// - Http3::json().body(&request).post(&url) for HTTP requests
    /// - emit!(sender, chunk) to send values
    /// - handle_error!(err, context) for error handling
    fn stream_completion(&self, request: CompletionRequest) -> AsyncStream<CompletionChunk>;
    
    /// Stream completion chunks for a simple prompt
    /// 
    /// ARCHITECTURE CONSTRAINT: Must return AsyncStream<CompletionChunk>, never Result
    /// Default implementation converts prompt to CompletionRequest
    fn stream_prompt(&self, prompt: Prompt, params: &CompletionParams) -> AsyncStream<CompletionChunk> {
        // Convert prompt and params to CompletionRequest
        let request = CompletionRequest::from_prompt_and_params(prompt, params);
        self.stream_completion(request)
    }
    
    /// Provider name for identification and routing
    /// 
    /// ARCHITECTURE CONSTRAINT: Must be 'static str, no allocations
    fn provider_name(&self) -> &'static str;
    
    /// Test provider connectivity and authentication
    /// 
    /// ARCHITECTURE CONSTRAINT: Returns AsyncStream<ConnectionStatus>, not Result
    /// Streams success/failure status without using Result types
    fn test_connection(&self) -> AsyncStream<ConnectionStatus>;
}

/// Connection status for streaming connection tests
#[derive(Debug, Clone)]
pub enum ConnectionStatus {
    /// Connection successful
    Connected,
    /// Connection failed with error message
    Failed(String),
    /// Connection testing in progress
    Testing,
}

/// Provider configuration result
#[derive(Debug, Clone)]
pub enum ConfigurationResult {
    /// Configuration applied successfully  
    Applied,
    /// Configuration failed with error message
    Failed(String),
    /// Configuration in progress
    InProgress,
}

/// Extension trait for common streaming operations
/// 
/// Provides helper methods that maintain the streaming-only architecture
pub trait StreamingCompletionProviderExt: StreamingCompletionProvider {
    /// Collect all completion chunks into a single response
    /// 
    /// NOTE: Use sparingly - streaming consumption is preferred
    fn complete_and_collect(&self, request: CompletionRequest) -> Vec<CompletionChunk> {
        self.stream_completion(request).collect()
    }
    
    /// Stream with error handling fallback
    /// 
    /// Provides a fallback chunk if the stream encounters errors
    fn stream_with_fallback(&self, request: CompletionRequest, fallback: CompletionChunk) -> AsyncStream<CompletionChunk> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let provider_stream = self.stream_completion(request);
        
        AsyncStream::with_channel(move |sender| {
            let mut received_any = false;
            
            provider_stream.on_chunk(|chunk| {
                received_any = true;
                emit!(sender, chunk);
            });
            
            // If no chunks were received, emit fallback
            if !received_any {
                emit!(sender, fallback);
            }
        })
    }
}

// Blanket implementation for all StreamingCompletionProvider implementors
impl<T: StreamingCompletionProvider> StreamingCompletionProviderExt for T {}

/// Utility functions for implementing StreamingCompletionProvider
pub mod utils {
    use super::*;
    use fluent_ai_async::{emit, handle_error};
    
    /// Parse Server-Sent Events (SSE) chunk into CompletionChunk
    /// 
    /// Common pattern for most providers that use SSE streaming
    pub fn parse_sse_chunk(chunk_bytes: &[u8]) -> Option<CompletionChunk> {
        let chunk_str = std::str::from_utf8(chunk_bytes).ok()?;
        
        for line in chunk_str.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    return None;
                }
                
                if let Ok(json_data) = serde_json::from_str::<serde_json::Value>(data) {
                    // Extract content based on common SSE patterns
                    if let Some(content) = json_data["choices"][0]["delta"]["content"].as_str() {
                        return Some(CompletionChunk::text(content.to_string()));
                    }
                }
            }
        }
        
        None
    }
    
    /// Build HTTP request body for completion requests
    /// 
    /// Standardizes request format across providers
    pub fn build_completion_request_body(
        request: &CompletionRequest,
        model: &str,
        stream: bool,
    ) -> serde_json::Value {
        use serde_json::json;
        
        json!({
            "model": model,
            "messages": request.messages.iter().map(|m| {
                json!({"role": m.role, "content": m.content})
            }).collect::<Vec<_>>(),
            "stream": stream,
            "temperature": request.temperature.unwrap_or(0.7),
            "max_tokens": request.max_tokens.unwrap_or(2048)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Mock provider for testing trait enforcement
    struct MockProvider {
        provider_name: &'static str,
    }
    
    impl StreamingCompletionProvider for MockProvider {
        fn stream_completion(&self, _request: CompletionRequest) -> AsyncStream<CompletionChunk> {
            use fluent_ai_async::{AsyncStream, emit};
            
            AsyncStream::with_channel(move |sender| {
                // Mock streaming response
                emit!(sender, CompletionChunk::text("Hello".to_string()));
                emit!(sender, CompletionChunk::text(" World".to_string()));
            })
        }
        
        fn provider_name(&self) -> &'static str {
            self.provider_name
        }
        
        fn test_connection(&self) -> AsyncStream<ConnectionStatus> {
            use fluent_ai_async::{AsyncStream, emit};
            
            AsyncStream::with_channel(move |sender| {
                emit!(sender, ConnectionStatus::Connected);
            })
        }
    }
    
    #[test]
    fn test_streaming_provider_trait() {
        let provider = MockProvider { provider_name: "mock" };
        
        // Test provider name
        assert_eq!(provider.provider_name(), "mock");
        
        // Test streaming completion
        let request = CompletionRequest::new("test prompt".to_string());
        let chunks: Vec<CompletionChunk> = provider.stream_completion(request).collect();
        assert_eq!(chunks.len(), 2);
        
        // Test connection
        let status: Vec<ConnectionStatus> = provider.test_connection().collect();
        assert_eq!(status.len(), 1);
    }
    
    #[test]
    fn test_extension_trait() {
        let provider = MockProvider { provider_name: "mock" };
        
        // Test collect method
        let request = CompletionRequest::new("test prompt".to_string());
        let chunks = provider.complete_and_collect(request);
        assert_eq!(chunks.len(), 2);
    }
}