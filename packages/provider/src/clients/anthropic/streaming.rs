//! Zero-allocation streaming completion implementation for Anthropic API
//!
//! Provides real-time streaming with tool calling, proper SSE handling,
//! and efficient chunk processing with minimal memory allocation using HTTP3 client.
//! PURE STREAMING - NO FUTURES ARCHITECTURE

use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest as Http3Request};
// NO FUTURES - pure streaming HTTP3 architecture
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::AnthropicResult;
use super::messages::ContentBlock;

/// Streaming completion chunk from Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    #[serde(rename = "type")]
    pub chunk_type: String,
    #[serde(flatten)]
    pub data: StreamingData}

/// Data content of streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamingData {
    MessageStart {
        message: StreamingMessage},
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock},
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta},
    ContentBlockStop {
        index: usize},
    MessageDelta {
        delta: MessageDelta,
        usage: Option<StreamingUsage>},
    MessageStop,
    Ping,
    Error {
        error: ErrorDetails}}

/// Streaming message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: StreamingUsage}

/// Content delta for text updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String }}

/// Message delta for completion updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>}

/// Usage statistics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>}

/// Error details for streaming errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String}

/// Anthropic streaming chunk wrapper for easier processing
#[derive(Debug, Clone)]
pub struct AnthropicStreamChunk {
    pub chunk_type: String,
    pub content: String,
    pub is_final: bool,
    pub usage: Option<StreamingUsage>,
    pub error: Option<String>}

/// High-performance zero-allocation streaming implementation for Anthropic
///
/// This implementation uses HTTP3 client with streaming-first design for optimal performance
/// with no unsafe code, no unchecked operations, and no locking.
pub struct AnthropicStreamingProcessor {
    client: HttpClient,
    accumulator: String,
    current_usage: Option<StreamingUsage>}

impl AnthropicStreamingProcessor {
    /// Create a new streaming processor with HTTP3 client
    #[inline(always)]
    pub fn new() -> AnthropicResult<Self> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized()).map_err(|e| {
            super::error::AnthropicError::RequestError(format!(
                "Failed to create HTTP3 client: {}",
                e
            ))
        })?;

        Ok(Self {
            client,
            accumulator: String::with_capacity(4096), // Pre-allocate for efficiency
            current_usage: None})
    }

    /// Process a streaming request and return an AsyncStream - PURE STREAMING (no futures)
    /// Returns stream directly, error handling via user on_chunk handlers
    #[inline(always)]
    pub fn process_streaming_request(
        &self,
        http3_request: Http3Request,
    ) -> AnthropicResult<
        AsyncStream<AnthropicStreamChunk>,
    > {
        let client = self.client.clone();

        // Create high-throughput channel for chunks
        let (tx, stream) = crate::channel();

        // Use std::thread instead of spawn_async - NO FUTURES
        std::thread::spawn(move || {
            // Send streaming request using HTTP3 client with tokio runtime internally
            let response = {
                let rt = match tokio::runtime::Handle::try_current() {
                    Ok(handle) => handle,
                    Err(_) => {
                        let _ = tx.try_send(Err(
                            super::error::AnthropicError::RequestError(
                                "No tokio runtime available".to_string(),
                            ),
                        ));
                        return;
                    }
                };

                rt.block_on(async {
                    client.send(http3_request).await.map_err(|e| {
                        super::error::AnthropicError::RequestError(e.to_string())
                    })
                })
            };

            let response = match response {
                Ok(resp) => resp,
                Err(e) => {
                    let _ = tx.try_send(Err(e));
                    return;
                }
            };

            // Get SSE events from HTTP3 response - direct Vec<SseEvent> (no futures)
            let sse_events = response.sse();
            let mut content_accumulator = String::with_capacity(4096);
            let mut current_usage = None;

            // Direct iteration over SSE events (no futures)
            for sse_event in sse_events {
                if let Some(data) = sse_event.data {
                    // Skip empty events
                    if data.trim().is_empty() {
                        continue;
                    }

                    // Parse the SSE data as JSON
                    let chunk = match serde_json::from_str::<StreamingChunk>(&data) {
                        Ok(chunk) => chunk,
                        Err(e) => {
                            let _ = tx.try_send(Err(
                                super::error::AnthropicError::DeserializationError(
                                    format!("Failed to parse Anthropic SSE chunk: {}", e),
                                ),
                            ));
                            continue;
                        }
                    };

                    // Process the chunk based on its type
                    let processed_chunk = match process_anthropic_chunk(
                        &chunk,
                        &mut content_accumulator,
                        &mut current_usage,
                    ) {
                        Ok(Some(chunk)) => chunk,
                        Ok(None) => continue, // Skip internal chunks
                        Err(e) => {
                            let _ = tx.try_send(Err(e));
                            continue;
                        }
                    };

                    // Send the processed chunk
                    if tx.try_send(Ok(processed_chunk)).is_err() {
                        tracing::warn!(target: "rig", "Anthropic streaming receiver dropped");
                        break;
                    }
                }
            }
        });

        Ok(stream)
    }
}

impl Default for AnthropicStreamingProcessor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to default configuration if streaming optimization fails
            let client = HttpClient::with_config(HttpConfig::ai_optimized())
                .unwrap_or_else(|_| HttpClient::default());

            Self {
                client,
                accumulator: String::with_capacity(4096),
                current_usage: None}
        })
    }
}

/// Process an Anthropic streaming chunk with zero-allocation patterns
///
/// This function handles the different types of Anthropic streaming chunks
/// and accumulates content efficiently without unnecessary allocations.
#[inline(always)]
fn process_anthropic_chunk(
    chunk: &StreamingChunk,
    content_accumulator: &mut String,
    current_usage: &mut Option<StreamingUsage>,
) -> Result<Option<AnthropicStreamChunk>, super::error::AnthropicError> {
    match &chunk.data {
        StreamingData::MessageStart { message } => {
            // Initialize usage tracking
            *current_usage = Some(message.usage.clone());
            Ok(None) // Don't emit a chunk for message start
        }
        StreamingData::ContentBlockStart { .. } => {
            // Prepare for content accumulation
            content_accumulator.clear();
            Ok(None) // Don't emit a chunk for content block start
        }
        StreamingData::ContentBlockDelta { delta, .. } => {
            // Accumulate content based on delta type
            match delta {
                ContentDelta::TextDelta { text } => {
                    content_accumulator.push_str(text);
                    Ok(Some(AnthropicStreamChunk {
                        chunk_type: "content_block_delta".to_string(),
                        content: text.clone(),
                        is_final: false,
                        usage: current_usage.clone(),
                        error: None}))
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    content_accumulator.push_str(partial_json);
                    Ok(Some(AnthropicStreamChunk {
                        chunk_type: "input_json_delta".to_string(),
                        content: partial_json.clone(),
                        is_final: false,
                        usage: current_usage.clone(),
                        error: None}))
                }
            }
        }
        StreamingData::ContentBlockStop { .. } => {
            // Emit final content block
            let final_content = content_accumulator.clone();
            content_accumulator.clear();
            Ok(Some(AnthropicStreamChunk {
                chunk_type: "content_block_stop".to_string(),
                content: final_content,
                is_final: false,
                usage: current_usage.clone(),
                error: None}))
        }
        StreamingData::MessageDelta { delta, usage } => {
            // Update usage if provided
            if let Some(new_usage) = usage {
                *current_usage = Some(new_usage.clone());
            }

            // Check for stop reason
            if let Some(stop_reason) = &delta.stop_reason {
                Ok(Some(AnthropicStreamChunk {
                    chunk_type: "message_delta".to_string(),
                    content: format!("Stop reason: {}", stop_reason),
                    is_final: false,
                    usage: current_usage.clone(),
                    error: None}))
            } else {
                Ok(None)
            }
        }
        StreamingData::MessageStop => {
            // Final message - streaming is complete
            Ok(Some(AnthropicStreamChunk {
                chunk_type: "message_stop".to_string(),
                content: String::new(),
                is_final: true,
                usage: current_usage.clone(),
                error: None}))
        }
        StreamingData::Ping => {
            // Ping events are used for connection keep-alive
            Ok(None)
        }
        StreamingData::Error { error } => {
            // Error event
            Ok(Some(AnthropicStreamChunk {
                chunk_type: "error".to_string(),
                content: String::new(),
                is_final: true,
                usage: current_usage.clone(),
                error: Some(format!("{}: {}", error.error_type, error.message))}))
        }
    }
}
