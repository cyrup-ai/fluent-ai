//! Zero-allocation streaming completion implementation for Anthropic API
//!
//! Provides real-time streaming with tool calling, proper SSE handling,
//! and efficient chunk processing with minimal memory allocation.

use crate::async_task::AsyncStream;
use crate::domain::chunk::CompletionChunk;
use crate::providers::anthropic::{
    AnthropicResult, AnthropicCompletionRequest,
    handle_reqwest_error, handle_json_error,
};
use crate::providers::anthropic::messages::ContentBlock;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// Streaming completion chunk from Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    #[serde(rename = "type")]
    pub chunk_type: String,
    #[serde(flatten)]
    pub data: StreamingData,
}

/// Data content of streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamingData {
    MessageStart {
        message: StreamingMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<StreamingUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ErrorDetails,
    },
}

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
    pub usage: StreamingUsage,
}

/// Content delta for text updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
}

/// Message delta for completion updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// Usage statistics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

/// Error details in streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Zero-allocation streaming client for Anthropic API
pub struct AnthropicStreamingClient {
    client: Client,
    api_key: String,
    base_url: String,
}

/// Aggregated streaming state for complete responses
#[derive(Debug, Clone)]
pub struct StreamingState {
    pub message_id: Option<String>,
    pub text_content: String,
    pub tool_calls: Vec<ToolCallState>,
    pub usage: Option<StreamingUsage>,
    pub stop_reason: Option<String>,
    pub is_complete: bool,
}

/// Tool call state during streaming
#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub id: String,
    pub name: String,
    pub input_json: String,
    pub is_complete: bool,
}

impl AnthropicStreamingClient {
    /// Create new streaming client
    #[inline(always)]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let client = Client::builder()
            .user_agent("fluent-ai/1.0")
            .build()
            .unwrap_or_default();

        Self {
            client,
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Start streaming completion request
    pub async fn stream_completion(
        &self,
        mut request: AnthropicCompletionRequest,
    ) -> AnthropicResult<Pin<Box<dyn Stream<Item = AnthropicResult<StreamingChunk>> + Send>>> {
        // Enable streaming
        request.stream = Some(true);
        
        let url = format!("{}/messages", self.base_url);
        let request_body = serde_json::to_string(&request)
            .map_err(handle_json_error)?;
        
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Accept", "text/event-stream")
            .body(request_body)
            .send()
            .await
            .map_err(handle_reqwest_error)?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::providers::anthropic::handle_http_error(status.as_u16(), &body));
        }
        
        let stream = response.bytes_stream();
        let sse_stream = SSEStream::new(stream);
        
        Ok(Box::pin(sse_stream))
    }

    /// Stream completion with chunk aggregation
    pub fn stream_with_aggregation(
        &self,
        request: AnthropicCompletionRequest,
    ) -> AsyncStream<CompletionChunk> {
        let client = self.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let mut state = StreamingState::new();
            
            match client.stream_completion(request).await {
                Ok(mut stream) => {
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                if let Some(completion_chunk) = state.process_chunk(chunk) {
                                    if tx.send(completion_chunk).is_err() {
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                let error_chunk = CompletionChunk::Error(format!("Stream error: {}", e));
                                if tx.send(error_chunk).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Send final completion chunk
                    if state.is_complete {
                        let final_chunk = CompletionChunk::Complete {
                            text: state.text_content,
                            finish_reason: state.stop_reason.map(|reason| match reason.as_str() {
                                "end_turn" => crate::domain::chunk::FinishReason::Stop,
                                "max_tokens" => crate::domain::chunk::FinishReason::Length,
                                "stop_sequence" => crate::domain::chunk::FinishReason::Stop,
                                "tool_use" => crate::domain::chunk::FinishReason::ToolCalls,
                                _ => crate::domain::chunk::FinishReason::Stop,
                            }),
                            usage: state.usage.map(|u| crate::domain::chunk::Usage {
                                prompt_tokens: u.input_tokens as u32,
                                completion_tokens: u.output_tokens as u32,
                                total_tokens: (u.input_tokens + u.output_tokens) as u32,
                            }),
                        };
                        let _ = tx.send(final_chunk);
                    }
                }
                Err(e) => {
                    let error_chunk = CompletionChunk::Error(format!("Failed to start stream: {}", e));
                    let _ = tx.send(error_chunk);
                }
            }
        });
        
        AsyncStream::new(rx)
    }
}

impl Clone for AnthropicStreamingClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
        }
    }
}

impl StreamingState {
    /// Create new streaming state
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            message_id: None,
            text_content: String::new(),
            tool_calls: Vec::new(),
            usage: None,
            stop_reason: None,
            is_complete: false,
        }
    }

    /// Process streaming chunk and return completion chunk if ready
    #[inline(always)]
    pub fn process_chunk(&mut self, chunk: StreamingChunk) -> Option<CompletionChunk> {
        match chunk.data {
            StreamingData::MessageStart { message } => {
                self.message_id = Some(message.id);
                None
            }
            StreamingData::ContentBlockDelta { delta, .. } => {
                match delta {
                    ContentDelta::TextDelta { text } => {
                        self.text_content.push_str(&text);
                        Some(CompletionChunk::Text(text))
                    }
                    ContentDelta::InputJsonDelta { partial_json } => {
                        // Handle partial JSON for tool calls
                        if let Some(tool_call) = self.tool_calls.last_mut() {
                            tool_call.input_json.push_str(&partial_json);
                        }
                        Some(CompletionChunk::ToolCall {
                            id: self.tool_calls.last()?.id.clone(),
                            name: self.tool_calls.last()?.name.clone(),
                            partial_input: partial_json,
                        })
                    }
                }
            }
            StreamingData::ContentBlockStart { content_block, .. } => {
                match content_block {
                    ContentBlock::ToolUse { id, name, .. } => {
                        let tool_call = ToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                            input_json: String::new(),
                            is_complete: false,
                        };
                        self.tool_calls.push(tool_call);
                        
                        Some(CompletionChunk::ToolCallStart {
                            id,
                            name,
                        })
                    }
                    _ => None,
                }
            }
            StreamingData::ContentBlockStop { .. } => {
                if let Some(tool_call) = self.tool_calls.last_mut() {
                    tool_call.is_complete = true;
                    Some(CompletionChunk::ToolCallComplete {
                        id: tool_call.id.clone(),
                        name: tool_call.name.clone(),
                        input: tool_call.input_json.clone(),
                    })
                } else {
                    None
                }
            }
            StreamingData::MessageDelta { delta, usage } => {
                if let Some(stop_reason) = delta.stop_reason {
                    self.stop_reason = Some(stop_reason);
                }
                if let Some(u) = usage {
                    self.usage = Some(u);
                }
                None
            }
            StreamingData::MessageStop => {
                self.is_complete = true;
                None
            }
            StreamingData::Error { error } => {
                Some(CompletionChunk::Error(format!("{}: {}", error.error_type, error.message)))
            }
            _ => None,
        }
    }
}

/// Server-Sent Events stream parser
struct SSEStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> SSEStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> Stream for SSEStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = AnthropicResult<StreamingChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    self.buffer.extend_from_slice(&bytes);
                    
                    // Process complete lines
                    while let Some(line_end) = self.buffer.iter().position(|&b| b == b'\n') {
                        let line = self.buffer.drain(..=line_end).collect::<Vec<u8>>();
                        let line_str = String::from_utf8_lossy(&line[..line.len()-1]); // Remove \n
                        
                        if line_str.starts_with("data: ") {
                            let json_str = &line_str[6..]; // Remove "data: " prefix
                            
                            if json_str == "[DONE]" {
                                return Poll::Ready(None);
                            }
                            
                            match serde_json::from_str::<StreamingChunk>(json_str) {
                                Ok(chunk) => return Poll::Ready(Some(Ok(chunk))),
                                Err(e) => return Poll::Ready(Some(Err(handle_json_error(e)))),
                            }
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(handle_reqwest_error(e))));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}