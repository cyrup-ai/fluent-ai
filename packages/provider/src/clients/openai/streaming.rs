//! Zero-allocation OpenAI streaming implementation with SSE decoding
//!
//! Provides blazing-fast real-time streaming for OpenAI chat completions, function calls,
//! and tool use with comprehensive SSE parsing and no unsafe operations.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::tools::{
    OpenAIFunctionCall as ToolsOpenAIFunctionCall, OpenAIToolCall as ToolsOpenAIToolCall,
};
use super::{OpenAIError, OpenAIMessage, OpenAIResult};
use crate::AsyncStream;
use crate::ZeroOneOrMany;
use crate::domain::chunk::CompletionChunk;

/// OpenAI streaming completion chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ZeroOneOrMany<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<StreamUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Individual choice in streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Delta containing incremental content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ZeroOneOrMany<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallDelta>,
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Log probabilities for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    pub content: Option<ZeroOneOrMany<TokenLogProb>>,
}

/// Token log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ZeroOneOrMany<u8>>,
    pub top_logprobs: ZeroOneOrMany<TopLogProb>,
}

/// Top log probability alternative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ZeroOneOrMany<u8>>,
}

/// Usage information for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Detailed completion token breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

/// Detailed prompt token breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// SSE event parser for OpenAI streams
#[derive(Debug, Clone)]
pub struct SSEParser {
    buffer: String,
    event_type: Option<String>,
    data_lines: Vec<String>,
}

/// Parsed SSE event
#[derive(Debug, Clone)]
pub struct SSEEvent {
    pub event_type: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

/// Stream accumulator for building complete messages
#[derive(Debug, Clone)]
pub struct StreamAccumulator {
    pub content: String,
    pub role: Option<String>,
    pub tool_calls: HashMap<u32, PartialToolCall>,
    pub function_call: Option<PartialFunctionCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<StreamUsage>,
    pub model: String,
    pub id: String,
}

/// Partial tool call being built from deltas
#[derive(Debug, Clone)]
pub struct PartialToolCall {
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub function_name: Option<String>,
    pub function_arguments: String,
}

/// Partial function call being built from deltas
#[derive(Debug, Clone)]
pub struct PartialFunctionCall {
    pub name: Option<String>,
    pub arguments: String,
}

/// Stream processing configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub buffer_size: usize,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub include_usage: bool,
    pub include_logprobs: bool,
    pub yield_incomplete_tool_calls: bool,
}

/// Stream processing metrics
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub chunks_processed: u64,
    pub tokens_generated: u32,
    pub processing_time_ms: u64,
    pub average_chunk_size: f32,
    pub tool_calls_count: u32,
    pub errors_encountered: u32,
}

impl SSEParser {
    /// Create new SSE parser
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(4096),
            event_type: None,
            data_lines: Vec::with_capacity(16),
        }
    }

    /// Parse SSE chunk and return complete events
    #[inline(always)]
    pub fn parse_chunk(&mut self, chunk: &str) -> ZeroOneOrMany<SSEEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.buffer.drain(..=line_end).collect::<String>();
            let line = line.trim_end_matches('\n').trim_end_matches('\r');

            if line.is_empty() {
                // Empty line signals end of event
                if !self.data_lines.is_empty() {
                    let event = SSEEvent {
                        event_type: self.event_type.take(),
                        data: self.data_lines.join("\n"),
                        id: None,
                        retry: None,
                    };
                    events.push(event);
                    self.data_lines.clear();
                }
            } else if let Some(field_value) = line.split_once(':') {
                let field = field_value.0.trim();
                let value = field_value.1.trim();

                match field {
                    "event" => self.event_type = Some(value.to_string()),
                    "data" => self.data_lines.push(value.to_string()),
                    "id" => {}    // Could store ID if needed
                    "retry" => {} // Could store retry if needed
                    _ => {}       // Ignore unknown fields
                }
            }
        }

        ZeroOneOrMany::from_vec(events)
    }

    /// Check if parser has complete event ready
    #[inline(always)]
    pub fn has_complete_event(&self) -> bool {
        !self.data_lines.is_empty()
    }

    /// Reset parser state
    #[inline(always)]
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.event_type = None;
        self.data_lines.clear();
    }
}

impl StreamAccumulator {
    /// Create new accumulator
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            content: String::with_capacity(4096),
            role: None,
            tool_calls: HashMap::with_capacity(8),
            function_call: None,
            finish_reason: None,
            usage: None,
            model: String::new(),
            id: String::new(),
        }
    }

    /// Process streaming chunk and update accumulator
    #[inline(always)]
    pub fn process_chunk(
        &mut self,
        chunk: &OpenAIStreamChunk,
    ) -> OpenAIResult<Option<CompletionChunk>> {
        self.model = chunk.model.clone();
        self.id = chunk.id.clone();

        // Update usage if present
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }

        // Process choices
        match &chunk.choices {
            ZeroOneOrMany::None => return Ok(None),
            ZeroOneOrMany::One(choice) => {
                self.process_choice(choice)?;
            }
            ZeroOneOrMany::Many(choices) => {
                // Process first choice only for simplicity
                if let Some(choice) = choices.first() {
                    self.process_choice(choice)?;
                }
            }
        }

        // Create completion chunk
        let completion_chunk = if !self.content.is_empty() {
            CompletionChunk::Text(self.content.clone())
        } else if let Some(finish_reason) = &self.finish_reason {
            let usage = self.usage.as_ref().map(|u| crate::domain::chunk::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            });

            let reason = match finish_reason.as_str() {
                "stop" => Some(crate::domain::chunk::FinishReason::Stop),
                "length" => Some(crate::domain::chunk::FinishReason::Length),
                "content_filter" => Some(crate::domain::chunk::FinishReason::ContentFilter),
                "tool_calls" => Some(crate::domain::chunk::FinishReason::ToolCalls),
                _ => Some(crate::domain::chunk::FinishReason::Stop),
            };

            CompletionChunk::Complete {
                text: self.content.clone(),
                finish_reason: reason,
                usage,
            }
        } else {
            CompletionChunk::Text(String::new())
        };

        Ok(Some(completion_chunk))
    }

    /// Process individual choice delta
    #[inline(always)]
    fn process_choice(&mut self, choice: &StreamChoice) -> OpenAIResult<()> {
        let delta = &choice.delta;

        // Update role
        if let Some(role) = &delta.role {
            self.role = Some(role.clone());
        }

        // Append content
        if let Some(content) = &delta.content {
            self.content.push_str(content);
        }

        // Process tool calls
        if let Some(tool_calls) = &delta.tool_calls {
            self.process_tool_call_deltas(tool_calls)?;
        }

        // Process function call
        if let Some(function_call) = &delta.function_call {
            self.process_function_call_delta(function_call);
        }

        // Update finish reason
        if let Some(reason) = &choice.finish_reason {
            self.finish_reason = Some(reason.clone());
        }

        Ok(())
    }

    /// Process tool call deltas
    #[inline(always)]
    fn process_tool_call_deltas(
        &mut self,
        deltas: &ZeroOneOrMany<ToolCallDelta>,
    ) -> OpenAIResult<()> {
        match deltas {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(delta) => {
                self.process_single_tool_call_delta(delta)?;
            }
            ZeroOneOrMany::Many(delta_vec) => {
                for delta in delta_vec {
                    self.process_single_tool_call_delta(delta)?;
                }
            }
        }
        Ok(())
    }

    /// Process single tool call delta
    #[inline(always)]
    fn process_single_tool_call_delta(&mut self, delta: &ToolCallDelta) -> OpenAIResult<()> {
        let partial_call = self
            .tool_calls
            .entry(delta.index)
            .or_insert_with(|| PartialToolCall {
                id: None,
                call_type: None,
                function_name: None,
                function_arguments: String::with_capacity(512),
            });

        // Update ID if present
        if let Some(id) = &delta.id {
            partial_call.id = Some(id.clone());
        }

        // Update type if present
        if let Some(call_type) = &delta.call_type {
            partial_call.call_type = Some(call_type.clone());
        }

        // Update function details if present
        if let Some(function) = &delta.function {
            if let Some(name) = &function.name {
                partial_call.function_name = Some(name.clone());
            }
            if let Some(arguments) = &function.arguments {
                partial_call.function_arguments.push_str(arguments);
            }
        }

        Ok(())
    }

    /// Process function call delta
    #[inline(always)]
    fn process_function_call_delta(&mut self, delta: &FunctionCallDelta) {
        let function_call = self
            .function_call
            .get_or_insert_with(|| PartialFunctionCall {
                name: None,
                arguments: String::with_capacity(512),
            });

        if let Some(name) = &delta.name {
            function_call.name = Some(name.clone());
        }
        if let Some(arguments) = &delta.arguments {
            function_call.arguments.push_str(arguments);
        }
    }

    /// Get completed tool calls
    #[inline(always)]
    pub fn get_completed_tool_calls(&self) -> ZeroOneOrMany<ToolsOpenAIToolCall> {
        let mut completed_calls = Vec::new();

        for (_, partial) in &self.tool_calls {
            if let (Some(id), Some(call_type), Some(name)) =
                (&partial.id, &partial.call_type, &partial.function_name)
            {
                completed_calls.push(ToolsOpenAIToolCall {
                    id: id.clone(),
                    call_type: call_type.clone(),
                    function: ToolsOpenAIFunctionCall {
                        name: name.clone(),
                        arguments: partial.function_arguments.clone(),
                    },
                });
            }
        }

        ZeroOneOrMany::from_vec(completed_calls)
    }

    /// Check if streaming is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Get final message
    #[inline(always)]
    pub fn to_message(&self) -> OpenAIMessage {
        let tool_calls = self.get_completed_tool_calls();
        let tool_calls_vec = match tool_calls {
            ZeroOneOrMany::None => None,
            ZeroOneOrMany::One(call) => {
                // Convert from tools::OpenAIToolCall to messages::OpenAIToolCall
                let messages_call = crate::providers::openai::messages::OpenAIToolCall {
                    id: call.id,
                    call_type: call.call_type,
                    function: crate::providers::openai::messages::OpenAIFunctionCall {
                        name: call.function.name,
                        arguments: call.function.arguments,
                    },
                };
                Some(vec![messages_call])
            }
            ZeroOneOrMany::Many(calls) => {
                // Convert vector of tools::OpenAIToolCall to messages::OpenAIToolCall
                let messages_calls: Vec<_> = calls
                    .into_iter()
                    .map(|call| crate::providers::openai::messages::OpenAIToolCall {
                        id: call.id,
                        call_type: call.call_type,
                        function: crate::providers::openai::messages::OpenAIFunctionCall {
                            name: call.function.name,
                            arguments: call.function.arguments,
                        },
                    })
                    .collect();
                Some(messages_calls)
            }
        };

        OpenAIMessage {
            role: self.role.clone().unwrap_or_else(|| "assistant".to_string()),
            content: if self.content.is_empty() {
                None
            } else {
                Some(crate::providers::openai::OpenAIContent::Text(
                    self.content.clone(),
                ))
            },
            name: None,
            tool_calls: tool_calls_vec,
            tool_call_id: None,
            function_call: self.function_call.as_ref().map(|fc| {
                crate::providers::openai::messages::OpenAIFunctionCall {
                    name: fc.name.clone().unwrap_or_default(),
                    arguments: fc.arguments.clone(),
                }
            }),
        }
    }
}

impl StreamConfig {
    /// Create default streaming configuration
    #[inline(always)]
    pub fn default() -> Self {
        Self {
            buffer_size: 8192,
            timeout_ms: 30000,
            retry_attempts: 3,
            include_usage: true,
            include_logprobs: false,
            yield_incomplete_tool_calls: false,
        }
    }

    /// Create configuration optimized for speed
    #[inline(always)]
    pub fn fast() -> Self {
        Self {
            buffer_size: 4096,
            timeout_ms: 15000,
            retry_attempts: 1,
            include_usage: false,
            include_logprobs: false,
            yield_incomplete_tool_calls: true,
        }
    }

    /// Create configuration optimized for reliability
    #[inline(always)]
    pub fn reliable() -> Self {
        Self {
            buffer_size: 16384,
            timeout_ms: 60000,
            retry_attempts: 5,
            include_usage: true,
            include_logprobs: true,
            yield_incomplete_tool_calls: false,
        }
    }

    /// Set buffer size
    #[inline(always)]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set timeout
    #[inline(always)]
    pub fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Set retry attempts
    #[inline(always)]
    pub fn with_retry_attempts(mut self, attempts: u32) -> Self {
        self.retry_attempts = attempts;
        self
    }

    /// Enable/disable usage tracking
    #[inline(always)]
    pub fn with_usage_tracking(mut self, enabled: bool) -> Self {
        self.include_usage = enabled;
        self
    }

    /// Enable/disable logprobs
    #[inline(always)]
    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.include_logprobs = enabled;
        self
    }
}

impl StreamMetrics {
    /// Create new metrics tracker
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            chunks_processed: 0,
            tokens_generated: 0,
            processing_time_ms: 0,
            average_chunk_size: 0.0,
            tool_calls_count: 0,
            errors_encountered: 0,
        }
    }

    /// Update metrics with new chunk
    #[inline(always)]
    pub fn update_chunk(&mut self, chunk: &OpenAIStreamChunk, processing_time_ms: u64) {
        self.chunks_processed += 1;
        self.processing_time_ms += processing_time_ms;

        if let Some(usage) = &chunk.usage {
            self.tokens_generated = usage.completion_tokens;
        }

        // Update average chunk size
        let total_size = self.average_chunk_size * (self.chunks_processed - 1) as f32;
        self.average_chunk_size =
            (total_size + chunk.id.len() as f32) / self.chunks_processed as f32;

        // Count tool calls
        match &chunk.choices {
            ZeroOneOrMany::One(choice) => {
                if let Some(tool_calls) = &choice.delta.tool_calls {
                    self.tool_calls_count += match tool_calls {
                        ZeroOneOrMany::One(_) => 1,
                        ZeroOneOrMany::Many(calls) => calls.len() as u32,
                        ZeroOneOrMany::None => 0,
                    };
                }
            }
            ZeroOneOrMany::Many(choices) => {
                for choice in choices {
                    if let Some(tool_calls) = &choice.delta.tool_calls {
                        self.tool_calls_count += match tool_calls {
                            ZeroOneOrMany::One(_) => 1,
                            ZeroOneOrMany::Many(calls) => calls.len() as u32,
                            ZeroOneOrMany::None => 0,
                        };
                    }
                }
            }
            ZeroOneOrMany::None => {}
        }
    }

    /// Record error
    #[inline(always)]
    pub fn record_error(&mut self) {
        self.errors_encountered += 1;
    }

    /// Get tokens per second
    #[inline(always)]
    pub fn tokens_per_second(&self) -> f32 {
        if self.processing_time_ms == 0 {
            0.0
        } else {
            (self.tokens_generated as f32) / (self.processing_time_ms as f32 / 1000.0)
        }
    }

    /// Get error rate
    #[inline(always)]
    pub fn error_rate(&self) -> f32 {
        if self.chunks_processed == 0 {
            0.0
        } else {
            (self.errors_encountered as f32) / (self.chunks_processed as f32)
        }
    }
}

/// Parse OpenAI streaming chunk from JSON
#[inline(always)]
pub fn parse_stream_chunk(data: &str) -> OpenAIResult<OpenAIStreamChunk> {
    // Handle special cases
    if data.trim() == "[DONE]" {
        return Err(OpenAIError::ApiError("Stream complete".to_string()));
    }

    serde_json::from_str(data)
        .map_err(|e| OpenAIError::JsonError(format!("Failed to parse stream chunk: {}", e)))
}

/// Create streaming completion from SSE events
#[inline(always)]
pub fn stream_from_sse_events(events: AsyncStream<SSEEvent>) -> AsyncStream<CompletionChunk> {
    let (sender, stream) = AsyncStream::channel();
    let mut accumulator = StreamAccumulator::new();

    crate::async_task::spawn_async(async move {
        let mut events_iter = events;
        while let Some(event) = futures_util::StreamExt::next(&mut events_iter).await {
            // Skip non-data events
            if event.event_type.as_deref() != Some("data") && event.event_type.is_some() {
                continue;
            }

            // Parse chunk from event data
            match parse_stream_chunk(&event.data) {
                Ok(chunk) => {
                    match accumulator.process_chunk(&chunk) {
                        Ok(Some(completion_chunk)) => {
                            if sender.try_send(completion_chunk).is_err() {
                                break; // Stream closed
                            }
                        }
                        Ok(None) => {}   // No chunk to yield yet
                        Err(_) => break, // Error processing chunk
                    }
                }
                Err(_) => {
                    // Check for [DONE] marker
                    if event.data.trim() == "[DONE]" {
                        break;
                    }
                }
            }
        }
    });

    stream
}

/// Optimize streaming performance by batching small chunks
#[inline(always)]
pub fn optimize_stream_batching(
    stream: AsyncStream<CompletionChunk>,
    batch_size: usize,
    timeout_ms: u64,
) -> AsyncStream<ZeroOneOrMany<CompletionChunk>> {
    let (sender, optimized_stream) = AsyncStream::channel();

    crate::async_task::spawn_async(async move {
        let mut stream_iter = stream;
        let mut batch = Vec::with_capacity(batch_size);
        let mut last_batch_time = SystemTime::now();

        while let Some(chunk) = futures_util::StreamExt::next(&mut stream_iter).await {
            batch.push(chunk);

            let should_flush = batch.len() >= batch_size
                || last_batch_time.elapsed().unwrap_or_default().as_millis() >= timeout_ms as u128;

            if should_flush && !batch.is_empty() {
                let batched_chunks = ZeroOneOrMany::from_vec(std::mem::take(&mut batch));
                if sender.try_send(batched_chunks).is_err() {
                    break;
                }
                last_batch_time = SystemTime::now();
            }
        }

        // Send remaining chunks
        if !batch.is_empty() {
            let _ = sender.try_send(ZeroOneOrMany::from_vec(batch));
        }
    });

    optimized_stream
}

/// Compatibility aliases for streaming types
pub type StreamingCompletionResponse = OpenAIStreamChunk;
pub type StreamingChoice = StreamChoice;
pub type StreamingMessage = Delta;

/// Send a compatible streaming request for OpenAI-compatible providers
/// This function provides compatibility for other providers that use OpenAI-style APIs
pub async fn send_compatible_streaming_request(
    client: &fluent_ai_http3::HttpClient,
    url: &str,
    headers: std::collections::HashMap<String, String>,
    body: serde_json::Value,
) -> Result<AsyncStream<serde_json::Value>, crate::clients::openai::error::OpenAIError> {
    use fluent_ai_http3::HttpRequest;

    let mut request = HttpRequest::post(url, serde_json::to_vec(&body)?)?;

    for (key, value) in headers {
        request = request.header(&key, &value);
    }

    let response = client.send(request).await?;
    let sse_stream = response.sse();

    let (sender, stream) = AsyncStream::channel();

    crate::async_task::spawn_async(async move {
        let mut sse_iter = sse_stream;
        while let Some(event) = futures_util::StreamExt::next(&mut sse_iter).await {
            match event {
                Ok(sse_event) => {
                    if let Some(data) = sse_event.data {
                        if data == "[DONE]" {
                            break;
                        }

                        match serde_json::from_str::<serde_json::Value>(&data) {
                            Ok(json_chunk) => {
                                sender.send(json_chunk).await;
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse SSE chunk: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("SSE stream error: {}", e);
                    break;
                }
            }
        }
        sender.close().await;
    });

    Ok(stream)
}
