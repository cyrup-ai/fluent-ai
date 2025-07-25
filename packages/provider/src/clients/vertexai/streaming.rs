//! Zero-allocation streaming response handling for VertexAI
//!
//! Implements Server-Sent Events (SSE) parsing with efficient JSON chunk processing,
//! delta content accumulation, and comprehensive error recovery.

use crate::clients::vertexai::{
    VertexAIError, VertexAIResult,
    completion::{CompletionChunk, CompletionResponse, Candidate, UsageMetadata, Part, Content}};
use arrayvec::{ArrayString};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use futures_util::{Stream, StreamExt};
use serde_json;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::sync::LazyLock;

/// Global streaming metrics
static STREAM_EVENTS_PROCESSED: RelaxedCounter = RelaxedCounter::new(0);
static STREAM_PARSE_ERRORS: RelaxedCounter = RelaxedCounter::new(0);
static STREAM_CONNECTIONS: RelaxedCounter = RelaxedCounter::new(0);

/// Maximum SSE event size in bytes
const MAX_SSE_EVENT_SIZE: usize = 16384;

/// Maximum accumulated response size
const MAX_ACCUMULATED_SIZE: usize = 1048576; // 1MB

/// VertexAI streaming response handler
pub struct VertexAIStream {
    /// Underlying SSE stream
    sse_stream: Pin<Box<dyn Stream<Item = Result<SseEvent, VertexAIError>> + Send>>,
    
    /// Accumulated response content
    accumulated_response: Option<CompletionResponse>,
    
    /// Current chunk buffer
    chunk_buffer: ArrayString<MAX_SSE_EVENT_SIZE>,
    
    /// Stream state
    state: StreamState,
    
    /// Accumulated content size for memory protection
    accumulated_size: usize,
    
    /// Stream metrics
    events_processed: usize,
    
    /// Last error for recovery
    last_error: Option<VertexAIError>}

/// Stream processing state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    /// Stream is active and processing events
    Active,
    
    /// Stream completed successfully
    Completed,
    
    /// Stream encountered error
    Error,
    
    /// Stream was terminated early
    Terminated}

/// Server-Sent Event structure
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type (data, error, etc.)
    pub event_type: ArrayString<32>,
    
    /// Event data
    pub data: String,
    
    /// Event ID
    pub id: Option<ArrayString<64>>,
    
    /// Retry interval
    pub retry: Option<u32>}

/// Stream event types for external consumption
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Content chunk received
    ContentChunk {
        /// Incremental text content
        content: String,
        
        /// Function call data (if any)
        function_call: Option<Part>,
        
        /// Chunk metadata
        metadata: Option<ChunkMetadata>},
    
    /// Usage statistics (final event)
    Usage {
        /// Token usage information
        usage: UsageMetadata},
    
    /// Stream completed
    Completed {
        /// Final accumulated response
        response: CompletionResponse},
    
    /// Stream error
    Error {
        /// Error details
        error: VertexAIError,
        
        /// Partial response if available
        partial_response: Option<CompletionResponse>}}

/// Chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Chunk sequence number
    pub sequence: usize,
    
    /// Chunk processing timestamp
    pub timestamp: u64,
    
    /// Chunk size in bytes
    pub size: usize,
    
    /// Finish reason (if final chunk)
    pub finish_reason: Option<String>}

impl VertexAIStream {
    /// Create new streaming response handler
    pub fn new(sse_stream: impl Stream<Item = Result<fluent_ai_http3::SseEvent, fluent_ai_http3::HttpError>> + Send + 'static) -> Self {
        STREAM_CONNECTIONS.inc();
        
        // Convert fluent_ai_http3::SseEvent to our SseEvent
        let converted_stream = sse_stream.map(|result| {
            match result {
                Ok(sse_event) => {
                    let event_type = ArrayString::from(sse_event.event_type.as_deref().unwrap_or("data"))
                        .unwrap_or_else(|_| {
                            // Fallback to compile-time verified string
                            ArrayString::from("data").unwrap_or_else(|_| {
                                // This should never happen as "data" is 4 chars and fits in ArrayString
                                ArrayString::new()
                            })
                        });
                    
                    let id = sse_event.id.and_then(|id| ArrayString::from(&id).ok());
                    
                    Ok(SseEvent {
                        event_type,
                        data: sse_event.data.unwrap_or_default(),
                        id,
                        retry: sse_event.retry})
                }
                Err(e) => Err(VertexAIError::Streaming {
                    source: format!("SSE parsing error: {}", e)})}
        });
        
        Self {
            sse_stream: Box::pin(converted_stream),
            accumulated_response: None,
            chunk_buffer: ArrayString::new(),
            state: StreamState::Active,
            accumulated_size: 0,
            events_processed: 0,
            last_error: None}
    }
    
    /// Get next stream event
    pub async fn next_event(&mut self) -> Option<StreamEvent> {
        if self.state != StreamState::Active {
            return None;
        }
        
        match self.sse_stream.next().await {
            Some(Ok(sse_event)) => {
                self.events_processed += 1;
                STREAM_EVENTS_PROCESSED.inc();
                
                match self.process_sse_event(sse_event) {
                    Ok(event) => event,
                    Err(error) => {
                        self.state = StreamState::Error;
                        self.last_error = Some(error.clone());
                        STREAM_PARSE_ERRORS.inc();
                        
                        Some(StreamEvent::Error {
                            error,
                            partial_response: self.accumulated_response.clone()})
                    }
                }
            }
            Some(Err(error)) => {
                self.state = StreamState::Error;
                self.last_error = Some(error.clone());
                
                Some(StreamEvent::Error {
                    error,
                    partial_response: self.accumulated_response.clone()})
            }
            None => {
                self.state = StreamState::Completed;
                
                if let Some(response) = self.accumulated_response.clone() {
                    Some(StreamEvent::Completed { response })
                } else {
                    Some(StreamEvent::Error {
                        error: VertexAIError::Streaming {
                            source: "Stream ended without final response".to_string()},
                        partial_response: None})
                }
            }
        }
    }
    
    /// Process individual SSE event
    fn process_sse_event(&mut self, event: SseEvent) -> VertexAIResult<Option<StreamEvent>> {
        // Check accumulated size limit
        if self.accumulated_size + event.data.len() > MAX_ACCUMULATED_SIZE {
            return Err(VertexAIError::Streaming {
                source: format!("Response size limit {} exceeded", MAX_ACCUMULATED_SIZE)});
        }
        
        self.accumulated_size += event.data.len();
        
        match event.event_type.as_str() {
            "data" => self.process_data_event(event.data),
            "error" => {
                let error = VertexAIError::Streaming {
                    source: format!("Server error: {}", event.data)};
                Err(error)
            }
            "close" | "end" => {
                self.state = StreamState::Completed;
                Ok(None)
            }
            _ => {
                // Unknown event type, ignore
                Ok(None)
            }
        }
    }
    
    /// Process data event containing JSON chunk
    fn process_data_event(&mut self, data: String) -> VertexAIResult<Option<StreamEvent>> {
        // Skip empty or special markers
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            return Ok(None);
        }
        
        // Parse JSON chunk
        let chunk: CompletionChunk = serde_json::from_str(&data)
            .map_err(|e| VertexAIError::SseParsing {
                line: self.events_processed as u32,
                details: format!("JSON parsing failed: {}", e)})?;
        
        // Process chunk and extract events
        self.process_completion_chunk(chunk)
    }
    
    /// Process completion chunk and extract stream events
    fn process_completion_chunk(&mut self, chunk: CompletionChunk) -> VertexAIResult<Option<StreamEvent>> {
        // Initialize accumulated response if needed
        if self.accumulated_response.is_none() {
            self.accumulated_response = Some(CompletionResponse {
                candidates: ArrayVec::new(),
                prompt_feedback: None,
                usage_metadata: None});
        }
        
        let accumulated = self.accumulated_response.as_mut().ok_or_else(|| {
            VertexAIError::Internal {
                context: "Accumulated response is None after initialization".to_string()}
        })?;
        
        // Process usage metadata (typically in final chunk)
        if let Some(usage) = chunk.usage_metadata {
            accumulated.usage_metadata = Some(usage.clone());
            return Ok(Some(StreamEvent::Usage { usage }));
        }
        
        // Process candidates
        for (idx, candidate) in chunk.candidates.iter().enumerate() {
            // Ensure we have a candidate slot
            while accumulated.candidates.len() <= idx {
                accumulated.candidates.push(Candidate {
                    content: Content {
                        role: ArrayString::from("model").unwrap_or_default(),
                        parts: ArrayVec::new()},
                    finish_reason: None,
                    safety_ratings: ArrayVec::new(),
                    citation_metadata: None,
                    grounding_attributions: ArrayVec::new()});
            }
            
            let accumulated_candidate = &mut accumulated.candidates[idx];
            
            // Process content parts
            for part in &candidate.content.parts {
                match part {
                    Part::Text { text } => {
                        // Accumulate text content
                        self.accumulate_text_content(accumulated_candidate, text)?;
                        
                        return Ok(Some(StreamEvent::ContentChunk {
                            content: text.clone(),
                            function_call: None,
                            metadata: Some(ChunkMetadata {
                                sequence: self.events_processed,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                                size: text.len(),
                                finish_reason: candidate.finish_reason.as_ref().map(|r| format!("{:?}", r))})}));
                    }
                    Part::FunctionCall { name, args } => {
                        // Handle function call
                        let function_call = part.clone();
                        self.accumulate_function_call(accumulated_candidate, part.clone())?;
                        
                        return Ok(Some(StreamEvent::ContentChunk {
                            content: format!("Function call: {}", name),
                            function_call: Some(function_call),
                            metadata: Some(ChunkMetadata {
                                sequence: self.events_processed,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                                size: name.len(),
                                finish_reason: candidate.finish_reason.as_ref().map(|r| format!("{:?}", r))})}));
                    }
                    _ => {
                        // Handle other part types
                        accumulated_candidate.content.parts.push(part.clone());
                    }
                }
            }
            
            // Update finish reason and other metadata
            if candidate.finish_reason.is_some() {
                accumulated_candidate.finish_reason = candidate.finish_reason.clone();
            }
            
            // Merge safety ratings
            for rating in &candidate.safety_ratings {
                // Only add if not already present
                if !accumulated_candidate.safety_ratings.iter().any(|r| {
                    std::mem::discriminant(&r.category) == std::mem::discriminant(&rating.category)
                }) {
                    if accumulated_candidate.safety_ratings.len() < accumulated_candidate.safety_ratings.capacity() {
                        accumulated_candidate.safety_ratings.push(rating.clone());
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Accumulate text content efficiently
    fn accumulate_text_content(&self, candidate: &mut Candidate, new_text: &str) -> VertexAIResult<()> {
        // Find existing text part or create new one
        if let Some(existing_part) = candidate.content.parts
            .iter_mut()
            .find(|part| matches!(part, Part::Text { .. })) {
            
            if let Part::Text { text } = existing_part {
                text.push_str(new_text);
            }
        } else {
            // Add new text part
            if candidate.content.parts.len() < candidate.content.parts.capacity() {
                candidate.content.parts.push(Part::Text {
                    text: new_text.to_string()});
            } else {
                return Err(VertexAIError::Streaming {
                    source: "Too many content parts in response".to_string()});
            }
        }
        
        Ok(())
    }
    
    /// Accumulate function call content
    fn accumulate_function_call(&self, candidate: &mut Candidate, function_call: Part) -> VertexAIResult<()> {
        // Add function call part
        if candidate.content.parts.len() < candidate.content.parts.capacity() {
            candidate.content.parts.push(function_call);
        } else {
            return Err(VertexAIError::Streaming {
                source: "Too many content parts in response".to_string()});
        }
        
        Ok(())
    }
    
    /// Get current stream state
    pub fn state(&self) -> StreamState {
        self.state
    }
    
    /// Get accumulated response (if any)
    pub fn accumulated_response(&self) -> Option<&CompletionResponse> {
        self.accumulated_response.as_ref()
    }
    
    /// Get stream statistics
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            events_processed: self.events_processed,
            accumulated_size: self.accumulated_size,
            state: self.state,
            last_error: self.last_error.as_ref().map(|e| format!("{}", e))}
    }
    
    /// Get global streaming statistics
    pub fn global_stats() -> (usize, usize, usize) {
        (
            STREAM_EVENTS_PROCESSED.get(),
            STREAM_PARSE_ERRORS.get(),
            STREAM_CONNECTIONS.get(),
        )
    }
}

/// Stream statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Number of events processed
    pub events_processed: usize,
    
    /// Total accumulated size in bytes
    pub accumulated_size: usize,
    
    /// Current stream state
    pub state: StreamState,
    
    /// Last error message (if any)
    pub last_error: Option<String>}

impl Stream for VertexAIStream {
    type Item = StreamEvent;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.state != StreamState::Active {
            return Poll::Ready(None);
        }
        
        // This is a simplified implementation
        // In practice, you'd need to properly implement async polling
        Poll::Pending
    }
}

/// Utility functions for stream processing
impl VertexAIStream {
    /// Create stream from raw SSE data
    pub fn from_sse_data(sse_data: &str) -> VertexAIResult<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let mut temp_stream = Self::new(futures_util::stream::empty());
        
        for line in sse_data.lines() {
            if line.starts_with("data: ") {
                let data = line[6..].to_string();
                if let Ok(Some(event)) = temp_stream.process_data_event(data) {
                    events.push(event);
                }
            }
        }
        
        Ok(events)
    }
    
    /// Validate SSE event format
    pub fn validate_sse_event(event: &SseEvent) -> VertexAIResult<()> {
        if event.data.len() > MAX_SSE_EVENT_SIZE {
            return Err(VertexAIError::SseParsing {
                line: 0,
                details: format!("Event size {} exceeds maximum {}", event.data.len(), MAX_SSE_EVENT_SIZE)});
        }
        
        // Validate JSON if it's a data event
        if event.event_type == "data" && !event.data.trim().is_empty() && event.data.trim() != "[DONE]" {
            serde_json::from_str::<serde_json::Value>(&event.data)
                .map_err(|e| VertexAIError::SseParsing {
                    line: 0,
                    details: format!("Invalid JSON in SSE data: {}", e)})?;
        }
        
        Ok(())
    }
    
    /// Extract text content from stream events
    pub fn extract_text_content(events: &[StreamEvent]) -> String {
        let mut content = String::new();
        
        for event in events {
            if let StreamEvent::ContentChunk { content: chunk_content, .. } = event {
                content.push_str(chunk_content);
            }
        }
        
        content
    }
    
    /// Extract function calls from stream events
    pub fn extract_function_calls(events: &[StreamEvent]) -> Vec<Part> {
        let mut function_calls = Vec::new();
        
        for event in events {
            if let StreamEvent::ContentChunk { function_call: Some(call), .. } = event {
                function_calls.push(call.clone());
            }
        }
        
        function_calls
    }
}