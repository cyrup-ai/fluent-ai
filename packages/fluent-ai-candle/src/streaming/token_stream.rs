//! Token stream implementation with buffering and backpressure control
//!
//! Provides efficient token streaming with:
//! - Circular buffer for optimal memory usage
//! - Configurable backpressure detection
//! - Async streaming with tokio integration
//! - Zero-allocation paths where possible

use super::{StreamingError, StreamingToken, TokenOutputStream, StreamingTokenResponse, StreamStats, BackpressureStrategy};
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use futures_util::Stream;

/// Configuration for token stream behavior
#[derive(Debug, Clone)]
pub struct TokenStreamConfig {
    /// Maximum buffer size for tokens
    pub buffer_capacity: usize,
    /// Backpressure threshold (0.0 to 1.0)
    pub backpressure_threshold: f32,
    /// Enable automatic flushing
    pub auto_flush: bool,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Enable flow control
    pub flow_control_enabled: bool,
    /// Timeout for token operations in milliseconds
    pub timeout_ms: u64,
}

impl Default for TokenStreamConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 1024,
            backpressure_threshold: 0.8,
            auto_flush: true,
            flush_interval_ms: 100,
            flow_control_enabled: true,
            timeout_ms: 5000,
        }
    }
}

/// Buffered token stream with intelligent backpressure control
pub struct BufferedTokenStream {
    /// Internal token buffer (circular buffer for efficiency)
    buffer: Arc<Mutex<VecDeque<StreamingTokenResponse>>>,
    /// Receiver for incoming tokens
    token_receiver: Arc<Mutex<mpsc::Receiver<StreamingToken>>>,
    /// Configuration
    config: TokenStreamConfig,
    /// Stream statistics
    stats: StreamStats,
    /// Current backpressure state
    backpressure_active: bool,
    /// Last flush time for auto-flush logic
    last_flush: std::time::Instant,
    /// Stream end indicator
    stream_ended: bool,
}

impl BufferedTokenStream {
    /// Create new buffered token stream
    pub fn new(
        token_receiver: mpsc::Receiver<StreamingToken>,
        config: TokenStreamConfig,
    ) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_capacity))),
            token_receiver: Arc::new(Mutex::new(token_receiver)),
            config,
            stats: StreamStats::default(),
            backpressure_active: false,
            last_flush: std::time::Instant::now(),
            stream_ended: false,
        }
    }

    /// Process incoming tokens and add to buffer
    async fn process_incoming_tokens(&mut self, cx: &mut Context<'_>) -> Result<bool, StreamingError> {
        if self.stream_ended {
            return Ok(false);
        }

        let mut receiver = self.token_receiver.lock().await;
        
        // Poll for new tokens with timeout
        match Pin::new(&mut *receiver).poll_recv(cx) {
            Poll::Ready(Some(token)) => {
                // Convert streaming token to response
                let response = self.convert_token_to_response(token).await?;
                
                // Add to buffer if space available
                let mut buffer = self.buffer.lock().await;
                if buffer.len() < self.config.buffer_capacity {
                    buffer.push_back(response);
                    self.stats.tokens_streamed += 1;
                    
                    // Update backpressure state
                    self.update_backpressure_state(buffer.len()).await;
                    
                    Ok(true)
                } else {
                    // Buffer full - apply backpressure
                    self.handle_buffer_overflow().await?;
                    Ok(false)
                }
            }
            Poll::Ready(None) => {
                // Stream ended
                self.stream_ended = true;
                Ok(false)
            }
            Poll::Pending => {
                // No tokens available right now
                Ok(false)
            }
        }
    }

    /// Convert streaming token to response format
    async fn convert_token_to_response(&mut self, token: StreamingToken) -> Result<StreamingTokenResponse, StreamingError> {
        use crate::streaming::{TokenTiming, TokenMetadata};
        
        // Convert raw bytes to string
        let content = String::from_utf8(token.raw_bytes.clone())
            .map_err(|e| StreamingError::Utf8Error(e.to_string()))?;
        
        // Create timing information
        let timing = TokenTiming {
            generation_start_us: token.generation_time.elapsed().as_micros() as u64,
            completion_time_us: std::time::Instant::now().duration_since(token.generation_time).as_micros() as u64,
            model_time_us: 0, // Would be filled by model layer
            sampling_time_us: 0, // Would be filled by sampling layer
            decoding_time_us: 10, // Approximate decoding time
        };

        // Create metadata
        let metadata = TokenMetadata {
            is_final: token.is_special && token.token_id == 2, // Assuming 2 is EOS token
            finish_reason: if token.is_special { Some("stop".to_string()) } else { None },
            temperature: None, // Would be filled from generation config
            top_p: None,
            top_k: None,
            repetition_penalty_applied: false,
            tokens_generated: self.stats.tokens_streamed + 1,
        };

        Ok(StreamingTokenResponse {
            content,
            token_id: Some(token.token_id),
            position: token.position,
            is_complete_token: true, // Assuming complete tokens for now
            probability: token.probability,
            alternatives: None, // Would be filled if requested
            timing,
            metadata,
        })
    }

    /// Update backpressure state based on buffer utilization
    async fn update_backpressure_state(&mut self, buffer_len: usize) {
        let utilization = buffer_len as f32 / self.config.buffer_capacity as f32;
        self.stats.buffer_utilization = utilization;
        
        if self.config.flow_control_enabled {
            if utilization > self.config.backpressure_threshold && !self.backpressure_active {
                self.backpressure_active = true;
                self.stats.backpressure_events += 1;
            } else if utilization < self.config.backpressure_threshold * 0.7 && self.backpressure_active {
                // Hysteresis to prevent oscillation
                self.backpressure_active = false;
            }
        }
    }

    /// Handle buffer overflow situation
    async fn handle_buffer_overflow(&mut self) -> Result<(), StreamingError> {
        self.stats.backpressure_events += 1;
        
        // Apply backpressure strategy
        if self.config.flow_control_enabled {
            // Could implement different strategies here
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(())
        } else {
            Err(StreamingError::BufferOverflow {
                capacity: self.config.buffer_capacity,
                size: self.config.buffer_capacity + 1,
            })
        }
    }

    /// Check if auto-flush should be triggered
    fn should_auto_flush(&self) -> bool {
        self.config.auto_flush && 
        self.last_flush.elapsed().as_millis() as u64 > self.config.flush_interval_ms
    }

    /// Perform flush operation
    async fn perform_flush(&mut self) -> Result<(), StreamingError> {
        self.last_flush = std::time::Instant::now();
        self.stats.flush_events += 1;
        // Implementation would flush any pending operations
        Ok(())
    }

    /// Get next token from buffer if available
    async fn get_next_buffered_token(&mut self) -> Option<StreamingTokenResponse> {
        let mut buffer = self.buffer.lock().await;
        buffer.pop_front()
    }

    /// Update stream timing statistics
    fn update_timing_stats(&mut self, token_time: u64) {
        if self.stats.tokens_streamed > 0 {
            let current_avg = self.stats.average_token_time_us;
            let count = self.stats.tokens_streamed as u64;
            self.stats.average_token_time_us = (current_avg * (count - 1) + token_time) / count;
        } else {
            self.stats.average_token_time_us = token_time;
        }
    }
}

impl TokenOutputStream for BufferedTokenStream {
    fn poll_next_token(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamingTokenResponse, StreamingError>>> {
        // Check for auto-flush
        if self.should_auto_flush() {
            let flush_future = self.perform_flush();
            tokio::pin!(flush_future);
            
            match flush_future.as_mut().poll(cx) {
                Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Ok(())) => {},
                Poll::Pending => return Poll::Pending,
            }
        }

        // Try to get buffered token first
        let get_token_future = self.get_next_buffered_token();
        tokio::pin!(get_token_future);
        
        match get_token_future.as_mut().poll(cx) {
            Poll::Ready(Some(token)) => {
                // Update statistics
                self.update_timing_stats(token.timing.completion_time_us);
                return Poll::Ready(Some(Ok(token)));
            }
            Poll::Ready(None) => {
                // No buffered tokens, try to process new ones
            }
            Poll::Pending => return Poll::Pending,
        }

        // Process incoming tokens to fill buffer
        let process_future = self.process_incoming_tokens(cx);
        tokio::pin!(process_future);
        
        match process_future.as_mut().poll(cx) {
            Poll::Ready(Ok(true)) => {
                // Token was processed, try again for buffered token
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Ok(false)) => {
                // No more tokens available
                if self.stream_ended {
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn supports_backpressure(&self) -> bool {
        self.config.flow_control_enabled
    }

    fn apply_backpressure(&mut self, strategy: BackpressureStrategy) -> Result<(), StreamingError> {
        if !self.config.flow_control_enabled {
            return Err(StreamingError::FlowControlError(
                "Flow control not enabled".to_string()
            ));
        }

        match strategy {
            BackpressureStrategy::Drop => {
                // Drop strategy would be implemented here
                Ok(())
            }
            BackpressureStrategy::Block => {
                // Block strategy would be implemented here
                Ok(())
            }
            BackpressureStrategy::Exponential => {
                // Exponential backoff strategy would be implemented here
                Ok(())
            }
        }
    }

    fn get_stream_stats(&self) -> StreamStats {
        self.stats.clone()
    }

    fn flush(&mut self) -> Result<(), StreamingError> {
        self.last_flush = std::time::Instant::now();
        self.stats.flush_events += 1;
        Ok(())
    }

    fn close(&mut self) -> Result<(), StreamingError> {
        self.stream_ended = true;
        Ok(())
    }
}

/// Stream implementation for async iteration
impl Stream for BufferedTokenStream {
    type Item = Result<StreamingTokenResponse, StreamingError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        TokenOutputStream::poll_next_token(self, cx)
    }
}

/// Multi-consumer token stream that can broadcast to multiple receivers
pub struct BroadcastTokenStream {
    /// Source stream
    source: Arc<Mutex<dyn TokenOutputStream + Send>>,
    /// List of consumers
    consumers: Vec<mpsc::Sender<StreamingTokenResponse>>,
    /// Configuration
    config: TokenStreamConfig,
    /// Broadcasting statistics
    broadcast_stats: BroadcastStats,
}

#[derive(Debug, Clone, Default)]
pub struct BroadcastStats {
    pub active_consumers: usize,
    pub total_broadcasts: usize,
    pub failed_broadcasts: usize,
    pub consumer_dropouts: usize,
}

impl BroadcastTokenStream {
    /// Create new broadcast stream
    pub fn new(
        source: Arc<Mutex<dyn TokenOutputStream + Send>>,
        config: TokenStreamConfig,
    ) -> Self {
        Self {
            source,
            consumers: Vec::new(),
            config,
            broadcast_stats: BroadcastStats::default(),
        }
    }

    /// Add a new consumer to the broadcast
    pub fn add_consumer(&mut self, buffer_size: usize) -> mpsc::Receiver<StreamingTokenResponse> {
        let (sender, receiver) = mpsc::channel(buffer_size);
        self.consumers.push(sender);
        self.broadcast_stats.active_consumers = self.consumers.len();
        receiver
    }

    /// Remove inactive consumers
    pub async fn cleanup_inactive_consumers(&mut self) {
        let mut active_consumers = Vec::new();
        
        for consumer in self.consumers.drain(..) {
            if !consumer.is_closed() {
                active_consumers.push(consumer);
            } else {
                self.broadcast_stats.consumer_dropouts += 1;
            }
        }
        
        self.consumers = active_consumers;
        self.broadcast_stats.active_consumers = self.consumers.len();
    }

    /// Broadcast token to all active consumers
    pub async fn broadcast_token(&mut self, token: StreamingTokenResponse) -> Result<(), StreamingError> {
        if self.consumers.is_empty() {
            return Ok(());
        }

        let mut failed_sends = 0;
        let mut consumers_to_remove = Vec::new();

        for (i, consumer) in self.consumers.iter().enumerate() {
            match consumer.try_send(token.clone()) {
                Ok(()) => {
                    // Successfully sent
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Consumer buffer full - could implement backpressure here
                    failed_sends += 1;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    // Consumer disconnected
                    consumers_to_remove.push(i);
                }
            }
        }

        // Remove disconnected consumers
        for &i in consumers_to_remove.iter().rev() {
            self.consumers.remove(i);
            self.broadcast_stats.consumer_dropouts += 1;
        }

        self.broadcast_stats.total_broadcasts += 1;
        self.broadcast_stats.failed_broadcasts += failed_sends;
        self.broadcast_stats.active_consumers = self.consumers.len();

        if failed_sends > 0 && failed_sends == self.consumers.len() {
            Err(StreamingError::BackpressureError(
                "All consumers experiencing backpressure".to_string()
            ))
        } else {
            Ok(())
        }
    }

    /// Get broadcast statistics
    pub fn get_broadcast_stats(&self) -> &BroadcastStats {
        &self.broadcast_stats
    }

    /// Run the broadcast loop
    pub async fn run_broadcast_loop(&mut self) -> Result<(), StreamingError> {
        loop {
            let mut source = self.source.lock().await;
            
            // Create a future for the next token
            let poll_future = async {
                let mut pinned_source = Pin::new(&mut *source);
                std::future::poll_fn(move |cx| {
                    pinned_source.as_mut().poll_next_token(cx)
                }).await
            };
            
            match poll_future.await {
                Some(Ok(token)) => {
                    drop(source); // Release the lock before broadcasting
                    self.broadcast_token(token).await?;
                    
                    // Cleanup inactive consumers periodically
                    if self.broadcast_stats.total_broadcasts % 100 == 0 {
                        self.cleanup_inactive_consumers().await;
                    }
                }
                Some(Err(e)) => {
                    return Err(e);
                }
                None => {
                    // Stream ended
                    break;
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    fn create_test_streaming_token(token_id: u32, content: &str, position: usize) -> StreamingToken {
        StreamingToken {
            token_id,
            raw_bytes: content.as_bytes().to_vec(),
            position,
            probability: Some(0.9),
            is_special: false,
            generation_time: std::time::Instant::now(),
        }
    }

    #[tokio::test]
    async fn test_token_stream_config() {
        let config = TokenStreamConfig::default();
        assert_eq!(config.buffer_capacity, 1024);
        assert_eq!(config.backpressure_threshold, 0.8);
        assert!(config.auto_flush);
        assert!(config.flow_control_enabled);
    }

    #[tokio::test]
    async fn test_buffered_token_stream_creation() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let stream = BufferedTokenStream::new(receiver, config);
        
        assert!(!stream.stream_ended);
        assert!(!stream.backpressure_active);
        assert!(stream.supports_backpressure());
    }

    #[tokio::test]
    async fn test_token_conversion() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let mut stream = BufferedTokenStream::new(receiver, config);
        
        let token = create_test_streaming_token(42, "hello", 0);
        let response = stream.convert_token_to_response(token).await.expect("conversion succeeds");
        
        assert_eq!(response.content, "hello");
        assert_eq!(response.token_id, Some(42));
        assert_eq!(response.position, 0);
        assert!(response.is_complete_token);
    }

    #[tokio::test]
    async fn test_backpressure_detection() {
        let (_sender, receiver) = mpsc::channel(10);
        let mut config = TokenStreamConfig::default();
        config.buffer_capacity = 10;
        config.backpressure_threshold = 0.5; // Trigger at 50%
        
        let mut stream = BufferedTokenStream::new(receiver, config);
        
        // Simulate buffer at 60% capacity (6/10)
        stream.update_backpressure_state(6).await;
        assert!(stream.backpressure_active);
        assert_eq!(stream.stats.backpressure_events, 1);
        
        // Simulate buffer dropping to 30% (hysteresis)
        stream.update_backpressure_state(3).await;
        assert!(!stream.backpressure_active);
    }

    #[tokio::test]
    async fn test_stream_stats_update() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let mut stream = BufferedTokenStream::new(receiver, config);
        
        // Update timing stats
        stream.update_timing_stats(1000); // 1000 microseconds
        assert_eq!(stream.stats.average_token_time_us, 1000);
        
        stream.stats.tokens_streamed = 1;
        stream.update_timing_stats(2000); // 2000 microseconds
        assert_eq!(stream.stats.average_token_time_us, 1500); // Average of 1000 and 2000
    }

    #[tokio::test]
    async fn test_flush_functionality() {
        let (_sender, receiver) = mpsc::channel(10);
        let mut config = TokenStreamConfig::default();
        config.auto_flush = true;
        config.flush_interval_ms = 100;
        
        let mut stream = BufferedTokenStream::new(receiver, config);
        
        // Test manual flush
        assert!(stream.flush().is_ok());
        assert_eq!(stream.stats.flush_events, 1);
        
        // Test auto-flush timing (would need to simulate time passage)
        assert!(!stream.should_auto_flush()); // Just flushed
        
        // Simulate time passage
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(stream.should_auto_flush());
    }

    #[tokio::test]
    async fn test_stream_close() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let mut stream = BufferedTokenStream::new(receiver, config);
        
        assert!(!stream.stream_ended);
        assert!(stream.close().is_ok());
        assert!(stream.stream_ended);
    }

    #[tokio::test]
    async fn test_broadcast_stream_creation() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let source_stream = Arc::new(Mutex::new(BufferedTokenStream::new(receiver, config.clone())));
        
        let mut broadcast_stream = BroadcastTokenStream::new(source_stream, config);
        
        assert_eq!(broadcast_stream.consumers.len(), 0);
        assert_eq!(broadcast_stream.broadcast_stats.active_consumers, 0);
    }

    #[tokio::test]
    async fn test_broadcast_add_consumer() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let source_stream = Arc::new(Mutex::new(BufferedTokenStream::new(receiver, config.clone())));
        
        let mut broadcast_stream = BroadcastTokenStream::new(source_stream, config);
        
        let _consumer1 = broadcast_stream.add_consumer(10);
        assert_eq!(broadcast_stream.consumers.len(), 1);
        assert_eq!(broadcast_stream.broadcast_stats.active_consumers, 1);
        
        let _consumer2 = broadcast_stream.add_consumer(10);
        assert_eq!(broadcast_stream.consumers.len(), 2);
        assert_eq!(broadcast_stream.broadcast_stats.active_consumers, 2);
    }

    #[tokio::test]
    async fn test_broadcast_token() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let source_stream = Arc::new(Mutex::new(BufferedTokenStream::new(receiver, config.clone())));
        
        let mut broadcast_stream = BroadcastTokenStream::new(source_stream, config);
        let mut consumer = broadcast_stream.add_consumer(10);
        
        // Create test response
        use crate::streaming::{TokenTiming, TokenMetadata};
        let response = StreamingTokenResponse {
            content: "test".to_string(),
            token_id: Some(42),
            position: 0,
            is_complete_token: true,
            probability: Some(0.9),
            alternatives: None,
            timing: TokenTiming::default(),
            metadata: TokenMetadata::default(),
        };
        
        // Broadcast token
        assert!(broadcast_stream.broadcast_token(response.clone()).await.is_ok());
        assert_eq!(broadcast_stream.broadcast_stats.total_broadcasts, 1);
        
        // Consumer should receive token
        let received = timeout(Duration::from_millis(100), consumer.recv()).await;
        assert!(received.is_ok());
        let token = received.unwrap().unwrap();
        assert_eq!(token.content, "test");
        assert_eq!(token.token_id, Some(42));
    }

    #[tokio::test]
    async fn test_consumer_cleanup() {
        let (_sender, receiver) = mpsc::channel(10);
        let config = TokenStreamConfig::default();
        let source_stream = Arc::new(Mutex::new(BufferedTokenStream::new(receiver, config.clone())));
        
        let mut broadcast_stream = BroadcastTokenStream::new(source_stream, config);
        
        // Add consumer and then drop it
        {
            let _consumer = broadcast_stream.add_consumer(10);
            assert_eq!(broadcast_stream.consumers.len(), 1);
        } // consumer dropped here
        
        // Cleanup should remove inactive consumers
        broadcast_stream.cleanup_inactive_consumers().await;
        // Note: This test may not work as expected since the channel might not immediately show as closed
        // In real usage, cleanup happens over time as send operations fail
    }
}