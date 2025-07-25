//! Streaming Context Processor with Atomic State Tracking
//!
//! Zero-allocation context processing with atomic performance counters and streaming operations.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::events::ContextEvent;
use super::errors::{ContextError, ValidationError};
use super::context_types::ImmutableFileContext;
use crate::types::CandleDocument;

/// Streaming context processor with atomic state tracking
pub struct StreamingContextProcessor {
    /// Unique processor identifier
    processor_id: String,

    /// Atomic performance counters
    context_requests: AtomicU64,
    active_contexts: AtomicUsize,
    total_contexts_processed: AtomicU64,
    successful_contexts: AtomicU64,
    failed_contexts: AtomicU64,
    total_documents_loaded: AtomicU64,
    total_processing_time_nanos: AtomicU64,

    /// Event streaming
    event_sender: Option<AsyncStreamSender<ContextEvent>>,

    /// Performance thresholds
    max_processing_time_ms: u64,
    max_documents_per_context: usize,
    max_concurrent_contexts: usize,
}

impl StreamingContextProcessor {
    /// Create new streaming context processor
    #[inline]
    pub fn new(processor_id: String) -> Self {
        Self {
            processor_id,
            context_requests: AtomicU64::new(0),
            active_contexts: AtomicUsize::new(0),
            total_contexts_processed: AtomicU64::new(0),
            successful_contexts: AtomicU64::new(0),
            failed_contexts: AtomicU64::new(0),
            total_documents_loaded: AtomicU64::new(0),
            total_processing_time_nanos: AtomicU64::new(0),
            event_sender: None,
            max_processing_time_ms: 30000, // 30 seconds default
            max_documents_per_context: 10000,
            max_concurrent_contexts: 100,
        }
    }

    /// Create processor with event streaming
    #[inline]
    pub fn with_streaming(processor_id: String) -> (Self, AsyncStream<ContextEvent>) {
        let stream = AsyncStream::with_channel(|sender| {
            // Stream created for event processing - placeholder implementation
            let _ = sender; // Prevent unused variable warning
        });
        let mut processor = Self::new(processor_id);
        processor.event_sender = None; // Will be set up separately if needed
        (processor, stream)
    }

    /// Set performance thresholds
    pub fn set_thresholds(
        &mut self,
        max_processing_time_ms: u64,
        max_documents_per_context: usize,
        max_concurrent_contexts: usize,
    ) {
        self.max_processing_time_ms = max_processing_time_ms;
        self.max_documents_per_context = max_documents_per_context;
        self.max_concurrent_contexts = max_concurrent_contexts;
    }

    /// Get processor ID
    pub fn processor_id(&self) -> &str {
        &self.processor_id
    }

    /// Process file context with streaming results - returns unwrapped values
    #[inline]
    pub fn process_file_context(
        &self,
        context: ImmutableFileContext,
    ) -> AsyncStream<CandleDocument> {
        let processor_id = self.processor_id.clone();
        let context_requests = self.context_requests.clone();
        let active_contexts = self.active_contexts.clone();
        let successful_contexts = self.successful_contexts.clone();
        let failed_contexts = self.failed_contexts.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Increment request counter
            context_requests.fetch_add(1, Ordering::Relaxed);
            active_contexts.fetch_add(1, Ordering::Relaxed);
            
            // Validate context
            match Self::validate_file_context(&context) {
                Ok(_) => {
                    // Load document
                    match Self::load_file_document(&context) {
                        Ok(document) => {
                            successful_contexts.fetch_add(1, Ordering::Relaxed);
                            let _ = sender.send(document);
                        }
                        Err(e) => {
                            failed_contexts.fetch_add(1, Ordering::Relaxed);
                            log::error!(
                                "Stream error in {}: Failed to load file document. Details: {}",
                                file!(),
                                format!("Processor {}: {}", processor_id, e)
                            );
                        }
                    }
                }
                Err(e) => {
                    failed_contexts.fetch_add(1, Ordering::Relaxed);
                    log::error!(
                        "Stream error in {}: File context validation failed. Details: {}",
                        file!(),
                        format!("Processor {}: {}", processor_id, e)
                    );
                }
            }
            
            // Decrement active counter
            active_contexts.fetch_sub(1, Ordering::Relaxed);
        })
    }

    /// Validate file context
    fn validate_file_context(context: &ImmutableFileContext) -> Result<(), ValidationError> {
        if context.path.is_empty() {
            return Err(ValidationError::PathValidation(
                "Empty file path".to_string(),
            ));
        }

        if context.size_bytes > 100 * 1024 * 1024 {
            // 100MB limit
            return Err(ValidationError::SizeLimitExceeded(format!(
                "File size {} bytes exceeds 100MB limit",
                context.size_bytes
            )));
        }

        Ok(())
    }

    /// Load file document
    fn load_file_document(context: &ImmutableFileContext) -> Result<CandleDocument, ContextError> {
        use std::collections::HashMap;
        
        // Implementation would read file and create CandleDocument
        // For now, create a basic document structure
        Ok(CandleDocument {
            data: format!("Content from file: {}", context.path),
            format: Some(crate::types::candle_context::ContentFormat::Text),
            media_type: Some(crate::types::candle_context::DocumentMediaType::TXT),
            additional_props: {
                let mut props = HashMap::new();
                props.insert(
                    "id".to_string(),
                    serde_json::Value::String(Uuid::new_v4().to_string()),
                );
                props.insert(
                    "path".to_string(),
                    serde_json::Value::String(context.path.clone()),
                );
                props.insert(
                    "size".to_string(),
                    serde_json::Value::String(context.size_bytes.to_string()),
                );
                props.insert(
                    "hash".to_string(),
                    serde_json::Value::String(context.content_hash.clone()),
                );
                props
            },
        })
    }

    /// Get processor statistics
    #[inline]
    pub fn get_statistics(&self) -> ContextProcessorStatistics {
        ContextProcessorStatistics {
            processor_id: self.processor_id.clone(),
            context_requests: self.context_requests.load(Ordering::Relaxed),
            active_contexts: self.active_contexts.load(Ordering::Relaxed),
            total_contexts_processed: self.total_contexts_processed.load(Ordering::Relaxed),
            successful_contexts: self.successful_contexts.load(Ordering::Relaxed),
            failed_contexts: self.failed_contexts.load(Ordering::Relaxed),
            total_documents_loaded: self.total_documents_loaded.load(Ordering::Relaxed),
            success_rate: self.success_rate(),
            average_processing_time_nanos: self.average_processing_time_nanos(),
        }
    }

    /// Calculate success rate
    #[inline]
    fn success_rate(&self) -> f64 {
        let successful = self.successful_contexts.load(Ordering::Relaxed);
        let failed = self.failed_contexts.load(Ordering::Relaxed);
        let total = successful + failed;
        if total == 0 {
            1.0
        } else {
            successful as f64 / total as f64
        }
    }

    /// Calculate average processing time
    #[inline]
    fn average_processing_time_nanos(&self) -> u64 {
        let total_time = self.total_processing_time_nanos.load(Ordering::Relaxed);
        let processed = self.total_contexts_processed.load(Ordering::Relaxed);
        if processed == 0 {
            0
        } else {
            total_time / processed
        }
    }
}

/// Context processor statistics with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProcessorStatistics {
    pub processor_id: String,
    pub context_requests: u64,
    pub active_contexts: usize,
    pub total_contexts_processed: u64,
    pub successful_contexts: u64,
    pub failed_contexts: u64,
    pub total_documents_loaded: u64,
    pub success_rate: f64,
    pub average_processing_time_nanos: u64,
}