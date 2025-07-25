//! Streaming message formatter with atomic operations
//!
//! Provides zero-allocation, lock-free streaming formatting with atomic
//! state tracking and comprehensive event emission.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use fluent_ai_async::{AsyncStream, AsyncStreamSender};

use super::content::ImmutableMessageContent;
use super::error::{FormatError, FormatResult};
use super::options::ImmutableFormatOptions;

/// Formatting event for streaming operations
#[derive(Debug, Clone)]
pub enum FormattingEvent {

    /// Formatting started
    Started {
        content_id: u64,
        content_type: String,
        timestamp_nanos: u64},
    /// Formatting progress
    Progress {
        content_id: u64,
        progress_percent: f32,
        stage: String},
    /// Formatting completed
    Completed {
        content_id: u64,
        result: ImmutableMessageContent,
        duration_nanos: u64},
    /// Formatting failed
    Failed {
        content_id: u64,
        error: FormatError,
        duration_nanos: u64},
    /// Partial result available
    PartialResult {
        content_id: u64,
        partial_content: String}}

/// Streaming message formatter with atomic state tracking
pub struct StreamingMessageFormatter {
    /// Content counter (atomic)
    content_counter: AtomicU64,
    /// Active formatting operations (atomic)
    active_operations: AtomicUsize,
    /// Total operations (atomic)
    total_operations: AtomicU64,
    /// Successful operations (atomic)
    successful_operations: AtomicU64,
    /// Failed operations (atomic)
    failed_operations: AtomicU64,
    /// Event stream sender
    event_sender: Option<AsyncStreamSender<FormattingEvent>>,
    /// Formatter configuration
    options: ImmutableFormatOptions}

impl std::fmt::Debug for StreamingMessageFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingMessageFormatter")
            .field(
                "content_counter",
                &self
                    .content_counter
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "active_operations",
                &self
                    .active_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "total_operations",
                &self
                    .total_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "successful_operations",
                &self
                    .successful_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "failed_operations",
                &self
                    .failed_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("event_sender", &self.event_sender.is_some())
            .field("options", &self.options)
            .finish()
    }
}

impl StreamingMessageFormatter {
    /// Create new streaming message formatter
    #[inline]
    pub fn new(options: ImmutableFormatOptions) -> FormatResult<Self> {
        options.validate()?;
        Ok(Self {
            content_counter: AtomicU64::new(0),
            active_operations: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            event_sender: None,
            options})
    }

    /// Create formatter with event streaming
    #[inline]
    pub fn with_streaming(
        options: ImmutableFormatOptions,
    ) -> FormatResult<(Self, AsyncStream<FormattingEvent>)> {
        options.validate()?;
        let stream = AsyncStream::with_channel(|_sender| {
            // Stream is created but not used directly
        });
        let formatter = Self {
            content_counter: AtomicU64::new(0),
            active_operations: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            event_sender: None, // Will be set up separately if needed
            options};
        Ok((formatter, stream))
    }

    /// Format content with streaming events
    #[inline]
    pub fn format_content(&self, content: &ImmutableMessageContent) -> FormatResult<u64> {
        // Validate content first
        content.validate()?;

        // Generate content ID
        let content_id = self.content_counter.fetch_add(1, Ordering::Relaxed);

        // Update counters
        self.active_operations.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);

        // Send started event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(FormattingEvent::Started {
                content_id,
                content_type: content.content_type().to_string(),
                timestamp_nanos: Self::current_timestamp_nanos()});
        }

        // TODO: Implement actual formatting logic here
        // This would integrate with markdown parsers, syntax highlighters, etc.

        Ok(content_id)
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get formatting statistics (atomic reads)
    #[inline]
    pub fn stats(&self) -> FormatterStats {
        FormatterStats {
            active_operations: self.active_operations.load(Ordering::Relaxed) as u64,
            total_operations: self.total_operations.load(Ordering::Relaxed),
            successful_operations: self.successful_operations.load(Ordering::Relaxed),
            failed_operations: self.failed_operations.load(Ordering::Relaxed)}
    }

    /// Get formatter options (borrowed reference)
    #[inline]
    pub fn options(&self) -> &ImmutableFormatOptions {
        &self.options
    }

    /// Update formatter options
    #[inline]
    pub fn update_options(&mut self, options: ImmutableFormatOptions) -> FormatResult<()> {
        options.validate()?;
        self.options = options;
        Ok(())
    }
    
    /// Get color scheme from options
    #[inline]
    pub fn color_scheme(&self) -> &super::themes::ImmutableColorScheme {
        self.options.color_scheme()
    }
    
    /// Get format rules from options
    #[inline]
    pub fn format_rules(&self) -> &[super::styles::ImmutableCustomFormatRule] {
        self.options.format_rules()
    }
    
    /// Get syntax theme
    #[inline]
    pub fn syntax_theme(&self) -> super::themes::SyntaxTheme {
        self.options.syntax_theme()
    }
    
    /// Get output format
    #[inline]
    pub fn output_format(&self) -> super::themes::OutputFormat {
        self.options.output_format()
    }
}

/// Formatter statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatterStats {
    pub active_operations: u64,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64}

impl FormatterStats {
    /// Calculate success rate as percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_operations + self.failed_operations;
        if completed == 0 {
            0.0
        } else {
            (self.successful_operations as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }
}

/// Legacy compatibility alias for StreamingMessageFormatter
#[deprecated(note = "Use StreamingMessageFormatter instead for zero-allocation streaming")]
pub type MessageFormatter = StreamingMessageFormatter;

// Explicit Send + Sync implementations for thread safety
// Send + Sync are automatically derived since all fields implement Send + Sync

// FormattingEvent is Send + Sync safe because:
// - All fields are either primitives, strings, or error types
// - ImmutableMessageContent is designed to be thread-safe
// - FormatError is Send + Sync

// StreamingMessageFormatter is Send + Sync safe because:
// - Uses atomic operations for all counters
// - AsyncStreamSender is designed to be thread-safe
// - ImmutableFormatOptions uses only owned data