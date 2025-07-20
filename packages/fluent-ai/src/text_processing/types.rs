//! Core types and constants for text processing
//!
//! Provides blazing-fast, zero-allocation types and constants optimized for SIMD operations
//! with production-ready performance and ergonomic APIs.

use arrayvec::ArrayVec;
use thiserror::Error;

/// Standard text processing dimensions optimized for SIMD
pub const MAX_TOKEN_LENGTH: usize = 128;
pub const MAX_TOKENS_PER_BATCH: usize = 512;
pub const MAX_PATTERN_LENGTH: usize = 64;
pub const MAX_PATTERNS_PER_SET: usize = 16;
pub const TEXT_BUFFER_SIZE: usize = 4096;

/// Token types with explicit discriminants for performance
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Word = 0,
    Number = 1,
    Punctuation = 2,
    Whitespace = 3,
    Symbol = 4,
    Unknown = 5,
}

/// Zero-allocation token with stack-allocated content
#[derive(Debug, Clone)]
pub struct Token {
    content: ArrayVec<u8, MAX_TOKEN_LENGTH>,
    token_type: TokenType,
    start_offset: usize,
    end_offset: usize,
}

impl Token {
    /// Create new token with zero allocation
    #[inline(always)]
    pub fn new(
        content: &[u8],
        token_type: TokenType,
        start: usize,
        end: usize,
    ) -> Result<Self, TextProcessingError> {
        if content.len() > MAX_TOKEN_LENGTH {
            return Err(TextProcessingError::TokenTooLarge(content.len()));
        }

        let mut token_content = ArrayVec::new();
        token_content
            .try_extend_from_slice(content)
            .map_err(|_| TextProcessingError::TokenTooLarge(content.len()))?;

        Ok(Self {
            content: token_content,
            token_type,
            start_offset: start,
            end_offset: end,
        })
    }

    /// Get token content as string slice
    #[inline(always)]
    pub fn as_str(&self) -> Result<&str, TextProcessingError> {
        std::str::from_utf8(&self.content).map_err(|_| TextProcessingError::InvalidUtf8)
    }

    /// Get token type
    #[inline(always)]
    pub fn token_type(&self) -> TokenType {
        self.token_type
    }

    /// Get token offsets
    #[inline(always)]
    pub fn offsets(&self) -> (usize, usize) {
        (self.start_offset, self.end_offset)
    }

    /// Get token length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if token is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get raw content bytes
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        &self.content
    }
}

/// Text analysis statistics
#[derive(Debug, Clone, PartialEq)]
pub struct TextStats {
    pub character_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub average_word_length: f32,
    pub reading_level: f32,
    pub complexity_score: f32,
}

impl Default for TextStats {
    fn default() -> Self {
        Self {
            character_count: 0,
            word_count: 0,
            sentence_count: 0,
            paragraph_count: 0,
            average_word_length: 0.0,
            reading_level: 0.0,
            complexity_score: 0.0,
        }
    }
}

/// Pattern matching result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_id: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    pub matched_text: ArrayVec<u8, MAX_PATTERN_LENGTH>,
    pub confidence: f32,
}

impl PatternMatch {
    /// Create new pattern match
    #[inline(always)]
    pub fn new(
        pattern_id: usize,
        start_offset: usize,
        end_offset: usize,
        matched_text: &[u8],
        confidence: f32,
    ) -> Result<Self, TextProcessingError> {
        if matched_text.len() > MAX_PATTERN_LENGTH {
            return Err(TextProcessingError::PatternTooLarge(matched_text.len()));
        }

        let mut text_content = ArrayVec::new();
        text_content
            .try_extend_from_slice(matched_text)
            .map_err(|_| TextProcessingError::PatternTooLarge(matched_text.len()))?;

        Ok(Self {
            pattern_id,
            start_offset,
            end_offset,
            matched_text: text_content,
            confidence,
        })
    }

    /// Get matched text as string
    #[inline(always)]
    pub fn as_str(&self) -> Result<&str, TextProcessingError> {
        std::str::from_utf8(&self.matched_text).map_err(|_| TextProcessingError::InvalidUtf8)
    }

    /// Get match length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.end_offset - self.start_offset
    }

    /// Check if match is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.start_offset == self.end_offset
    }
}

/// Performance metrics for text processing operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub total_operations: u64,
    pub total_processing_time_nanos: u64,
    pub average_tokenization_time_nanos: u64,
    pub average_pattern_matching_time_nanos: u64,
    pub average_analysis_time_nanos: u64,
    pub simd_operations_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl PerformanceStats {
    /// Calculate cache hit ratio
    #[inline(always)]
    pub fn cache_hit_ratio(&self) -> f32 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        }
    }

    /// Calculate average operation time
    #[inline(always)]
    pub fn average_operation_time_nanos(&self) -> u64 {
        if self.total_operations == 0 {
            0
        } else {
            self.total_processing_time_nanos / self.total_operations
        }
    }
}

/// Text processing error types
#[derive(Error, Debug, Clone)]
pub enum TextProcessingError {
    #[error("Token too large: {0} bytes (max: {MAX_TOKEN_LENGTH})")]
    TokenTooLarge(usize),

    #[error("Pattern too large: {0} bytes (max: {MAX_PATTERN_LENGTH})")]
    PatternTooLarge(usize),

    #[error("Invalid UTF-8 sequence")]
    InvalidUtf8,

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("SIMD operation failed: {0}")]
    SimdError(String),

    #[error("Buffer overflow: attempted to write {0} bytes")]
    BufferOverflow(usize),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Operation timeout after {0}ms")]
    Timeout(u64),
}

/// Result type for text processing operations
pub type TextProcessingResult<T> = Result<T, TextProcessingError>;

/// Configuration for text processing operations
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Maximum processing time in milliseconds
    pub timeout_ms: u64,
    /// Enable performance monitoring
    pub enable_metrics: bool,
    /// Cache size for pattern matching
    pub cache_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            timeout_ms: 5000,
            enable_metrics: true,
            cache_size: 1024,
            batch_size: MAX_TOKENS_PER_BATCH,
        }
    }
}

/// Text processing context for operations
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Configuration settings
    pub config: ProcessingConfig,
    /// Operation start time
    pub start_time: std::time::Instant,
    /// Current operation ID
    pub operation_id: u64,
}

impl ProcessingContext {
    /// Create new processing context
    #[inline(always)]
    pub fn new(config: ProcessingConfig, operation_id: u64) -> Self {
        Self {
            config,
            start_time: std::time::Instant::now(),
            operation_id,
        }
    }

    /// Check if operation has timed out
    #[inline(always)]
    pub fn is_timeout(&self) -> bool {
        self.start_time.elapsed().as_millis() > self.config.timeout_ms as u128
    }

    /// Get elapsed time in nanoseconds
    #[inline(always)]
    pub fn elapsed_nanos(&self) -> u64 {
        self.start_time.elapsed().as_nanos() as u64
    }
}
