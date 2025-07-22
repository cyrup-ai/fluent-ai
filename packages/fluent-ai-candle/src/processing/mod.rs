//! Unified LogitsProcessor system for production-ready sampling strategies
//!
//! This module provides a comprehensive, high-performance processing system with:
//! - Zero allocation patterns and lock-free operations
//! - Sophisticated error handling without unwrap/expect
//! - Context-aware processing with SIMD optimizations
//! - Composable processor architecture
//! - Production-ready numerical stability

pub mod context;
pub mod error;
pub mod processors;
pub mod traits;

// Core trait definitions
// Context integration
pub use context::{ContextBuilder, ProcessingContext};
// Error system
pub use error::ProcessingError;
// Processor implementations
pub use processors::{
    CompositeProcessor, RepetitionPenaltyProcessor, TemperatureProcessor, TopKProcessor,
    TopPProcessor,
};
pub use traits::{LogitsProcessor, ProcessingResult};

/// Processing module version for compatibility tracking
pub const VERSION: &str = "1.0.0";

/// Maximum supported vocabulary size for bounded allocation
pub const MAX_VOCABULARY_SIZE: usize = 128_000;

/// Maximum context window for token history tracking  
pub const MAX_CONTEXT_WINDOW: usize = 8_192;

/// Default processing context size for most use cases
pub const DEFAULT_CONTEXT_SIZE: usize = 1_024;

/// High-level processing interface combining all capabilities
#[derive(Debug)]
pub struct ProcessingEngine {
    /// Main processor chain
    processor: CompositeProcessor,
    /// Processing context for token tracking
    context: ProcessingContext,
    /// Processing metrics for performance monitoring
    metrics: ProcessingMetrics,
}

impl ProcessingEngine {
    /// Create new processing engine with configuration
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Result<Self, ProcessingError> {
        if vocab_size > MAX_VOCABULARY_SIZE {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Vocabulary size {} exceeds maximum {}",
                vocab_size, MAX_VOCABULARY_SIZE
            )));
        }

        let context = ProcessingContext::new(vocab_size, DEFAULT_CONTEXT_SIZE)?;
        let processor = CompositeProcessor::new(Vec::new())?;
        let metrics = ProcessingMetrics::new();

        Ok(Self {
            processor,
            context,
            metrics,
        })
    }

    /// Create processing engine with custom context size
    #[inline(always)]
    pub fn with_context_size(
        vocab_size: usize,
        context_size: usize,
    ) -> Result<Self, ProcessingError> {
        if vocab_size > MAX_VOCABULARY_SIZE {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Vocabulary size {} exceeds maximum {}",
                vocab_size, MAX_VOCABULARY_SIZE
            )));
        }

        if context_size > MAX_CONTEXT_WINDOW {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Context size {} exceeds maximum {}",
                context_size, MAX_CONTEXT_WINDOW
            )));
        }

        let context = ProcessingContext::new(vocab_size, context_size)?;
        let processor = CompositeProcessor::new(Vec::new())?;
        let metrics = ProcessingMetrics::new();

        Ok(Self {
            processor,
            context,
            metrics,
        })
    }

    /// Set the processor chain
    #[inline(always)]
    pub fn set_processor(&mut self, processor: CompositeProcessor) {
        self.processor = processor;
    }

    /// Process logits with full pipeline
    #[inline(always)]
    pub fn process_logits(&mut self, logits: &mut [f32]) -> ProcessingResult<()> {
        let start_time = std::time::Instant::now();

        // Validate logits array
        if logits.len() != self.context.vocab_size() {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Logits array size {} does not match vocabulary size {}",
                logits.len(),
                self.context.vocab_size()
            )));
        }

        // Process through the pipeline
        self.processor.process_logits(logits, &self.context)?;

        // Record processing metrics
        let processing_time = start_time.elapsed();
        self.metrics.record_processing(processing_time);

        Ok(())
    }

    /// Add token to context after generation
    #[inline(always)]
    pub fn add_token(&mut self, token_id: u32) -> ProcessingResult<()> {
        self.context.add_token(token_id)?;
        self.metrics.record_token();
        Ok(())
    }

    /// Reset engine for new sequence
    #[inline(always)]
    pub fn reset(&mut self) {
        self.context.reset();
        self.metrics.reset_sequence();
    }

    /// Get processing context
    #[inline(always)]
    pub fn context(&self) -> &ProcessingContext {
        &self.context
    }

    /// Get processing metrics
    #[inline(always)]
    pub fn metrics(&self) -> &ProcessingMetrics {
        &self.metrics
    }
}

/// Performance metrics for processing operations
#[derive(Debug)]
pub struct ProcessingMetrics {
    /// Total processing operations
    total_operations: std::sync::atomic::AtomicU64,
    /// Total processing time in nanoseconds
    total_processing_time: std::sync::atomic::AtomicU64,
    /// Total tokens processed
    total_tokens: std::sync::atomic::AtomicU64,
    /// Current sequence length
    sequence_length: std::sync::atomic::AtomicU32,
}

impl ProcessingMetrics {
    /// Create new metrics instance
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            total_operations: std::sync::atomic::AtomicU64::new(0),
            total_processing_time: std::sync::atomic::AtomicU64::new(0),
            total_tokens: std::sync::atomic::AtomicU64::new(0),
            sequence_length: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Record a processing operation
    #[inline(always)]
    pub fn record_processing(&self, duration: std::time::Duration) {
        self.total_operations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_processing_time.fetch_add(
            duration.as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Record a token generation
    #[inline(always)]
    pub fn record_token(&self) {
        self.total_tokens
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.sequence_length
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Reset sequence-level metrics
    #[inline(always)]
    pub fn reset_sequence(&self) {
        self.sequence_length
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get average processing time per operation
    #[inline(always)]
    pub fn average_processing_time(&self) -> std::time::Duration {
        let total_ops = self
            .total_operations
            .load(std::sync::atomic::Ordering::Relaxed);
        if total_ops == 0 {
            return std::time::Duration::ZERO;
        }

        let total_time = self
            .total_processing_time
            .load(std::sync::atomic::Ordering::Relaxed);
        std::time::Duration::from_nanos(total_time / total_ops)
    }

    /// Get tokens per second throughput
    #[inline(always)]
    pub fn tokens_per_second(&self) -> f64 {
        let total_tokens = self.total_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let total_time = self
            .total_processing_time
            .load(std::sync::atomic::Ordering::Relaxed);

        if total_time == 0 {
            return 0.0;
        }

        (total_tokens as f64) / ((total_time as f64) / 1_000_000_000.0)
    }

    /// Get current sequence length
    #[inline(always)]
    pub fn sequence_length(&self) -> u32 {
        self.sequence_length
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total operations count
    #[inline(always)]
    pub fn total_operations(&self) -> u64 {
        self.total_operations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total tokens processed
    #[inline(always)]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for ProcessingMetrics {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating processing engines with custom configurations
#[derive(Debug)]
pub struct ProcessingEngineBuilder {
    vocab_size: usize,
    context_size: Option<usize>,
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl ProcessingEngineBuilder {
    /// Create new builder with vocabulary size
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            context_size: None,
            processors: Vec::new(),
        }
    }

    /// Set context window size
    #[inline(always)]
    pub fn context_size(mut self, size: usize) -> Self {
        self.context_size = Some(size);
        self
    }

    /// Add a processor to the chain
    #[inline(always)]
    pub fn add_processor(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Add temperature processor
    #[inline(always)]
    pub fn temperature(self, temperature: f32) -> ProcessingResult<Self> {
        let processor = Box::new(TemperatureProcessor::new(temperature)?);
        Ok(self.add_processor(processor))
    }

    /// Add top-k processor
    #[inline(always)]
    pub fn top_k(self, k: usize) -> ProcessingResult<Self> {
        let processor = Box::new(TopKProcessor::new(k)?);
        Ok(self.add_processor(processor))
    }

    /// Add top-p processor
    #[inline(always)]
    pub fn top_p(self, p: f32) -> ProcessingResult<Self> {
        let processor = Box::new(TopPProcessor::new(p)?);
        Ok(self.add_processor(processor))
    }

    /// Add repetition penalty processor
    #[inline(always)]
    pub fn repetition_penalty(
        self,
        penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> ProcessingResult<Self> {
        let processor = Box::new(RepetitionPenaltyProcessor::new(
            penalty,
            frequency_penalty,
            presence_penalty,
        )?);
        Ok(self.add_processor(processor))
    }

    /// Build the processing engine
    pub fn build(self) -> ProcessingResult<ProcessingEngine> {
        let context_size = self.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
        let mut engine = ProcessingEngine::with_context_size(self.vocab_size, context_size)?;

        if !self.processors.is_empty() {
            let composite = CompositeProcessor::new(self.processors)?;
            engine.set_processor(composite);
        }

        Ok(engine)
    }
}

/// Utility functions for processing system
pub mod utils {
    use super::*;

    /// Create a standard text generation processing engine
    #[inline(always)]
    pub fn standard_text_generation(
        vocab_size: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
    ) -> ProcessingResult<ProcessingEngine> {
        let mut builder = ProcessingEngineBuilder::new(vocab_size);

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = repetition_penalty {
            builder = builder.repetition_penalty(penalty, 0.0, 0.0)?;
        }

        // Add temperature scaling
        builder = builder.temperature(temperature)?;

        // Add top-k filtering
        if let Some(k) = top_k {
            builder = builder.top_k(k)?;
        }

        // Add top-p nucleus sampling
        if let Some(p) = top_p {
            builder = builder.top_p(p)?;
        }

        builder.build()
    }

    /// Create creative writing processing engine
    #[inline(always)]
    pub fn creative_writing(vocab_size: usize) -> ProcessingResult<ProcessingEngine> {
        standard_text_generation(
            vocab_size,
            0.85,       // Higher temperature for creativity
            None,       // No top-k limit
            Some(0.92), // Nucleus sampling
            Some(1.15), // Moderate repetition penalty
        )
    }

    /// Create code generation processing engine
    #[inline(always)]
    pub fn code_generation(vocab_size: usize) -> ProcessingResult<ProcessingEngine> {
        standard_text_generation(
            vocab_size,
            0.2,        // Low temperature for precision
            Some(20),   // Focused vocabulary
            Some(0.95), // High nucleus threshold
            Some(1.05), // Minimal repetition penalty
        )
    }

    /// Create conversational processing engine
    #[inline(always)]
    pub fn conversation(vocab_size: usize) -> ProcessingResult<ProcessingEngine> {
        standard_text_generation(
            vocab_size,
            0.7,       // Balanced temperature
            Some(40),  // Moderate vocabulary focus
            Some(0.9), // Standard nucleus sampling
            Some(1.1), // Standard repetition penalty
        )
    }

    /// Validate logits array for numerical stability
    #[inline(always)]
    pub fn validate_logits(logits: &[f32]) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Err(ProcessingError::InvalidConfiguration(
                "Empty logits array".to_string(),
            ));
        }

        // Check for NaN or infinite values
        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(ProcessingError::NumericalError(format!(
                    "Non-finite logit at index {}: {}",
                    i, logit
                )));
            }
        }

        Ok(())
    }

    /// Calculate entropy of logits distribution
    #[inline(always)]
    pub fn calculate_entropy(logits: &[f32]) -> ProcessingResult<f32> {
        validate_logits(logits)?;

        // Find max for numerical stability
        let max_logit = logits
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ProcessingError::NumericalError("No valid logits found".to_string()))?;

        // Compute softmax with stability
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        if exp_sum <= 0.0 {
            return Err(ProcessingError::NumericalError(
                "Invalid softmax normalization".to_string(),
            ));
        }

        // Calculate entropy: -Σ(p * log(p))
        let entropy: f32 = logits
            .iter()
            .map(|&x| {
                let prob = (x - max_logit).exp() / exp_sum;
                if prob > 0.0 {
                    -prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum();

        Ok(entropy)
    }
}
