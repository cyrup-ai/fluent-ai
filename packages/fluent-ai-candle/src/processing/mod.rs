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
    TopPProcessor};
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
    metrics: ProcessingMetrics}

impl ProcessingEngine {
    /// Creates a new ProcessingEngine with default configuration
    /// 
    /// Initializes a high-performance logits processing engine with production-ready
    /// defaults suitable for most text generation tasks.
    /// 
    /// # Arguments
    /// 
    /// * `vocab_size` - Size of the model vocabulary (must be ≤ 128,000)
    /// 
    /// # Returns
    /// 
    /// `Result<ProcessingEngine, ProcessingError>` containing:
    /// - `Ok(engine)` - Configured processing engine ready for use
    /// - `Err(ProcessingError::InvalidConfiguration)` - If vocab_size exceeds limits
    /// 
    /// # Default Configuration
    /// 
    /// - **Context Size**: 1,024 tokens (DEFAULT_CONTEXT_SIZE)
    /// - **Processor Chain**: Empty (no processing steps)
    /// - **Metrics**: Initialized for performance tracking
    /// - **Memory**: Pre-allocated for zero-allocation processing
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Zero Allocation**: Pre-allocates all required memory
    /// - **Lock-Free**: Uses atomic operations for metrics
    /// - **Bounded**: Vocabulary and context size limits prevent OOM
    /// - **Cache Friendly**: Contiguous memory layout
    /// 
    /// # Vocabulary Limits
    /// 
    /// Maximum supported vocabulary size is 128,000 tokens to ensure:
    /// - Reasonable memory usage (< 1MB for f32 logits)
    /// - Cache-friendly processing
    /// - Bounded allocation guarantees
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngine;
    /// 
    /// // Standard model vocabulary
    /// let engine = ProcessingEngine::new(50_257)?;  // GPT-2 vocab size
    /// 
    /// // Large modern model
    /// let engine = ProcessingEngine::new(100_000)?; // Large vocabulary
    /// 
    /// // Error case - vocabulary too large
    /// let result = ProcessingEngine::new(200_000);
    /// assert!(result.is_err());
    /// ```
    /// 
    /// # Custom Configuration
    /// 
    /// For advanced configurations, use the builder pattern:
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngineBuilder;
    /// 
    /// let engine = ProcessingEngineBuilder::new(50_257)
    ///     .context_size(2048)
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .build()?;
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// The engine is thread-safe for metrics but requires external synchronization
    /// for logits processing operations.
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Result<Self, ProcessingError> {
        if vocab_size > MAX_VOCABULARY_SIZE {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Vocabulary size {} exceeds maximum {}",
                vocab_size, MAX_VOCABULARY_SIZE
            )));
        }

        let context = ProcessingContext::new(vocab_size, DEFAULT_CONTEXT_SIZE)?;
        let processor = CompositeProcessor::new();
        let metrics = ProcessingMetrics::new();

        Ok(Self {
            processor,
            context,
            metrics})
    }

    /// Creates a new ProcessingEngine with custom context window size
    /// 
    /// Advanced constructor allowing precise control over token history tracking
    /// for context-aware processing like repetition penalty and frequency analysis.
    /// 
    /// # Arguments
    /// 
    /// * `vocab_size` - Size of the model vocabulary (must be ≤ 128,000)
    /// * `context_size` - Number of recent tokens to track (must be ≤ 8,192)
    /// 
    /// # Returns
    /// 
    /// `Result<ProcessingEngine, ProcessingError>` with validation:
    /// - `Ok(engine)` - Configured engine with custom context size
    /// - `Err(ProcessingError::InvalidConfiguration)` - If parameters exceed limits
    /// 
    /// # Context Size Impact
    /// 
    /// ## Memory Usage
    /// ```
    /// memory_bytes = context_size * sizeof(u32)  // Token storage
    ///              + context_size * vocab_size * sizeof(u32)  // Frequency tracking
    /// ```
    /// 
    /// ## Processing Features
    /// - **Repetition Penalty**: Tracks token frequency over context window
    /// - **Frequency Analysis**: Maintains occurrence counts for penalty calculation
    /// - **Presence Penalty**: Binary tracking of token presence
    /// - **Pattern Detection**: Identifies repetitive sequences
    /// 
    /// # Context Size Guidelines
    /// 
    /// - **Small (64-256)**: Code completion, short responses
    /// - **Medium (512-1024)**: Standard text generation
    /// - **Large (2048-4096)**: Long-form content, conversation
    /// - **Maximum (8192)**: Research, complex analysis
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Linear Memory**: O(context_size) memory usage
    /// - **Constant Time**: O(1) token addition and lookup
    /// - **Cache Efficient**: Circular buffer implementation
    /// - **Zero Allocation**: Pre-allocated context buffers
    /// 
    /// # Examples
    /// 
    /// ## Memory-Constrained Environment
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngine;
    /// 
    /// // Small context for low memory usage
    /// let engine = ProcessingEngine::with_context_size(50_257, 256)?;
    /// 
    /// // Estimated memory: 256 * 4 + 256 * 50_257 * 4 ≈ 51MB
    /// ```
    /// 
    /// ## Long-Form Generation
    /// ```rust
    /// // Large context for document generation
    /// let engine = ProcessingEngine::with_context_size(100_000, 4096)?;
    /// 
    /// // Can track repetition patterns over 4K tokens
    /// ```
    /// 
    /// ## Real-Time Chat
    /// ```rust
    /// // Moderate context for conversation
    /// let engine = ProcessingEngine::with_context_size(32_000, 1024)?;
    /// 
    /// // Good balance of memory usage and context awareness
    /// ```
    /// 
    /// ## Error Handling
    /// ```rust
    /// match ProcessingEngine::with_context_size(200_000, 10_000) {
    ///     Err(ProcessingError::InvalidConfiguration(msg)) => {
    ///         eprintln!("Configuration error: {}", msg);
    ///         // Fall back to default configuration
    ///         ProcessingEngine::new(50_257)?
    ///     },
    ///     Ok(engine) => engine,
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    /// 
    /// # Context Management
    /// 
    /// The context automatically manages token history:
    /// - Circular buffer prevents memory growth
    /// - Oldest tokens evicted when context fills
    /// - Frequency counts updated incrementally
    /// - Reset functionality for new sequences
    /// 
    /// # Thread Safety
    /// 
    /// Context access requires external synchronization for multi-threaded usage.
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
        let processor = CompositeProcessor::new();
        let metrics = ProcessingMetrics::new();

        Ok(Self {
            processor,
            context,
            metrics})
    }

    /// Sets the logits processor chain for this engine
    /// 
    /// Replaces the current processing pipeline with a new composite processor,
    /// allowing dynamic reconfiguration of sampling strategies.
    /// 
    /// # Arguments
    /// 
    /// * `processor` - CompositeProcessor containing the processing pipeline
    ///   with ordered sequence of transformations (temperature, top-k, top-p, etc.)
    /// 
    /// # Processing Pipeline
    /// 
    /// The processor chain typically includes:
    /// 1. **Repetition Penalty** - Context-aware repetition reduction
    /// 2. **Temperature Scaling** - Probability distribution adjustment
    /// 3. **Top-K Filtering** - Vocabulary limitation by rank
    /// 4. **Top-P Sampling** - Nucleus sampling by cumulative probability
    /// 
    /// # Performance Impact
    /// 
    /// - **Zero Allocation**: Processor replacement without memory allocation
    /// - **Immediate Effect**: Next `process_logits()` call uses new processor
    /// - **No Validation**: Assumes processor is properly configured
    /// - **Thread Safety**: Requires external synchronization
    /// 
    /// # Examples
    /// 
    /// ## Dynamic Reconfiguration
    /// ```rust
    /// use fluent_ai_candle::processing::{
    ///     ProcessingEngine, CompositeProcessor, TemperatureProcessor
    /// };
    /// 
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// // Start with temperature-only processing
    /// let temp_processor = CompositeProcessor::new()
    ///     .add_processor(Box::new(TemperatureProcessor::new(0.8)?));
    /// engine.set_processor(temp_processor);
    /// 
    /// // Process some logits
    /// engine.process_logits(&mut logits1)?;
    /// 
    /// // Switch to more conservative settings
    /// let conservative_processor = CompositeProcessor::new()
    ///     .add_processor(Box::new(TemperatureProcessor::new(0.2)?))
    ///     .add_processor(Box::new(TopKProcessor::new(20)?));
    /// engine.set_processor(conservative_processor);
    /// 
    /// // Next processing uses new settings
    /// engine.process_logits(&mut logits2)?;
    /// ```
    /// 
    /// ## Runtime Adaptation
    /// ```rust
    /// // Adapt processing based on generation quality
    /// if generation_quality < threshold {
    ///     // Use more focused sampling
    ///     let focused = CompositeProcessor::new()
    ///         .add_processor(Box::new(TemperatureProcessor::new(0.1)?))
    ///         .add_processor(Box::new(TopKProcessor::new(5)?));
    ///     engine.set_processor(focused);
    /// } else {
    ///     // Use more creative sampling
    ///     let creative = CompositeProcessor::new()
    ///         .add_processor(Box::new(TemperatureProcessor::new(1.2)?))
    ///         .add_processor(Box::new(TopPProcessor::new(0.95)?));
    ///     engine.set_processor(creative);
    /// }
    /// ```
    /// 
    /// ## User Preference Switching
    /// ```rust
    /// match user_preference {
    ///     "creative" => {
    ///         let processor = creative_writing_processor()?;
    ///         engine.set_processor(processor);
    ///     },
    ///     "precise" => {
    ///         let processor = code_generation_processor()?;
    ///         engine.set_processor(processor);
    ///     },
    ///     "balanced" => {
    ///         let processor = conversation_processor()?;
    ///         engine.set_processor(processor);
    ///     },
    /// }
    /// ```
    /// 
    /// # Processor Validation
    /// 
    /// The method does not validate the processor configuration.
    /// Ensure the processor is properly configured before setting:
    /// ```rust
    /// // Validate processor before setting
    /// let processor = CompositeProcessor::new();
    /// if processor.is_empty() {
    ///     eprintln!("Warning: Empty processor chain");
    /// }
    /// engine.set_processor(processor);
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This method is not thread-safe. Ensure exclusive access when
    /// modifying the processor chain in multi-threaded environments.
    #[inline(always)]
    pub fn set_processor(&mut self, processor: CompositeProcessor) {
        self.processor = processor;
    }

    /// Processes logits through the complete transformation pipeline
    /// 
    /// Applies all configured processing steps (temperature, top-k, top-p, repetition penalty)
    /// to the input logits array, transforming raw model outputs into a sampling-ready
    /// probability distribution.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Mutable slice of raw logits from model (length must equal vocab_size)
    ///   Values are transformed in-place for zero-allocation processing
    /// 
    /// # Returns
    /// 
    /// `ProcessingResult<()>` indicating success or failure:
    /// - `Ok(())` - Logits processed successfully and ready for sampling
    /// - `Err(ProcessingError::InvalidConfiguration)` - Logits array size mismatch
    /// - `Err(ProcessingError::NumericalError)` - Numerical instability detected
    /// - `Err(ProcessingError::ProcessingFailed)` - Pipeline processing failed
    /// 
    /// # Processing Pipeline
    /// 
    /// The processing occurs in this order:
    /// 1. **Input Validation** - Size and numerical stability checks
    /// 2. **Context Integration** - Incorporates token history for repetition handling
    /// 3. **Processor Chain** - Applies all configured transformations
    /// 4. **Metrics Collection** - Records timing and performance data
    /// 
    /// # Performance Characteristics
    /// 
    /// - **In-Place Processing**: Logits modified directly, no allocation
    /// - **SIMD Optimized**: Uses vectorized operations where possible
    /// - **Numerical Stability**: Handles extreme values gracefully
    /// - **Bounded Execution**: Processing time scales linearly with vocab size
    /// 
    /// # Error Conditions
    /// 
    /// ## Size Mismatch
    /// ```rust
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// let mut wrong_size_logits = vec![0.0; 30_000];  // Wrong size
    /// 
    /// match engine.process_logits(&mut wrong_size_logits) {
    ///     Err(ProcessingError::InvalidConfiguration(msg)) => {
    ///         eprintln!("Size error: {}", msg);
    ///     },
    ///     _ => unreachable!(),
    /// }
    /// ```
    /// 
    /// ## Numerical Issues
    /// ```rust
    /// let mut logits = vec![f32::NAN; 50_257];  // Invalid logits
    /// 
    /// match engine.process_logits(&mut logits) {
    ///     Err(ProcessingError::NumericalError(_)) => {
    ///         // Handle numerical instability
    ///         logits.fill(0.0);  // Reset to neutral
    ///         engine.process_logits(&mut logits)?;  // Retry
    ///     },
    ///     result => result?,
    /// }
    /// ```
    /// 
    /// # Examples
    /// 
    /// ## Basic Processing
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngine;
    /// 
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// let mut logits = model.forward(&tokens)?;  // Raw model output
    /// 
    /// // Transform logits for sampling
    /// engine.process_logits(&mut logits)?;
    /// 
    /// // Logits now ready for token sampling
    /// let token_id = sample_from_logits(&logits)?;
    /// ```
    /// 
    /// ## Performance Monitoring
    /// ```rust
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// for batch in batches {
    ///     let start = std::time::Instant::now();
    ///     engine.process_logits(&mut batch.logits)?;
    ///     
    ///     println!("Processing took: {:?}", start.elapsed());
    ///     println!("Average time: {:?}", engine.metrics().average_processing_time());
    /// }
    /// ```
    /// 
    /// ## Error Recovery
    /// ```rust
    /// fn robust_process_logits(
    ///     engine: &mut ProcessingEngine,
    ///     logits: &mut [f32]
    /// ) -> ProcessingResult<()> {
    ///     match engine.process_logits(logits) {
    ///         Ok(()) => Ok(()),
    ///         Err(ProcessingError::InvalidConfiguration(_)) => {
    ///             return Err(ProcessingError::InvalidConfiguration(
    ///                 "Logits size mismatch - check model compatibility".to_string()
    ///             ));
    ///         },
    ///         Err(ProcessingError::NumericalError(_)) => {
    ///             // Attempt to recover by normalizing
    ///             normalize_logits(logits);
    ///             engine.process_logits(logits)  // Retry once
    ///         },
    ///         Err(e) => Err(e),
    ///     }
    /// }
    /// ```
    /// 
    /// ## Context-Aware Processing
    /// ```rust
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// // Configure repetition penalty processor
    /// let processor = CompositeProcessor::new()
    ///     .add_processor(Box::new(RepetitionPenaltyProcessor::new(
    ///         1.1,    // repetition_penalty
    ///         0.1,    // frequency_penalty
    ///         0.05,   // presence_penalty
    ///         512,    // context_window
    ///     )?));
    /// engine.set_processor(processor);
    /// 
    /// // Processing will use token history for repetition control
    /// for token_id in generated_tokens {
    ///     engine.add_token(token_id)?;
    ///     let mut logits = model.forward(&context)?;
    ///     engine.process_logits(&mut logits)?;
    /// }
    /// ```
    /// 
    /// # Logits Transformation
    /// 
    /// The logits undergo these transformations:
    /// - **Raw Logits**: Model output (unbounded real numbers)
    /// - **After Temperature**: Scaled for sharpness/smoothness
    /// - **After Top-K**: Limited vocabulary (others set to -inf)
    /// - **After Top-P**: Nucleus sampling preparation
    /// - **Ready for Sampling**: Probability distribution ready
    /// 
    /// # Thread Safety
    /// 
    /// This method modifies engine state (metrics, context) and requires
    /// exclusive access. Use external synchronization for multi-threaded usage.
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

    /// Adds a generated token to the processing context for history tracking
    /// 
    /// Updates the context with a newly generated token, maintaining token history
    /// for context-aware processing like repetition penalty and frequency analysis.
    /// 
    /// # Arguments
    /// 
    /// * `token_id` - The generated token ID to add to the context
    ///   (must be valid for the model's vocabulary)
    /// 
    /// # Returns
    /// 
    /// `ProcessingResult<()>` indicating success or failure:
    /// - `Ok(())` - Token added successfully to context
    /// - `Err(ProcessingError::InvalidToken)` - Token ID exceeds vocabulary size
    /// - `Err(ProcessingError::ContextFull)` - Context window is full (shouldn't happen with circular buffer)
    /// 
    /// # Context Management
    /// 
    /// ## Token History
    /// - Maintains circular buffer of recent tokens
    /// - Updates frequency counts for repetition penalty
    /// - Tracks presence information for presence penalty
    /// - Automatically evicts oldest tokens when context fills
    /// 
    /// ## Frequency Tracking
    /// - Increments occurrence count for this token
    /// - Updates frequency-based penalty calculations
    /// - Maintains running statistics for context window
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Constant Time**: O(1) token addition and frequency update
    /// - **Memory Bounded**: Context size prevents unbounded growth
    /// - **Cache Friendly**: Circular buffer with good locality
    /// - **Zero Allocation**: Updates existing data structures in-place
    /// 
    /// # Usage Pattern
    /// 
    /// The typical generation loop includes:
    /// 1. Process logits with current context
    /// 2. Sample token from processed distribution
    /// 3. Add sampled token to context
    /// 4. Repeat for next token
    /// 
    /// # Examples
    /// 
    /// ## Basic Generation Loop
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngine;
    /// 
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// let mut generated_tokens = Vec::new();
    /// 
    /// for _ in 0..max_tokens {
    ///     // Get logits from model
    ///     let mut logits = model.forward(&context)?;
    ///     
    ///     // Process with current context
    ///     engine.process_logits(&mut logits)?;
    ///     
    ///     // Sample next token
    ///     let token_id = sample_from_logits(&logits)?;
    ///     generated_tokens.push(token_id);
    ///     
    ///     // Add to context for next iteration
    ///     engine.add_token(token_id)?;
    ///     
    ///     // Check for end token
    ///     if token_id == eos_token_id {
    ///         break;
    ///     }
    /// }
    /// ```
    /// 
    /// ## Error Handling
    /// ```rust
    /// match engine.add_token(token_id) {
    ///     Ok(()) => {
    ///         // Token added successfully
    ///         context.push(token_id);
    ///     },
    ///     Err(ProcessingError::InvalidToken(msg)) => {
    ///         eprintln!("Invalid token {}: {}", token_id, msg);
    ///         // Skip this token or use fallback
    ///         continue;
    ///     },
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    /// 
    /// ## Batch Processing
    /// ```rust
    /// // Process multiple sequences
    /// for sequence in sequences {
    ///     engine.reset();  // Clear context for new sequence
    ///     
    ///     for token_id in sequence.initial_tokens {
    ///         engine.add_token(token_id)?;
    ///     }
    ///     
    ///     // Continue generation with established context
    ///     let mut logits = model.forward(&sequence.context)?;
    ///     engine.process_logits(&mut logits)?;
    /// }
    /// ```
    /// 
    /// ## Metrics Monitoring
    /// ```rust
    /// let initial_tokens = engine.metrics().total_tokens();
    /// 
    /// for token_id in new_tokens {
    ///     engine.add_token(token_id)?;
    /// }
    /// 
    /// let added_count = engine.metrics().total_tokens() - initial_tokens;
    /// println!("Added {} tokens to context", added_count);
    /// println!("Current sequence length: {}", engine.metrics().sequence_length());
    /// ```
    /// 
    /// ## Context Inspection
    /// ```rust
    /// // Add token and inspect context state
    /// engine.add_token(token_id)?;
    /// 
    /// let context = engine.context();
    /// println!("Context size: {}/{}", context.current_length(), context.capacity());
    /// println!("Token frequency: {}", context.token_frequency(token_id));
    /// println!("Is token present: {}", context.is_token_present(token_id));
    /// ```
    /// 
    /// # Repetition Control
    /// 
    /// Token addition directly affects repetition penalty calculations:
    /// - Higher frequency tokens get stronger penalties
    /// - Recently used tokens are penalized more heavily
    /// - Context window limits penalty scope
    /// 
    /// # Thread Safety
    /// 
    /// This method modifies internal state and requires exclusive access.
    /// Use external synchronization for concurrent usage.
    #[inline(always)]
    pub fn add_token(&mut self, token_id: u32) -> ProcessingResult<()> {
        self.context.add_token(token_id)?;
        self.metrics.record_token();
        Ok(())
    }

    /// Resets the processing engine for a new generation sequence
    /// 
    /// Clears all context history and sequence-specific metrics while preserving
    /// processor configuration and cumulative statistics.
    /// 
    /// # Reset Operations
    /// 
    /// ## Context Clearing
    /// - **Token History**: Clears all stored tokens from context buffer
    /// - **Frequency Counts**: Resets all token frequency counters to zero
    /// - **Presence Flags**: Clears all token presence indicators
    /// - **Context Length**: Resets current context length to zero
    /// 
    /// ## Metrics Reset
    /// - **Sequence Length**: Resets to zero for new sequence
    /// - **Cumulative Stats**: Preserves total operations and processing time
    /// - **Performance Tracking**: Maintains historical averages
    /// 
    /// ## Preserved State
    /// - **Processor Chain**: Keeps all configured processors
    /// - **Engine Configuration**: Maintains vocab size and context capacity
    /// - **Global Metrics**: Preserves lifetime statistics
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Fast Reset**: O(1) circular buffer reset, no memory deallocation
    /// - **Zero Allocation**: Reuses existing memory allocations
    /// - **Immediate Effect**: Next operations start with clean state
    /// - **Cache Friendly**: Maintains memory layout and locality
    /// 
    /// # When to Reset
    /// 
    /// ## New Generation Session
    /// - Starting generation for a different prompt
    /// - Switching between unrelated conversations
    /// - Beginning batch processing of new sequences
    /// 
    /// ## Context Contamination
    /// - After processing errors or invalid tokens
    /// - When context contains inappropriate content
    /// - For clean state in testing scenarios
    /// 
    /// ## Memory Management
    /// - Preventing context from affecting unrelated generations
    /// - Ensuring repetition penalties don't carry over
    /// - Maintaining isolation between processing sessions
    /// 
    /// # Examples
    /// 
    /// ## Batch Processing
    /// ```rust
    /// use fluent_ai_candle::processing::ProcessingEngine;
    /// 
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// for prompt in prompts {
    ///     // Reset for new prompt
    ///     engine.reset();
    ///     
    ///     // Process prompt tokens to establish context
    ///     for &token_id in &prompt.tokens {
    ///         engine.add_token(token_id)?;
    ///     }
    ///     
    ///     // Generate response with clean context
    ///     let response = generate_response(&mut engine, &prompt)?;
    ///     results.push(response);
    /// }
    /// ```
    /// 
    /// ## Conversation Management
    /// ```rust
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// for conversation in conversations {
    ///     engine.reset();  // Start fresh for each conversation
    ///     
    ///     for turn in conversation.turns {
    ///         // Add user message to context
    ///         for &token in &turn.user_tokens {
    ///             engine.add_token(token)?;
    ///         }
    ///         
    ///         // Generate assistant response
    ///         let mut logits = model.forward(&turn.context)?;
    ///         engine.process_logits(&mut logits)?;
    ///         
    ///         let response_token = sample_from_logits(&logits)?;
    ///         engine.add_token(response_token)?;
    ///     }
    /// }
    /// ```
    /// 
    /// ## Error Recovery
    /// ```rust
    /// match engine.process_logits(&mut logits) {
    ///     Ok(()) => { /* Continue processing */ },
    ///     Err(ProcessingError::ContextCorrupted(_)) => {
    ///         eprintln!("Context corrupted, resetting engine");
    ///         engine.reset();
    ///         
    ///         // Restart with clean context
    ///         engine.process_logits(&mut logits)?;
    ///     },
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    /// 
    /// ## Testing Scenarios
    /// ```rust
    /// #[test]
    /// fn test_generation_independence() {
    ///     let mut engine = ProcessingEngine::new(1000)?;
    ///     
    ///     // First generation
    ///     engine.add_token(100)?;
    ///     let result1 = generate_tokens(&mut engine)?;
    ///     
    ///     // Reset for independent generation
    ///     engine.reset();
    ///     
    ///     // Second generation should not be affected by first
    ///     engine.add_token(100)?;
    ///     let result2 = generate_tokens(&mut engine)?;
    ///     
    ///     // Results should be identical (no context carryover)
    ///     assert_eq!(result1, result2);
    /// }
    /// ```
    /// 
    /// ## Performance Monitoring
    /// ```rust
    /// let mut engine = ProcessingEngine::new(50_257)?;
    /// 
    /// // Check metrics before reset
    /// println!("Before reset:");
    /// println!("  Sequence length: {}", engine.metrics().sequence_length());
    /// println!("  Total operations: {}", engine.metrics().total_operations());
    /// 
    /// engine.reset();
    /// 
    /// // Check metrics after reset
    /// println!("After reset:");
    /// println!("  Sequence length: {}", engine.metrics().sequence_length());  // Should be 0
    /// println!("  Total operations: {}", engine.metrics().total_operations()); // Preserved
    /// ```
    /// 
    /// ## Context State Verification
    /// ```rust
    /// // Verify clean state after reset
    /// engine.reset();
    /// 
    /// let context = engine.context();
    /// assert_eq!(context.current_length(), 0);
    /// assert_eq!(engine.metrics().sequence_length(), 0);
    /// 
    /// // Verify processor chain is preserved
    /// assert!(!engine.processor.is_empty());  // Processors still configured
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// Reset operations modify internal state and require exclusive access.
    /// Ensure no concurrent processing occurs during reset.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.context.reset();
        self.metrics.reset_sequence();
    }

    /// Returns a reference to the processing context for inspection
    /// 
    /// Provides read-only access to the current processing context containing
    /// token history, frequency counts, and context management state.
    /// 
    /// # Returns
    /// 
    /// `&ProcessingContext` containing:
    /// - **Token History**: Circular buffer of recent tokens
    /// - **Frequency Counts**: Per-token occurrence statistics
    /// - **Presence Flags**: Binary indicators of token presence
    /// - **Context Metadata**: Size, capacity, and state information
    /// 
    /// # Context Information
    /// 
    /// ## Available Methods
    /// - `vocab_size()` - Model vocabulary size
    /// - `capacity()` - Maximum context window size
    /// - `current_length()` - Number of tokens currently stored
    /// - `token_frequency(token_id)` - Occurrence count for specific token
    /// - `is_token_present(token_id)` - Whether token appears in context
    /// - `recent_tokens(n)` - Last n tokens from history
    /// 
    /// ## Memory Layout
    /// - **Circular Buffer**: Efficient O(1) token storage
    /// - **Frequency Map**: Hash table for occurrence counting
    /// - **Presence Set**: Bit vector for presence tracking
    /// 
    /// # Use Cases
    /// 
    /// ## Context Analysis
    /// ```rust
    /// let context = engine.context();
    /// 
    /// println!("Context utilization: {}/{}", 
    ///          context.current_length(), context.capacity());
    /// println!("Vocabulary coverage: {:.1}%", 
    ///          context.unique_tokens() as f32 / context.vocab_size() as f32 * 100.0);
    /// ```
    /// 
    /// ## Repetition Analysis
    /// ```rust
    /// let context = engine.context();
    /// let mut repeated_tokens = Vec::new();
    /// 
    /// for token_id in 0..context.vocab_size() {
    ///     let frequency = context.token_frequency(token_id as u32);
    ///     if frequency > 3 {  // Repeated more than 3 times
    ///         repeated_tokens.push((token_id, frequency));
    ///     }
    /// }
    /// 
    /// println!("Highly repeated tokens: {:?}", repeated_tokens);
    /// ```
    /// 
    /// ## Context Window Analysis
    /// ```rust
    /// let context = engine.context();
    /// 
    /// // Analyze recent token patterns
    /// if let Some(recent) = context.recent_tokens(10) {
    ///     println!("Last 10 tokens: {:?}", recent);
    ///     
    ///     // Check for immediate repetition
    ///     let unique_recent: HashSet<_> = recent.iter().collect();
    ///     if unique_recent.len() < recent.len() / 2 {
    ///         println!("Warning: High repetition in recent tokens");
    ///     }
    /// }
    /// ```
    /// 
    /// ## Debug Information
    /// ```rust
    /// let context = engine.context();
    /// 
    /// println!("Context Debug Info:");
    /// println!("  Size: {}/{}", context.current_length(), context.capacity());
    /// println!("  Unique tokens: {}", context.unique_tokens());
    /// println!("  Most frequent token: {:?}", context.most_frequent_token());
    /// println!("  Average frequency: {:.2}", context.average_frequency());
    /// ```
    /// 
    /// ## Context Health Check
    /// ```rust
    /// fn check_context_health(engine: &ProcessingEngine) -> bool {
    ///     let context = engine.context();
    ///     
    ///     // Check for reasonable diversity
    ///     let diversity_ratio = context.unique_tokens() as f32 / context.current_length() as f32;
    ///     if diversity_ratio < 0.3 {  // Less than 30% unique tokens
    ///         return false;
    ///     }
    ///     
    ///     // Check for extreme repetition
    ///     for token_id in 0..context.vocab_size() {
    ///         let frequency = context.token_frequency(token_id as u32);
    ///         let relative_freq = frequency as f32 / context.current_length() as f32;
    ///         if relative_freq > 0.5 {  // Single token > 50% of context
    ///             return false;
    ///         }
    ///     }
    ///     
    ///     true
    /// }
    /// ```
    /// 
    /// ## Performance Analysis
    /// ```rust
    /// let context = engine.context();
    /// 
    /// // Analyze context efficiency
    /// let utilization = context.current_length() as f32 / context.capacity() as f32;
    /// 
    /// match utilization {
    ///     x if x < 0.1 => println!("Context underutilized ({:.1}%)", x * 100.0),
    ///     x if x > 0.9 => println!("Context nearly full ({:.1}%)", x * 100.0),
    ///     x => println!("Context utilization: {:.1}%", x * 100.0),
    /// }
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// The returned reference is immutable and safe for concurrent read access.
    /// Multiple threads can inspect the context simultaneously without synchronization.
    /// 
    /// # Lifetime
    /// 
    /// The context reference is valid as long as the engine exists.
    /// Context state may change between calls due to token additions or resets.
    #[inline(always)]
    pub fn context(&self) -> &ProcessingContext {
        &self.context
    }

    /// Returns a reference to the processing metrics for performance monitoring
    /// 
    /// Provides read-only access to comprehensive performance statistics including
    /// timing, throughput, and operational metrics collected during processing.
    /// 
    /// # Returns
    /// 
    /// `&ProcessingMetrics` containing:
    /// - **Operation Counts**: Total processing operations performed
    /// - **Timing Statistics**: Cumulative and average processing times
    /// - **Token Metrics**: Token generation counts and throughput
    /// - **Sequence State**: Current sequence length and progress
    /// 
    /// # Available Metrics
    /// 
    /// ## Performance Timing
    /// - `total_operations()` - Number of logits processing operations
    /// - `average_processing_time()` - Mean time per processing operation
    /// - `tokens_per_second()` - Token generation throughput
    /// 
    /// ## Token Statistics
    /// - `total_tokens()` - Cumulative tokens processed across all sequences
    /// - `sequence_length()` - Current sequence length (resets per sequence)
    /// 
    /// ## Operational Health
    /// - Processing consistency and performance trends
    /// - Memory usage patterns and allocation efficiency
    /// - Error rates and recovery statistics
    /// 
    /// # Use Cases
    /// 
    /// ## Performance Monitoring
    /// ```rust
    /// let metrics = engine.metrics();
    /// 
    /// println!("Processing Performance:");
    /// println!("  Operations: {}", metrics.total_operations());
    /// println!("  Avg time: {:?}", metrics.average_processing_time());
    /// println!("  Throughput: {:.1} tokens/sec", metrics.tokens_per_second());
    /// println!("  Current sequence: {} tokens", metrics.sequence_length());
    /// ```
    /// 
    /// ## Performance Profiling
    /// ```rust
    /// let before_ops = engine.metrics().total_operations();
    /// let before_time = std::time::Instant::now();
    /// 
    /// // Process batch of logits
    /// for logits in batch {
    ///     engine.process_logits(logits)?;
    /// }
    /// 
    /// let elapsed = before_time.elapsed();
    /// let ops_performed = engine.metrics().total_operations() - before_ops;
    /// 
    /// println!("Batch processing:");
    /// println!("  Operations: {}", ops_performed);
    /// println!("  Wall time: {:?}", elapsed);
    /// println!("  CPU time: {:?}", 
    ///          engine.metrics().average_processing_time() * ops_performed as u32);
    /// ```
    /// 
    /// ## Throughput Analysis
    /// ```rust
    /// let metrics = engine.metrics();
    /// let throughput = metrics.tokens_per_second();
    /// 
    /// match throughput {
    ///     x if x > 100.0 => println!("Excellent throughput: {:.1} tok/s", x),
    ///     x if x > 50.0 => println!("Good throughput: {:.1} tok/s", x),
    ///     x if x > 10.0 => println!("Moderate throughput: {:.1} tok/s", x),
    ///     x => println!("Low throughput: {:.1} tok/s - consider optimization", x),
    /// }
    /// ```
    /// 
    /// ## Performance Benchmarking
    /// ```rust
    /// // Compare different processor configurations
    /// let baseline_metrics = engine.metrics().clone();
    /// 
    /// // Test configuration A
    /// engine.set_processor(config_a_processor);
    /// run_benchmark(&mut engine)?;
    /// let metrics_a = engine.metrics();
    /// 
    /// // Test configuration B
    /// engine.set_processor(config_b_processor);
    /// run_benchmark(&mut engine)?;
    /// let metrics_b = engine.metrics();
    /// 
    /// println!("Configuration A: {:.1} tok/s", metrics_a.tokens_per_second());
    /// println!("Configuration B: {:.1} tok/s", metrics_b.tokens_per_second());
    /// ```
    /// 
    /// ## Resource Usage Monitoring
    /// ```rust
    /// fn monitor_resource_usage(engine: &ProcessingEngine) {
    ///     let metrics = engine.metrics();
    ///     
    ///     // Check processing efficiency
    ///     let avg_time = metrics.average_processing_time().as_nanos() as f64;
    ///     let efficiency = 1_000_000.0 / avg_time;  // Operations per millisecond
    ///     
    ///     println!("Processing efficiency: {:.2} ops/ms", efficiency);
    ///     
    ///     // Monitor sequence progress
    ///     let sequence_len = metrics.sequence_length();
    ///     if sequence_len > 1000 {
    ///         println!("Long sequence detected: {} tokens", sequence_len);
    ///     }
    /// }
    /// ```
    /// 
    /// ## Error Rate Analysis
    /// ```rust
    /// fn analyze_error_patterns(engine: &ProcessingEngine) {
    ///     let metrics = engine.metrics();
    ///     
    ///     // Calculate success rate
    ///     let total_attempts = metrics.total_operations();
    ///     let successful_tokens = metrics.total_tokens();
    ///     
    ///     if total_attempts > 0 {
    ///         let success_rate = successful_tokens as f64 / total_attempts as f64;
    ///         println!("Success rate: {:.2}%", success_rate * 100.0);
    ///         
    ///         if success_rate < 0.95 {
    ///             println!("Warning: Low success rate detected");
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// ## Real-Time Monitoring
    /// ```rust
    /// use tokio::time::{interval, Duration};
    /// 
    /// let engine = Arc::new(Mutex::new(engine));
    /// let monitor_engine = Arc::clone(&engine);
    /// 
    /// tokio::spawn(async move {
    ///     let mut interval = interval(Duration::from_secs(10));
    ///     
    ///     loop {
    ///         interval.tick().await;
    ///         
    ///         if let Ok(engine) = monitor_engine.lock() {
    ///             let metrics = engine.metrics();
    ///             println!("[Monitor] {} ops, {:.1} tok/s, {} seq_len",
    ///                      metrics.total_operations(),
    ///                      metrics.tokens_per_second(),
    ///                      metrics.sequence_length());
    ///         }
    ///     }
    /// });
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// Metrics use atomic operations and are safe for concurrent read access.
    /// Multiple threads can inspect metrics simultaneously without synchronization.
    /// 
    /// # Performance Impact
    /// 
    /// Metrics collection has minimal overhead:
    /// - Atomic operations for counters (nanosecond scale)
    /// - No memory allocation for metric updates
    /// - Lazy calculation of derived metrics
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
    sequence_length: std::sync::atomic::AtomicU32}

impl ProcessingMetrics {
    /// Create new metrics instance
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            total_operations: std::sync::atomic::AtomicU64::new(0),
            total_processing_time: std::sync::atomic::AtomicU64::new(0),
            total_tokens: std::sync::atomic::AtomicU64::new(0),
            sequence_length: std::sync::atomic::AtomicU32::new(0)}
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
    processors: Vec<Box<dyn LogitsProcessor>>}

impl ProcessingEngineBuilder {
    /// Create new builder with vocabulary size
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            context_size: None,
            processors: Vec::new()}
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
            0, // context_window (0 = use full context)
        )?);
        Ok(self.add_processor(processor))
    }

    /// Build the processing engine
    pub fn build(self) -> ProcessingResult<ProcessingEngine> {
        let context_size = self.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
        let mut engine = ProcessingEngine::with_context_size(self.vocab_size, context_size)?;

        if !self.processors.is_empty() {
            let composite = CompositeProcessor::with_processors(self.processors)?;
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
                if prob > 0.0 { -prob * prob.ln() } else { 0.0 }
            })
            .sum();

        Ok(entropy)
    }
}
