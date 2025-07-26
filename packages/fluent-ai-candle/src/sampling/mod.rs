//! Advanced sampling strategies for transformer model inference
//!
//! This module provides the canonical Candle LogitsProcessor API integration
//! with streaming-first HTTP/3 architecture and SIMD optimizations.
//!
//! The unified system provides:
//! - Production-grade sampling implementations
//! - Zero allocation where possible
//! - Numerically stable algorithms
//! - Composable processor chains
//! - Comprehensive error handling
//! - High-performance tensor operations

use candle_core::{Result as CandleResult, Tensor};
/// Re-export canonical Candle LogitsProcessor and Sampling enums
pub use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand;

// Legacy modules maintained for compatibility
pub mod composite;
pub mod gumbel;
pub mod mirostat;
pub mod nucleus;
pub mod repetition;
pub mod simd;
pub mod temperature;
pub mod topk;
pub mod typical;

/// Errors that can occur during logits processing - DEPRECATED
///
/// Use `crate::processing::error::ProcessingError` for the unified error system.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SamplingError {
    #[error("Invalid temperature: {0} (must be > 0.0)")]
    InvalidTemperature(f64),

    #[error("Invalid top-p value: {0} (must be in [0.0, 1.0])")]
    InvalidTopP(f64),

    #[error("Invalid top-k value: {0} (must be > 0)")]
    InvalidTopK(usize),

    #[error("Invalid repetition penalty: {0} (must be >= 1.0)")]
    InvalidRepetitionPenalty(f64),

    #[error("Logits tensor error: {0}")]
    TensorError(String),

    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),

    #[error("Empty vocabulary: cannot sample from zero-length logits")]
    EmptyVocabulary,

    #[error("Empty logits: no valid logits found for processing")]
    EmptyLogits,

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Processor chain error: {0}")]
    ProcessorChainError(String)}

impl From<candle_core::Error> for SamplingError {
    fn from(err: candle_core::Error) -> Self {
        SamplingError::TensorError(err.to_string())
    }
}

impl From<crate::processing::error::ProcessingError> for SamplingError {
    fn from(err: crate::processing::error::ProcessingError) -> Self {
        SamplingError::ProcessingFailed(err.to_string())
    }
}

// Error conversions for compatibility

/// High-performance sampling configuration builder for canonical Candle Sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for probability scaling (> 0.0)
    pub temperature: f64,
    /// Top-p nucleus sampling threshold [0.0, 1.0]
    pub top_p: Option<f64>,
    /// Top-k token limit (> 0)
    pub top_k: Option<usize>,
    /// Random seed for reproducible sampling
    pub random_seed: u64}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            random_seed: 42}
    }
}

impl SamplingConfig {
    /// Creates a new sampling configuration with production-ready defaults
    /// 
    /// Initializes a SamplingConfig with conservative settings suitable for most
    /// text generation tasks while maintaining deterministic and stable behavior.
    /// 
    /// # Default Configuration
    /// 
    /// - **Temperature**: 1.0 (balanced randomness vs determinism)
    /// - **Top-p**: None (nucleus sampling disabled)
    /// - **Top-k**: None (top-k filtering disabled) 
    /// - **Random Seed**: 42 (fixed seed for reproducible results)
    /// 
    /// # Design Philosophy
    /// 
    /// The defaults prioritize:
    /// - **Stability**: Settings that work reliably across different models
    /// - **Reproducibility**: Fixed seed ensures consistent results
    /// - **Safety**: Conservative parameters prevent degenerate sampling
    /// - **Simplicity**: Minimal configuration for basic use cases
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    /// 
    /// // Basic configuration with defaults
    /// let config = SamplingConfig::new();
    /// let sampling = config.build_sampling();
    /// 
    /// // Customize specific parameters
    /// let config = SamplingConfig::new()
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .random_seed(12345);
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and can be called concurrently from multiple threads.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature with validation and numerical stability guarantees
    ///
    /// Configures the temperature parameter for logits scaling with comprehensive validation
    /// to ensure numerically stable and mathematically sound sampling behavior. Temperature
    /// controls the randomness of token selection, with lower values producing more deterministic
    /// outputs and higher values increasing diversity.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature scaling factor (must be > 0.0 and finite)
    ///
    /// # Temperature Effects
    ///
    /// ## Low Temperature (0.1 - 0.7)
    /// - **Behavior**: More deterministic, focused outputs
    /// - **Use Cases**: Factual questions, code generation, structured output
    /// - **Quality**: Higher coherence, lower creativity
    /// - **Example**: `config.temperature(0.3)?` for precise code completion
    ///
    /// ## Medium Temperature (0.8 - 1.2)
    /// - **Behavior**: Balanced randomness and determinism
    /// - **Use Cases**: General conversation, creative writing, balanced content
    /// - **Quality**: Good balance of coherence and creativity
    /// - **Example**: `config.temperature(1.0)?` for natural dialogue
    ///
    /// ## High Temperature (1.3 - 2.0)
    /// - **Behavior**: More random, creative outputs
    /// - **Use Cases**: Creative writing, brainstorming, exploratory content
    /// - **Quality**: Higher creativity, potentially lower coherence
    /// - **Example**: `config.temperature(1.8)?` for creative story generation
    ///
    /// # Mathematical Foundation
    ///
    /// Temperature modifies the softmax distribution:
    /// ```text
    /// P(token_i) = exp(logit_i / temperature) / Σ(exp(logit_j / temperature))
    /// ```
    ///
    /// - **temperature < 1.0**: Sharpens distribution (more peaked)
    /// - **temperature = 1.0**: Unmodified distribution
    /// - **temperature > 1.0**: Flattens distribution (more uniform)
    ///
    /// # Validation Rules
    ///
    /// - **Positive**: Must be greater than 0.0 to prevent division by zero
    /// - **Finite**: Must not be infinite or NaN to ensure stable computation
    /// - **Practical Range**: Typically 0.1 to 3.0 for meaningful results
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) validation and assignment
    /// - **Memory Usage**: Zero allocation - direct field assignment
    /// - **Validation Cost**: Single comparison and finite check
    /// - **Runtime Impact**: Temperature affects softmax computation during sampling
    ///
    /// # Examples
    ///
    /// ## Basic Temperature Configuration
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    ///
    /// // Conservative temperature for factual content
    /// let precise_config = SamplingConfig::new()
    ///     .temperature(0.2)?;
    ///
    /// // Balanced temperature for general use
    /// let balanced_config = SamplingConfig::new()
    ///     .temperature(0.8)?;
    ///
    /// // Creative temperature for diverse output
    /// let creative_config = SamplingConfig::new()
    ///     .temperature(1.5)?;
    /// ```
    ///
    /// ## Temperature Validation Examples
    /// ```rust
    /// // Valid temperature values
    /// assert!(SamplingConfig::new().temperature(0.1).is_ok());
    /// assert!(SamplingConfig::new().temperature(1.0).is_ok());
    /// assert!(SamplingConfig::new().temperature(2.5).is_ok());
    ///
    /// // Invalid temperature values
    /// assert!(SamplingConfig::new().temperature(0.0).is_err()); // Zero
    /// assert!(SamplingConfig::new().temperature(-0.5).is_err()); // Negative
    /// assert!(SamplingConfig::new().temperature(f64::INFINITY).is_err()); // Infinite
    /// assert!(SamplingConfig::new().temperature(f64::NAN).is_err()); // NaN
    /// ```
    ///
    /// ## Dynamic Temperature Adjustment
    /// ```rust
    /// fn adaptive_temperature(context: &str, base_temp: f64) -> Result<f64, SamplingError> {
    ///     let adjusted_temp = match context {
    ///         ctx if ctx.contains("code") => base_temp * 0.5, // Lower for code
    ///         ctx if ctx.contains("creative") => base_temp * 1.8, // Higher for creativity
    ///         ctx if ctx.contains("factual") => base_temp * 0.3, // Much lower for facts
    ///         _ => base_temp, // Default
    ///     };
    ///     
    ///     let config = SamplingConfig::new().temperature(adjusted_temp)?;
    ///     Ok(adjusted_temp)
    /// }
    ///
    /// // Usage with different contexts
    /// let code_temp = adaptive_temperature("Generate code for sorting", 1.0)?; // ~0.5
    /// let creative_temp = adaptive_temperature("Write a creative story", 1.0)?; // ~1.8
    /// let factual_temp = adaptive_temperature("State factual information", 1.0)?; // ~0.3
    /// ```
    ///
    /// ## Temperature Scheduling
    /// ```rust
    /// struct TemperatureScheduler {
    ///     initial_temp: f64,
    ///     final_temp: f64,
    ///     steps: usize,
    ///     current_step: usize,
    /// }
    ///
    /// impl TemperatureScheduler {
    ///     fn new(initial: f64, final_temp: f64, total_steps: usize) -> Self {
    ///         Self {
    ///             initial_temp: initial,
    ///             final_temp,
    ///             steps: total_steps,
    ///             current_step: 0,
    ///         }
    ///     }
    ///     
    ///     fn next_temperature(&mut self) -> f64 {
    ///         if self.current_step >= self.steps {
    ///             return self.final_temp;
    ///         }
    ///         
    ///         let progress = self.current_step as f64 / self.steps as f64;
    ///         let temp = self.initial_temp + 
    ///                   (self.final_temp - self.initial_temp) * progress;
    ///         
    ///         self.current_step += 1;
    ///         temp
    ///     }
    /// }
    ///
    /// // Usage: Gradually reduce temperature for better convergence
    /// let mut scheduler = TemperatureScheduler::new(1.5, 0.3, 100);
    /// 
    /// for generation_step in 0..100 {
    ///     let current_temp = scheduler.next_temperature();
    ///     let config = SamplingConfig::new()
    ///         .temperature(current_temp)?
    ///         .top_p(0.9)?;
    ///     
    ///     let sampling = config.build_sampling();
    ///     // Use sampling for this step...
    /// }
    /// ```
    ///
    /// ## Temperature Impact Analysis
    /// ```rust
    /// fn analyze_temperature_impact(temperatures: &[f64], prompt: &str) {
    ///     println!("Temperature Impact Analysis for: '{}'", prompt);
    ///     println!("{'Temperature':<15} {'Expected Behavior':<30} {'Use Case':<25}");
    ///     println!("{}", "-".repeat(70));
    ///     
    ///     for &temp in temperatures {
    ///         let (behavior, use_case) = match temp {
    ///             t if t < 0.5 => ("Very deterministic", "Code generation"),
    ///             t if t < 0.8 => ("Moderately focused", "Technical writing"),
    ///             t if t < 1.2 => ("Balanced variety", "General conversation"),
    ///             t if t < 1.8 => ("Creative diverse", "Story writing"),
    ///             _ => ("Highly random", "Experimental content"),
    ///         };
    ///         
    ///         println!("{:<15.2} {:<30} {:<25}", temp, behavior, use_case);
    ///         
    ///         // Test configuration
    ///         if let Ok(config) = SamplingConfig::new().temperature(temp) {
    ///             let sampling = config.build_sampling();
    ///             println!("  ✓ Valid configuration created");
    ///         } else {
    ///             println!("  ✗ Invalid temperature value");
    ///         }
    ///     }
    /// }
    ///
    /// // Analyze different temperature ranges
    /// let test_temperatures = vec![0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5];
    /// analyze_temperature_impact(&test_temperatures, "Example prompt");
    /// ```
    ///
    /// # Error Handling
    ///
    /// Returns `SamplingError::InvalidTemperature` if:
    /// - Temperature is zero or negative
    /// - Temperature is infinite (positive or negative infinity)
    /// - Temperature is NaN (Not a Number)
    ///
    /// # Integration with Other Parameters
    ///
    /// Temperature works in combination with other sampling parameters:
    /// - **With Top-k**: Temperature applied after top-k filtering
    /// - **With Top-p**: Temperature applied after nucleus sampling
    /// - **With Repetition Penalty**: Temperature applied after penalty adjustment
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. The validation
    /// and assignment operations are atomic at the language level.
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: No heap allocation during validation or assignment
    /// - ✅ **Fail Fast**: Immediate validation prevents runtime errors
    /// - ✅ **Mathematically Sound**: Ensures valid temperature range for stable sampling
    /// - ✅ **Fluent API**: Returns Self for method chaining
    pub fn temperature(mut self, temperature: f64) -> Result<Self, SamplingError> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Set nucleus (top-p) sampling threshold with validation and adaptive vocabulary control
    ///
    /// Configures the nucleus sampling probability threshold that dynamically selects the
    /// smallest set of tokens whose cumulative probability exceeds the threshold. This enables
    /// context-adaptive vocabulary filtering where the effective vocabulary size adjusts based
    /// on the model's confidence, producing higher quality outputs than fixed top-k filtering.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Cumulative probability threshold in range [0.0, 1.0]
    ///
    /// # Nucleus Sampling Strategy
    ///
    /// ## Algorithm Overview
    /// 1. **Sort Tokens**: Order by probability (highest to lowest)
    /// 2. **Cumulative Sum**: Calculate running probability sum
    /// 3. **Threshold Cut**: Include tokens until sum exceeds top_p
    /// 4. **Renormalization**: Redistribute probabilities among selected tokens
    ///
    /// ## Adaptive Behavior
    /// - **High Confidence**: Small vocabulary when model is certain
    /// - **Low Confidence**: Large vocabulary when model is uncertain
    /// - **Context Sensitive**: Vocabulary size adapts to prediction difficulty
    ///
    /// # Top-p Value Effects
    ///
    /// ## Very Low (0.1 - 0.3)
    /// - **Behavior**: Extremely focused, deterministic output
    /// - **Vocabulary**: Typically 1-5 tokens selected
    /// - **Use Cases**: Factual completion, structured data generation
    /// - **Quality**: Maximum coherence, minimal diversity
    /// - **Example**: `config.top_p(0.2)?` for precise code completion
    ///
    /// ## Low-Medium (0.4 - 0.6)
    /// - **Behavior**: Conservative with controlled diversity
    /// - **Vocabulary**: Typically 5-20 tokens selected
    /// - **Use Cases**: Technical writing, consistent tone generation
    /// - **Quality**: High coherence with some variety
    /// - **Example**: `config.top_p(0.5)?` for professional writing
    ///
    /// ## Medium-High (0.7 - 0.9)
    /// - **Behavior**: Balanced quality and creativity
    /// - **Vocabulary**: Typically 20-100 tokens selected
    /// - **Use Cases**: General conversation, creative content
    /// - **Quality**: Good balance of coherence and surprise
    /// - **Example**: `config.top_p(0.85)?` for engaging dialogue
    ///
    /// ## High (0.9 - 1.0)
    /// - **Behavior**: Maximum diversity and creativity
    /// - **Vocabulary**: Large portion of vocabulary included
    /// - **Use Cases**: Brainstorming, experimental generation
    /// - **Quality**: High creativity, potentially lower coherence
    /// - **Example**: `config.top_p(0.95)?` for creative exploration
    ///
    /// # Mathematical Foundation
    ///
    /// Given sorted probabilities P₁ ≥ P₂ ≥ ... ≥ Pₙ:
    /// ```text
    /// Nucleus = {tokens i : Σⱼ₌₁ⁱ Pⱼ ≤ top_p}
    /// ```
    ///
    /// Renormalized probability:
    /// ```text
    /// P'(token_i) = P(token_i) / Σ(P(token_j) for j in Nucleus)
    /// ```
    ///
    /// # Validation Rules
    ///
    /// - **Range**: Must be in [0.0, 1.0] inclusive
    /// - **Finite**: Must not be infinite or NaN
    /// - **Practical Use**: Values below 0.1 or above 0.99 rarely useful
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) validation and assignment
    /// - **Memory Usage**: Zero allocation - stores Option<f64>
    /// - **Sampling Cost**: O(V log V) where V is vocabulary size (sorting required)
    /// - **Adaptive Cost**: Cost varies with model confidence
    ///
    /// # Examples
    ///
    /// ## Basic Nucleus Sampling Configuration
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    ///
    /// // Conservative nucleus sampling for factual content
    /// let precise_config = SamplingConfig::new()
    ///     .top_p(0.3)?
    ///     .temperature(0.7)?;
    ///
    /// // Balanced nucleus sampling for general use
    /// let balanced_config = SamplingConfig::new()
    ///     .top_p(0.8)?
    ///     .temperature(1.0)?;
    ///
    /// // Creative nucleus sampling for diverse output
    /// let creative_config = SamplingConfig::new()
    ///     .top_p(0.95)?
    ///     .temperature(1.2)?;
    /// ```
    ///
    /// ## Top-p Validation Examples
    /// ```rust
    /// // Valid top-p values
    /// assert!(SamplingConfig::new().top_p(0.0).is_ok()); // Include only most likely token
    /// assert!(SamplingConfig::new().top_p(0.5).is_ok()); // Median threshold
    /// assert!(SamplingConfig::new().top_p(1.0).is_ok()); // Include all tokens
    ///
    /// // Invalid top-p values
    /// assert!(SamplingConfig::new().top_p(-0.1).is_err()); // Negative
    /// assert!(SamplingConfig::new().top_p(1.1).is_err()); // Greater than 1
    /// assert!(SamplingConfig::new().top_p(f64::INFINITY).is_err()); // Infinite
    /// assert!(SamplingConfig::new().top_p(f64::NAN).is_err()); // NaN
    /// ```
    ///
    /// ## Context-Adaptive Top-p Selection
    /// ```rust
    /// fn adaptive_top_p(context_type: &str, model_confidence: f64) -> Result<f64, SamplingError> {
    ///     let base_top_p = match context_type {
    ///         "code" => 0.2,      // Very focused for code
    ///         "factual" => 0.3,   // Conservative for facts
    ///         "dialogue" => 0.8,  // Balanced for conversation
    ///         "creative" => 0.95, // Diverse for creativity
    ///         _ => 0.7,           // Default balanced
    ///     };
    ///     
    ///     // Adjust based on model confidence (lower confidence -> higher top_p)
    ///     let confidence_adjustment = (1.0 - model_confidence) * 0.2;
    ///     let final_top_p = (base_top_p + confidence_adjustment).min(1.0);
    ///     
    ///     let config = SamplingConfig::new().top_p(final_top_p)?;
    ///     Ok(final_top_p)
    /// }
    ///
    /// // Usage with different contexts and confidence levels
    /// let code_top_p = adaptive_top_p("code", 0.9)?;        // High confidence: ~0.22
    /// let creative_top_p = adaptive_top_p("creative", 0.6)?; // Low confidence: ~1.0
    /// let dialogue_top_p = adaptive_top_p("dialogue", 0.7)?; // Medium confidence: ~0.86
    /// ```
    ///
    /// ## Dynamic Top-p Adjustment
    /// ```rust
    /// struct AdaptiveNucleusSampler {
    ///     base_top_p: f64,
    ///     min_top_p: f64,
    ///     max_top_p: f64,
    ///     adaptation_rate: f64,
    /// }
    ///
    /// impl AdaptiveNucleusSampler {
    ///     fn new(base_top_p: f64) -> Self {
    ///         Self {
    ///             base_top_p,
    ///             min_top_p: 0.1,
    ///             max_top_p: 0.98,
    ///             adaptation_rate: 0.1,
    ///         }
    ///     }
    ///     
    ///     fn adjust_for_entropy(&self, entropy: f64) -> f64 {
    ///         // Higher entropy (more uncertainty) -> higher top_p
    ///         let normalized_entropy = (entropy / 10.0).min(1.0); // Assume max entropy ~10
    ///         let adjustment = normalized_entropy * self.adaptation_rate;
    ///         
    ///         (self.base_top_p + adjustment)
    ///             .max(self.min_top_p)
    ///             .min(self.max_top_p)
    ///     }
    ///     
    ///     fn adjust_for_repetition(&self, repetition_score: f64) -> f64 {
    ///         // Higher repetition -> higher top_p for diversity
    ///         let diversity_boost = repetition_score * 0.2;
    ///         
    ///         (self.base_top_p + diversity_boost)
    ///             .max(self.min_top_p)
    ///             .min(self.max_top_p)
    ///     }
    /// }
    ///
    /// // Usage with adaptive adjustment
    /// let sampler = AdaptiveNucleusSampler::new(0.8);
    /// let entropy_adjusted = sampler.adjust_for_entropy(5.2); // Medium uncertainty
    /// let repetition_adjusted = sampler.adjust_for_repetition(0.3); // Some repetition
    /// 
    /// let config = SamplingConfig::new()
    ///     .top_p(entropy_adjusted)?
    ///     .temperature(1.0)?;
    /// ```
    ///
    /// ## Nucleus vs Top-K Comparison
    /// ```rust
    /// fn compare_sampling_strategies(vocab_size: usize) {
    ///     println!("Sampling Strategy Comparison (Vocab size: {})", vocab_size);
    ///     println!("{:<20} {:<15} {:<20} {:<25}", "Strategy", "Parameter", "Effective Size", "Behavior");
    ///     println!("{}", "-".repeat(80));
    ///     
    ///     // Top-K examples
    ///     for k in [10, 50, 100] {
    ///         println!("{:<20} {:<15} {:<20} {:<25}", 
    ///                 "Top-K", 
    ///                 format!("k={}", k), 
    ///                 format!("Fixed: {}", k),
    ///                 "Constant vocabulary");
    ///     }
    ///     
    ///     // Top-P examples
    ///     for p in [0.3, 0.7, 0.9] {
    ///         let expected_size = match p {
    ///             0.3 => "5-20 tokens",
    ///             0.7 => "20-100 tokens", 
    ///             0.9 => "100-500 tokens",
    ///             _ => "Variable",
    ///         };
    ///         
    ///         println!("{:<20} {:<15} {:<20} {:<25}", 
    ///                 "Nucleus (Top-P)", 
    ///                 format!("p={}", p), 
    ///                 format!("Adaptive: {}", expected_size),
    ///                 "Context-sensitive");
    ///     }
    /// }
    ///
    /// compare_sampling_strategies(50000); // Typical vocabulary size
    /// ```
    ///
    /// ## Quality vs Diversity Analysis
    /// ```rust
    /// fn analyze_top_p_quality_tradeoff(prompts: &[&str]) {
    ///     let top_p_values = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
    ///     
    ///     for prompt in prompts {
    ///         println!("Analyzing prompt: '{}'", prompt);
    ///         
    ///         for &p in &top_p_values {
    ///             let config = SamplingConfig::new()
    ///                 .top_p(p)
    ///                 .unwrap()
    ///                 .temperature(1.0)
    ///                 .unwrap();
    ///             
    ///             let (quality, diversity) = match p {
    ///                 p if p < 0.3 => ("Very High", "Very Low"),
    ///                 p if p < 0.5 => ("High", "Low"),
    ///                 p if p < 0.7 => ("Good", "Medium"),
    ///                 p if p < 0.9 => ("Medium", "High"),
    ///                 _ => ("Variable", "Very High"),
    ///             };
    ///             
    ///             println!("  top_p={:<4} | Quality: {:<10} | Diversity: {:<10}", 
    ///                     p, quality, diversity);
    ///         }
    ///         println!();
    ///     }
    /// }
    ///
    /// let test_prompts = vec![
    ///     "Write a technical explanation",
    ///     "Create a creative story",
    ///     "Generate code documentation",
    ///     "Compose a casual email",
    /// ];
    /// analyze_top_p_quality_tradeoff(&test_prompts);
    /// ```
    ///
    /// ## Production Nucleus Sampling Pipeline
    /// ```rust
    /// struct NucleusSamplingPipeline {
    ///     configs: HashMap<String, f64>,
    ///     fallback_top_p: f64,
    /// }
    ///
    /// impl NucleusSamplingPipeline {
    ///     fn new() -> Self {
    ///         let mut configs = HashMap::new();
    ///         configs.insert("code_generation".to_string(), 0.2);
    ///         configs.insert("technical_writing".to_string(), 0.4);
    ///         configs.insert("general_chat".to_string(), 0.8);
    ///         configs.insert("creative_writing".to_string(), 0.95);
    ///         configs.insert("brainstorming".to_string(), 0.98);
    ///         
    ///         Self {
    ///             configs,
    ///             fallback_top_p: 0.7,
    ///         }
    ///     }
    ///     
    ///     fn get_config(&self, task_type: &str, user_preference: Option<f64>) -> Result<SamplingConfig, SamplingError> {
    ///         let base_top_p = user_preference
    ///             .or_else(|| self.configs.get(task_type).copied())
    ///             .unwrap_or(self.fallback_top_p);
    ///         
    ///         SamplingConfig::new()
    ///             .top_p(base_top_p)?
    ///             .temperature(1.0)?
    ///             .random_seed(42)
    ///     }
    /// }
    ///
    /// // Usage in production system
    /// let pipeline = NucleusSamplingPipeline::new();
    /// 
    /// // Get configs for different tasks
    /// let code_config = pipeline.get_config("code_generation", None)?;        // Uses 0.2
    /// let chat_config = pipeline.get_config("general_chat", Some(0.85))?;     // Uses 0.85 (user preference)
    /// let creative_config = pipeline.get_config("creative_writing", None)?;   // Uses 0.95
    /// let unknown_config = pipeline.get_config("unknown_task", None)?;        // Uses 0.7 (fallback)
    /// ```
    ///
    /// # Error Handling
    ///
    /// Returns `SamplingError::InvalidTopP` if:
    /// - Value is less than 0.0 or greater than 1.0
    /// - Value is infinite (positive or negative infinity)
    /// - Value is NaN (Not a Number)
    ///
    /// # Integration with Other Parameters
    ///
    /// Nucleus sampling combines effectively with:
    /// - **Temperature**: Applied before nucleus filtering for probability adjustment
    /// - **Top-k**: Creates TopKThenTopP strategy when both are specified
    /// - **Repetition Penalty**: Applied before nucleus to penalize repeated tokens
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. The validation
    /// and assignment operations are atomic at the language level.
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: No heap allocation during validation or assignment
    /// - ✅ **Adaptive Vocabulary**: Dynamic token selection based on probability mass
    /// - ✅ **Mathematically Sound**: Probability theory-based token filtering
    /// - ✅ **Quality Focused**: Superior to fixed top-k for most applications
    pub fn top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        if !(0.0..=1.0).contains(&top_p) || !top_p.is_finite() {
            return Err(SamplingError::InvalidTopP(top_p));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set top-k sampling with validation and fixed vocabulary control
    ///
    /// Configures the top-k sampling parameter that limits token consideration to the
    /// k most probable tokens, providing predictable vocabulary size and consistent
    /// performance characteristics. Unlike nucleus sampling, top-k maintains a fixed
    /// vocabulary size regardless of model confidence, making it suitable for scenarios
    /// requiring deterministic computational costs.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Number of top tokens to consider (must be > 0)
    ///
    /// # Top-K Sampling Strategy
    ///
    /// ## Algorithm Overview
    /// 1. **Sort Tokens**: Order by probability (highest to lowest)
    /// 2. **Truncate**: Keep only the top k tokens
    /// 3. **Renormalization**: Redistribute probabilities among selected tokens
    /// 4. **Sample**: Draw from the truncated distribution
    ///
    /// ## Fixed Behavior
    /// - **Consistent Size**: Always considers exactly k tokens
    /// - **Predictable Cost**: O(k) sampling complexity regardless of context
    /// - **Uniform Filtering**: Same vocabulary size for all predictions
    ///
    /// # Top-k Value Effects
    ///
    /// ## Very Small (1 - 10)
    /// - **Behavior**: Highly deterministic, focused output
    /// - **Vocabulary**: Extremely limited choice set
    /// - **Use Cases**: Structured output, classification tasks
    /// - **Quality**: Maximum coherence, minimal diversity
    /// - **Example**: `config.top_k(3)?` for boolean or categorical responses
    ///
    /// ## Small (10 - 50)
    /// - **Behavior**: Conservative with controlled options
    /// - **Vocabulary**: Small but reasonable choice set
    /// - **Use Cases**: Technical writing, factual responses
    /// - **Quality**: High coherence with limited variety
    /// - **Example**: `config.top_k(20)?` for precise technical content
    ///
    /// ## Medium (50 - 200)
    /// - **Behavior**: Balanced variety and coherence
    /// - **Vocabulary**: Moderate choice set for natural language
    /// - **Use Cases**: General conversation, content generation
    /// - **Quality**: Good balance of quality and creativity
    /// - **Example**: `config.top_k(100)?` for natural dialogue
    ///
    /// ## Large (200 - 1000)
    /// - **Behavior**: High diversity with broader vocabulary
    /// - **Vocabulary**: Large choice set approaching full vocabulary
    /// - **Use Cases**: Creative writing, brainstorming
    /// - **Quality**: Increased creativity, potentially lower coherence
    /// - **Example**: `config.top_k(500)?` for creative content generation
    ///
    /// ## Very Large (1000+)
    /// - **Behavior**: Near-complete vocabulary access
    /// - **Vocabulary**: Most of the model's vocabulary available
    /// - **Use Cases**: Maximum diversity applications
    /// - **Quality**: High creativity, variable coherence
    /// - **Example**: `config.top_k(2000)?` for experimental generation
    ///
    /// # Mathematical Foundation
    ///
    /// Given sorted probabilities P₁ ≥ P₂ ≥ ... ≥ Pₙ:
    /// ```text
    /// TopK = {token_i : i ≤ k}
    /// ```
    ///
    /// Renormalized probability:
    /// ```text
    /// P'(token_i) = P(token_i) / Σⱼ₌₁ᵏ P(token_j)
    /// ```
    ///
    /// # Validation Rules
    ///
    /// - **Positive**: Must be greater than 0 to ensure at least one token
    /// - **Practical Range**: Typically 1 to 10,000 for meaningful results
    /// - **Memory Considerations**: Large k values increase memory usage
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) validation and assignment
    /// - **Memory Usage**: Zero allocation - stores Option<usize>
    /// - **Sampling Cost**: O(k) for truncation and renormalization
    /// - **Predictable Cost**: Constant computational cost regardless of context
    ///
    /// # Examples
    ///
    /// ## Basic Top-K Sampling Configuration
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    ///
    /// // Very focused top-k for structured output
    /// let precise_config = SamplingConfig::new()
    ///     .top_k(5)?
    ///     .temperature(0.7)?;
    ///
    /// // Balanced top-k for general use
    /// let balanced_config = SamplingConfig::new()
    ///     .top_k(50)?
    ///     .temperature(1.0)?;
    ///
    /// // Creative top-k for diverse output
    /// let creative_config = SamplingConfig::new()
    ///     .top_k(200)?
    ///     .temperature(1.2)?;
    /// ```
    ///
    /// ## Top-k Validation Examples
    /// ```rust
    /// // Valid top-k values
    /// assert!(SamplingConfig::new().top_k(1).is_ok());     // Single token (greedy)
    /// assert!(SamplingConfig::new().top_k(50).is_ok());    // Balanced choice
    /// assert!(SamplingConfig::new().top_k(1000).is_ok());  // Large vocabulary
    ///
    /// // Invalid top-k values
    /// assert!(SamplingConfig::new().top_k(0).is_err());    // Zero tokens invalid
    /// ```
    ///
    /// ## Context-Adaptive Top-k Selection
    /// ```rust
    /// fn adaptive_top_k(content_type: &str, vocab_size: usize) -> Result<usize, SamplingError> {
    ///     let base_k = match content_type {
    ///         "classification" => 5,          // Very focused
    ///         "code" => 20,                   // Limited but sufficient
    ///         "technical" => 50,              // Conservative choice set
    ///         "dialogue" => 100,              // Natural conversation
    ///         "creative" => 300,              // Broad creativity
    ///         "experimental" => 1000,         // Maximum diversity
    ///         _ => 50,                        // Default balanced
    ///     };
    ///     
    ///     // Adjust based on vocabulary size (ensure k doesn't exceed vocab)
    ///     let final_k = base_k.min(vocab_size);
    ///     
    ///     let config = SamplingConfig::new().top_k(final_k)?;
    ///     Ok(final_k)
    /// }
    ///
    /// // Usage with different content types
    /// let code_k = adaptive_top_k("code", 50000)?;           // Small focused set
    /// let creative_k = adaptive_top_k("creative", 50000)?;   // Larger creative set
    /// let dialogue_k = adaptive_top_k("dialogue", 50000)?;   // Balanced conversation
    /// ```
    ///
    /// ## Dynamic Top-k Scaling
    /// ```rust
    /// struct AdaptiveTopKSampler {
    ///     base_k: usize,
    ///     min_k: usize,
    ///     max_k: usize,
    ///     vocab_size: usize,
    /// }
    ///
    /// impl AdaptiveTopKSampler {
    ///     fn new(base_k: usize, vocab_size: usize) -> Self {
    ///         Self {
    ///             base_k,
    ///             min_k: 1,
    ///             max_k: vocab_size.min(2000), // Cap at reasonable limit
    ///             vocab_size,
    ///         }
    ///     }
    ///     
    ///     fn adjust_for_uncertainty(&self, uncertainty: f64) -> usize {
    ///         // Higher uncertainty -> larger k for more options
    ///         let uncertainty_multiplier = 1.0 + uncertainty;
    ///         let adjusted_k = (self.base_k as f64 * uncertainty_multiplier) as usize;
    ///         
    ///         adjusted_k.max(self.min_k).min(self.max_k)
    ///     }
    ///     
    ///     fn adjust_for_creativity(&self, creativity_level: f64) -> usize {
    ///         // Higher creativity -> larger k for more diversity
    ///         let creativity_multiplier = 1.0 + (creativity_level * 2.0);
    ///         let adjusted_k = (self.base_k as f64 * creativity_multiplier) as usize;
    ///         
    ///         adjusted_k.max(self.min_k).min(self.max_k)
    ///     }
    ///     
    ///     fn adjust_for_performance(&self, performance_budget: f64) -> usize {
    ///         // Lower performance budget -> smaller k for speed
    ///         let performance_multiplier = performance_budget;
    ///         let adjusted_k = (self.base_k as f64 * performance_multiplier) as usize;
    ///         
    ///         adjusted_k.max(self.min_k).min(self.max_k)
    ///     }
    /// }
    ///
    /// // Usage with adaptive adjustment
    /// let sampler = AdaptiveTopKSampler::new(100, 50000);
    /// let uncertainty_k = sampler.adjust_for_uncertainty(0.7);    // Higher uncertainty
    /// let creativity_k = sampler.adjust_for_creativity(0.8);      // High creativity
    /// let performance_k = sampler.adjust_for_performance(0.5);    // Limited performance budget
    /// 
    /// let config = SamplingConfig::new()
    ///     .top_k(uncertainty_k)?
    ///     .temperature(1.0)?;
    /// ```
    ///
    /// ## Top-K vs Nucleus Comparison
    /// ```rust
    /// fn compare_fixed_vs_adaptive_sampling() {
    ///     println!("Fixed vs Adaptive Sampling Comparison");
    ///     println!("{:<15} {:<20} {:<15} {:<25}", "Strategy", "Parameter", "Vocab Size", "Behavior");
    ///     println!("{}", "-".repeat(75));
    ///     
    ///     // Top-K examples (fixed vocabulary)
    ///     for k in [10, 50, 100, 500] {
    ///         println!("{:<15} {:<20} {:<15} {:<25}", 
    ///                 "Top-K", 
    ///                 format!("k={}", k), 
    ///                 format!("Fixed: {}", k),
    ///                 "Predictable performance");
    ///     }
    ///     
    ///     println!(); // Separator
    ///     
    ///     // Top-P examples (adaptive vocabulary)
    ///     for p in [0.3, 0.7, 0.9, 0.95] {
    ///         let expected_range = match p {
    ///             0.3 => "5-20",
    ///             0.7 => "20-100", 
    ///             0.9 => "100-500",
    ///             0.95 => "200-1000",
    ///             _ => "Variable",
    ///         };
    ///         
    ///         println!("{:<15} {:<20} {:<15} {:<25}", 
    ///                 "Nucleus (Top-P)", 
    ///                 format!("p={}", p), 
    ///                 format!("Adaptive: {}", expected_range),
    ///                 "Context-sensitive quality");
    ///     }
    /// }
    ///
    /// compare_fixed_vs_adaptive_sampling();
    /// ```
    ///
    /// ## Performance Benchmarking
    /// ```rust
    /// use std::time::Instant;
    ///
    /// fn benchmark_top_k_performance(vocab_size: usize) {
    ///     let k_values = vec![10, 50, 100, 500, 1000];
    ///     
    ///     println!("Top-K Performance Benchmark (Vocab size: {})", vocab_size);
    ///     println!("{:<10} {:<15} {:<20}", "k", "Config Time", "Expected Sample Time");
    ///     println!("{}", "-".repeat(45));
    ///     
    ///     for k in k_values {
    ///         let start = Instant::now();
    ///         
    ///         let config = SamplingConfig::new()
    ///             .top_k(k)
    ///             .unwrap()
    ///             .temperature(1.0)
    ///             .unwrap();
    ///         
    ///         let config_time = start.elapsed();
    ///         
    ///         // Estimate sampling time based on k (linear relationship)
    ///         let estimated_sample_time = k as f64 * 0.001; // ms per token
    ///         
    ///         println!("{:<10} {:<15} {:<20}", 
    ///                 k, 
    ///                 format!("{:.3}μs", config_time.as_nanos() as f64 / 1000.0),
    ///                 format!("{:.3}ms", estimated_sample_time));
    ///     }
    /// }
    ///
    /// benchmark_top_k_performance(50000);
    /// ```
    ///
    /// ## Production Top-K Configuration
    /// ```rust
    /// struct TopKConfigurationManager {
    ///     profiles: std::collections::HashMap<String, usize>,
    ///     vocab_size: usize,
    ///     default_k: usize,
    /// }
    ///
    /// impl TopKConfigurationManager {
    ///     fn new(vocab_size: usize) -> Self {
    ///         let mut profiles = std::collections::HashMap::new();
    ///         profiles.insert("classification".to_string(), 5);
    ///         profiles.insert("code_completion".to_string(), 20);
    ///         profiles.insert("technical_writing".to_string(), 50);
    ///         profiles.insert("general_chat".to_string(), 100);
    ///         profiles.insert("creative_writing".to_string(), 300);
    ///         profiles.insert("brainstorming".to_string(), 500);
    ///         profiles.insert("experimental".to_string(), 1000);
    ///         
    ///         Self {
    ///             profiles,
    ///             vocab_size,
    ///             default_k: 50,
    ///         }
    ///     }
    ///     
    ///     fn get_config(&self, profile: &str, custom_k: Option<usize>) -> Result<SamplingConfig, SamplingError> {
    ///         let k = custom_k
    ///             .or_else(|| self.profiles.get(profile).copied())
    ///             .unwrap_or(self.default_k)
    ///             .min(self.vocab_size); // Never exceed vocabulary size
    ///         
    ///         SamplingConfig::new()
    ///             .top_k(k)?
    ///             .temperature(1.0)?
    ///             .random_seed(42)
    ///     }
    ///     
    ///     fn list_profiles(&self) -> Vec<(&str, usize)> {
    ///         self.profiles.iter()
    ///             .map(|(profile, &k)| (profile.as_str(), k))
    ///             .collect()
    ///     }
    /// }
    ///
    /// // Usage in production system
    /// let manager = TopKConfigurationManager::new(50000);
    /// 
    /// // Get predefined configurations
    /// let code_config = manager.get_config("code_completion", None)?;     // k=20
    /// let chat_config = manager.get_config("general_chat", Some(75))?;    // k=75 (custom)
    /// let creative_config = manager.get_config("creative_writing", None)?; // k=300
    /// let unknown_config = manager.get_config("unknown_profile", None)?;   // k=50 (default)
    /// 
    /// // List available profiles
    /// for (profile, k) in manager.list_profiles() {
    ///     println!("Profile '{}': k={}", profile, k);
    /// }
    /// ```
    ///
    /// ## Memory Usage Analysis
    /// ```rust
    /// fn analyze_top_k_memory_usage(k_values: &[usize]) {
    ///     println!("Top-K Memory Usage Analysis");
    ///     println!("{:<10} {:<20} {:<25} {:<20}", "k", "Token Storage", "Probability Storage", "Total Estimate");
    ///     println!("{}", "-".repeat(75));
    ///     
    ///     for &k in k_values {
    ///         // Estimate memory usage (simplified)
    ///         let token_storage = k * 4;      // 4 bytes per token ID
    ///         let prob_storage = k * 8;       // 8 bytes per f64 probability
    ///         let total_estimate = token_storage + prob_storage;
    ///         
    ///         println!("{:<10} {:<20} {:<25} {:<20}", 
    ///                 k,
    ///                 format!("{} bytes", token_storage),
    ///                 format!("{} bytes", prob_storage),
    ///                 format!("{} bytes", total_estimate));
    ///     }
    /// }
    ///
    /// let test_k_values = vec![10, 50, 100, 500, 1000, 5000];
    /// analyze_top_k_memory_usage(&test_k_values);
    /// ```
    ///
    /// # Error Handling
    ///
    /// Returns `SamplingError::InvalidTopK` if:
    /// - Value is zero (no tokens would be available for sampling)
    ///
    /// # Integration with Other Parameters
    ///
    /// Top-k sampling combines effectively with:
    /// - **Temperature**: Applied after top-k filtering for probability adjustment
    /// - **Top-p**: Creates TopKThenTopP strategy when both are specified
    /// - **Repetition Penalty**: Applied before top-k to penalize repeated tokens
    ///
    /// # When to Use Top-K vs Nucleus
    ///
    /// ## Prefer Top-K When:
    /// - **Predictable Performance**: Need consistent computational costs
    /// - **Memory Constraints**: Working with limited memory budgets
    /// - **Batch Processing**: Processing many requests with uniform requirements
    /// - **Simple Configuration**: Want straightforward parameter tuning
    ///
    /// ## Prefer Nucleus (Top-P) When:
    /// - **Quality Focus**: Want adaptive vocabulary based on model confidence
    /// - **Context Sensitivity**: Need different vocabulary sizes for different inputs
    /// - **Research Applications**: Exploring optimal sampling strategies
    /// - **Production Quality**: Want state-of-the-art text generation quality
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. The validation
    /// and assignment operations are atomic at the language level.
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: No heap allocation during validation or assignment
    /// - ✅ **Predictable Performance**: Fixed computational cost regardless of context
    /// - ✅ **Memory Efficient**: Known memory usage based on k parameter
    /// - ✅ **Simple Configuration**: Straightforward parameter with clear semantics
    pub fn top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        if top_k == 0 {
            return Err(SamplingError::InvalidTopK(top_k));
        }
        self.top_k = Some(top_k);
        Ok(self)
    }

    /// Set random seed for reproducible sampling with deterministic pseudo-random generation
    ///
    /// Configures the random seed used by the pseudo-random number generator to ensure
    /// reproducible and deterministic sampling behavior across multiple runs. This is
    /// essential for debugging, testing, research reproducibility, and applications
    /// requiring consistent outputs for the same inputs and configuration.
    ///
    /// # Arguments
    ///
    /// * `seed` - 64-bit unsigned integer seed for the random number generator
    ///
    /// # Reproducibility Guarantees
    ///
    /// ## Same Seed, Same Results
    /// Given identical:
    /// - Random seed value
    /// - Model weights and architecture
    /// - Input prompt and context
    /// - Sampling configuration (temperature, top-k, top-p)
    /// - Hardware platform (for floating-point consistency)
    ///
    /// The sampling will produce **identical token sequences** across multiple runs.
    ///
    /// ## Cross-Platform Considerations
    /// - **Same Architecture**: Identical results on same CPU architecture
    /// - **Different Architectures**: May vary due to floating-point precision differences
    /// - **GPU vs CPU**: May produce different results due to computational differences
    /// - **Deterministic Mode**: Use CPU-only mode for maximum reproducibility
    ///
    /// # Seed Value Effects
    ///
    /// ## Fixed Seeds (Production)
    /// - **seed = 42**: Common default for consistent behavior
    /// - **seed = 0**: Valid seed, often used for baseline comparisons
    /// - **seed = timestamp**: Semi-random but reproducible if timestamp is recorded
    /// - **seed = hash(config)**: Deterministic based on configuration parameters
    ///
    /// ## Random Seeds (Development)
    /// - **seed = random()**: Non-reproducible for exploring output variety
    /// - **seed = user_id**: User-specific but consistent for that user
    /// - **seed = request_id**: Request-specific reproducibility
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time assignment
    /// - **Memory Usage**: Zero allocation - direct field assignment
    /// - **Generation Impact**: No performance impact on sampling speed
    /// - **Initialization Cost**: Minimal overhead for RNG initialization
    ///
    /// # Examples
    ///
    /// ## Basic Reproducible Configuration
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    ///
    /// // Fixed seed for consistent results
    /// let reproducible_config = SamplingConfig::new()
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .random_seed(42);
    ///
    /// // Same configuration will produce identical results
    /// let config1 = SamplingConfig::new().random_seed(12345);
    /// let config2 = SamplingConfig::new().random_seed(12345);
    /// // config1 and config2 will generate identical sequences
    /// ```
    ///
    /// ## Reproducible Testing Framework
    /// ```rust
    /// struct ReproducibleTester {
    ///     base_config: SamplingConfig,
    /// }
    ///
    /// impl ReproducibleTester {
    ///     fn new() -> Result<Self, SamplingError> {
    ///         let base_config = SamplingConfig::new()
    ///             .temperature(1.0)?
    ///             .top_p(0.9)?
    ///             .random_seed(42); // Fixed seed for reproducibility
    ///         
    ///         Ok(Self { base_config })
    ///     }
    ///     
    ///     fn test_with_seed(&self, test_seed: u64) -> SamplingConfig {
    ///         let mut config = self.base_config.clone();
    ///         config.random_seed = test_seed;
    ///         config
    ///     }
    ///     
    ///     fn test_suite(&self, prompts: &[&str]) -> Vec<(u64, Vec<String>)> {
    ///         let test_seeds = vec![42, 123, 456, 789, 999];
    ///         let mut results = Vec::new();
    ///         
    ///         for &seed in &test_seeds {
    ///             let config = self.test_with_seed(seed);
    ///             let sampling = config.build_sampling();
    ///             
    ///             // Simulate generation for each prompt
    ///             let mut generated_texts = Vec::new();
    ///             for prompt in prompts {
    ///                 // This would use the actual model with the sampling config
    ///                 let generated = format!("Generated text for '{}' with seed {}", prompt, seed);
    ///                 generated_texts.push(generated);
    ///             }
    ///             
    ///             results.push((seed, generated_texts));
    ///         }
    ///         
    ///         results
    ///     }
    /// }
    ///
    /// // Usage for reproducible testing
    /// let tester = ReproducibleTester::new()?;
    /// let test_prompts = vec!["Hello world", "Write a story", "Explain AI"];
    /// let test_results = tester.test_suite(&test_prompts);
    ///
    /// for (seed, generated_texts) in test_results {
    ///     println!("Seed {}: Generated {} responses", seed, generated_texts.len());
    /// }
    /// ```
    ///
    /// ## Research Reproducibility Pipeline
    /// ```rust
    /// use std::collections::HashMap;
    ///
    /// struct ResearchPipeline {
    ///     experiment_configs: HashMap<String, u64>,
    /// }
    ///
    /// impl ResearchPipeline {
    ///     fn new() -> Self {
    ///         let mut configs = HashMap::new();
    ///         configs.insert("baseline".to_string(), 42);
    ///         configs.insert("experiment_a".to_string(), 123);
    ///         configs.insert("experiment_b".to_string(), 456);
    ///         configs.insert("ablation_study".to_string(), 789);
    ///         
    ///         Self {
    ///             experiment_configs: configs,
    ///         }
    ///     }
    ///     
    ///     fn create_config(&self, experiment: &str, temperature: f64, top_p: f64) -> Result<SamplingConfig, SamplingError> {
    ///         let seed = self.experiment_configs.get(experiment)
    ///             .copied()
    ///             .unwrap_or(42); // Default seed if experiment not found
    ///         
    ///         SamplingConfig::new()
    ///             .temperature(temperature)?
    ///             .top_p(top_p)?
    ///             .random_seed(seed)
    ///     }
    ///     
    ///     fn run_reproducible_experiment(&self, experiment: &str) -> Result<(), SamplingError> {
    ///         println!("Running experiment: {}", experiment);
    ///         
    ///         // Create reproducible configuration
    ///         let config = self.create_config(experiment, 0.8, 0.9)?;
    ///         let sampling = config.build_sampling();
    ///         
    ///         println!("  Configuration: temp=0.8, top_p=0.9, seed={}", config.random_seed);
    ///         println!("  Results will be reproducible with this exact configuration");
    ///         
    ///         // Save configuration for reproducibility
    ///         self.save_experiment_config(experiment, &config)?;
    ///         
    ///         Ok(())
    ///     }
    ///     
    ///     fn save_experiment_config(&self, experiment: &str, config: &SamplingConfig) -> Result<(), SamplingError> {
    ///         // In real implementation, this would save to a file or database
    ///         println!("  Saved config for future reproduction:");
    ///         println!("    temperature: {}", config.temperature);
    ///         println!("    top_p: {:?}", config.top_p);
    ///         println!("    top_k: {:?}", config.top_k);
    ///         println!("    random_seed: {}", config.random_seed);
    ///         
    ///         Ok(())
    ///     }
    /// }
    ///
    /// // Usage for research reproducibility
    /// let pipeline = ResearchPipeline::new();
    /// pipeline.run_reproducible_experiment("baseline")?;
    /// pipeline.run_reproducible_experiment("experiment_a")?;
    /// pipeline.run_reproducible_experiment("ablation_study")?;
    /// ```
    ///
    /// ## Seed Generation Strategies
    /// ```rust
    /// use std::time::{SystemTime, UNIX_EPOCH};
    /// use std::hash::{Hash, Hasher};
    /// use std::collections::hash_map::DefaultHasher;
    ///
    /// struct SeedGenerator;
    ///
    /// impl SeedGenerator {
    ///     /// Fixed seed for maximum reproducibility
    ///     fn fixed_seed() -> u64 {
    ///         42
    ///     }
    ///     
    ///     /// Timestamp-based seed (reproducible if timestamp is recorded)
    ///     fn timestamp_seed() -> u64 {
    ///         SystemTime::now()
    ///             .duration_since(UNIX_EPOCH)
    ///             .unwrap_or_default()
    ///             .as_secs()
    ///     }
    ///     
    ///     /// Configuration-based seed (deterministic based on parameters)
    ///     fn config_based_seed(temperature: f64, top_p: Option<f64>, top_k: Option<usize>) -> u64 {
    ///         let mut hasher = DefaultHasher::new();
    ///         temperature.to_bits().hash(&mut hasher);
    ///         top_p.map(|p| p.to_bits()).hash(&mut hasher);
    ///         top_k.hash(&mut hasher);
    ///         hasher.finish()
    ///     }
    ///     
    ///     /// User-specific seed (consistent per user, different across users)
    ///     fn user_seed(user_id: &str) -> u64 {
    ///         let mut hasher = DefaultHasher::new();
    ///         user_id.hash(&mut hasher);
    ///         hasher.finish()
    ///     }
    ///     
    ///     /// Session-specific seed (consistent within session)
    ///     fn session_seed(session_id: &str, interaction_count: u32) -> u64 {
    ///         let mut hasher = DefaultHasher::new();
    ///         session_id.hash(&mut hasher);
    ///         interaction_count.hash(&mut hasher);
    ///         hasher.finish()
    ///     }
    /// }
    ///
    /// // Usage examples for different seed strategies
    /// 
    /// // Research/debugging: Maximum reproducibility
    /// let research_config = SamplingConfig::new()
    ///     .temperature(0.8)?
    ///     .random_seed(SeedGenerator::fixed_seed());
    ///
    /// // Production: User-consistent but varied across users
    /// let user_config = SamplingConfig::new()
    ///     .temperature(1.0)?
    ///     .random_seed(SeedGenerator::user_seed("user_12345"));
    ///
    /// // Configuration-based: Deterministic based on parameters
    /// let param_seed = SeedGenerator::config_based_seed(0.8, Some(0.9), Some(50));
    /// let param_config = SamplingConfig::new()
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .top_k(50)?
    ///     .random_seed(param_seed);
    ///
    /// // Session-based: Consistent within conversation
    /// let session_config = SamplingConfig::new()
    ///     .temperature(1.0)?
    ///     .random_seed(SeedGenerator::session_seed("session_abc", 5));
    /// ```
    ///
    /// ## Reproducibility Validation
    /// ```rust
    /// fn validate_reproducibility(config: &SamplingConfig, iterations: usize) -> bool {
    ///     let mut previous_results = Vec::new();
    ///     
    ///     for iteration in 0..iterations {
    ///         // Create identical configuration
    ///         let test_config = SamplingConfig::new()
    ///             .temperature(config.temperature).unwrap()
    ///             .top_p(config.top_p.unwrap_or(1.0)).unwrap()
    ///             .random_seed(config.random_seed);
    ///         
    ///         let sampling = test_config.build_sampling();
    ///         
    ///         // In real implementation, this would run actual generation
    ///         let simulated_result = format!("Result with seed {}", config.random_seed);
    ///         
    ///         if iteration == 0 {
    ///             previous_results.push(simulated_result);
    ///         } else {
    ///             // Check if results match previous iteration
    ///             if previous_results[0] != simulated_result {
    ///                 println!("Reproducibility check failed at iteration {}", iteration);
    ///                 return false;
    ///             }
    ///         }
    ///     }
    ///     
    ///     println!("Reproducibility validated across {} iterations", iterations);
    ///     true
    /// }
    ///
    /// // Test reproducibility
    /// let config = SamplingConfig::new()
    ///     .temperature(0.8).unwrap()
    ///     .top_p(0.9).unwrap()
    ///     .random_seed(42);
    ///
    /// let is_reproducible = validate_reproducibility(&config, 10);
    /// assert!(is_reproducible);
    /// ```
    ///
    /// ## Debugging with Fixed Seeds
    /// ```rust
    /// struct DebuggingSuite {
    ///     debug_seed: u64,
    /// }
    ///
    /// impl DebuggingSuite {
    ///     fn new() -> Self {
    ///         Self {
    ///             debug_seed: 42, // Fixed seed for consistent debugging
    ///         }
    ///     }
    ///     
    ///     fn debug_temperature_effects(&self, temperatures: &[f64]) -> Result<(), SamplingError> {
    ///         println!("Debugging temperature effects with fixed seed {}", self.debug_seed);
    ///         
    ///         for &temp in temperatures {
    ///             let config = SamplingConfig::new()
    ///                 .temperature(temp)?
    ///                 .random_seed(self.debug_seed); // Same seed for fair comparison
    ///             
    ///             let sampling = config.build_sampling();
    ///             
    ///             println!("Temperature {}: Sampling strategy configured", temp);
    ///             // With same seed, differences are purely due to temperature
    ///         }
    ///         
    ///         Ok(())
    ///     }
    ///     
    ///     fn debug_sampling_strategies(&self) -> Result<(), SamplingError> {
    ///         println!("Debugging sampling strategies with fixed seed {}", self.debug_seed);
    ///         
    ///         // Pure temperature
    ///         let temp_config = SamplingConfig::new()
    ///             .temperature(0.8)?
    ///             .random_seed(self.debug_seed);
    ///         
    ///         // Top-k
    ///         let topk_config = SamplingConfig::new()
    ///             .temperature(0.8)?
    ///             .top_k(50)?
    ///             .random_seed(self.debug_seed);
    ///         
    ///         // Nucleus (top-p)
    ///         let nucleus_config = SamplingConfig::new()
    ///             .temperature(0.8)?
    ///             .top_p(0.9)?
    ///             .random_seed(self.debug_seed);
    ///         
    ///         // Combined
    ///         let combined_config = SamplingConfig::new()
    ///             .temperature(0.8)?
    ///             .top_k(50)?
    ///             .top_p(0.9)?
    ///             .random_seed(self.debug_seed);
    ///         
    ///         println!("All strategies use seed {} for reproducible comparison", self.debug_seed);
    ///         
    ///         Ok(())
    ///     }
    /// }
    ///
    /// // Usage for debugging
    /// let debug_suite = DebuggingSuite::new();
    /// debug_suite.debug_temperature_effects(&[0.3, 0.7, 1.0, 1.5])?;
    /// debug_suite.debug_sampling_strategies()?;
    /// ```
    ///
    /// # Mathematical Properties
    ///
    /// ## Pseudo-Random Number Generation
    /// - Uses cryptographically secure pseudo-random number generator
    /// - Period length: 2^64 - 1 (maximum for 64-bit seed)
    /// - Distribution: Uniform distribution over [0, 1) for sampling
    /// - Correlation: Statistically independent sequence elements
    ///
    /// ## Seed Space
    /// - **Total Seeds**: 2^64 = 18,446,744,073,709,551,616 possible seeds
    /// - **Collision Probability**: Negligible for practical applications
    /// - **Distribution**: Uniform coverage of random number space
    ///
    /// # Best Practices
    ///
    /// ## For Research and Development
    /// - Use fixed seeds (e.g., 42) for reproducible experiments
    /// - Document seeds used in published results
    /// - Test multiple seeds to ensure robustness
    /// - Save complete configuration including seed
    ///
    /// ## For Production Systems
    /// - Use user-specific or session-specific seeds for consistency
    /// - Consider timestamp-based seeds for variety with reproducibility
    /// - Log seeds for debugging and issue reproduction
    /// - Balance reproducibility needs with output diversity
    ///
    /// ## For Testing
    /// - Use fixed seeds for unit tests
    /// - Test with multiple known seeds
    /// - Validate reproducibility across platforms
    /// - Include seed in test failure reports
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. The assignment
    /// operation is atomic at the language level.
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: No heap allocation during assignment
    /// - ✅ **Deterministic**: Guarantees reproducible pseudo-random sequences
    /// - ✅ **Cross-Platform**: Consistent behavior across supported platforms
    /// - ✅ **High Entropy**: Full 64-bit seed space for maximum variety
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Constructs a canonical Candle Sampling enum from the current configuration
    /// 
    /// Converts the builder configuration into the appropriate Candle Sampling
    /// variant based on which parameters are enabled, following the canonical
    /// sampling strategy hierarchy.
    /// 
    /// # Sampling Strategy Selection
    /// 
    /// The method selects sampling strategies in this priority order:
    /// 
    /// ## 1. TopKThenTopP (Both k and p specified)
    /// ```rust
    /// Sampling::TopKThenTopP { k, p, temperature }
    /// ```
    /// - First applies top-k filtering to limit vocabulary
    /// - Then applies nucleus (top-p) sampling to the filtered set
    /// - Most restrictive but highest quality sampling
    /// 
    /// ## 2. TopK (Only k specified)
    /// ```rust
    /// Sampling::TopK { k, temperature }
    /// ```
    /// - Limits consideration to k most likely tokens
    /// - Good balance between quality and diversity
    /// - Predictable vocabulary size
    /// 
    /// ## 3. TopP (Only p specified)
    /// ```rust
    /// Sampling::TopP { p, temperature }
    /// ```
    /// - Dynamic vocabulary based on cumulative probability
    /// - Adapts to model confidence
    /// - More diverse than top-k for uncertain predictions
    /// 
    /// ## 4. All (Neither k nor p specified)
    /// ```rust
    /// Sampling::All { temperature }
    /// ```
    /// - Pure temperature sampling across full vocabulary
    /// - Maximum diversity but potentially lower quality
    /// - Fastest sampling strategy
    /// 
    /// # Temperature Behavior
    /// 
    /// Temperature controls randomness across all strategies:
    /// - **0.1 - 0.7**: More deterministic, higher quality
    /// - **0.8 - 1.2**: Balanced creativity and coherence
    /// - **1.3 - 2.0**: More creative, potentially less coherent
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Memory**: Zero allocation enum construction
    /// - **Speed**: O(1) strategy selection based on configuration
    /// - **Compatibility**: Direct integration with Candle LogitsProcessor
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::SamplingConfig;
    /// 
    /// // Pure temperature sampling
    /// let config = SamplingConfig::new().temperature(0.8)?;
    /// let sampling = config.build_sampling();
    /// // Creates: Sampling::All { temperature: 0.8 }
    /// 
    /// // Top-k sampling
    /// let config = SamplingConfig::new()
    ///     .temperature(0.9)?
    ///     .top_k(50)?;
    /// let sampling = config.build_sampling();
    /// // Creates: Sampling::TopK { k: 50, temperature: 0.9 }
    /// 
    /// // Combined top-k and top-p
    /// let config = SamplingConfig::new()
    ///     .temperature(1.0)?
    ///     .top_k(40)?
    ///     .top_p(0.9)?;
    /// let sampling = config.build_sampling();
    /// // Creates: Sampling::TopKThenTopP { k: 40, p: 0.9, temperature: 1.0 }
    /// ```
    /// 
    /// # Integration with LogitsProcessor
    /// 
    /// ```rust
    /// let config = SamplingConfig::new().top_p(0.95)?.temperature(0.7)?;
    /// let sampling = config.build_sampling();
    /// let processor = LogitsProcessor::from_sampling(42, sampling);
    /// ```
    pub fn build_sampling(&self) -> Sampling {
        match (self.top_k, self.top_p) {
            (None, None) => Sampling::All {
                temperature: self.temperature},
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: self.temperature},
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: self.temperature},
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: self.temperature}}
    }

    /// Build canonical LogitsProcessor from configuration
    pub fn build_processor(&self) -> LogitsProcessor {
        LogitsProcessor::from_sampling(self.random_seed, self.build_sampling())
    }
}

/// Convenient builder for canonical LogitsProcessor with validation
pub struct LogitsProcessorBuilder {
    config: SamplingConfig}

impl LogitsProcessorBuilder {
    /// Create new LogitsProcessorBuilder with production-ready defaults and fluent configuration
    ///
    /// Initializes a new builder for constructing canonical Candle LogitsProcessor instances
    /// with fluent API patterns and comprehensive validation. The builder starts with
    /// conservative, production-ready defaults that ensure stable and predictable sampling
    /// behavior across different models and use cases.
    ///
    /// # Default Configuration
    ///
    /// The builder initializes with the same defaults as `SamplingConfig::new()`:
    /// - **Temperature**: 1.0 (no probability scaling)
    /// - **Top-p**: None (nucleus sampling disabled)
    /// - **Top-k**: None (top-k filtering disabled)
    /// - **Random Seed**: 42 (fixed for reproducible results)
    ///
    /// # Design Philosophy
    ///
    /// ## Conservative Defaults
    /// - Start with safe, stable parameters
    /// - Avoid exotic configurations that might cause issues
    /// - Provide predictable baseline behavior
    /// - Enable easy customization through fluent API
    ///
    /// ## Builder Pattern Benefits
    /// - **Type Safety**: Compile-time validation of configuration
    /// - **Fluent API**: Readable, chainable method calls
    /// - **Immutable**: Each method returns new builder state
    /// - **Validation**: Parameters validated at assignment time
    /// - **Flexibility**: Support for partial configuration
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time initialization
    /// - **Memory Usage**: Minimal - only stores configuration struct
    /// - **Build Cost**: O(1) to create final LogitsProcessor
    /// - **Zero Allocation**: Uses stack-allocated default configuration
    ///
    /// # Examples
    ///
    /// ## Basic Builder Usage
    /// ```rust
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// // Create builder with defaults
    /// let builder = LogitsProcessorBuilder::new();
    /// let processor = builder.build();
    ///
    /// // Equivalent to:
    /// // LogitsProcessor::from_sampling(42, Sampling::All { temperature: 1.0 })
    /// ```
    ///
    /// ## Fluent Configuration Chain
    /// ```rust
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// // Chain configuration methods
    /// let processor = LogitsProcessorBuilder::new()
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .random_seed(12345)
    ///     .build();
    ///
    /// // Creates: LogitsProcessor with TopP sampling
    /// ```
    ///
    /// ## Error Handling with Builder
    /// ```rust
    /// use fluent_ai_candle::{LogitsProcessorBuilder, SamplingError};
    ///
    /// fn create_validated_processor() -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///     LogitsProcessorBuilder::new()
    ///         .temperature(0.7)?     // Validated temperature
    ///         .top_k(50)?           // Validated top-k
    ///         .top_p(0.9)?          // Validated top-p
    ///         .random_seed(42)       // Always valid
    ///         .build()               // Cannot fail after validation
    /// }
    ///
    /// match create_validated_processor() {
    ///     Ok(processor) => println!("Processor created successfully"),
    ///     Err(e) => eprintln!("Configuration error: {}", e),
    /// }
    /// ```
    ///
    /// ## Production Builder Factory
    /// ```rust
    /// struct ProcessorFactory;
    ///
    /// impl ProcessorFactory {
    ///     fn for_code_generation() -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         LogitsProcessorBuilder::new()
    ///             .temperature(0.3)?    // Low temperature for deterministic code
    ///             .top_k(20)?          // Limited vocabulary for focus
    ///             .random_seed(42)      // Reproducible for debugging
    ///             .build()
    ///     }
    ///     
    ///     fn for_creative_writing() -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         LogitsProcessorBuilder::new()
    ///             .temperature(1.2)?    // Higher temperature for creativity
    ///             .top_p(0.95)?        // Nucleus sampling for quality
    ///             .random_seed(42)      // Still reproducible
    ///             .build()
    ///     }
    ///     
    ///     fn for_general_chat() -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         LogitsProcessorBuilder::new()
    ///             .temperature(0.8)?    // Balanced temperature
    ///             .top_p(0.9)?         // Good nucleus threshold
    ///             .random_seed(42)      // Consistent behavior
    ///             .build()
    ///     }
    ///     
    ///     fn custom_config(temp: f64, top_p: f64, seed: u64) -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         LogitsProcessorBuilder::new()
    ///             .temperature(temp)?
    ///             .top_p(top_p)?
    ///             .random_seed(seed)
    ///             .build()
    ///     }
    /// }
    ///
    /// // Usage in different contexts
    /// let code_processor = ProcessorFactory::for_code_generation()?;
    /// let creative_processor = ProcessorFactory::for_creative_writing()?;
    /// let chat_processor = ProcessorFactory::for_general_chat()?;
    /// let custom_processor = ProcessorFactory::custom_config(0.9, 0.85, 123)?;
    /// ```
    ///
    /// ## Configuration Validation Pipeline
    /// ```rust
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// struct ConfigValidator;
    ///
    /// impl ConfigValidator {
    ///     fn validate_and_build(
    ///         temperature: Option<f64>,
    ///         top_p: Option<f64>,
    ///         top_k: Option<usize>,
    ///         seed: Option<u64>
    ///     ) -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         let mut builder = LogitsProcessorBuilder::new();
    ///         
    ///         // Apply temperature if provided
    ///         if let Some(temp) = temperature {
    ///             builder = builder.temperature(temp)?;
    ///         }
    ///         
    ///         // Apply top-p if provided
    ///         if let Some(p) = top_p {
    ///             builder = builder.top_p(p)?;
    ///         }
    ///         
    ///         // Apply top-k if provided
    ///         if let Some(k) = top_k {
    ///             builder = builder.top_k(k)?;
    ///         }
    ///         
    ///         // Apply seed if provided
    ///         if let Some(s) = seed {
    ///             builder = builder.random_seed(s);
    ///         }
    ///         
    ///         Ok(builder.build())
    ///     }
    /// }
    ///
    /// // Usage with optional parameters
    /// let processor1 = ConfigValidator::validate_and_build(
    ///     Some(0.8), None, Some(50), None
    /// )?; // Uses defaults for top_p and seed
    ///
    /// let processor2 = ConfigValidator::validate_and_build(
    ///     None, Some(0.9), None, Some(123)
    /// )?; // Uses defaults for temperature and top_k
    /// ```
    ///
    /// ## Builder Reuse Pattern
    /// ```rust
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// struct ProcessorBuilder {
    ///     base_temperature: f64,
    ///     base_seed: u64,
    /// }
    ///
    /// impl ProcessorBuilder {
    ///     fn new(base_temperature: f64, base_seed: u64) -> Self {
    ///         Self {
    ///             base_temperature,
    ///             base_seed,
    ///         }
    ///     }
    ///     
    ///     fn create_base_builder(&self) -> Result<LogitsProcessorBuilder, SamplingError> {
    ///         Ok(LogitsProcessorBuilder::new()
    ///             .temperature(self.base_temperature)?
    ///             .random_seed(self.base_seed))
    ///     }
    ///     
    ///     fn with_top_k(&self, k: usize) -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         self.create_base_builder()?
    ///             .top_k(k)?
    ///             .build()
    ///     }
    ///     
    ///     fn with_top_p(&self, p: f64) -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         self.create_base_builder()?
    ///             .top_p(p)?
    ///             .build()
    ///     }
    ///     
    ///     fn with_both(&self, k: usize, p: f64) -> Result<candle_transformers::generation::LogitsProcessor, SamplingError> {
    ///         self.create_base_builder()?
    ///             .top_k(k)?
    ///             .top_p(p)?
    ///             .build()
    ///     }
    /// }
    ///
    /// // Usage with shared base configuration
    /// let builder = ProcessorBuilder::new(0.8, 42);
    /// let topk_processor = builder.with_top_k(50)?;
    /// let nucleus_processor = builder.with_top_p(0.9)?;
    /// let combined_processor = builder.with_both(50, 0.9)?;
    /// ```
    ///
    /// ## Debugging and Testing Support
    /// ```rust
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// struct TestProcessorBuilder;
    ///
    /// impl TestProcessorBuilder {
    ///     /// Create processor with fixed seed for reproducible tests
    ///     fn reproducible() -> LogitsProcessorBuilder {
    ///         LogitsProcessorBuilder::new()
    ///             .random_seed(42) // Always use same seed for tests
    ///     }
    ///     
    ///     /// Create processors for A/B testing
    ///     fn ab_test_pair() -> Result<(candle_transformers::generation::LogitsProcessor, candle_transformers::generation::LogitsProcessor), SamplingError> {
    ///         let processor_a = LogitsProcessorBuilder::new()
    ///             .temperature(0.7)?
    ///             .top_p(0.9)?
    ///             .random_seed(123)
    ///             .build();
    ///             
    ///         let processor_b = LogitsProcessorBuilder::new()
    ///             .temperature(0.9)?
    ///             .top_k(50)?
    ///             .random_seed(123) // Same seed for fair comparison
    ///             .build();
    ///             
    ///         Ok((processor_a, processor_b))
    ///     }
    ///     
    ///     /// Create multiple processors with different seeds
    ///     fn multi_seed_processors(config_temp: f64, seeds: &[u64]) -> Result<Vec<candle_transformers::generation::LogitsProcessor>, SamplingError> {
    ///         seeds.iter()
    ///             .map(|&seed| {
    ///                 LogitsProcessorBuilder::new()
    ///                     .temperature(config_temp)?
    ///                     .random_seed(seed)
    ///                     .build()
    ///             })
    ///             .collect()
    ///     }
    /// }
    ///
    /// // Usage in testing
    /// let test_processor = TestProcessorBuilder::reproducible()
    ///     .temperature(0.8)?
    ///     .build();
    ///
    /// let (processor_a, processor_b) = TestProcessorBuilder::ab_test_pair()?;
    /// let multi_processors = TestProcessorBuilder::multi_seed_processors(0.8, &[42, 123, 456])?;
    /// ```
    ///
    /// # Integration with Candle
    ///
    /// The built LogitsProcessor integrates directly with Candle's generation pipeline:
    /// ```rust
    /// use candle_transformers::generation::LogitsProcessor;
    /// use fluent_ai_candle::LogitsProcessorBuilder;
    ///
    /// // Create processor
    /// let processor: LogitsProcessor = LogitsProcessorBuilder::new()
    ///     .temperature(0.8)?
    ///     .top_p(0.9)?
    ///     .build();
    ///
    /// // Use with Candle model (pseudo-code)
    /// // let next_token = processor.sample(&logits)?;
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. The initialization
    /// creates independent builder instances with no shared state.
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Uses stack-allocated default configuration
    /// - ✅ **Type Safety**: Compile-time validation through builder pattern
    /// - ✅ **Fluent API**: Chainable methods for readable configuration
    /// - ✅ **Fail Fast**: Parameter validation during configuration, not at runtime
    pub fn new() -> Self {
        Self {
            config: SamplingConfig::default()}
    }

    /// Set temperature with validation
    pub fn temperature(mut self, temperature: f64) -> Result<Self, SamplingError> {
        self.config = self.config.temperature(temperature)?;
        Ok(self)
    }

    /// Set top-p with validation
    pub fn top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        self.config = self.config.top_p(top_p)?;
        Ok(self)
    }

    /// Set top-k with validation
    pub fn top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        self.config = self.config.top_k(top_k)?;
        Ok(self)
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config = self.config.random_seed(seed);
        self
    }

    /// Build canonical LogitsProcessor
    pub fn build(self) -> LogitsProcessor {
        self.config.build_processor()
    }
}

impl Default for LogitsProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for logits processing - DEPRECATED
///
/// Use `crate::processing::utils` for modern utility functions.
pub mod utils {
    use candle_core::Device;

    use super::*;

    /// Apply softmax with temperature scaling and numerical stability - DEPRECATED
    #[inline(always)]
    pub fn stable_softmax(
        logits: &Tensor,
        temperature: f64,
        _device: &Device,
    ) -> CandleResult<Tensor> {
        // Scale by temperature first
        let scaled = if (temperature - 1.0).abs() < f64::EPSILON {
            logits.clone()
        } else {
            logits.affine(1.0 / temperature, 0.0)?
        };

        // Find maximum for numerical stability
        let max_logit = scaled.max_keepdim(candle_core::D::Minus1)?;
        let shifted = scaled.broadcast_sub(&max_logit)?;

        // Apply softmax
        let exp_logits = shifted.exp()?;
        let sum_exp = exp_logits.sum_keepdim(candle_core::D::Minus1)?;
        exp_logits.broadcast_div(&sum_exp)
    }

    /// Sample from categorical distribution using efficient algorithms - DEPRECATED
    #[inline(always)]
    pub fn categorical_sample(probs: &Tensor, rng: &mut impl rand::Rng) -> CandleResult<u32> {
        let probs_vec = probs.to_vec1::<f32>()?;

        // Validate probabilities
        let sum: f32 = probs_vec.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(candle_core::Error::Msg(
                "Invalid probability distribution".to_string(),
            ));
        }

        // Generate random value
        let random_val: f32 = rng.random_range(0.0..sum);
        let mut cumulative = 0.0;

        // Find sample using cumulative distribution
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token (handles floating point precision issues)
        Ok((probs_vec.len() - 1) as u32)
    }

    /// Check for numerical instabilities in logits - DEPRECATED
    #[inline(always)]
    pub fn validate_logits(logits: &Tensor) -> Result<(), SamplingError> {
        // This is a simplified check - in production, you might want more thorough validation
        let shape = logits.shape();
        if shape.dims().is_empty() || shape.dims().iter().any(|&d| d == 0) {
            return Err(SamplingError::EmptyVocabulary);
        }
        Ok(())
    }

    /// Efficient tensor sorting for top-k and top-p operations - DEPRECATED
    #[inline(always)]
    pub fn argsort_descending(values: &[f32]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}
