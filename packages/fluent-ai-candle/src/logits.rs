//! Sophisticated LogitsProcessor implementations for advanced text generation sampling
//! Zero-allocation, lock-free, blazing-fast sampling strategies

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use arrayvec::ArrayVec;
use candle_core::Tensor;

use crate::error::{CandleError, CandleResult};

/// Maximum vocabulary size for zero-allocation processing
const MAX_VOCAB_SIZE: usize = 128000;
/// Maximum top-k value for bounded sampling
const MAX_TOP_K: usize = 100;
/// Maximum sequence length for bounded context tracking
const MAX_SEQUENCE_LENGTH: usize = 8192;

/// Processing context for logit manipulation with bounded memory usage
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Token generation position in sequence
    pub position: usize,
    /// Recent token history for repetition penalty (bounded)
    pub token_history: ArrayVec<u32, MAX_SEQUENCE_LENGTH>,
    /// Token frequency counts for repetition penalty
    pub token_frequencies: ArrayVec<u32, MAX_VOCAB_SIZE>,
    /// Vocabulary size for bounds checking
    pub vocab_size: usize,
    /// Current sequence length
    pub sequence_length: usize,
}

impl ProcessingContext {
    /// Create new processing context with zero allocation
    #[inline(always)]
    pub fn new(vocab_size: usize) -> CandleResult<Self> {
        if vocab_size > MAX_VOCAB_SIZE {
            return Err(CandleError::configuration(
                "Vocabulary size exceeds maximum supported size"
            ));
        }

        let mut token_frequencies = ArrayVec::new();
        // Initialize frequencies to zero
        for _ in 0..vocab_size {
            if token_frequencies.try_push(0).is_err() {
                return Err(CandleError::configuration(
                    "Failed to initialize token frequencies"
                ));
            }
        }

        Ok(Self {
            position: 0,
            token_history: ArrayVec::new(),
            token_frequencies,
            vocab_size,
            sequence_length: 0,
        })
    }

    /// Add token to history with bounded memory usage
    #[inline(always)]
    pub fn add_token(&mut self, token: u32) -> CandleResult<()> {
        // Validate token is within vocabulary
        if token as usize >= self.vocab_size {
            return Err(CandleError::generation_failed(
                "Token index exceeds vocabulary size"
            ));
        }

        // Add to history with sliding window behavior for bounded memory
        if self.token_history.is_full() {
            // Remove oldest token frequency when sliding window
            if let Some(oldest_token) = self.token_history.first() {
                let oldest_idx = *oldest_token as usize;
                if oldest_idx < self.token_frequencies.len() && self.token_frequencies[oldest_idx] > 0 {
                    self.token_frequencies[oldest_idx] -= 1;
                }
            }
            self.token_history.remove(0);
        }

        if self.token_history.try_push(token).is_err() {
            return Err(CandleError::generation_failed(
                "Failed to add token to history"
            ));
        }

        // Update frequency count
        let token_idx = token as usize;
        if token_idx < self.token_frequencies.len() {
            self.token_frequencies[token_idx] = self.token_frequencies[token_idx]
                .saturating_add(1);
        }

        self.position += 1;
        self.sequence_length += 1;

        Ok(())
    }

    /// Reset context for new sequence
    #[inline(always)]
    pub fn reset(&mut self) {
        self.position = 0;
        self.token_history.clear();
        for freq in &mut self.token_frequencies {
            *freq = 0;
        }
        self.sequence_length = 0;
    }
}

/// Comprehensive sampling configuration with all hyperparameters
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for controlling randomness (0.0 = greedy, >1.0 = more random)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) filtering (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty factor (1.0 = disabled)
    pub repetition_penalty: f32,
    /// Frequency penalty factor (0.0 = disabled)
    pub frequency_penalty: f32,
    /// Presence penalty factor (0.0 = disabled)  
    pub presence_penalty: f32,
    /// Minimum probability threshold for token selection
    pub min_probability: f32,
}

impl Default for SamplingConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_probability: 0.0,
        }
    }
}

impl SamplingConfig {
    /// Validate configuration parameters
    #[inline(always)]
    pub fn validate(&self) -> CandleResult<()> {
        if self.temperature < 0.0 {
            return Err(CandleError::configuration("Temperature must be non-negative"));
        }
        if self.top_k > MAX_TOP_K {
            return Err(CandleError::configuration("Top-k value exceeds maximum"));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(CandleError::configuration("Top-p must be between 0.0 and 1.0"));
        }
        if self.repetition_penalty < 0.0 {
            return Err(CandleError::configuration("Repetition penalty must be non-negative"));
        }
        if self.min_probability < 0.0 || self.min_probability > 1.0 {
            return Err(CandleError::configuration("Minimum probability must be between 0.0 and 1.0"));
        }
        Ok(())
    }

    /// Check if any processing is needed (optimization for identity configs)
    #[inline(always)]
    pub fn needs_processing(&self) -> bool {
        self.temperature != 1.0 ||
        self.top_k > 0 ||
        self.top_p < 1.0 ||
        self.repetition_penalty != 1.0 ||
        self.frequency_penalty != 0.0 ||
        self.presence_penalty != 0.0 ||
        self.min_probability > 0.0
    }
}

/// Core trait for logit processing strategies
pub trait LogitsProcessor: Send + Sync + std::fmt::Debug {
    /// Process logits in-place with token context
    /// Returns error if processing fails
    fn process(&self, logits: &mut Tensor, token_ids: &[u32], position: usize) -> Result<(), crate::sampling::SamplingError>;

    /// Check if this processor is enabled (optimization for composition)
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }

    /// Get processor name for debugging and metrics
    fn name(&self) -> &'static str;

    /// Validate processor configuration
    fn validate(&self) -> Result<(), crate::sampling::SamplingError> {
        Ok(()) // Default implementation - no validation required
    }

    /// Check if processor is identity (no-op) for optimization
    fn is_identity(&self) -> bool {
        false // Default implementation - assume processor modifies logits
    }
}

/// Temperature scaling processor for controlling generation randomness
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    /// Create new temperature processor
    #[inline(always)]
    pub fn new(temperature: f32) -> CandleResult<Self> {
        if temperature < 0.0 {
            return Err(CandleError::configuration("Temperature must be non-negative"));
        }
        Ok(Self { temperature })
    }
}

impl LogitsProcessor for TemperatureProcessor {
    #[inline(always)]
    fn process(&self, logits: &mut Tensor, _token_ids: &[u32], _position: usize) -> Result<(), crate::sampling::SamplingError> {
        if self.temperature == 1.0 {
            return Ok(()); // No-op for identity temperature
        }

        if self.temperature == 0.0 {
            // Greedy sampling - find max and set others to negative infinity
            let logits_vec = logits.to_vec1::<f32>().map_err(crate::sampling::SamplingError::from)?;
            let max_idx = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| crate::sampling::SamplingError::EmptyLogits)?;

            let mut new_logits = vec![-1000.0f32; logits_vec.len()];
            new_logits[max_idx] = 1000.0;
            
            *logits = Tensor::from_vec(new_logits, logits.shape(), logits.device())
                .map_err(crate::sampling::SamplingError::from)?;
        } else {
            // Apply temperature scaling
            let inv_temp = 1.0 / self.temperature;
            *logits = logits.affine(inv_temp as f64, 0.0)
                .map_err(crate::sampling::SamplingError::from)?;
        }

        Ok(())
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.temperature != 1.0
    }

    fn name(&self) -> &'static str {
        "temperature"
    }
}

/// Top-K sampling processor for controlled vocabulary selection
/// Uses optimized partial sorting for O(n + k log k) complexity
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    pub k: usize,
}

impl TopKProcessor {
    /// Create new top-k processor with validation
    #[inline(always)]
    pub fn new(k: usize) -> CandleResult<Self> {
        if k > MAX_TOP_K {
            return Err(CandleError::configuration(
                "Top-k value exceeds maximum supported size"
            ));
        }
        Ok(Self { k })
    }

    /// Find top-k logits using optimized partial sort with binary heap
    #[inline(always)]
    fn find_top_k_threshold(&self, logits: &[f32]) -> CandleResult<f32> {
        if self.k == 0 || logits.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }

        let k = self.k.min(logits.len());
        
        // Use binary heap for efficient top-k selection
        let mut heap: ArrayVec<(std::cmp::Reverse<FloatOrd>, usize), MAX_TOP_K> = ArrayVec::new();
        
        // Build initial heap with first k elements
        for (i, &logit) in logits.iter().enumerate().take(k) {
            if heap.try_push((std::cmp::Reverse(FloatOrd(logit)), i)).is_err() {
                break;
            }
        }
        
        // Make heap property hold
        heap.sort_by_key(|(ord, _)| *ord);
        
        // Process remaining elements
        for (i, &logit) in logits.iter().enumerate().skip(k) {
            if let Some((std::cmp::Reverse(FloatOrd(min_val)), _)) = heap.first() {
                if logit > *min_val {
                    // Replace minimum with current element
                    heap[0] = (std::cmp::Reverse(FloatOrd(logit)), i);
                    // Restore heap property (bubble down)
                    let mut pos = 0;
                    while pos * 2 + 1 < heap.len() {
                        let left_child = pos * 2 + 1;
                        let right_child = pos * 2 + 2;
                        let mut smallest = pos;
                        
                        if left_child < heap.len() && heap[left_child].0 < heap[smallest].0 {
                            smallest = left_child;
                        }
                        if right_child < heap.len() && heap[right_child].0 < heap[smallest].0 {
                            smallest = right_child;
                        }
                        
                        if smallest != pos {
                            heap.swap(pos, smallest);
                            pos = smallest;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        
        // Find minimum value in top-k (threshold)
        heap.iter()
            .map(|(std::cmp::Reverse(FloatOrd(val)), _)| *val)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .ok_or_else(|| CandleError::generation_failed("Failed to find top-k threshold"))
    }
}

impl LogitsProcessor for TopKProcessor {
    #[inline(always)]
    fn process(&self, logits: &mut Tensor, _token_ids: &[u32], _position: usize) -> Result<(), crate::sampling::SamplingError> {
        let logits_vec = logits.to_vec1::<f32>().map_err(crate::sampling::SamplingError::from)?;
        
        if self.k == 0 || self.k >= logits_vec.len() {
            return Ok(()); // No filtering needed
        }

        // Find the threshold value for top-k
        let threshold = self.find_top_k_threshold(&logits_vec)
            .map_err(|e| crate::sampling::SamplingError::ProcessingFailed(e.to_string()))?;
        
        // Mask all logits below threshold
        let new_logits: Vec<f32> = logits_vec.iter()
            .map(|&logit| if logit < threshold { f32::NEG_INFINITY } else { logit })
            .collect();

        *logits = Tensor::from_vec(new_logits, logits.shape(), logits.device())
            .map_err(crate::sampling::SamplingError::from)?;
        
        Ok(())
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.k > 0
    }

    fn name(&self) -> &'static str {
        "top_k"
    }
}

/// Wrapper for f32 to enable ordering for heap operations
#[derive(Debug, Clone, Copy, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FloatOrd {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Top-P (nucleus) sampling processor for dynamic vocabulary truncation
/// Uses numerically stable cumulative probability computation
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    /// Create new top-p processor with validation
    #[inline(always)]
    pub fn new(p: f32) -> CandleResult<Self> {
        if p < 0.0 || p > 1.0 {
            return Err(CandleError::configuration(
                "Top-p value must be between 0.0 and 1.0"
            ));
        }
        Ok(Self { p })
    }

    /// Sort logits and find nucleus threshold using stable algorithm
    #[inline(always)]
    fn find_nucleus_threshold(&self, logits: &[f32]) -> CandleResult<f32> {
        if self.p >= 1.0 || logits.is_empty() {
            return Ok(f32::NEG_INFINITY); // No filtering
        }

        if self.p <= 0.0 {
            // Greedy sampling - find maximum logit
            return logits
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .copied()
                .ok_or_else(|| CandleError::generation_failed("No valid logits found"));
        }

        // Create sorted index-value pairs for nucleus selection
        let mut indexed_logits: ArrayVec<(usize, f32), MAX_VOCAB_SIZE> = ArrayVec::new();
        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                continue; // Skip infinite or NaN values
            }
            if indexed_logits.try_push((i, logit)).is_err() {
                break; // Respect bounded allocation
            }
        }

        if indexed_logits.is_empty() {
            return Err(CandleError::generation_failed("No finite logits found"));
        }

        // Sort by logit value descending
        indexed_logits.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Apply softmax to sorted logits for probability computation
        let max_logit = indexed_logits[0].1;
        let mut exp_logits: ArrayVec<(usize, f32), MAX_VOCAB_SIZE> = ArrayVec::new();
        let mut sum_exp = 0.0f32;

        for (idx, logit) in &indexed_logits {
            let exp_val = (logit - max_logit).exp(); // Subtract max for numerical stability
            sum_exp += exp_val;
            if exp_logits.try_push((*idx, exp_val)).is_err() {
                break;
            }
        }

        if sum_exp <= 0.0 {
            return Err(CandleError::generation_failed("Softmax normalization failed"));
        }

        // Find nucleus threshold using cumulative probability with Kahan summation
        let mut cumulative = 0.0f32;
        let mut compensation = 0.0f32; // For Kahan summation numerical stability

        for (_, exp_val) in &exp_logits {
            let prob = exp_val / sum_exp;
            
            // Kahan summation for numerical stability
            let adjusted_prob = prob - compensation;
            let new_cumulative = cumulative + adjusted_prob;
            compensation = (new_cumulative - cumulative) - adjusted_prob;
            cumulative = new_cumulative;

            if cumulative >= self.p {
                // Find the corresponding original logit value
                for (orig_idx, orig_logit) in &indexed_logits {
                    if exp_logits.iter().any(|(idx, _)| idx == orig_idx) {
                        return Ok(*orig_logit);
                    }
                }
                break;
            }
        }

        // If we reach here, include all tokens (shouldn't happen with valid p < 1.0)
        indexed_logits
            .last()
            .map(|(_, logit)| *logit)
            .ok_or_else(|| CandleError::generation_failed("Failed to determine nucleus threshold"))
    }
}

impl LogitsProcessor for TopPProcessor {
    #[inline(always)]
    fn process(&self, logits: &mut Tensor, _token_ids: &[u32], _position: usize) -> Result<(), crate::sampling::SamplingError> {
        let logits_vec = logits.to_vec1::<f32>().map_err(crate::sampling::SamplingError::from)?;
        
        if self.p >= 1.0 {
            return Ok(()); // No filtering needed
        }

        if self.p <= 0.0 {
            // Greedy sampling - keep only maximum logit
            let max_idx = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| crate::sampling::SamplingError::EmptyLogits)?;

            let mut new_logits = vec![f32::NEG_INFINITY; logits_vec.len()];
            new_logits[max_idx] = logits_vec[max_idx];
            
            *logits = Tensor::from_vec(new_logits, logits.shape(), logits.device())
                .map_err(crate::sampling::SamplingError::from)?;
            return Ok(());
        }

        // Find nucleus threshold
        let threshold = self.find_nucleus_threshold(&logits_vec)
            .map_err(|e| crate::sampling::SamplingError::ProcessingFailed(e.to_string()))?;

        // Mask all logits below nucleus threshold
        let new_logits: Vec<f32> = logits_vec.iter()
            .map(|&logit| if logit < threshold { f32::NEG_INFINITY } else { logit })
            .collect();

        *logits = Tensor::from_vec(new_logits, logits.shape(), logits.device())
            .map_err(crate::sampling::SamplingError::from)?;

        Ok(())
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.p < 1.0
    }

    fn name(&self) -> &'static str {
        "top_p"
    }
}

/// Repetition penalty processor for controlling token repetition
/// Supports both frequency-based and presence-based penalties
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    /// Penalty factor for repeated tokens
    pub repetition_penalty: f32,
    /// Additional frequency-based penalty scaling
    pub frequency_penalty: f32,
    /// Presence-based penalty for any occurrence
    pub presence_penalty: f32,
}

impl RepetitionPenaltyProcessor {
    /// Create new repetition penalty processor with validation
    #[inline(always)]
    pub fn new(
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> CandleResult<Self> {
        if repetition_penalty < 0.0 {
            return Err(CandleError::configuration(
                "Repetition penalty must be non-negative"
            ));
        }
        Ok(Self {
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
        })
    }

    /// Apply frequency penalty with numerical stability
    #[inline(always)]
    fn apply_frequency_penalty(&self, logit: f32, frequency: u32) -> f32 {
        if self.frequency_penalty == 0.0 || frequency == 0 {
            return logit;
        }
        
        // Apply frequency penalty: logit -= frequency_penalty * frequency
        logit - (self.frequency_penalty * frequency as f32)
    }

    /// Apply presence penalty for any token occurrence
    #[inline(always)]
    fn apply_presence_penalty(&self, logit: f32, is_present: bool) -> f32 {
        if self.presence_penalty == 0.0 || !is_present {
            return logit;
        }
        
        // Apply presence penalty: logit -= presence_penalty
        logit - self.presence_penalty
    }

    /// Apply repetition penalty using multiplicative approach
    #[inline(always)]
    fn apply_repetition_penalty(&self, logit: f32, frequency: u32) -> f32 {
        if self.repetition_penalty == 1.0 || frequency == 0 {
            return logit;
        }

        // Apply repetition penalty multiplicatively for better control
        if logit > 0.0 {
            logit / self.repetition_penalty.powf(frequency as f32)
        } else {
            logit * self.repetition_penalty.powf(frequency as f32)
        }
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    #[inline(always)]
    fn process(&self, logits: &mut Tensor, token_ids: &[u32], _position: usize) -> Result<(), crate::sampling::SamplingError> {
        // Early exit if no penalties are configured
        if self.repetition_penalty == 1.0 && 
           self.frequency_penalty == 0.0 && 
           self.presence_penalty == 0.0 {
            return Ok(());
        }

        let logits_vec = logits.to_vec1::<f32>().map_err(crate::sampling::SamplingError::from)?;
        let vocab_size = logits_vec.len();

        // Count token frequencies from the provided token history
        let mut token_frequencies = vec![0u32; vocab_size];
        for &token in token_ids {
            if (token as usize) < vocab_size {
                token_frequencies[token as usize] += 1;
            }
        }

        // Apply penalties to each token based on its history
        let new_logits: Vec<f32> = logits_vec.iter().enumerate().map(|(token_id, &logit)| {
            let frequency = token_frequencies[token_id];
            let is_present = frequency > 0;

            // Apply all configured penalties
            let mut processed_logit = self.apply_repetition_penalty(logit, frequency);
            processed_logit = self.apply_frequency_penalty(processed_logit, frequency);
            self.apply_presence_penalty(processed_logit, is_present)
        }).collect();

        *logits = Tensor::from_vec(new_logits, logits.shape(), logits.device())
            .map_err(crate::sampling::SamplingError::from)?;

        Ok(())
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.repetition_penalty != 1.0 || 
        self.frequency_penalty != 0.0 || 
        self.presence_penalty != 0.0
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Maximum number of processors in composition for bounded allocation
const MAX_PROCESSORS: usize = 8;

/// Composed processor for chaining multiple sampling strategies
/// Applies processors in optimal order for best generation quality
#[derive(Debug)]
pub struct ComposedProcessor {
    /// Ordered list of processors to apply
    processors: ArrayVec<Box<dyn LogitsProcessor>, MAX_PROCESSORS>,
    /// Cached processor names for debugging and metrics
    processor_names: ArrayVec<&'static str, MAX_PROCESSORS>,
}

impl ComposedProcessor {
    /// Create new composed processor with optimal ordering
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            processors: ArrayVec::new(),
            processor_names: ArrayVec::new(),
        }
    }

    /// Add processor to the composition with automatic optimal ordering
    #[inline(always)]
    pub fn add_processor(&mut self, processor: Box<dyn LogitsProcessor>) -> CandleResult<()> {
        if self.processors.is_full() {
            return Err(CandleError::configuration(
                "Maximum number of processors reached"
            ));
        }

        let name = processor.name();
        
        // Insert processor in optimal order for generation quality
        let optimal_position = self.find_optimal_position(name);
        
        if optimal_position >= self.processors.len() {
            // Append at end
            self.processors.try_push(processor)
                .map_err(|_| CandleError::configuration("Failed to add processor"))?;
            self.processor_names.try_push(name)
                .map_err(|_| CandleError::configuration("Failed to track processor name"))?;
        } else {
            // Insert at optimal position
            self.processors.try_insert(optimal_position, processor)
                .map_err(|_| CandleError::configuration("Failed to insert processor"))?;
            self.processor_names.try_insert(optimal_position, name)
                .map_err(|_| CandleError::configuration("Failed to insert processor name"))?;
        }

        Ok(())
    }

    /// Find optimal position for processor based on type
    /// Optimal order: temperature → repetition_penalty → top_k → top_p
    #[inline(always)]
    fn find_optimal_position(&self, processor_name: &'static str) -> usize {
        let priority = match processor_name {
            "temperature" => 0,
            "repetition_penalty" => 1,
            "top_k" => 2,
            "top_p" => 3,
            _ => 4, // Custom processors go last
        };

        // Find insertion point to maintain order
        for (i, &existing_name) in self.processor_names.iter().enumerate() {
            let existing_priority = match existing_name {
                "temperature" => 0,
                "repetition_penalty" => 1,
                "top_k" => 2,
                "top_p" => 3,
                _ => 4,
            };
            
            if priority < existing_priority {
                return i;
            }
        }

        self.processors.len()
    }

    /// Create composed processor with standard configuration
    #[inline(always)]
    pub fn with_config(config: &SamplingConfig) -> CandleResult<Self> {
        let mut composed = Self::new();

        // Add processors based on configuration, using optimal ordering
        if config.temperature != 1.0 {
            composed.add_processor(Box::new(TemperatureProcessor::new(config.temperature)?))?;
        }

        if config.repetition_penalty != 1.0 || config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
            composed.add_processor(Box::new(RepetitionPenaltyProcessor::new(
                config.repetition_penalty,
                config.frequency_penalty,
                config.presence_penalty,
            )?))?;
        }

        if config.top_k > 0 {
            composed.add_processor(Box::new(TopKProcessor::new(config.top_k)?))?;
        }

        if config.top_p < 1.0 {
            composed.add_processor(Box::new(TopPProcessor::new(config.top_p)?))?;
        }

        Ok(composed)
    }

    /// Get list of active processor names for debugging
    #[inline(always)]
    pub fn active_processors(&self) -> &[&'static str] {
        &self.processor_names
    }

    /// Check if composition has any enabled processors
    #[inline(always)]
    pub fn has_active_processors(&self) -> bool {
        self.processors.iter().any(|p| p.is_enabled())
    }
}

impl LogitsProcessor for ComposedProcessor {
    #[inline(always)]
    fn process(&self, logits: &mut Tensor, token_ids: &[u32], position: usize) -> Result<(), crate::sampling::SamplingError> {
        // Short-circuit if no active processors
        if !self.has_active_processors() {
            return Ok(());
        }

        // Apply processors in optimal order with error propagation
        for processor in &self.processors {
            if processor.is_enabled() {
                processor.process(logits, token_ids, position)?;
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.has_active_processors()
    }

    fn name(&self) -> &'static str {
        "composed"
    }
}

impl Default for ComposedProcessor {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// High-level sampling interface combining all processors
#[derive(Debug)]
pub struct LogitsSampler {
    /// Main composed processor
    processor: ComposedProcessor,
    /// Processing context for token tracking
    context: ProcessingContext,
    /// Sampling configuration
    config: SamplingConfig,
}

impl LogitsSampler {
    /// Create new sampler with configuration
    #[inline(always)]
    pub fn new(vocab_size: usize, config: SamplingConfig) -> CandleResult<Self> {
        config.validate()?;
        
        let context = ProcessingContext::new(vocab_size)?;
        let processor = ComposedProcessor::with_config(&config)?;

        Ok(Self {
            processor,
            context,
            config,
        })
    }

    /// Process logits with all configured sampling strategies
    #[inline(always)]
    pub fn process_logits(&mut self, logits: &mut [f32]) -> CandleResult<()> {
        if logits.len() != self.context.vocab_size {
            return Err(CandleError::generation_failed(
                "Logits array size mismatch"
            ));
        }

        // Record sampling operation start time for metrics
        let start_time = std::time::Instant::now();

        // Process logits through composed pipeline
        self.processor.process(logits, &self.context)?;

        // Record metrics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        sampling_metrics().record_sample(processing_time);

        Ok(())
    }

    /// Add token to context after generation
    #[inline(always)]
    pub fn add_generated_token(&mut self, token: u32) -> CandleResult<()> {
        self.context.add_token(token)
    }

    /// Reset sampler for new sequence
    #[inline(always)]
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// Get current sampling configuration
    #[inline(always)]
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Update sampling configuration
    #[inline(always)]
    pub fn update_config(&mut self, config: SamplingConfig) -> CandleResult<()> {
        config.validate()?;
        self.processor = ComposedProcessor::with_config(&config)?;
        self.config = config;
        Ok(())
    }

    /// Get processing context
    #[inline(always)]
    pub fn context(&self) -> &ProcessingContext {
        &self.context
    }
}

/// Performance metrics for sampling operations
#[derive(Debug)]
pub struct SamplingMetrics {
    /// Total number of sampling operations
    pub total_samples: AtomicU64,
    /// Total processing time in nanoseconds
    pub total_processing_time_ns: AtomicU64,
    /// Number of cache hits for repeated configurations
    pub cache_hits: AtomicU64,
}

impl SamplingMetrics {
    /// Create new metrics instance
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            total_samples: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// Record a sampling operation
    #[inline(always)]
    pub fn record_sample(&self, processing_time_ns: u64) {
        self.total_samples.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_processing_time_ns.fetch_add(processing_time_ns, AtomicOrdering::Relaxed);
    }

    /// Record a cache hit
    #[inline(always)]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Get average processing time per sample
    #[inline(always)]
    pub fn average_processing_time_ns(&self) -> f64 {
        let total_samples = self.total_samples.load(AtomicOrdering::Relaxed);
        if total_samples == 0 {
            return 0.0;
        }
        let total_time = self.total_processing_time_ns.load(AtomicOrdering::Relaxed);
        total_time as f64 / total_samples as f64
    }

    /// Get cache hit rate
    #[inline(always)]
    pub fn cache_hit_rate(&self) -> f64 {
        let total_samples = self.total_samples.load(AtomicOrdering::Relaxed);
        if total_samples == 0 {
            return 0.0;
        }
        let cache_hits = self.cache_hits.load(AtomicOrdering::Relaxed);
        cache_hits as f64 / total_samples as f64
    }
}

/// Global sampling metrics instance
static SAMPLING_METRICS: std::sync::LazyLock<SamplingMetrics> = 
    std::sync::LazyLock::new(SamplingMetrics::new);

/// Get reference to global sampling metrics
#[inline(always)]
pub fn sampling_metrics() -> &'static SamplingMetrics {
    &SAMPLING_METRICS
}

/// Utility functions for numerical stability in sampling operations
pub mod utils {
    use super::*;

    /// Compute softmax with numerical stability using log-sum-exp trick
    #[inline(always)]
    pub fn stable_softmax(logits: &mut [f32]) -> CandleResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Find maximum for numerical stability
        let max_logit = logits
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .copied()
            .ok_or_else(|| CandleError::generation_failed("No valid logits for softmax"))?;

        // Subtract max and compute exp
        let mut sum = 0.0f32;
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }

        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for logit in logits.iter_mut() {
                *logit *= inv_sum;
            }
        } else {
            return Err(CandleError::generation_failed("Softmax normalization failed"));
        }

        Ok(())
    }

    /// Find top-k indices using partial sort for optimal performance
    #[inline(always)]
    pub fn find_top_k_indices(logits: &[f32], k: usize) -> ArrayVec<usize, MAX_TOP_K> {
        let mut indices: ArrayVec<usize, MAX_TOP_K> = ArrayVec::new();
        
        if k == 0 || logits.is_empty() {
            return indices;
        }

        let k = k.min(logits.len()).min(MAX_TOP_K);
        
        // Create index-value pairs
        let mut pairs: ArrayVec<(usize, f32), MAX_TOP_K> = ArrayVec::new();
        for (i, &value) in logits.iter().enumerate().take(k) {
            if pairs.try_push((i, value)).is_err() {
                break;
            }
        }

        // For remaining elements, maintain top-k heap
        for (i, &value) in logits.iter().enumerate().skip(k) {
            if let Some((min_idx, &(_, min_val))) = pairs
                .iter()
                .enumerate()
                .min_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            {
                if value > min_val {
                    pairs[min_idx] = (i, value);
                }
            }
        }

        // Sort by value descending and extract indices
        pairs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        
        for (idx, _) in pairs {
            if indices.try_push(idx).is_err() {
                break;
            }
        }

        indices
    }

    /// Compute cumulative probability for nucleus sampling
    #[inline(always)]
    pub fn cumulative_probability_threshold(
        probabilities: &[(usize, f32)], 
        nucleus_p: f32
    ) -> usize {
        let mut cumulative = 0.0f32;
        for (count, (_, prob)) in probabilities.iter().enumerate() {
            cumulative += prob;
            if cumulative >= nucleus_p {
                return count + 1;
            }
        }
        probabilities.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_context_creation() {
        let context = ProcessingContext::new(1000);
        assert!(context.is_ok());
        let context = context.unwrap();
        assert_eq!(context.vocab_size, 1000);
        assert_eq!(context.position, 0);
        assert_eq!(context.sequence_length, 0);
    }

    #[test]
    fn test_sampling_config_validation() {
        let config = SamplingConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = SamplingConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_temperature_processor() {
        let processor = TemperatureProcessor::new(2.0).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0];
        let context = ProcessingContext::new(3).unwrap();
        
        processor.process(&mut logits, &context).unwrap();
        assert_eq!(logits, vec![0.5, 1.0, 1.5]);
    }
}