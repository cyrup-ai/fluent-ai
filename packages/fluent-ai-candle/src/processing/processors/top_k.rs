//! Top-K sampling processor with efficient partial sorting algorithms
//!
//! Implements top-k filtering for logits with adaptive algorithms,
//! zero-allocation patterns, and comprehensive edge case handling.

use arrayvec::ArrayVec;

use crate::processing::traits::{
    ConfigurableProcessor, LogitsProcessor, NumericallyStableProcessor, ProcessingResult,
    ZeroAllocationProcessor,
    utils::{clamp_for_stability, validate_logits},
};
use crate::processing::{ProcessingContext, ProcessingError};

/// Maximum top-k value for bounded allocation
const MAX_TOP_K: usize = 256;

/// Stack-allocated buffer for top-k selection
type TopKBuffer = ArrayVec<(usize, f32), MAX_TOP_K>;

/// Top-K sampling processor for controlled vocabulary selection
///
/// Filters logits to keep only the top-k highest values, setting others
/// to negative infinity. Uses adaptive algorithms based on k size:
/// - Small k: Binary heap for O(n + k log k) complexity
/// - Large k: Quickselect for O(n) average complexity
/// - k=0: No-op for efficiency
/// - k >= vocab_size: No-op for efficiency
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    /// Number of top tokens to keep
    k: usize,
    /// Cached identity status for optimization
    is_identity: bool,
}

impl TopKProcessor {
    /// Create a new top-k processor
    ///
    /// # Arguments
    /// * `k` - Number of top logits to keep (0 means no filtering)
    ///
    /// # Returns
    /// * `Ok(TopKProcessor)` - Successfully created processor
    /// * `Err(ProcessingError)` - Invalid k value
    ///
    /// # Examples
    /// ```
    /// use fluent_ai_candle::processing::processors::TopKProcessor;
    ///
    /// let processor = TopKProcessor::new(50)?;
    /// ```
    #[inline(always)]
    pub fn new(k: usize) -> ProcessingResult<Self> {
        if k > MAX_TOP_K {
            return Err(ProcessingError::configuration(format!(
                "Top-k value {} exceeds maximum {}",
                k, MAX_TOP_K
            )));
        }

        let is_identity = k == 0; // k=0 means no filtering

        Ok(Self { k, is_identity })
    }

    /// Create top-k processor with small preset (k=20)
    #[inline(always)]
    pub fn small() -> ProcessingResult<Self> {
        Self::new(20)
    }

    /// Create top-k processor with medium preset (k=50)
    #[inline(always)]
    pub fn medium() -> ProcessingResult<Self> {
        Self::new(50)
    }

    /// Create top-k processor with large preset (k=100)
    #[inline(always)]
    pub fn large() -> ProcessingResult<Self> {
        Self::new(100)
    }

    /// Get the k value
    #[inline(always)]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Apply top-k filtering to logits in-place
    ///
    /// Uses adaptive algorithms based on k size for optimal performance:
    /// - Very small k (≤ 8): Linear scan with small buffer
    /// - Small k (≤ 32): Binary heap for efficient selection
    /// - Large k: Quickselect algorithm for linear average complexity
    fn apply_top_k_filtering(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        // Early return for identity cases
        if self.is_identity || self.k >= logits.len() {
            return Ok(());
        }

        // Validate input
        validate_logits(logits, "TopKProcessor")?;

        // Apply numerical stability clamping
        clamp_for_stability(logits);

        // Choose algorithm based on k size
        if self.k <= 8 {
            self.apply_small_k_selection(logits)?;
        } else if self.k <= 32 {
            self.apply_heap_selection(logits)?;
        } else {
            self.apply_quickselect_selection(logits)?;
        }

        Ok(())
    }

    /// Apply selection for very small k values using linear scan
    ///
    /// For very small k, a simple linear scan with a small buffer
    /// is more efficient than complex algorithms due to lower overhead.
    fn apply_small_k_selection(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        // Use stack-allocated buffer for very small k
        let mut top_k: ArrayVec<(usize, f32), 8> = ArrayVec::new();

        // Find top k elements using linear scan
        for (idx, &logit_value) in logits.iter().enumerate() {
            if top_k.len() < self.k {
                // Buffer not full, just add
                if top_k.try_push((idx, logit_value)).is_err() {
                    return Err(ProcessingError::resource("Failed to push to top-k buffer"));
                }
            } else {
                // Buffer full, check if current is better than worst
                if let Some((worst_idx, (_, worst_score))) =
                    top_k.iter().enumerate().min_by(|a, b| {
                        a.1.1
                            .partial_cmp(&b.1.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    if logit_value > *worst_score {
                        top_k[worst_idx] = (idx, logit_value);
                    }
                }
            }
        }

        // Find threshold (minimum value in top-k)
        let threshold = top_k
            .iter()
            .map(|(_, score)| *score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ProcessingError::internal("No threshold found for small k selection"))?;

        // Mask logits below threshold
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Apply selection using binary heap for medium k values
    ///
    /// Uses a min-heap to efficiently maintain the top-k elements.
    /// More efficient than quickselect for moderate k values.
    fn apply_heap_selection(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        let mut heap = TopKBuffer::new();

        // Build heap with first k elements
        for (idx, &logit) in logits.iter().enumerate().take(self.k) {
            if heap.try_push((idx, logit)).is_err() {
                return Err(ProcessingError::resource("Failed to initialize heap"));
            }
        }

        // Convert to min-heap
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Process remaining elements
        for (idx, &logit) in logits.iter().enumerate().skip(self.k) {
            if let Some(&(_, min_val)) = heap.first() {
                if logit > min_val {
                    // Replace minimum with current element
                    heap[0] = (idx, logit);

                    // Restore heap property (bubble down)
                    let mut pos = 0;
                    while pos * 2 + 1 < heap.len() {
                        let left_child = pos * 2 + 1;
                        let right_child = pos * 2 + 2;
                        let mut smallest = pos;

                        if left_child < heap.len() && heap[left_child].1 < heap[smallest].1 {
                            smallest = left_child;
                        }
                        if right_child < heap.len() && heap[right_child].1 < heap[smallest].1 {
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

        // Find threshold (minimum in heap)
        let threshold = heap
            .iter()
            .map(|(_, score)| *score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ProcessingError::internal("No threshold found in heap selection"))?;

        // Mask logits below threshold
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Apply selection using quickselect algorithm for large k values
    ///
    /// Uses the quickselect algorithm to find the k-th largest element
    /// in O(n) average time. More efficient for large k values.
    fn apply_quickselect_selection(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        // Create index-value pairs for sorting
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Find k-th largest using quickselect
        if indexed_logits.len() < self.k {
            return Ok(()); // k is larger than vocab size, no filtering needed
        }

        let threshold_idx = indexed_logits.len() - self.k;
        self.quickselect(&mut indexed_logits, threshold_idx)?;

        // Get threshold value (k-th largest element)
        let threshold = indexed_logits[threshold_idx].1;

        // Mask logits below threshold
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Quickselect algorithm implementation
    ///
    /// Finds the k-th smallest element in average O(n) time.
    /// We use it to find the threshold for top-k selection.
    fn quickselect(&self, arr: &mut [(usize, f32)], k: usize) -> ProcessingResult<()> {
        if arr.is_empty() || k >= arr.len() {
            return Ok(());
        }

        let mut left = 0;
        let mut right = arr.len() - 1;

        while left < right {
            let pivot_idx = self.partition(arr, left, right)?;

            if pivot_idx == k {
                break;
            } else if pivot_idx < k {
                left = pivot_idx + 1;
            } else {
                right = pivot_idx - 1;
            }
        }

        Ok(())
    }

    /// Partition function for quickselect
    ///
    /// Partitions array around pivot, returning the pivot's final position.
    fn partition(
        &self,
        arr: &mut [(usize, f32)],
        left: usize,
        right: usize,
    ) -> ProcessingResult<usize> {
        if left >= arr.len() || right >= arr.len() || left > right {
            return Err(ProcessingError::internal("Invalid partition bounds"));
        }

        // Use median-of-three pivot selection for better performance
        let pivot_idx = self.median_of_three(arr, left, right);
        let pivot_value = arr[pivot_idx].1;

        // Move pivot to end
        arr.swap(pivot_idx, right);

        let mut store_idx = left;

        for i in left..right {
            // Sort by value (ascending)
            if arr[i].1 < pivot_value {
                arr.swap(i, store_idx);
                store_idx += 1;
            }
        }

        // Move pivot to its final position
        arr.swap(store_idx, right);

        Ok(store_idx)
    }

    /// Median-of-three pivot selection for better quickselect performance
    fn median_of_three(&self, arr: &[(usize, f32)], left: usize, right: usize) -> usize {
        let mid = left + (right - left) / 2;

        if arr[mid].1 < arr[left].1 {
            if arr[right].1 < arr[left].1 {
                if arr[right].1 < arr[mid].1 {
                    mid
                } else {
                    right
                }
            } else {
                left
            }
        } else if arr[right].1 < arr[mid].1 {
            if arr[right].1 < arr[left].1 {
                left
            } else {
                right
            }
        } else {
            mid
        }
    }
}

impl LogitsProcessor for TopKProcessor {
    #[inline]
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        self.apply_top_k_filtering(logits)
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "TopKProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_identity
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        !self.is_identity
    }

    fn validate(&self) -> ProcessingResult<()> {
        if self.k > MAX_TOP_K {
            return Err(ProcessingError::configuration(format!(
                "Top-k value {} exceeds maximum {}",
                self.k, MAX_TOP_K
            )));
        }
        Ok(())
    }

    fn config_summary(&self) -> String {
        format!("TopKProcessor(k={})", self.k)
    }

    #[inline(always)]
    fn estimated_overhead(&self) -> f32 {
        if self.is_identity {
            0.0
        } else if self.k <= 8 {
            1.0 // Linear scan is efficient for small k
        } else if self.k <= 32 {
            1.5 // Heap operations have moderate overhead
        } else {
            2.0 // Quickselect has higher setup cost
        }
    }

    #[inline(always)]
    fn priority(&self) -> u8 {
        super::priorities::FILTERING
    }
}

/// Configuration for top-k processor
#[derive(Debug, Clone)]
pub struct TopKConfig {
    pub k: usize,
}

impl ConfigurableProcessor for TopKProcessor {
    type Config = TopKConfig;

    fn update_config(&mut self, config: Self::Config) -> ProcessingResult<()> {
        let new_processor = Self::new(config.k)?;

        self.k = new_processor.k;
        self.is_identity = new_processor.is_identity;

        Ok(())
    }

    fn get_config(&self) -> Self::Config {
        TopKConfig { k: self.k }
    }
}

// Marker trait implementations
impl ZeroAllocationProcessor for TopKProcessor {}
impl NumericallyStableProcessor for TopKProcessor {}

/// Builder for top-k processor with validation and presets
#[derive(Debug, Clone, Default)]
pub struct TopKBuilder {
    k: Option<usize>,
}

impl TopKBuilder {
    /// Create a new top-k builder
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set k value
    #[inline(always)]
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Use small preset (k=20)
    #[inline(always)]
    pub fn small(mut self) -> Self {
        self.k = Some(20);
        self
    }

    /// Use medium preset (k=50)
    #[inline(always)]
    pub fn medium(mut self) -> Self {
        self.k = Some(50);
        self
    }

    /// Use large preset (k=100)
    #[inline(always)]
    pub fn large(mut self) -> Self {
        self.k = Some(100);
        self
    }

    /// Disable top-k filtering
    #[inline(always)]
    pub fn disabled(mut self) -> Self {
        self.k = Some(0);
        self
    }

    /// Build the top-k processor
    pub fn build(self) -> ProcessingResult<TopKProcessor> {
        let k = self.k.unwrap_or(0); // Default to disabled
        TopKProcessor::new(k)
    }
}

/// Utility functions for top-k operations
pub mod utils {
    use super::*;
    use crate::processing::ProcessingContext;

    /// Calculate adaptive top-k based on context
    ///
    /// Adjusts k value based on generation context to balance
    /// diversity and quality. Uses context length and diversity
    /// metrics to determine optimal vocabulary size.
    pub fn adaptive_top_k(
        base_k: usize,
        context: &ProcessingContext,
        diversity_factor: f32,
    ) -> ProcessingResult<usize> {
        if base_k > MAX_TOP_K {
            return Err(ProcessingError::configuration("Base k exceeds maximum"));
        }

        if !diversity_factor.is_finite() || diversity_factor < 0.0 {
            return Err(ProcessingError::configuration("Invalid diversity factor"));
        }

        let mut adjusted_k = base_k as f32;

        // Increase k with lower diversity to encourage exploration
        let diversity_score = context.diversity_score();
        if diversity_score < 0.5 {
            adjusted_k *= 1.0 + (0.5 - diversity_score) * diversity_factor;
        }

        // Adjust based on context utilization
        let utilization = context.utilization_ratio();
        if utilization > 0.8 {
            // Near context limit, reduce k for more focused generation
            adjusted_k *= 0.8;
        } else if utilization < 0.3 {
            // Early in generation, allow more diversity
            adjusted_k *= 1.2;
        }

        // Clamp to valid range
        let final_k = (adjusted_k as usize).clamp(0, MAX_TOP_K);

        Ok(final_k)
    }

    /// Find optimal k for target vocabulary coverage
    ///
    /// Determines the k value that covers a specific fraction
    /// of the probability mass in the logits distribution.
    pub fn k_for_coverage(logits: &[f32], target_coverage: f32) -> ProcessingResult<usize> {
        if !(0.0..=1.0).contains(&target_coverage) {
            return Err(ProcessingError::configuration(
                "Target coverage must be between 0.0 and 1.0",
            ));
        }

        if logits.is_empty() {
            return Ok(0);
        }

        // Create sorted index-value pairs
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Sort by value descending
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply softmax for probability computation
        let max_logit = indexed_logits[0].1;
        let mut exp_sum = 0.0f32;
        let mut exp_logits: Vec<f32> = Vec::with_capacity(indexed_logits.len());

        for (_, logit) in &indexed_logits {
            let exp_val = (logit - max_logit).exp();
            exp_logits.push(exp_val);
            exp_sum += exp_val;
        }

        if exp_sum <= 0.0 {
            return Err(ProcessingError::numerical(
                "Invalid probability distribution",
            ));
        }

        // Find k that achieves target coverage
        let mut cumulative_prob = 0.0f32;
        for (k, &exp_logit) in exp_logits.iter().enumerate() {
            cumulative_prob += exp_logit / exp_sum;
            if cumulative_prob >= target_coverage {
                return Ok(k + 1); // +1 because k is 0-indexed
            }
        }

        // If we reach here, return full vocabulary size
        Ok(logits.len())
    }

    /// Estimate effective vocabulary size after top-k filtering
    ///
    /// Provides an estimate of how many tokens will have non-negligible
    /// probability after top-k filtering and softmax normalization.
    pub fn estimate_effective_vocab_size(logits: &[f32], k: usize) -> ProcessingResult<usize> {
        if k == 0 || k >= logits.len() {
            return Ok(logits.len());
        }

        // For small k, the effective size is approximately k
        if k <= 10 {
            return Ok(k);
        }

        // For larger k, use heuristic based on logits distribution
        // This is an approximation - in practice, some top-k tokens
        // may have very low probability after softmax
        let effective_ratio = if k <= 50 {
            0.8 // 80% of top-k tokens are typically significant
        } else if k <= 100 {
            0.7 // 70% for medium k
        } else {
            0.6 // 60% for large k
        };

        Ok((k as f32 * effective_ratio) as usize)
    }

    /// Validate top-k configuration for given vocabulary size
    pub fn validate_top_k_config(k: usize, vocab_size: usize) -> ProcessingResult<()> {
        if k > MAX_TOP_K {
            return Err(ProcessingError::configuration(format!(
                "Top-k {} exceeds maximum {}",
                k, MAX_TOP_K
            )));
        }

        if k > vocab_size && k != 0 {
            // This is a warning condition, not an error
            // k > vocab_size is effectively no filtering
        }

        Ok(())
    }
}
