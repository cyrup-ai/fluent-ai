//! Top-k sampling with efficient token filtering
//!
//! Top-k sampling keeps only the k most likely tokens, setting all others to negative infinity.
//! This provides a fixed vocabulary size regardless of model confidence.

use candle_core::Tensor;

use super::SamplingError;
// Removed unused import: crate::processing::traits::LogitsProcessor

/// Top-k sampling processor with optimized filtering algorithms
///
/// Top-k sampling works by:
/// 1. Find the k-th largest logit value
/// 2. Set all logits smaller than this threshold to -infinity
/// 3. Preserve relative ordering of top k tokens
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    top_k: usize,
    #[allow(dead_code)] // Legacy field for optimization
    is_identity: bool, // Optimization when top_k >= vocab_size
}

impl TopKProcessor {
    /// Create new top-k processor with validation
    ///
    /// # Arguments
    /// * `top_k` - Number of top tokens to keep (must be > 0)
    ///
    /// # Errors
    /// Returns `SamplingError::InvalidTopK` if top_k is 0
    pub fn new(top_k: usize) -> Result<Self, SamplingError> {
        if top_k == 0 {
            return Err(SamplingError::InvalidTopK(top_k));
        }

        Ok(Self {
            top_k,
            is_identity: false, // Will be set dynamically based on vocab size
        })
    }

    /// Get current top-k value
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Update top-k with validation
    pub fn set_top_k(&mut self, top_k: usize) -> Result<(), SamplingError> {
        if top_k == 0 {
            return Err(SamplingError::InvalidTopK(top_k));
        }
        self.top_k = top_k;
        Ok(())
    }

    /// Apply top-k filtering with optimized algorithms
    ///
    /// Uses different strategies based on k and vocabulary size:
    /// - Small k: Partial sort (quickselect-based)
    /// - Large k: Full sort with early termination
    /// - k >= vocab_size: No-op (identity)
    #[allow(dead_code)]
    fn apply_top_k_filtering(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        let logits_vec = logits
            .to_vec1::<f32>()
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        if logits_vec.is_empty() {
            return Err(SamplingError::EmptyVocabulary);
        }

        let vocab_size = logits_vec.len();

        // No filtering needed if k >= vocab_size
        if self.top_k >= vocab_size {
            return Ok(logits.clone());
        }

        // Find k-th largest value using efficient algorithm
        let threshold = self.find_kth_largest(&logits_vec)?;

        // Apply threshold mask
        self.apply_threshold_mask(logits, &logits_vec, threshold)
    }

    /// Find k-th largest value using optimized selection algorithm
    ///
    /// Uses quickselect for O(n) expected time complexity when k is small,
    /// falls back to partial sort for larger k values.
    #[allow(dead_code)]
    #[inline(always)]
    fn find_kth_largest(&self, values: &[f32]) -> Result<f32, SamplingError> {
        let mut sorted_values = values.to_vec();

        // Use different strategies based on k relative to vocab size
        let vocab_size = values.len();

        if self.top_k <= vocab_size / 4 {
            // For small k, use partial sort (more efficient)
            self.quickselect_kth_largest(&mut sorted_values)
        } else {
            // For larger k, full sort might be more cache-friendly
            self.partial_sort_kth_largest(&mut sorted_values)
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn quickselect_kth_largest(&self, values: &mut [f32]) -> Result<f32, SamplingError> {
        // Use nth_element equivalent (partial_sort to find k-th largest)
        let k_index = self.top_k - 1; // 0-indexed

        if k_index >= values.len() {
            return Err(SamplingError::InvalidTopK(self.top_k));
        }

        // Partial sort to find k-th largest element
        values.select_nth_unstable_by(k_index, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(values[k_index])
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn partial_sort_kth_largest(&self, values: &mut [f32]) -> Result<f32, SamplingError> {
        let k_index = self.top_k - 1;

        if k_index >= values.len() {
            return Err(SamplingError::InvalidTopK(self.top_k));
        }

        // Sort in descending order, but only up to k elements
        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        Ok(values[k_index])
    }

    /// Apply threshold mask to zero out tokens below k-th largest
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_threshold_mask(
        &self,
        logits: &Tensor,
        original_values: &[f32],
        threshold: f32,
    ) -> Result<Tensor, SamplingError> {
        // Count how many values are >= threshold (handle ties correctly)
        let mut kept_count = 0;
        let mut threshold_indices: Vec<usize> = Vec::new();

        // First pass: count values > threshold and collect ties
        for (i, &val) in original_values.iter().enumerate() {
            if val > threshold {
                kept_count += 1;
            } else if (val - threshold).abs() < f32::EPSILON {
                threshold_indices.push(i);
            }
        }

        // Handle ties at the threshold
        let remaining_slots = self.top_k.saturating_sub(kept_count);

        // Sort ties by original index for deterministic behavior
        threshold_indices.sort();
        let keep_tie_count = remaining_slots.min(threshold_indices.len());

        // Create mask
        let mask_vec: Vec<f32> = original_values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                if val > threshold {
                    0.0 // Keep token (add 0 to logit)
                } else if (val - threshold).abs() < f32::EPSILON
                    && threshold_indices
                        .iter()
                        .position(|&idx| idx == i)
                        .unwrap_or(usize::MAX)
                        < keep_tie_count
                {
                    0.0 // Keep this tied token
                } else {
                    f32::NEG_INFINITY // Filter out token
                }
            })
            .collect();

        let device = logits.device();
        let mask = Tensor::from_vec(mask_vec, (original_values.len(),), device)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Apply mask to logits
        logits
            .broadcast_add(&mask)
            .map_err(|e| SamplingError::TensorError(e.to_string()))
    }

    /// Validate that exactly k tokens are kept (for testing)
    #[cfg(test)]
    fn count_finite_tokens(&self, logits: &Tensor) -> usize {
        let values = logits.to_vec1::<f32>().unwrap_or_default();
        values.iter().filter(|&&x| x.is_finite()).count()
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for TopKProcessor {
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn validate(&self) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn name(&self) -> &'static str {
//         "TopKProcessor"
//     }
//
//     fn is_identity(&self) -> bool {
//         self.is_identity
//     }
// }

// Implement common traits for ergonomics
impl std::fmt::Display for TopKProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TopK({})", self.top_k)
    }
}

impl PartialEq for TopKProcessor {
    fn eq(&self, other: &Self) -> bool {
        self.top_k == other.top_k
    }
}
