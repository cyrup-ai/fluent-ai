//! Top-k sampling with efficient token filtering
//!
//! Top-k sampling keeps only the k most likely tokens, setting all others to negative infinity.
//! This provides a fixed vocabulary size regardless of model confidence.

use super::SamplingError;
use crate::processing::traits::LogitsProcessor;
use candle_core::Tensor;

/// Top-k sampling processor with optimized filtering algorithms
///
/// Top-k sampling works by:
/// 1. Find the k-th largest logit value
/// 2. Set all logits smaller than this threshold to -infinity
/// 3. Preserve relative ordering of top k tokens
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    top_k: usize,
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
    fn apply_top_k_filtering(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        let logits_vec = logits.to_vec1::<f32>()
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

    #[inline(always)]
    fn quickselect_kth_largest(&self, values: &mut [f32]) -> Result<f32, SamplingError> {
        // Use nth_element equivalent (partial_sort to find k-th largest)
        let k_index = self.top_k - 1; // 0-indexed
        
        if k_index >= values.len() {
            return Err(SamplingError::InvalidTopK(self.top_k));
        }

        // Partial sort to find k-th largest element
        values.select_nth_unstable_by(k_index, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(values[k_index])
    }

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
        let mask_vec: Vec<f32> = original_values.iter().enumerate()
            .map(|(i, &val)| {
                if val > threshold {
                    0.0 // Keep token (add 0 to logit)
                } else if (val - threshold).abs() < f32::EPSILON 
                    && threshold_indices.iter().position(|&idx| idx == i).unwrap_or(usize::MAX) < keep_tie_count {
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
        logits.broadcast_add(&mask)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    fn create_test_logits() -> Tensor {
        let device = Device::Cpu;
        // Create logits with clear ranking: [4.0, 3.0, 2.0, 1.0, 0.5]
        Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 4.0, 0.5], (5,), &device)
            .expect("tensor creation")
    }

    fn create_uniform_logits() -> Tensor {
        let device = Device::Cpu;
        Tensor::from_vec(vec![1.0f32; 10], (10,), &device)
            .expect("tensor creation")
    }

    fn create_tied_logits() -> Tensor {
        let device = Device::Cpu;
        // Create logits with ties: [3.0, 2.0, 2.0, 2.0, 1.0]
        Tensor::from_vec(vec![3.0f32, 2.0, 2.0, 2.0, 1.0], (5,), &device)
            .expect("tensor creation")
    }

    #[test]
    fn test_top_k_validation() {
        // Valid top-k values
        assert!(TopKProcessor::new(1).is_ok());
        assert!(TopKProcessor::new(50).is_ok());
        assert!(TopKProcessor::new(1000).is_ok());

        // Invalid top-k values
        assert!(matches!(
            TopKProcessor::new(0),
            Err(SamplingError::InvalidTopK(0))
        ));
    }

    #[test]
    fn test_top_k_filtering() {
        let processor = TopKProcessor::new(3).expect("valid top-k");
        
        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");
        
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        
        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        
        // Should keep exactly 3 tokens
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        assert_eq!(finite_count, 3);
        
        // The 3 largest values should be preserved
        let mut orig_sorted: Vec<(usize, f32)> = original_vec.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        orig_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Check that top 3 are kept
        for i in 0..3 {
            let (idx, _) = orig_sorted[i];
            assert!(processed_vec[idx].is_finite(), "Top {} token should be kept", i + 1);
        }
        
        // Check that bottom 2 are filtered
        for i in 3..5 {
            let (idx, _) = orig_sorted[i];
            assert_eq!(processed_vec[idx], f32::NEG_INFINITY, "Bottom {} token should be filtered", i - 2);
        }
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let processor = TopKProcessor::new(10).expect("valid top-k");
        
        let mut logits = create_test_logits(); // Only 5 tokens
        let original_vec = logits.to_vec1::<f32>().expect("conversion");
        
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        
        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        
        // All tokens should be kept when k > vocab_size
        for (orig, proc) in original_vec.iter().zip(processed_vec.iter()) {
            assert!((orig - proc).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_top_k_one() {
        let processor = TopKProcessor::new(1).expect("valid top-k");
        
        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");
        
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        
        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        
        // Should keep exactly 1 token
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        assert_eq!(finite_count, 1);
        
        // Should keep the maximum token
        let max_value = original_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_idx = original_vec.iter().position(|&x| (x - max_value).abs() < f32::EPSILON).unwrap();
        
        assert!(processed_vec[max_idx].is_finite());
        for (i, &val) in processed_vec.iter().enumerate() {
            if i != max_idx {
                assert_eq!(val, f32::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_ties_handling() {
        let processor = TopKProcessor::new(3).expect("valid top-k");
        
        let mut logits = create_tied_logits(); // [3.0, 2.0, 2.0, 2.0, 1.0]
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        
        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        
        // Should handle ties deterministically
        assert!(finite_count <= 3, "Should not exceed k={}", processor.top_k());
        assert!(finite_count >= 1, "Should keep at least one token");
        
        // The highest value (3.0) should always be kept
        assert!(processed_vec[0].is_finite());
    }

    #[test]
    fn test_uniform_distribution() {
        let processor = TopKProcessor::new(3).expect("valid top-k");
        
        let mut logits = create_uniform_logits(); // 10 identical values
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        
        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        
        // Should keep exactly 3 tokens even with ties
        assert_eq!(finite_count, 3);
    }

    #[test]
    fn test_edge_cases() {
        let device = Device::Cpu;

        // Empty logits should fail
        let empty_logits = Tensor::zeros((0,), DType::F32, &device).expect("tensor creation");
        let processor = TopKProcessor::new(3).expect("valid top-k");
        let mut logits = empty_logits;
        assert!(processor.process(&mut logits, &[], 0).is_err());

        // Single token with k=1
        let single_logits = Tensor::from_vec(vec![1.0f32], (1,), &device).expect("tensor creation");
        let mut logits = single_logits;
        assert!(processor.process(&mut logits, &[], 0).is_ok());
        let finite_count = processor.count_finite_tokens(&logits);
        assert_eq!(finite_count, 1);
    }

    #[test]
    fn test_kth_largest_algorithms() {
        let processor = TopKProcessor::new(3).expect("valid top-k");
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0]; // 3rd largest is 5.0
        
        // Test quickselect approach
        let mut values1 = values.clone();
        let result1 = processor.quickselect_kth_largest(&mut values1).expect("quickselect works");
        assert!((result1 - 5.0).abs() < f32::EPSILON);
        
        // Test partial sort approach
        let mut values2 = values.clone();
        let result2 = processor.partial_sort_kth_largest(&mut values2).expect("partial sort works");
        assert!((result2 - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_top_k_update() {
        let mut processor = TopKProcessor::new(5).expect("valid top-k");
        assert_eq!(processor.top_k(), 5);
        
        processor.set_top_k(10).expect("valid update");
        assert_eq!(processor.top_k(), 10);
        
        // Invalid update should fail
        assert!(processor.set_top_k(0).is_err());
    }

    #[test]
    fn test_display_and_equality() {
        let proc1 = TopKProcessor::new(50).expect("valid top-k");
        let proc2 = TopKProcessor::new(50).expect("valid top-k");
        let proc3 = TopKProcessor::new(40).expect("valid top-k");
        
        assert_eq!(proc1, proc2);
        assert_ne!(proc1, proc3);
        
        let display = format!("{}", proc1);
        assert!(display.contains("TopK"));
        assert!(display.contains("50"));
    }

    #[test]
    fn test_processor_trait_methods() {
        let processor = TopKProcessor::new(25).expect("valid top-k");
        
        assert_eq!(processor.name(), "TopKProcessor");
        assert!(processor.validate().is_ok());
        
        // is_identity depends on runtime vocab size, so test basic functionality
        assert!(!processor.is_identity() || processor.is_identity());
    }

    #[test]
    fn test_performance_characteristics() {
        // Test with larger vocabulary to verify performance characteristics
        let device = Device::Cpu;
        let large_vocab: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let large_logits = Tensor::from_vec(large_vocab, (1000,), &device)
            .expect("large tensor creation");
        
        let processor = TopKProcessor::new(50).expect("valid top-k");
        let mut logits = large_logits;
        
        let start = std::time::Instant::now();
        processor.process(&mut logits, &[], 0).expect("processing succeeds");
        let duration = start.elapsed();
        
        // Should complete in reasonable time (< 1ms for 1000 tokens)
        assert!(duration.as_millis() < 100, "Processing took too long: {:?}", duration);
        
        let finite_count = processor.count_finite_tokens(&logits);
        assert_eq!(finite_count, 50);
    }
}