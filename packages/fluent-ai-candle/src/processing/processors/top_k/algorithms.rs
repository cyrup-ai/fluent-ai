//! Selection algorithms for Top-K filtering
//!
//! Efficient algorithms for finding and filtering top-k elements from logits.

use arrayvec::ArrayVec;

use crate::processing::traits::{ProcessingResult, utils::{clamp_for_stability, validate_logits}};
use crate::processing::ProcessingError;

use super::core::{MAX_TOP_K, TopKBuffer};

/// Selection algorithms for top-k filtering
pub struct SelectionAlgorithms;

impl SelectionAlgorithms {
    /// Apply selection for very small k values using linear scan
    ///
    /// For very small k, a simple linear scan with a small buffer
    /// is more efficient than complex algorithms due to lower overhead.
    pub fn linear_scan_selection(logits: &mut [f32], k: usize) -> ProcessingResult<()> {
        // Use stack-allocated buffer for very small k
        let mut top_k: ArrayVec<(usize, f32), 8> = ArrayVec::new();

        // Find top k elements using linear scan
        for (idx, &logit_value) in logits.iter().enumerate() {
            if top_k.len() < k {
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
    pub fn heap_selection(logits: &mut [f32], k: usize) -> ProcessingResult<()> {
        let mut heap = TopKBuffer::new();

        // Build heap with first k elements
        for (idx, &logit) in logits.iter().enumerate().take(k) {
            if heap.try_push((idx, logit)).is_err() {
                return Err(ProcessingError::resource("Failed to initialize heap"));
            }
        }

        // Convert to min-heap
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Process remaining elements
        for (idx, &logit) in logits.iter().enumerate().skip(k) {
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
    pub fn quickselect_selection(logits: &mut [f32], k: usize) -> ProcessingResult<()> {
        // Create index-value pairs for sorting
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Find k-th largest using quickselect
        if indexed_logits.len() < k {
            return Ok(()); // k is larger than vocab size, no filtering needed
        }

        Self::quickselect(&mut indexed_logits, k)?;

        // Find threshold (k-th largest value)
        let threshold = indexed_logits[k - 1].1;

        // Mask logits below threshold
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Quickselect algorithm to find k-th largest element
    ///
    /// Partially sorts the array so that the k-th largest element
    /// is in the correct position. Uses median-of-three pivot selection
    /// for better performance on partially sorted data.
    fn quickselect(arr: &mut [(usize, f32)], k: usize) -> ProcessingResult<()> {
        if k == 0 || k > arr.len() {
            return Err(ProcessingError::configuration("Invalid k for quickselect"));
        }

        let mut left = 0;
        let mut right = arr.len() - 1;
        let target = k - 1; // 0-indexed

        while left < right {
            // Use median-of-three for pivot selection
            let pivot_idx = Self::median_of_three(arr, left, right);
            let partition_idx = Self::partition(arr, left, right, pivot_idx)?;

            if partition_idx == target {
                break;
            } else if partition_idx < target {
                left = partition_idx + 1;
            } else {
                right = partition_idx - 1;
            }
        }

        Ok(())
    }

    /// Partition array around pivot (3-way partitioning for stability)
    ///
    /// Rearranges array so elements greater than pivot come before it.
    /// Uses reverse order (largest first) for top-k selection.
    fn partition(
        arr: &mut [(usize, f32)],
        left: usize,
        right: usize,
        pivot_idx: usize,
    ) -> ProcessingResult<usize> {
        if pivot_idx < left || pivot_idx > right {
            return Err(ProcessingError::internal("Invalid pivot index"));
        }

        let pivot_value = arr[pivot_idx].1;

        // Move pivot to end
        arr.swap(pivot_idx, right);

        let mut store_idx = left;

        // Partition: elements > pivot go to left side
        for i in left..right {
            if arr[i].1 > pivot_value {
                arr.swap(i, store_idx);
                store_idx += 1;
            }
        }

        // Move pivot to final position
        arr.swap(store_idx, right);

        Ok(store_idx)
    }

    /// Select median-of-three as pivot for better quickselect performance
    ///
    /// Chooses the median of left, right, and middle elements to avoid
    /// worst-case O(n²) performance on sorted/reverse-sorted data.
    fn median_of_three(arr: &[(usize, f32)], left: usize, right: usize) -> usize {
        let mid = left + (right - left) / 2;

        let left_val = arr[left].1;
        let mid_val = arr[mid].1;
        let right_val = arr[right].1;

        if (left_val <= mid_val && mid_val <= right_val) || (right_val <= mid_val && mid_val <= left_val) {
            mid
        } else if (mid_val <= left_val && left_val <= right_val) || (right_val <= left_val && left_val <= mid_val) {
            left
        } else {
            right
        }
    }
}