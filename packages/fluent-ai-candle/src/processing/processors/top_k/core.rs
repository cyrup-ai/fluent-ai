//! Core Top-K processor implementation
//!
//! Main TopKProcessor struct with core functionality and algorithm selection logic.

use arrayvec::ArrayVec;

use crate::processing::traits::{
    LogitsProcessor, NumericallyStableProcessor, ProcessingResult,
    ZeroAllocationProcessor,
    utils::{clamp_for_stability, validate_logits}};
use crate::processing::{ProcessingContext, ProcessingError};

use super::algorithms::SelectionAlgorithms;

/// Maximum top-k value for bounded allocation
pub const MAX_TOP_K: usize = 256;

/// Stack-allocated buffer for top-k selection
pub type TopKBuffer = ArrayVec<(usize, f32), MAX_TOP_K>;

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
    pub(super) k: usize,
    /// Cached identity status for optimization
    pub(super) is_identity: bool}

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
    pub fn apply_top_k_filtering(&self, logits: &mut [f32]) -> ProcessingResult<()> {
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
            SelectionAlgorithms::linear_scan_selection(logits, self.k)?;
        } else if self.k <= 32 {
            SelectionAlgorithms::heap_selection(logits, self.k)?;
        } else {
            SelectionAlgorithms::quickselect_selection(logits, self.k)?;
        }

        Ok(())
    }
}

impl LogitsProcessor for TopKProcessor {
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        self.apply_top_k_filtering(logits)
    }

    fn name(&self) -> &'static str {
        "TopK"
    }

    fn is_identity(&self) -> bool {
        self.is_identity
    }

    fn is_enabled(&self) -> bool {
        self.k > 0
    }

    fn validate(&self) -> ProcessingResult<()> {
        if self.k > MAX_TOP_K {
            return Err(ProcessingError::configuration(format!(
                "Top-k {} exceeds maximum {}",
                self.k, MAX_TOP_K
            )));
        }
        Ok(())
    }

    fn config_summary(&self) -> String {
        if self.is_identity {
            "TopK(disabled)".to_string()
        } else {
            format!("TopK(k={})", self.k)
        }
    }

    fn estimated_overhead(&self) -> f32 {
        if self.is_identity {
            0.0 // No processing overhead
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
        super::super::priorities::FILTERING
    }
}

// Marker trait implementations
impl ZeroAllocationProcessor for TopKProcessor {}
impl NumericallyStableProcessor for TopKProcessor {}
