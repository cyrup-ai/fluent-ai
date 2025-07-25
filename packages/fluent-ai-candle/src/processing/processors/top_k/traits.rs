//! LogitsProcessor trait implementation for TopKProcessor
//!
//! Provides the LogitsProcessor interface compliance with performance
//! optimizations and comprehensive validation.

use super::core::{TopKProcessor, MAX_TOP_K};
use crate::processing::traits::{LogitsProcessor, ProcessingResult};
use crate::processing::{ProcessingContext, ProcessingError};

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
            0.1 // Linear scan is very fast
        } else if self.k <= 32 {
            0.2 // Heap operations
        } else {
            0.3 // Quickselect overhead
        }
    }

    #[inline(always)]
    fn priority(&self) -> u8 {
        super::super::priorities::FILTERING
    }
}