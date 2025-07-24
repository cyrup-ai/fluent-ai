//! Dummy model implementation for testing
//!
//! Provides a simple dummy model implementation used for testing and
//! placeholder functionality with zero-allocation patterns.

use candle_core::{Module, Tensor};

/// Dummy model implementation for testing with zero-allocation design
pub struct DummyModel;

impl Module for DummyModel {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Return logits for a small vocabulary (for testing)
        let batch_size = xs.dim(0)?;
        let seq_len = xs.dim(1)?;
        let vocab_size = 1000;

        // Create dummy logits with zero-allocation tensor creation
        Tensor::zeros((batch_size, seq_len, vocab_size), xs.dtype(), xs.device())
    }
}