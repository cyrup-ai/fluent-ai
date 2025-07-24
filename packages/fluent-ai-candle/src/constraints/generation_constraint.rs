//! Simple, optimized generation constraint trait for structured output
//!
//! This module provides the user's optimized GenerationConstraint trait
//! for zero-allocation, high-performance constraint validation.

use crate::error::CandleResult;

/// Simple, optimized constraint trait for structured generation
///
/// This trait provides a direct, zero-allocation interface for guiding
/// token generation with constraints like JSON schema validation.
pub trait GenerationConstraint {
    /// The constraint state type (must be Clone for efficiency)
    type State: Clone;

    /// Create a new initial constraint state
    fn new_state(&self) -> Self::State;

    /// Update constraint state with a validated token
    ///
    /// Returns true if the constraint is now complete
    fn update(&self, state: &mut Self::State, token: u32) -> CandleResult<bool>;

    /// Check if a token is valid for the current state without modifying state
    fn try_next(&self, state: &Self::State, token: u32) -> CandleResult<bool>;

    /// Check if the constraint is complete and generation can stop
    fn is_done(&self, state: &Self::State) -> bool;

    /// Get deterministic token sequence for batch processing optimization
    ///
    /// Returns tokens that must be generated next when only one valid path exists
    fn get_deterministic_sequence(&self, state: &Self::State) -> CandleResult<Vec<u32>>;
}
