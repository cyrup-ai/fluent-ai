//! Typestate markers for the builder pattern
//!
//! These zero-sized types enforce compile-time safety in the builder pattern.

/// Typestate marker indicating builder needs a prompt
pub struct NeedsPrompt;

/// Typestate marker indicating builder has a prompt
pub struct HasPrompt;