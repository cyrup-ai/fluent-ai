//! Optimized constraint system for structured output generation
//!
//! This module provides the user's optimized zero-allocation constraint system
//! with direct tokenizer integration and finite state machine parsing.

pub mod generation_constraint;
pub mod json;

// Re-export core types for ergonomic usage
pub use generation_constraint::GenerationConstraint;
pub use json::{
    JsonConstraint, JsonCurrentState, JsonStackItem, JsonState, NumberState,
    create_json_constraint_for_tokenizer};
