// src/cognitive/committee/mod.rs
//! A modular, multi-agent evaluation committee for code optimization.

// Expose the sub-modules.
pub mod agent;
pub mod committee_consensus;
pub mod committee_evaluators;
pub mod committee_evaluators_extension;
pub mod committee_types;
pub mod orchestrator;
pub mod relaxed_counter;

// Re-export the primary public-facing types for easy access.
pub use orchestrator::EvaluationCommittee;

pub use crate::cognitive::common::types::CommitteeEvent;
