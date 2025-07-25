//! Top-K sampling processor - decomposed module
//!
//! This module has been decomposed from a single 643-line file into focused submodules
//! for better maintainability and code organization. Each module is ≤300 lines.
//!
//! Architecture:
//! - `core`: Main TopKProcessor struct with adaptive algorithm selection (~180 lines)
//! - `algorithms`: Selection algorithms (linear scan, heap, quickselect) (~246 lines)
//! - `analysis`: Advanced analysis utilities for entropy and coverage (~290 lines)
//! - `config`: Configuration types and ConfigurableProcessor impl (~30 lines)
//! - `builder`: Builder pattern for fluent construction (~60 lines)
//! - `traits`: LogitsProcessor trait implementation (~65 lines)  
//! - `utils`: Utility functions for adaptive k and validation (~149 lines)

pub mod algorithms;
pub mod analysis;
pub mod builder;
pub mod config;
pub mod core;
pub mod traits;
pub mod utils;

// Re-export all public types and functions for backward compatibility
pub use algorithms::SelectionAlgorithms;
pub use analysis::{entropy_based_coverage, estimate_effective_vocab_size, k_for_coverage, DistributionMetrics};
pub use builder::TopKBuilder;
pub use config::TopKConfig;
pub use core::{TopKProcessor, MAX_TOP_K, TopKBuffer};
pub use utils::{adaptive_top_k, validate_top_k_config, perplexity_based_k, temperature_adjusted_k};

// Backward compatibility alias
pub use core::TopKProcessor as TopK;