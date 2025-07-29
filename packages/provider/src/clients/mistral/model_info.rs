//! ARCHITECTURAL COMPLIANCE NOTICE
//!
//! This file has been ELIMINATED to comply with the centralized model architecture.
//!
//! LOCAL MODEL ENUMERATION VIOLATIONS:
//! - All model structs (MistralLarge, Codestral, etc.) with hardcoded ModelConfig
//! - get_model_config() function delegating to centralized system  
//! - Helper functions replicating model information locally
//! - Constants and compile-time model lookups
//!
//! SOLUTION: Use model-info package as single source of truth
//! 
//! CORRECT USAGE:
//! ```rust
//! use model_info::{Provider, MistralProvider, ModelInfo};
//! 
//! let provider = Provider::Mistral(MistralProvider);
//! let model_info = provider.get_model_info("mistral-large-latest");
//! ```
//!
//! All model information now comes from the centralized model-info package.
//! This eliminates architectural violations and provides consistent model data.