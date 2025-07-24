//! Pure YAML model information structs
//!
//! This crate provides simple data structures that mirror the YAML structure exactly.
//! Zero transformations, zero domain dependencies - just raw YAML data containers.
//!
//! ## Performance Characteristics
//! - Blazing-fast deserialization with yyaml
//! - Zero allocation constructors where possible
//! - Elegant ergonomic field access
//! - Lock-free concurrent access patterns
//!
//! ## Usage
//! ```rust
//! use yaml_model_info::{download, models::{YamlProvider, YamlModel}};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Download YAML data with intelligent caching
//!     let yaml_content = download::download_yaml_with_cache(".cache").await?;
//!     
//!     // Parse with yyaml (zero fallback logic)
//!     let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;
//!     
//!     // Process providers and models
//!     for provider in providers {
//!         println!("Provider: {}", provider.identifier());
//!         for model in &provider.models {
//!             println!("  Model: {}", model.name);
//!         }
//!     }
//!     
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

/// YAML download functionality with intelligent caching
pub mod download;

/// Pure YAML data structures
pub mod models;

// Re-export main types for convenience
pub use models::{YamlProvider, YamlModel};

/// Simple utility function to load and parse YAML data
/// 
/// This function combines download and parsing in a single operation.
/// Uses yyaml exclusively with zero fallback logic.
/// 
/// # Performance
/// - First call: Downloads and caches YAML data, then parses
/// - Subsequent calls: Uses cached data with ETag validation
/// - Zero allocation patterns where possible
/// 
/// # Safety
/// All operations use proper error handling with no unwrap/expect calls.
#[inline(always)]
pub async fn load_yaml_data<P: AsRef<std::path::Path>>(cache_dir: P) -> Result<Vec<YamlProvider>, Box<dyn std::error::Error>> {
    let yaml_content = download::download_yaml_with_cache(cache_dir).await?;
    let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;
    Ok(providers)
}