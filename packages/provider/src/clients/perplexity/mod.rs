//! Perplexity API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::perplexity;
//!
//! let client = perplexity::Client::new("YOUR_API_KEY");
//!
//! let sonar_pro = client.completion_model(perplexity::SONAR_PRO);
//! ```

pub mod client;
pub mod completion;
pub mod streaming;

pub use client::{Client, PerplexityCompletionBuilder};
pub use completion::{CompletionModel, SONAR, SONAR_PRO};
pub mod types;
