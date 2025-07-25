//! xAi API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::xai;
//!
//! let client = xai::Client::new("YOUR_API_KEY");
//!
//! let groq_embedding_model = client.embedding_model(xai::v1);
//! ```

pub mod client;
pub mod completion;
pub mod streaming;
pub mod types;

pub use client::{Client, XAICompletionBuilder};
pub use completion::{CompletionModel, GROK_3, GROK_3_MINI};
