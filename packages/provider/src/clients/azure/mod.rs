//! Azure OpenAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::azure;
//!
//! let client = azure::Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
//!
//! let gpt4o = client.completion_model(azure::GPT_4O);
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;
pub mod transcription;

#[cfg(feature = "audio")]
pub mod audio_generation;
#[cfg(feature = "image")]
pub mod image_generation;

// Re-export main types
#[cfg(feature = "audio")]
pub use audio_generation::*;
pub use client::{AzureOpenAIAuth, Client, ClientBuilder};
pub use completion::*;
pub use embedding::*;
#[cfg(feature = "image")]
pub use image_generation::*;
pub use streaming::*;
pub use transcription::*;
