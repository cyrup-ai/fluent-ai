//! Google Cloud Vertex AI client implementation
//!
//! Zero-allocation, blazing-fast implementation with OAuth2 JWT authentication,
//! dynamic model support, and streaming capabilities.
//!
//! # Features
//! - Zero allocation hot paths with `arrayvec` and `smallvec`
//! - Lock-free model metadata with `crossbeam-skiplist`
//! - Hot-swappable configuration with `arc-swap`
//! - OAuth2 JWT authentication with service accounts
//! - Streaming Server-Sent Events support
//! - Comprehensive error handling
//! - HTTP/3 connection pooling via `fluent_ai_http3`
//!
//! # Example
//! ```rust,no_run
//! use fluent_ai_provider::clients::vertexai::{VertexAIClient, VertexAIConfig};
//! 
//! let config = VertexAIConfig::new("my-project", "us-central1", service_account_json)?;
//! let client = VertexAIClient::new(config)?;
//! 
//! let response = client.complete("gpt-4", "Hello, world!").await?;
//! ```

mod auth;
mod client;
mod completion;
mod config;
mod error;
mod models;
mod streaming;

pub use auth::{VertexAIAuth, ServiceAccountConfig, TokenManager};
pub use client::{VertexAIClient, VertexAIProvider};
pub use completion::{
    VertexAICompletionBuilder,
    CompletionRequest, CompletionResponse, CompletionChunk,
    GenerationConfig, SafetySettings, Content, Part,
};
pub use config::VertexAIConfig;
pub use error::{VertexAIError, VertexAIResult};
pub use models::{ModelConfig, ModelCapabilities, VertexAIModels};
pub use streaming::{VertexAIStream, StreamEvent};

// Re-export commonly used types from dependencies
pub use arrayvec::{ArrayString, ArrayVec};
pub use smallvec::{SmallVec, smallvec};

/// Type alias for zero-allocation header collections
pub type HeaderVec = SmallVec<[(&'static str, ArrayString<128>); 8]>;

/// Type alias for zero-allocation string storage
pub type VertexString = ArrayString<256>;

/// Type alias for project ID storage
pub type ProjectId = ArrayString<64>;

/// Type alias for region storage  
pub type Region = &'static str;