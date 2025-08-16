//! HTTP client implementation with async streaming support
//!
//! Provides HTTP/1.1, HTTP/2, and HTTP/3 client implementations
//! with zero-allocation streaming and async task primitives.

pub mod async_impl;
pub mod async_stream_service;
pub mod config;
pub mod connect;
pub mod dns;
pub mod error;
pub mod into_url;
pub mod proxy;
pub mod redirect;
pub mod response;
pub mod tls;
pub mod wasm;

// Re-export main client types
pub use async_impl::{Body, Client, ClientBuilder, Request, RequestBuilder, Response};
// Re-export service types
pub use async_stream_service::{AsyncStreamLayer, AsyncStreamService};
// Re-export config types
pub use config::{RequestConfig, RequestConfigValue};
// Re-export connection types
pub use connect::{Connect, HttpConnector};
// Re-export DNS types
pub use dns::{DnsResult, Name, Resolve};
// Re-export error types
pub use error::{Error, Result};
// Re-export URL types
pub use into_url::{IntoUrl, IntoUrlSealed};
// Re-export proxy types
pub use proxy::{Proxy, NoProxy, Intercept, Extra};
// Re-export redirect types
pub use redirect::{Action, Attempt, Policy};
// Response types are re-exported from async_impl module
// Re-export TLS types
pub use tls::{Certificate, Identity, TlsBackend, Version};
// Re-export WASM types
#[cfg(target_arch = "wasm32")]
pub use wasm::{AbortController, AbortSignal};
