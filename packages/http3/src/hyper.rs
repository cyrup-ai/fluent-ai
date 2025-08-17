//! Hyper compatibility module
//!
//! This module provides re-exports and compatibility layers for modules
//! that still reference the old `crate::hyper::` paths after refactoring.

// Re-export core client types
// Re-export body types for wasm compatibility - use bytes::Bytes as Body
pub use bytes::Bytes as Body;
// Re-export URL types
pub use url::Url;

pub use crate::client::HttpClient as Client;
// Re-export error types
pub use crate::error::HttpError as Error;
// Re-export request types
pub use crate::http::HttpRequest as Request;
// Re-export proxy types from our proxy module
pub use crate::proxy::Proxy;
// Re-export response types
pub use crate::streaming::response::HttpResponse as Response;

// Re-export request builder for wasm compatibility - create a stub for now
pub struct RequestBuilder;

// Re-export client builder for compatibility
pub struct ClientBuilder;

// Re-export canonical HttpChunk as HttpResponseChunk for backward compatibility
pub use crate::streaming::stream::chunks::HttpChunk as HttpResponseChunk;

// Nested modules for compatibility
pub mod error {
    pub use crate::error::HttpError as Error;
    pub type Result<T> = crate::error::HttpResult<T>;
}

#[cfg(feature = "wasm")]
pub mod wasm {
    #[cfg(feature = "wasm")]
    pub mod request {
        pub use crate::wasm::request::*;
    }
    #[cfg(feature = "wasm")]
    pub mod response {
        pub use crate::wasm::response::*;
    }
}
