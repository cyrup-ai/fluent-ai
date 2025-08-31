//! TLS Builder Interface
//!
//! This module provides a fluent, secure-by-default certificate management API.
//! All internal complexity is hidden behind the builder interface.

// Internal modules - not exposed publicly
pub(crate) mod certificate;
pub(crate) mod crl_cache;
pub(crate) mod errors;
pub(crate) mod key_encryption;
pub(crate) mod ocsp;
pub(crate) mod tls_config;
pub(crate) mod types;

// Public builder interface - the only public API
pub mod builder;
pub use builder::{CertificateAuthority, Tls};
