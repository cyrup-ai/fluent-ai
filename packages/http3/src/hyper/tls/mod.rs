//! TLS configuration and types
//!
//! A `Client` will use transport layer security (TLS) by default to connect to
//! HTTPS destinations.
//!
//! # Backends
//!
//! http3 supports several TLS backends, enabled with Cargo features.
//!
//! ## default-tls
//!
//! http3 will pick a TLS backend by default. This is true when the
//! `default-tls` feature is enabled.
//!
//! While it currently uses `native-tls`, the feature set is designed to only
//! enable configuration that is shared among available backends. This allows
//! http3 to change the default to `rustls` (or another) at some point in the
//! future.
//!
//! <div class="warning">This feature is enabled by default, and takes
//! precedence if any other crate enables it. This is true even if you declare
//! `features = []`. You must set `default-features = false` instead.</div>
//!
//! Since Cargo features are additive, other crates in your dependency tree can
//! cause the default backend to be enabled. If you wish to ensure your
//! `Client` uses a specific backend, call the appropriate builder methods
//! (such as [`use_rustls_tls()`][]).
//!
//! [`use_rustls_tls()`]: crate::ClientBuilder::use_rustls_tls()
//!
//! ## native-tls
//!
//! This backend uses the [native-tls][] crate. That will try to use the system
//! TLS on Windows and Mac, and OpenSSL on Linux targets.
//!
//! Enabling the feature explicitly allows for `native-tls`-specific
//! configuration options.
//!
//! [native-tls]: https://crates.io/crates/native-tls
//!
//! ## rustls-tls
//!
//! This backend uses the [rustls][] crate, a TLS library written in Rust.
//!
//! [rustls]: https://crates.io/crates/rustls

mod types;
mod certificate;
mod identity;
mod crl;
mod backend;

#[cfg(test)]
mod tests;

// Re-export main types
pub use types::{Certificate, Identity, TlsBackend, Version, IgnoreHostname};

#[cfg(feature = "__rustls")]
pub use types::CertificateRevocationList;

pub use backend::TlsConfigBuilder;

// Internal types for module organization
pub(crate) use types::{Cert, ClientCert, NoVerifier};
pub(crate) use backend::{build_native_tls_connector, build_rustls_config, build_rustls_config_dangerous};