//! TLS backend integration and configuration
//!
//! Production-quality backend abstraction supporting native-tls and rustls.

#[cfg(feature = "default-tls")]
use native_tls_crate;
#[cfg(feature = "__rustls")]
use rustls::{ClientConfig, RootCertStore};

use super::types::{NoVerifier, TlsBackend};

impl TlsBackend {
    /// Get the default TLS backend based on enabled features.
    pub fn default() -> TlsBackend {
        #[cfg(feature = "default-tls")]
        {
            return TlsBackend::Default;
        }

        #[cfg(any(
            all(feature = "__rustls", not(feature = "default-tls")),
            feature = "http3"
        ))]
        {
            return TlsBackend::Rustls;
        }
    }

    /// Check if this backend supports HTTP/3.
    pub fn supports_http3(&self) -> bool {
        match self {
            #[cfg(feature = "default-tls")]
            TlsBackend::Default => false,
            #[cfg(feature = "__rustls")]
            TlsBackend::Rustls => true,
            #[cfg(feature = "__rustls")]
            TlsBackend::BuiltRustls => true,
            TlsBackend::UnknownPreconfigured => false,
        }
    }

    /// Get a human-readable name for this backend.
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "default-tls")]
            TlsBackend::Default => "native-tls",
            #[cfg(feature = "__rustls")]
            TlsBackend::Rustls => "rustls",
            #[cfg(feature = "__rustls")]
            TlsBackend::BuiltRustls => "rustls-built",
            TlsBackend::UnknownPreconfigured => "unknown",
        }
    }
}

#[cfg(feature = "default-tls")]
pub(crate) fn build_native_tls_connector() -> crate::Result<native_tls_crate::TlsConnector> {
    let mut builder = native_tls_crate::TlsConnector::builder();

    // Configure default settings
    builder.danger_accept_invalid_hostnames(false);
    builder.danger_accept_invalid_certs(false);

    builder.build().map_err(crate::error::builder)
}

#[cfg(feature = "__rustls")]
pub(crate) fn build_rustls_config() -> crate::Result<ClientConfig> {
    let mut root_store = RootCertStore::empty();

    // Add system root certificates
    #[cfg(feature = "rustls-tls-webpki-roots")]
    {
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    }

    #[cfg(feature = "rustls-tls-native-roots")]
    {
        let native_certs =
            rustls_native_certs::load_native_certs().map_err(|e| crate::HttpError::Tls {
                message: format!("failed to load native certs: {}", e),
            })?;

        for cert in native_certs {
            root_store.add(cert).map_err(|e| crate::HttpError::Tls {
                message: format!("failed to add native cert: {}", e),
            })?;
        }
    }

    Ok(ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth())
}

#[cfg(feature = "__rustls")]
pub(crate) fn build_rustls_config_dangerous() -> crate::Result<ClientConfig> {
    let config = ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(std::sync::Arc::new(NoVerifier))
        .with_no_client_auth();

    Ok(config)
}

/// Configuration builder for TLS settings.
pub struct TlsConfigBuilder {
    backend: TlsBackend,
    accept_invalid_certs: bool,
    accept_invalid_hostnames: bool,
}

impl TlsConfigBuilder {
    /// Create a new TLS configuration builder.
    pub fn new() -> Self {
        Self {
            backend: TlsBackend::default(),
            accept_invalid_certs: false,
            accept_invalid_hostnames: false,
        }
    }

    /// Set the TLS backend to use.
    pub fn backend(mut self, backend: TlsBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Accept invalid certificates (dangerous).
    pub fn danger_accept_invalid_certs(mut self, accept: bool) -> Self {
        self.accept_invalid_certs = accept;
        self
    }

    /// Accept invalid hostnames (dangerous).
    pub fn danger_accept_invalid_hostnames(mut self, accept: bool) -> Self {
        self.accept_invalid_hostnames = accept;
        self
    }

    /// Build the TLS configuration.
    #[cfg(feature = "default-tls")]
    pub fn build_native(self) -> crate::Result<native_tls_crate::TlsConnector> {
        let mut builder = native_tls_crate::TlsConnector::builder();

        builder.danger_accept_invalid_certs(self.accept_invalid_certs);
        builder.danger_accept_invalid_hostnames(self.accept_invalid_hostnames);

        builder.build().map_err(crate::error::builder)
    }

    /// Build the rustls configuration.
    #[cfg(feature = "__rustls")]
    pub fn build_rustls(self) -> crate::Result<ClientConfig> {
        if self.accept_invalid_certs || self.accept_invalid_hostnames {
            build_rustls_config_dangerous()
        } else {
            build_rustls_config()
        }
    }
}

impl Default for TlsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
