//! H3 connector core implementation
//!
//! Main H3Connector struct with TLS configuration, QUIC endpoint setup,
//! and production-quality connection establishment.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::crypto::rustls::QuicClientConfig;
use quinn::{ClientConfig, Endpoint, TransportConfig};

use super::configuration::H3ClientConfig;
use super::types::DynResolver;

#[derive(Clone)]
pub struct H3Connector {
    pub(crate) resolver: DynResolver,
    pub(crate) endpoint: Endpoint,
    pub(crate) config: H3ClientConfig,
}

impl H3Connector {
    /// Create new H3Connector with production TLS configuration
    pub fn new() -> Option<Self> {
        // Create production TLS configuration with native certificates
        let mut root_store = rustls::RootCertStore::empty();

        // Load native certificates for production TLS
        let cert_result = rustls_native_certs::load_native_certs();
        for cert in &cert_result.certs {
            if let Err(e) = root_store.add(cert.clone()) {
                log::warn!("Failed to add native cert: {}", e);
            }
        }
        if let Some(first_error) = cert_result.errors.first() {
            log::warn!("Certificate loading errors: {}", first_error);
        }
        log::debug!("Loaded {} native certificates", cert_result.certs.len());

        // Create TLS configuration
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        // Create QUIC configuration with proper error handling
        let quic_client_config = match QuicClientConfig::try_from(tls_config) {
            Ok(config) => Arc::new(config),
            Err(e) => {
                log::error!("Failed to create QUIC crypto config: {}", e);
                return None;
            }
        };

        let mut quinn_config = ClientConfig::new(quic_client_config);

        // Configure transport parameters optimized for HTTP/3
        let mut transport_config = TransportConfig::default();
        transport_config.max_concurrent_bidi_streams(100_u32.into());
        transport_config.max_concurrent_uni_streams(100_u32.into());

        // Set idle timeout with proper error handling
        if let Ok(idle_timeout) = Duration::from_secs(30).try_into() {
            transport_config.max_idle_timeout(Some(idle_timeout));
        } else {
            log::warn!("Failed to set idle timeout, using default");
        }

        transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
        quinn_config.transport_config(Arc::new(transport_config));

        // Create endpoint with progressive fallback strategy
        let socket_addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0));
        let mut endpoint = match Endpoint::client(socket_addr) {
            Ok(ep) => ep,
            Err(_) => {
                log::debug!("IPv6 endpoint failed, trying IPv4");
                let ipv4_addr = SocketAddr::from(([0, 0, 0, 0], 0));
                match Endpoint::client(ipv4_addr) {
                    Ok(ep) => ep,
                    Err(e) => {
                        log::error!("Failed to create QUIC endpoint: {}", e);
                        return None;
                    }
                }
            }
        };

        endpoint.set_default_client_config(quinn_config);

        Some(Self {
            resolver: (),
            endpoint,
            config: H3ClientConfig::default(),
        })
    }

    /// Get reference to the QUIC endpoint
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Get reference to the H3 configuration
    pub fn config(&self) -> &H3ClientConfig {
        &self.config
    }

    /// Update H3 configuration
    pub fn with_config(mut self, config: H3ClientConfig) -> Self {
        self.config = config;
        self
    }
}
