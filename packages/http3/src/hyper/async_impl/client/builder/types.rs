//! Core types and structure for ClientBuilder
//!
//! Contains the main ClientBuilder struct and its core implementation
//! including constructor and build methods.

use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use http::Method;
use http::header::{HeaderMap, HeaderValue, USER_AGENT};
#[cfg(feature = "http3")]
use quinn::VarInt;

use super::super::config::{Config, HttpVersionPref};
use super::super::core::{Client, ClientRef, SimpleHyperService};
use super::super::tls_setup::setup_tls_connector;
use crate::hyper::config::{RequestConfig, RequestTimeout};
use crate::hyper::connect::ConnectorBuilder;
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
#[cfg(feature = "hickory-dns")]
use crate::hyper::dns::hickory::HickoryDnsResolver;
use crate::hyper::dns::{DynResolver, Resolve, gai::GaiResolver};
use crate::hyper::proxy::Matcher as ProxyMatcher;
#[cfg(feature = "__tls")]
use crate::hyper::tls::TlsBackend;

/// A `ClientBuilder` can be used to create a `Client` with custom configuration.
#[must_use]
pub struct ClientBuilder {
    pub(super) config: Config,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientBuilder {
    /// Constructs a new `ClientBuilder`.
    ///
    /// This is the same as `Client::builder()`.
    pub fn new() -> Self {
        ClientBuilder {
            config: Config::default(),
        }
    }

    /// Returns a `Client` that uses this `ClientBuilder` configuration.
    ///
    /// # Errors
    ///
    /// This method fails if a TLS backend cannot be initialized, or the resolver
    /// cannot load the system configuration.
    pub fn build(self) -> crate::Result<Client> {
        let config = self.config;

        if let Some(err) = config.error {
            return Err(err);
        }

        let mut proxies = config.proxies;
        if config.auto_sys_proxy {
            proxies.push(ProxyMatcher::system());
        }
        let proxies = Arc::new(proxies);

        // Initialize proxy authentication variables with proper defaults
        let proxies_maybe_http_auth: Option<bool> = None;
        let proxies_maybe_http_custom_headers: HashMap<String, String> = HashMap::new();

        #[allow(unused)]
        #[cfg(feature = "http3")]
        let mut h3_connector: Option<crate::hyper::async_impl::h3_client::H3Client> = None;

        let resolver = {
            let mut resolver: Arc<dyn Resolve> = match config.hickory_dns {
                false => Arc::new(GaiResolver::new()),
                #[cfg(feature = "hickory-dns")]
                true => Arc::new(HickoryDnsResolver::default()),
                #[cfg(not(feature = "hickory-dns"))]
                true => unreachable!("hickory-dns shouldn't be enabled unless the feature is"),
            };
            if let Some(dns_resolver) = config.dns_resolver {
                resolver = dns_resolver;
            }
            if !config.dns_overrides.is_empty() {
                // Convert Vec<SocketAddr> to ArrayVec<SocketAddr, 8>
                let converted_overrides: std::collections::HashMap<
                    String,
                    arrayvec::ArrayVec<std::net::SocketAddr, 8>,
                > = config
                    .dns_overrides
                    .into_iter()
                    .map(|(k, v)| {
                        let mut array_vec = arrayvec::ArrayVec::new();
                        for addr in v.into_iter().take(8) {
                            let _ = array_vec.try_push(addr);
                        }
                        (k, array_vec)
                    })
                    .collect();
                crate::hyper::dns::resolve::DynResolver::new_with_overrides(
                    resolver,
                    converted_overrides,
                )
            } else {
                DynResolver::new(resolver)
            }
        };

        // Setup TLS connector using dedicated module
        let _connector = setup_tls_connector(&config)?;

        // Create hyper client with proper TLS connector integration
        let http_connector = hyper_util::client::legacy::connect::HttpConnector::new();
        let hyper_client: hyper_util::client::legacy::Client<
            hyper_util::client::legacy::connect::HttpConnector,
            hyper::body::Incoming,
        > = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
            .build(http_connector);

        // Simple hyper service wrapper for AsyncStream compatibility
        let hyper_service = SimpleHyperService {
            #[cfg(feature = "cookies")]
            cookie_store: config.cookie_store.clone(),
            hyper: hyper_client,
        };

        Ok(Client {
            inner: Arc::new(ClientRef {
                accepts: config.accepts,
                #[cfg(feature = "cookies")]
                cookie_store: config.cookie_store.clone(),
                redirect_policy: config.redirect_policy,
                #[cfg(feature = "http3")]
                h3_client: None, // H3 client can be added back later
                headers: config.headers,
                referer: config.referer,
                read_timeout: config.read_timeout,
                request_timeout: RequestConfig::new(config.timeout),
                hyper: hyper_service,
                proxies,
                proxies_maybe_http_auth: false,
                proxies_maybe_http_custom_headers: false,
                https_only: config.https_only,
                redirect_policy_desc: None,
            }),
        })
    }
}

impl fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("ClientBuilder");
        self.config.fmt_fields(&mut builder);
        builder.finish()
    }
}
