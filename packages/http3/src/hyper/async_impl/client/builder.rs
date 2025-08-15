use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use http::header::{HeaderMap, HeaderValue, USER_AGENT};
use http::Method;

#[cfg(feature = "http3")]
use quinn::VarInt;

use super::config::{Config, HttpVersionPref};
use super::core::{Client, ClientRef, SimpleHyperService};
use super::tls_setup::setup_tls_connector;
use crate::hyper::connect::{ConnectorBuilder};
use crate::hyper::config::{RequestConfig, RequestTimeout};
use crate::hyper::dns::{DynResolver, Resolve, gai::GaiResolver};
#[cfg(feature = "hickory-dns")]
use crate::hyper::dns::hickory::HickoryDnsResolver;
use crate::hyper::proxy::Matcher as ProxyMatcher;

#[cfg(feature = "cookies")]
use crate::hyper::cookie;
#[cfg(feature = "__tls")]
use crate::hyper::tls::TlsBackend;

/// A `ClientBuilder` can be used to create a `Client` with custom configuration.
#[must_use]
pub struct ClientBuilder {
    config: Config,
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
                let converted_overrides: std::collections::HashMap<String, arrayvec::ArrayVec<std::net::SocketAddr, 8>> = 
                    config.dns_overrides.into_iter()
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
        let hyper_client: hyper_util::client::legacy::Client<hyper_util::client::legacy::connect::HttpConnector, hyper::body::Incoming> = 
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new()).build(http_connector);
        
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
                redirect_policy_desc: None
            }),
        })
    }

    // Higher-level options

    /// Sets the `User-Agent` header to be used by this client.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Name your user agent after your app?
    /// static APP_USER_AGENT: &str = concat!(
    ///     env!("CARGO_PKG_NAME"),
    ///     "/",
    ///     env!("CARGO_PKG_VERSION"),
    /// );
    ///
    /// let client = crate::hyper::Client::builder()
    ///     .user_agent(APP_USER_AGENT)
    ///     .build();
    /// let mut res_stream = client.get("https://www.rust-lang.org").send();
    /// if let Some(response) = res_stream.try_next() {
    ///     println!("Response: {:?}", response);
    /// }
    /// # // OK
    /// ```
    pub fn user_agent<V>(mut self, value: V) -> ClientBuilder
    where
        V: TryInto<HeaderValue>,
        V::Error: Into<http::Error>,
    {
        match value.try_into() {
            Ok(value) => {
                self.config.headers.insert(USER_AGENT, value);
            }
            Err(_e) => {
                self.config.error = Some(crate::HttpError::builder("Header conversion error".to_string()));
            }
        };
        self
    }
    
    /// Sets the default headers for every request.
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::hyper::header;
    /// 
    /// let mut headers = header::HeaderMap::new();
    /// headers.insert("X-MY-HEADER", header::HeaderValue::from_static("value"));
    ///
    /// // Consider marking security-sensitive headers with `set_sensitive`.
    /// let mut auth_value = header::HeaderValue::from_static("secret");
    /// auth_value.set_sensitive(true);
    /// headers.insert(header::AUTHORIZATION, auth_value);
    ///
    /// // get a client builder
    /// let client = crate::hyper::Client::builder()
    ///     .default_headers(headers)
    ///     .build();
    /// let mut res_stream = client.get("https://www.rust-lang.org").send();
    /// if let Some(response) = res_stream.try_next() {
    ///     println!("Response: {:?}", response);
    /// }
    /// ```
    pub fn default_headers(mut self, headers: HeaderMap) -> ClientBuilder {
        for (key, value) in headers.iter() {
            self.config.headers.insert(key, value.clone());
        }
        self
    }

    /// Enable a persistent cookie store for the client.
    ///
    /// Cookies received in responses will be preserved and included in
    /// additional requests.
    ///
    /// By default, no cookie store is used. Enabling the cookie store
    /// with `cookie_store(true)` will set the store to a default implementation.
    /// It is **not** necessary to call [cookie_store(true)](crate::ClientBuilder::cookie_store) if [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider)
    /// is used; calling [cookie_store(true)](crate::ClientBuilder::cookie_store) _after_ [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) will result
    /// in the provided `my_cookie_store` being **overridden** with a default implementation.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookie_store(mut self, enable: bool) -> ClientBuilder {
        if enable {
            self.cookie_provider(Arc::new(cookie::Jar::default()))
        } else {
            self.config.cookie_store = None;
            self
        }
    }

    /// Set the persistent cookie store for the client.
    ///
    /// Cookies received in responses will be passed to this store, and
    /// additional requests will query this store for cookies.
    ///
    /// By default, no cookie store is used. It is **not** necessary to also call
    /// [cookie_store(true)](crate::ClientBuilder::cookie_store) if [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) is used; calling
    /// [cookie_store(true)](crate::ClientBuilder::cookie_store) _after_ [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) will result
    /// in the provided `my_cookie_store` being **overridden** with a default implementation.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookie_provider<C: cookie::CookieStore + 'static>(
        mut self,
        cookie_store: Arc<C>,
    ) -> ClientBuilder {
        self.config.cookie_store = Some(cookie_store as _);
        self
    }

    /// Enable auto gzip decompression by checking the `Content-Encoding` response header.
    ///
    /// If auto gzip decompression is turned on:
    ///
    /// - When sending a request and if the request's headers do not already contain
    ///   an `Accept-Encoding` **and** `Range` values, the `Accept-Encoding` header is set to `gzip`.
    ///   The request body is **not** automatically compressed.
    /// - When receiving a response, if its headers contain a `Content-Encoding` value of
    ///   `gzip`, both `Content-Encoding` and `Content-Length` are removed from the
    ///   headers' set. The response body is automatically decompressed.
    ///
    /// If the `gzip` feature is turned on, the default option is enabled.
    ///
    /// # Optional
    ///
    /// This requires the optional `gzip` feature to be enabled
    #[cfg(feature = "gzip")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gzip")))]
    pub fn gzip(mut self, enable: bool) -> ClientBuilder {
        self.config.accepts.gzip = enable;
        self
    }

    /// Enable auto brotli decompression by checking the `Content-Encoding` response header.
    ///
    /// If auto brotli decompression is turned on:
    ///
    /// - When sending a request and if the request's headers do not already contain
    ///   an `Accept-Encoding` **and** `Range` values, the `Accept-Encoding` header is set to `br`.
    ///   The request body is **not** automatically compressed.
    /// - When receiving a response, if its headers contain a `Content-Encoding` value of
    ///   `br`, both `Content-Encoding` and `Content-Length` are removed from the
    ///   headers' set. The response body is automatically decompressed.
    ///
    /// If the `brotli` feature is turned on, the default option is enabled.
    ///
    /// # Optional
    ///
    /// This requires the optional `brotli` feature to be enabled
    #[cfg(feature = "brotli")]
    #[cfg_attr(docsrs, doc(cfg(feature = "brotli")))]
    pub fn brotli(mut self, enable: bool) -> ClientBuilder {
        self.config.accepts.brotli = enable;
        self
    }

    /// Enable auto zstd decompression by checking the `Content-Encoding` response header.
    ///
    /// If auto zstd decompression is turned on:
    ///
    /// - When sending a request and if the request's headers do not already contain
    ///   an `Accept-Encoding` **and** `Range` values, the `Accept-Encoding` header is set to `zstd`.
    ///   The request body is **not** automatically compressed.
    /// - When receiving a response, if its headers contain a `Content-Encoding` value of
    ///   `zstd`, both `Content-Encoding` and `Content-Length` are removed from the
    ///   headers' set. The response body is automatically decompressed.
    ///
    /// If the `zstd` feature is turned on, the default option is enabled.
    ///
    /// # Optional
    ///
    /// This requires the optional `zstd` feature to be enabled
    #[cfg(feature = "zstd")]
    #[cfg_attr(docsrs, doc(cfg(feature = "zstd")))]
    pub fn zstd(mut self, enable: bool) -> ClientBuilder {
        self.config.accepts.zstd = enable;
        self
    }

    /// Enable auto deflate decompression by checking the `Content-Encoding` response header.
    ///
    /// If auto deflate decompression is turned on:
    ///
    /// - When sending a request and if the request's headers do not already contain
    ///   an `Accept-Encoding` **and** `Range` values, the `Accept-Encoding` header is set to `deflate`.
    ///   The request body is **not** automatically compressed.
    /// - When receiving a response, if it's headers contain a `Content-Encoding` value that
    ///   equals to `deflate`, both values `Content-Encoding` and `Content-Length` are removed from the
    ///   headers' set. The response body is automatically decompressed.
    ///
    /// If the `deflate` feature is turned on, the default option is enabled.
    ///
    /// # Optional
    ///
    /// This requires the optional `deflate` feature to be enabled
    #[cfg(feature = "deflate")]
    #[cfg_attr(docsrs, doc(cfg(feature = "deflate")))]
    pub fn deflate(mut self, enable: bool) -> ClientBuilder {
        self.config.accepts.deflate = enable;
        self
    }

    /// Disable auto response body gzip decompression.
    ///
    /// This method exists even if the optional `gzip` feature is not enabled.
    /// This can be used to ensure a `Client` doesn't use gzip decompression
    /// even if another dependency were to enable the optional `gzip` feature.
    pub fn no_gzip(self) -> ClientBuilder {
        #[cfg(feature = "gzip")]
        {
            self.gzip(false)
        }

        #[cfg(not(feature = "gzip"))]
        {
            self
        }
    }

    /// Disable auto response body brotli decompression.
    ///
    /// This method exists even if the optional `brotli` feature is not enabled.
    /// This can be used to ensure a `Client` doesn't use brotli decompression
    /// even if another dependency were to enable the optional `brotli` feature.
    pub fn no_brotli(self) -> ClientBuilder {
        #[cfg(feature = "brotli")]
        {
            self.brotli(false)
        }

        #[cfg(not(feature = "brotli"))]
        {
            self
        }
    }

    /// Disable auto response body zstd decompression.
    ///
    /// This method exists even if the optional `zstd` feature is not enabled.
    /// This can be used to ensure a `Client` doesn't use zstd decompression
    /// even if another dependency were to enable the optional `zstd` feature.
    pub fn no_zstd(self) -> ClientBuilder {
        #[cfg(feature = "zstd")]
        {
            self.zstd(false)
        }

        #[cfg(not(feature = "zstd"))]
        {
            self
        }
    }

    /// Disable auto response body deflate decompression.
    ///
    /// This method exists even if the optional `deflate` feature is not enabled.
    /// This can be used to ensure a `Client` doesn't use deflate decompression
    /// even if another dependency were to enable the optional `deflate` feature.
    pub fn no_deflate(self) -> ClientBuilder {
        #[cfg(feature = "deflate")]
        {
            self.deflate(false)
        }

        #[cfg(not(feature = "deflate"))]
        {
            self
        }
    }

    /// Set a timeout for only the connect phase of a `Client`.
    ///
    /// Default is `None`.
    ///
    /// # Note
    ///
    /// This **requires** the futures be executed in a tokio runtime with
    /// a tokio timer enabled.
    pub fn connect_timeout(mut self, timeout: Duration) -> ClientBuilder {
        self.config.connect_timeout = Some(timeout);
        self
    }

    /// Set a timeout for the entire request.
    ///
    /// The timeout is applied from when the request starts connecting until the
    /// response body has finished.
    ///
    /// Default is no timeout.
    pub fn timeout(mut self, timeout: Duration) -> ClientBuilder {
        self.config.timeout = Some(timeout);
        self
    }

    /// Set the maximum idle connection per host allowed in the pool.
    pub fn pool_max_idle_per_host(mut self, max: usize) -> ClientBuilder {
        self.config.pool_max_idle_per_host = max;
        self
    }

    /// Only use HTTP/3.
    ///
    /// # Optional
    ///
    /// This requires the optional `http3` feature to be enabled.
    #[cfg(feature = "http3")]
    pub fn http3_prior_knowledge(mut self) -> ClientBuilder {
        self.config.http_version_pref = HttpVersionPref::Http3;
        self
    }

    /// Only use HTTP/2.
    ///
    /// # Optional
    ///
    /// This requires the optional `http2` feature to be enabled.
    #[cfg(feature = "http2")]
    pub fn http2_prior_knowledge(mut self) -> ClientBuilder {
        self.config.http_version_pref = HttpVersionPref::Http2;
        self
    }

    /// Set HTTP/3 maximum idle timeout.
    #[cfg(feature = "http3")]
    pub fn http3_max_idle_timeout(mut self, timeout: Duration) -> ClientBuilder {
        self.config.quic_max_idle_timeout = Some(timeout);
        self
    }

    /// Set HTTP/3 stream receive window.
    #[cfg(feature = "http3")]
    pub fn http3_stream_receive_window(mut self, window: VarInt) -> ClientBuilder {
        self.config.quic_stream_receive_window = Some(window);
        self
    }

    /// Set HTTP/3 connection receive window.
    #[cfg(feature = "http3")]
    pub fn http3_conn_receive_window(mut self, window: VarInt) -> ClientBuilder {
        self.config.quic_receive_window = Some(window);
        self
    }

    /// Set HTTP/3 send window.
    #[cfg(feature = "http3")]
    pub fn http3_send_window(mut self, window: u64) -> ClientBuilder {
        self.config.quic_send_window = Some(window);
        self
    }

    /// Enable HTTP/3 BBR congestion control.
    #[cfg(feature = "http3")]
    pub fn http3_congestion_bbr(mut self, enable: bool) -> ClientBuilder {
        self.config.quic_congestion_bbr = enable;
        self
    }

    /// Set HTTP/3 maximum field section size.
    #[cfg(feature = "http3")]
    pub fn http3_max_field_section_size(mut self, size: u64) -> ClientBuilder {
        self.config.h3_max_field_section_size = Some(size);
        self
    }

    /// Enable HTTP/3 GREASE.
    #[cfg(feature = "http3")]
    pub fn http3_send_grease(mut self, enable: bool) -> ClientBuilder {
        self.config.h3_send_grease = Some(enable);
        self
    }

    /// Set HTTP/2 maximum frame size.
    #[cfg(feature = "http2")]
    pub fn http2_max_frame_size(mut self, size: u32) -> ClientBuilder {
        self.config.http2_max_frame_size = Some(size);
        self
    }

    /// Set the pool idle timeout.
    pub fn pool_idle_timeout(mut self, timeout: Option<Duration>) -> ClientBuilder {
        self.config.pool_idle_timeout = timeout;
        self
    }

    /// Enable HTTP/2 adaptive window.
    #[cfg(feature = "http2")]
    pub fn http2_adaptive_window(mut self, enable: bool) -> ClientBuilder {
        self.config.http2_adaptive_window = enable;
        self
    }
}

impl fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("ClientBuilder");
        self.config.fmt_fields(&mut builder);
        builder.finish()
    }
}