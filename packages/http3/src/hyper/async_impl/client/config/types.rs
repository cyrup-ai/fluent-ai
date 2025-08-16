//! Core configuration types for HTTP client
//!
//! Contains the main Config struct and related type definitions
//! for HTTP client configuration management.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use http::header::{HeaderMap, HeaderValue};
#[cfg(feature = "http3")]
use quinn::VarInt;

#[cfg(feature = "__tls")]
use crate::hyper::Certificate;
#[cfg(any(feature = "native-tls", feature = "__rustls"))]
use crate::hyper::Identity;
use crate::hyper::async_impl::decoder::Accepts;
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
use crate::hyper::dns::Resolve;
use crate::hyper::proxy::Matcher as ProxyMatcher;
use crate::hyper::redirect;
#[cfg(feature = "__rustls")]
use crate::hyper::tls::CertificateRevocationList;
#[cfg(feature = "__tls")]
use crate::hyper::tls::{self, TlsBackend};

/// HTTP client configuration structure containing all client settings.
///
/// This structure holds all configuration options for the HTTP client including
/// TLS settings, connection parameters, HTTP version preferences, timeouts,
/// proxy settings, and protocol-specific options.
pub struct Config {
    // NOTE: When adding a new field, update `fmt::Debug for ClientBuilder`
    pub(super) accepts: Accepts,
    pub(super) headers: HeaderMap,
    #[cfg(feature = "__tls")]
    pub(super) hostname_verification: bool,
    #[cfg(feature = "__tls")]
    pub(super) certs_verification: bool,
    #[cfg(feature = "__tls")]
    pub(super) tls_sni: bool,
    pub(super) connect_timeout: Option<Duration>,
    pub(super) connection_verbose: bool,
    pub(super) pool_idle_timeout: Option<Duration>,
    pub(super) pool_max_idle_per_host: usize,
    pub(super) tcp_keepalive: Option<Duration>,
    pub(super) tcp_keepalive_interval: Option<Duration>,
    pub(super) tcp_keepalive_retries: Option<u32>,
    #[cfg(any(target_os = "android", target_os = "fuchsia", target_os = "linux"))]
    pub(super) tcp_user_timeout: Option<Duration>,
    #[cfg(any(feature = "native-tls", feature = "__rustls"))]
    pub(super) identity: Option<Identity>,
    pub(super) proxies: Vec<ProxyMatcher>,
    pub(super) auto_sys_proxy: bool,
    pub(super) redirect_policy: redirect::Policy,
    pub(super) referer: bool,
    pub(super) read_timeout: Option<Duration>,
    pub(super) timeout: Option<Duration>,
    #[cfg(feature = "__tls")]
    pub(super) root_certs: Vec<Certificate>,
    #[cfg(feature = "__tls")]
    pub(super) tls_built_in_root_certs: bool,
    #[cfg(feature = "rustls-tls-webpki-roots-no-provider")]
    pub(super) tls_built_in_certs_webpki: bool,
    #[cfg(feature = "rustls-tls-native-roots-no-provider")]
    pub(super) tls_built_in_certs_native: bool,
    #[cfg(feature = "__rustls")]
    pub(super) crls: Vec<CertificateRevocationList>,
    #[cfg(feature = "__tls")]
    pub(super) min_tls_version: Option<tls::Version>,
    #[cfg(feature = "__tls")]
    pub(super) max_tls_version: Option<tls::Version>,
    #[cfg(feature = "__tls")]
    pub(super) tls_info: bool,
    #[cfg(feature = "__tls")]
    pub(super) tls: TlsBackend,
    pub(super) connector_layers: Vec<crate::hyper::connect::BoxedConnectorLayer>,
    pub(super) http_version_pref: HttpVersionPref,
    pub(super) http09_responses: bool,
    pub(super) http1_title_case_headers: bool,
    pub(super) http1_allow_obsolete_multiline_headers_in_responses: bool,
    pub(super) http1_ignore_invalid_headers_in_responses: bool,
    pub(super) http1_allow_spaces_after_header_name_in_responses: bool,
    #[cfg(feature = "http2")]
    pub(super) http2_initial_stream_window_size: Option<u32>,
    #[cfg(feature = "http2")]
    pub(super) http2_initial_connection_window_size: Option<u32>,
    #[cfg(feature = "http2")]
    pub(super) http2_adaptive_window: bool,
    #[cfg(feature = "http2")]
    pub(super) http2_max_frame_size: Option<u32>,
    #[cfg(feature = "http2")]
    pub(super) http2_max_header_list_size: Option<u32>,
    #[cfg(feature = "http2")]
    pub(super) http2_keep_alive_interval: Option<Duration>,
    #[cfg(feature = "http2")]
    pub(super) http2_keep_alive_timeout: Option<Duration>,
    #[cfg(feature = "http2")]
    pub(super) http2_keep_alive_while_idle: bool,
    pub(super) local_address: Option<IpAddr>,
    #[cfg(any(
        target_os = "android",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "solaris",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos",
    ))]
    pub(super) interface: Option<String>,
    pub(super) nodelay: bool,
    #[cfg(feature = "cookies")]
    pub(super) cookie_store: Option<Arc<dyn cookie::CookieStore>>,
    pub(super) hickory_dns: bool,
    pub(super) error: Option<crate::Error>,
    pub(super) https_only: bool,
    #[cfg(feature = "http3")]
    pub(super) tls_enable_early_data: bool,
    #[cfg(feature = "http3")]
    pub(super) quic_max_idle_timeout: Option<Duration>,
    #[cfg(feature = "http3")]
    pub(super) quic_stream_receive_window: Option<VarInt>,
    #[cfg(feature = "http3")]
    pub(super) quic_receive_window: Option<VarInt>,
    #[cfg(feature = "http3")]
    pub(super) quic_send_window: Option<u64>,
    #[cfg(feature = "http3")]
    pub(super) quic_congestion_bbr: bool,
    #[cfg(feature = "http3")]
    pub(super) h3_max_field_section_size: Option<u64>,
    #[cfg(feature = "http3")]
    pub(super) h3_send_grease: Option<bool>,
    pub(super) dns_overrides: HashMap<String, Vec<SocketAddr>>,
    pub(super) dns_resolver: Option<Arc<dyn Resolve>>,
}

/// HTTP version preference for client connections.
///
/// Determines which HTTP protocol version the client should prefer
/// when establishing connections to servers.
pub enum HttpVersionPref {
    /// Use HTTP/1.1 only
    Http1,
    /// Use HTTP/2 with prior knowledge
    #[cfg(feature = "http2")]
    Http2,
    /// Use HTTP/3 over QUIC
    #[cfg(feature = "http3")]
    Http3,
    /// Allow all supported HTTP versions (auto-negotiation)
    All,
}
