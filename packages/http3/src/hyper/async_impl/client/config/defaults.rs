//! Default configuration values for HTTP client
//!
//! Contains the default implementation for Config struct with all
//! standard configuration values and initialization logic.

use std::collections::HashMap;
use std::time::Duration;

use http::header::{ACCEPT, HeaderMap, HeaderValue};

use super::types::{Config, HttpVersionPref};
use crate::hyper::async_impl::decoder::Accepts;
use crate::hyper::redirect;
#[cfg(feature = "__tls")]
use crate::hyper::tls::TlsBackend;

impl Config {
    /// Create a new Config with default values.
    ///
    /// Sets up sensible defaults for all configuration options including:
    /// - Accept header set to "*/*"
    /// - TLS verification enabled
    /// - Connection pooling with 90-second idle timeout
    /// - TCP keepalive enabled
    /// - System proxy detection enabled
    /// - Referer header enabled
    /// - Built-in root certificates enabled
    /// - All HTTP versions supported
    pub(super) fn default() -> Self {
        let mut headers: HeaderMap<HeaderValue> = HeaderMap::with_capacity(2);
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));

        Config {
            error: None,
            accepts: Accepts::default(),
            headers,
            #[cfg(feature = "__tls")]
            hostname_verification: true,
            #[cfg(feature = "__tls")]
            certs_verification: true,
            #[cfg(feature = "__tls")]
            tls_sni: true,
            connect_timeout: None,
            connection_verbose: false,
            pool_idle_timeout: Some(Duration::from_secs(90)),
            pool_max_idle_per_host: usize::MAX,
            tcp_keepalive: Some(Duration::from_secs(60)),
            tcp_keepalive_interval: None,
            tcp_keepalive_retries: None,
            #[cfg(any(target_os = "android", target_os = "fuchsia", target_os = "linux"))]
            tcp_user_timeout: None,
            proxies: Vec::new(),
            auto_sys_proxy: true,
            redirect_policy: redirect::Policy::default(),
            referer: true,
            read_timeout: None,
            timeout: None,
            #[cfg(feature = "__tls")]
            root_certs: Vec::new(),
            #[cfg(feature = "__tls")]
            tls_built_in_root_certs: true,
            #[cfg(feature = "rustls-tls-webpki-roots-no-provider")]
            tls_built_in_certs_webpki: true,
            #[cfg(feature = "rustls-tls-native-roots-no-provider")]
            tls_built_in_certs_native: true,
            #[cfg(any(feature = "native-tls", feature = "__rustls"))]
            identity: None,
            #[cfg(feature = "__rustls")]
            crls: vec![],
            #[cfg(feature = "__tls")]
            min_tls_version: None,
            #[cfg(feature = "__tls")]
            max_tls_version: None,
            #[cfg(feature = "__tls")]
            tls_info: false,
            #[cfg(feature = "__tls")]
            tls: TlsBackend::default(),
            connector_layers: Vec::new(),
            http_version_pref: HttpVersionPref::All,
            http09_responses: false,
            http1_title_case_headers: false,
            http1_allow_obsolete_multiline_headers_in_responses: false,
            http1_ignore_invalid_headers_in_responses: false,
            http1_allow_spaces_after_header_name_in_responses: false,
            #[cfg(feature = "http2")]
            http2_initial_stream_window_size: None,
            #[cfg(feature = "http2")]
            http2_initial_connection_window_size: None,
            #[cfg(feature = "http2")]
            http2_adaptive_window: false,
            #[cfg(feature = "http2")]
            http2_max_frame_size: None,
            #[cfg(feature = "http2")]
            http2_max_header_list_size: None,
            #[cfg(feature = "http2")]
            http2_keep_alive_interval: None,
            #[cfg(feature = "http2")]
            http2_keep_alive_timeout: None,
            #[cfg(feature = "http2")]
            http2_keep_alive_while_idle: false,
            local_address: None,
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
            interface: None,
            nodelay: true,
            hickory_dns: cfg!(feature = "hickory-dns"),
            #[cfg(feature = "cookies")]
            cookie_store: None,
            https_only: false,
            dns_overrides: HashMap::new(),
            #[cfg(feature = "http3")]
            tls_enable_early_data: false,
            #[cfg(feature = "http3")]
            quic_max_idle_timeout: None,
            #[cfg(feature = "http3")]
            quic_stream_receive_window: None,
            #[cfg(feature = "http3")]
            quic_receive_window: None,
            #[cfg(feature = "http3")]
            quic_send_window: None,
            #[cfg(feature = "http3")]
            quic_congestion_bbr: false,
            #[cfg(feature = "http3")]
            h3_max_field_section_size: None,
            #[cfg(feature = "http3")]
            h3_send_grease: None,
            dns_resolver: None,
        }
    }
}
