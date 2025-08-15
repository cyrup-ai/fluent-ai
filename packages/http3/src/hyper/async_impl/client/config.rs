use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use http::header::{HeaderMap, HeaderValue, ACCEPT, USER_AGENT};
use http::Method;

#[cfg(feature = "http3")]
use quinn::VarInt;

use crate::hyper::async_impl::decoder::Accepts;
use crate::hyper::redirect;
use crate::hyper::{IntoUrl, Proxy};
use crate::hyper::proxy::Matcher as ProxyMatcher;

#[cfg(feature = "__tls")]
use crate::hyper::Certificate;
#[cfg(any(feature = "native-tls", feature = "__rustls"))]
use crate::hyper::Identity;
#[cfg(feature = "__rustls")]
use crate::hyper::tls::CertificateRevocationList;
#[cfg(feature = "__tls")]
use crate::hyper::tls::{self, TlsBackend};
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
use crate::hyper::dns::{DynResolver, Resolve};

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

pub enum HttpVersionPref {
    Http1,
    #[cfg(feature = "http2")]
    Http2,
    #[cfg(feature = "http3")]
    Http3,
    All,
}

impl Config {
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

    pub(super) fn fmt_fields(&self, f: &mut fmt::DebugStruct<'_, '_>) {
        // Instead of deriving Debug, only print fields when their output
        // would provide relevant or interesting data.

        #[cfg(feature = "cookies")]
        {
            if let Some(_) = self.cookie_store {
                f.field("cookie_store", &true);
            }
        }

        f.field("accepts", &self.accepts);

        if !self.proxies.is_empty() {
            f.field("proxies", &self.proxies);
        }

        if !self.redirect_policy.is_default() {
            f.field("redirect_policy", &self.redirect_policy);
        }

        if self.referer {
            f.field("referer", &true);
        }

        f.field("default_headers", &self.headers);

        if self.http1_title_case_headers {
            f.field("http1_title_case_headers", &true);
        }

        if self.http1_allow_obsolete_multiline_headers_in_responses {
            f.field("http1_allow_obsolete_multiline_headers_in_responses", &true);
        }

        if self.http1_ignore_invalid_headers_in_responses {
            f.field("http1_ignore_invalid_headers_in_responses", &true);
        }

        if self.http1_allow_spaces_after_header_name_in_responses {
            f.field("http1_allow_spaces_after_header_name_in_responses", &true);
        }

        if matches!(self.http_version_pref, HttpVersionPref::Http1) {
            f.field("http1_only", &true);
        }

        #[cfg(feature = "http2")]
        if matches!(self.http_version_pref, HttpVersionPref::Http2) {
            f.field("http2_prior_knowledge", &true);
        }

        if let Some(ref d) = self.connect_timeout {
            f.field("connect_timeout", d);
        }

        if let Some(ref d) = self.timeout {
            f.field("timeout", d);
        }

        if let Some(ref v) = self.local_address {
            f.field("local_address", v);
        }

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
        if let Some(ref v) = self.interface {
            f.field("interface", v);
        }

        if self.nodelay {
            f.field("tcp_nodelay", &true);
        }

        #[cfg(feature = "__tls")]
        {
            if !self.hostname_verification {
                f.field("danger_accept_invalid_hostnames", &true);
            }
        }

        #[cfg(feature = "__tls")]
        {
            if !self.certs_verification {
                f.field("danger_accept_invalid_certs", &true);
            }

            if let Some(ref min_tls_version) = self.min_tls_version {
                f.field("min_tls_version", min_tls_version);
            }

            if let Some(ref max_tls_version) = self.max_tls_version {
                f.field("max_tls_version", max_tls_version);
            }

            f.field("tls_sni", &self.tls_sni);

            f.field("tls_info", &self.tls_info);
        }

        #[cfg(all(feature = "default-tls", feature = "__rustls"))]
        {
            f.field("tls_backend", &self.tls);
        }

        if !self.dns_overrides.is_empty() {
            f.field("dns_overrides", &self.dns_overrides);
        }

        #[cfg(feature = "http3")]
        {
            if self.tls_enable_early_data {
                f.field("tls_enable_early_data", &true);
            }
        }
    }
}