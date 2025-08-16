//! HTTP protocol version configuration methods for ClientBuilder
//!
//! Contains methods for configuring HTTP/2 and HTTP/3 protocol settings,
//! version preferences, and protocol-specific parameters.

use std::time::Duration;

#[cfg(feature = "http3")]
use quinn::VarInt;

use super::super::config::HttpVersionPref;
use super::types::ClientBuilder;

impl ClientBuilder {
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

    /// Enable HTTP/2 adaptive window.
    #[cfg(feature = "http2")]
    pub fn http2_adaptive_window(mut self, enable: bool) -> ClientBuilder {
        self.config.http2_adaptive_window = enable;
        self
    }
}
