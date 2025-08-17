//! HTTP protocol strategy pattern implementation
//!
//! Provides strategy enumeration for protocol selection with automatic fallback
//! and protocol-specific configuration management.

use std::time::Duration;

use crate::config::HttpConfig;
use crate::protocols::core::{HttpVersion, ProtocolConfig, TimeoutConfig};

/// Protocol selection strategy with fallback support
#[derive(Debug, Clone)]
pub enum HttpProtocolStrategy {
    /// Force HTTP/2 with specific configuration
    Http2(H2Config),
    /// Force HTTP/3 with specific configuration  
    Http3(H3Config),
    /// Force QUIC with Quiche implementation
    Quiche(QuicheConfig),
    /// Automatic selection with preference ordering
    Auto {
        prefer: Vec<HttpVersion>,
        fallback_chain: Vec<HttpVersion>,
        configs: ProtocolConfigs,
    },
}

impl Default for HttpProtocolStrategy {
    fn default() -> Self {
        Self::Auto {
            prefer: vec![HttpVersion::Http3, HttpVersion::Http2],
            fallback_chain: vec![HttpVersion::Http3, HttpVersion::Http2, HttpVersion::Http11],
            configs: ProtocolConfigs::default(),
        }
    }
}

impl HttpProtocolStrategy {
    /// Create AI-optimized strategy for streaming workloads
    pub fn ai_optimized() -> Self {
        Self::Auto {
            prefer: vec![HttpVersion::Http3],
            fallback_chain: vec![HttpVersion::Http3, HttpVersion::Http2],
            configs: ProtocolConfigs {
                h2: H2Config::ai_optimized(),
                h3: H3Config::ai_optimized(),
                quiche: QuicheConfig::ai_optimized(),
            },
        }
    }
    
    /// Create streaming-optimized strategy for real-time data
    pub fn streaming_optimized() -> Self {
        Self::Http3(H3Config::streaming_optimized())
    }
    
    /// Create low-latency strategy for interactive applications
    pub fn low_latency() -> Self {
        Self::Quiche(QuicheConfig::low_latency())
    }
}

/// Configuration bundle for all protocols
#[derive(Debug, Clone)]
pub struct ProtocolConfigs {
    pub h2: H2Config,
    pub h3: H3Config,
    pub quiche: QuicheConfig,
}

impl Default for ProtocolConfigs {
    fn default() -> Self {
        Self {
            h2: H2Config::default(),
            h3: H3Config::default(),
            quiche: QuicheConfig::default(),
        }
    }
}

/// HTTP/2 protocol configuration
#[derive(Debug, Clone)]
pub struct H2Config {
    pub max_concurrent_streams: u32,
    pub initial_window_size: u32,
    pub max_frame_size: u32,
    pub enable_push: bool,
    pub enable_connect_protocol: bool,
    pub keepalive_interval: Option<Duration>,
    pub keepalive_timeout: Duration,
    pub adaptive_window: bool,
    pub max_send_buffer_size: usize,
}

impl Default for H2Config {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 100,
            initial_window_size: 65535,
            max_frame_size: 16384,
            enable_push: false,
            enable_connect_protocol: true,
            keepalive_interval: Some(Duration::from_secs(30)),
            keepalive_timeout: Duration::from_secs(10),
            adaptive_window: true,
            max_send_buffer_size: 1024 * 1024,
        }
    }
}

impl H2Config {
    pub fn ai_optimized() -> Self {
        Self {
            max_concurrent_streams: 1000,
            initial_window_size: 1048576, // 1MB
            max_frame_size: 32768,
            enable_push: false,
            enable_connect_protocol: true,
            keepalive_interval: Some(Duration::from_secs(15)),
            keepalive_timeout: Duration::from_secs(5),
            adaptive_window: true,
            max_send_buffer_size: 4 * 1024 * 1024, // 4MB
        }
    }
}

impl ProtocolConfig for H2Config {
    fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_streams == 0 {
            return Err("max_concurrent_streams must be greater than 0".to_string());
        }
        if self.initial_window_size < 65535 {
            return Err("initial_window_size must be at least 65535".to_string());
        }
        if self.max_frame_size < 16384 || self.max_frame_size > 16777215 {
            return Err("max_frame_size must be between 16384 and 16777215".to_string());
        }
        Ok(())
    }
    
    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
            keepalive_timeout: Some(self.keepalive_timeout),
        }
    }
    
    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// HTTP/3 protocol configuration
#[derive(Debug, Clone)]
pub struct H3Config {
    pub max_idle_timeout: Duration,
    pub max_udp_payload_size: u16,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub enable_early_data: bool,
    pub enable_0rtt: bool,
    pub congestion_control: CongestionControl,
}

impl Default for H3Config {
    fn default() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(30),
            max_udp_payload_size: 1452,
            initial_max_data: 10485760, // 10MB
            initial_max_stream_data_bidi_local: 1048576, // 1MB
            initial_max_stream_data_bidi_remote: 1048576, // 1MB
            initial_max_stream_data_uni: 1048576, // 1MB
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Cubic,
        }
    }
}

impl H3Config {
    pub fn ai_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(60),
            max_udp_payload_size: 1452,
            initial_max_data: 104857600, // 100MB
            initial_max_stream_data_bidi_local: 10485760, // 10MB
            initial_max_stream_data_bidi_remote: 10485760, // 10MB
            initial_max_stream_data_uni: 10485760, // 10MB
            initial_max_streams_bidi: 1000,
            initial_max_streams_uni: 1000,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Bbr,
        }
    }
    
    pub fn streaming_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(300),
            max_udp_payload_size: 1452,
            initial_max_data: 1073741824, // 1GB
            initial_max_stream_data_bidi_local: 104857600, // 100MB
            initial_max_stream_data_bidi_remote: 104857600, // 100MB
            initial_max_stream_data_uni: 104857600, // 100MB
            initial_max_streams_bidi: 10000,
            initial_max_streams_uni: 10000,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Bbr,
        }
    }
}

impl ProtocolConfig for H3Config {
    fn validate(&self) -> Result<(), String> {
        if self.max_idle_timeout.as_secs() == 0 {
            return Err("max_idle_timeout must be greater than 0".to_string());
        }
        if self.max_udp_payload_size < 1200 {
            return Err("max_udp_payload_size must be at least 1200".to_string());
        }
        if self.initial_max_data == 0 {
            return Err("initial_max_data must be greater than 0".to_string());
        }
        Ok(())
    }
    
    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(5),
            idle_timeout: self.max_idle_timeout,
            keepalive_timeout: Some(self.max_idle_timeout / 2),
        }
    }
    
    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// Quiche QUIC configuration
#[derive(Debug, Clone)]
pub struct QuicheConfig {
    pub max_idle_timeout: Duration,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub max_udp_payload_size: u16,
    pub enable_early_data: bool,
    pub enable_hystart: bool,
    pub congestion_control: CongestionControl,
    pub max_connection_window: u64,
    pub max_stream_window: u64,
}

impl Default for QuicheConfig {
    fn default() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(30),
            initial_max_data: 10485760, // 10MB
            initial_max_stream_data_bidi_local: 1048576, // 1MB
            initial_max_stream_data_bidi_remote: 1048576, // 1MB
            initial_max_stream_data_uni: 1048576, // 1MB
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            max_udp_payload_size: 1452,
            enable_early_data: true,
            enable_hystart: true,
            congestion_control: CongestionControl::Cubic,
            max_connection_window: 25165824, // 24MB
            max_stream_window: 16777216, // 16MB
        }
    }
}

impl QuicheConfig {
    pub fn ai_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(60),
            initial_max_data: 104857600, // 100MB
            initial_max_stream_data_bidi_local: 10485760, // 10MB
            initial_max_stream_data_bidi_remote: 10485760, // 10MB
            initial_max_stream_data_uni: 10485760, // 10MB
            initial_max_streams_bidi: 1000,
            initial_max_streams_uni: 1000,
            max_udp_payload_size: 1452,
            enable_early_data: true,
            enable_hystart: true,
            congestion_control: CongestionControl::Bbr,
            max_connection_window: 268435456, // 256MB
            max_stream_window: 134217728, // 128MB
        }
    }
    
    pub fn low_latency() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(15),
            initial_max_data: 52428800, // 50MB
            initial_max_stream_data_bidi_local: 5242880, // 5MB
            initial_max_stream_data_bidi_remote: 5242880, // 5MB
            initial_max_stream_data_uni: 5242880, // 5MB
            initial_max_streams_bidi: 500,
            initial_max_streams_uni: 500,
            max_udp_payload_size: 1200, // Conservative for low latency
            enable_early_data: true,
            enable_hystart: false, // Disable for predictable latency
            congestion_control: CongestionControl::Bbr,
            max_connection_window: 67108864, // 64MB
            max_stream_window: 33554432, // 32MB
        }
    }
}

impl ProtocolConfig for QuicheConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_idle_timeout.as_secs() == 0 {
            return Err("max_idle_timeout must be greater than 0".to_string());
        }
        if self.initial_max_data == 0 {
            return Err("initial_max_data must be greater than 0".to_string());
        }
        if self.max_udp_payload_size < 1200 {
            return Err("max_udp_payload_size must be at least 1200".to_string());
        }
        Ok(())
    }
    
    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            idle_timeout: self.max_idle_timeout,
            keepalive_timeout: Some(self.max_idle_timeout / 3),
        }
    }
    
    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// Congestion control algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionControl {
    Reno,
    Cubic,
    Bbr,
    BbrV2,
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self::Cubic
    }
}
