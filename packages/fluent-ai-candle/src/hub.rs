//! Direct ProgressHub integration - no abstractions, just ProgressHub
//!
//! This module directly exposes ProgressHub's Client and types for blazing-fast,
//! zero-allocation model downloads with dual-backend support (XET CAS + QUIC).

use std::path::PathBuf;

// Direct re-exports of ProgressHub - no abstractions
pub use progresshub_client_selector::{Backend, Client, DownloadConfig};
pub use progresshub_config::ConfigTrait as ProgressHubConfig;
pub use progresshub_progress::{
    DownloadProgress, DownloadResult, ProgressData, ProgressHandler, create_progress_channel,
};

use crate::error::{CandleError, CandleResult};

/// Create ProgressHub client with cache directory
///
/// This is a thin convenience wrapper around ProgressHub's Client::new()
#[inline]
pub fn create_client(backend: Backend) -> CandleResult<Client> {
    Client::new(backend).map_err(|e| CandleError::InitializationError(e.to_string()))
}

/// Create download config with cache directory
///
/// This is a thin convenience wrapper around ProgressHub's DownloadConfig
#[inline]
pub fn create_download_config(cache_dir: PathBuf) -> DownloadConfig {
    DownloadConfig {
        destination: cache_dir,
        show_progress: false, // We handle progress via channels
        use_cache: true,
    }
}

/// Legacy type alias for backward compatibility during transition
pub type DownloadEvent = ProgressData;

/// Hub configuration for model downloading
///
/// This wraps ProgressHub configuration for backward compatibility
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Backend to use for downloads (XET or QUIC)
    pub backend: Backend,
    /// Cache directory for downloaded models
    pub cache_dir: PathBuf,
    /// Whether to show download progress
    pub show_progress: bool,
    /// Whether to use local cache
    pub use_cache: bool,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Xet, // Default to Xet backend
            cache_dir: std::env::var("HF_HOME")
                .or_else(|_| {
                    std::env::var("HOME").map(|home| format!("{}/.cache/huggingface", home))
                })
                .unwrap_or_else(|_| "/tmp/huggingface".to_string())
                .into(),
            show_progress: false,
            use_cache: true,
        }
    }
}
