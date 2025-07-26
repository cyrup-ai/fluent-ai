//! Direct ProgressHub integration - no abstractions, just ProgressHub
//!
//! This module directly exposes ProgressHub's Client and types for blazing-fast,
//! zero-allocation model downloads with dual-backend support (XET CAS + QUIC).

use std::path::PathBuf;

// Direct re-exports of ProgressHub - no abstractions
pub use progresshub_client_selector::{Backend, Client, DownloadConfig};
pub use progresshub_config::ConfigTrait as ProgressHubConfig;
pub use progresshub_progress::{
    DownloadProgress, DownloadResult, ProgressData, ProgressHandler, create_progress_channel};

use crate::error::{CandleError, CandleResult};

/// Create a ProgressHub client with specified backend for model downloading
///
/// Creates a new ProgressHub client instance using the specified backend for
/// downloading models from Hugging Face Hub or other model repositories.
/// Supports both XET CAS and QUIC backends for optimal performance.
///
/// # Arguments
///
/// * `backend` - The download backend to use (XET or QUIC)
///
/// # Returns
///
/// * `Ok(Client)` - Successfully created ProgressHub client
/// * `Err(CandleError)` - Client creation failed with error details
///
/// # Backends
///
/// - **XET**: Content-addressable storage with deduplication
/// - **QUIC**: HTTP/3 protocol for improved network performance
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::hub::{create_client, Backend};
///
/// // Create client with XET backend
/// let client = create_client(Backend::Xet)?;
///
/// // Create client with QUIC backend  
/// let client = create_client(Backend::Quic)?;
/// ```
#[inline]
pub fn create_client(backend: Backend) -> CandleResult<Client> {
    Client::new(backend).map_err(|e| CandleError::InitializationError(e.to_string()))
}

/// Create download configuration with specified cache directory
///
/// Creates a DownloadConfig instance configured for optimal model downloading
/// with caching enabled and progress handled via channels rather than console output.
/// The configuration enables local caching for better performance on repeated downloads.
///
/// # Arguments
///
/// * `cache_dir` - Directory path where downloaded models will be cached
///
/// # Returns
///
/// A configured DownloadConfig instance ready for use with ProgressHub
///
/// # Configuration
///
/// - **Caching**: Enabled by default for performance
/// - **Progress**: Disabled for console, handled via progress channels
/// - **Destination**: Set to the provided cache directory
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::hub::create_download_config;
/// use std::path::PathBuf;
///
/// let cache_dir = PathBuf::from("/tmp/model_cache");
/// let config = create_download_config(cache_dir);
/// ```
#[inline]
pub fn create_download_config(cache_dir: PathBuf) -> DownloadConfig {
    DownloadConfig {
        destination: cache_dir,
        show_progress: false, // We handle progress via channels
        use_cache: true,
    }
}

/// Legacy type alias for backward compatibility during ProgressHub transition
///
/// This alias maintains compatibility with existing code that uses DownloadEvent
/// while the underlying implementation transitions to ProgressHub's ProgressData.
/// New code should prefer using ProgressData directly.
pub type DownloadEvent = ProgressData;

/// Configuration for model hub downloading operations
///
/// Provides configuration options for downloading models from Hugging Face Hub
/// and other model repositories using ProgressHub. Includes backend selection,
/// caching options, and progress reporting preferences.
///
/// # Backends
///
/// - **XET**: Content-addressable storage with efficient deduplication
/// - **QUIC**: HTTP/3 protocol for improved network performance and reliability
///
/// # Caching
///
/// Supports local caching of downloaded models to avoid redundant downloads
/// and improve startup performance for repeated model loading.
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Backend to use for downloads (XET for deduplication or QUIC for performance)
    pub backend: Backend,
    /// Cache directory path where downloaded models will be stored
    pub cache_dir: PathBuf,
    /// Whether to show download progress indicators in console output
    pub show_progress: bool,
    /// Whether to use local caching to avoid redundant downloads
    pub use_cache: bool,
}

impl Default for HubConfig {
    /// Create default hub configuration with sensible defaults
    ///
    /// Creates a HubConfig with optimal defaults for most use cases:
    /// - XET backend for efficient content-addressable storage
    /// - Standard Hugging Face cache directory (following HF conventions)
    /// - Progress reporting disabled (handled via channels)
    /// - Local caching enabled for performance
    ///
    /// # Cache Directory Resolution
    ///
    /// The cache directory is determined in this order:
    /// 1. `HF_HOME` environment variable if set
    /// 2. `$HOME/.cache/huggingface` if HOME is available
    /// 3. `/tmp/huggingface` as fallback
    ///
    /// # Returns
    ///
    /// A HubConfig instance with production-ready default settings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::hub::HubConfig;
    ///
    /// let config = HubConfig::default();
    /// assert_eq!(config.use_cache, true);
    /// assert_eq!(config.show_progress, false);
    /// ```
    fn default() -> Self {
        Self {
            backend: Backend::Xet, // Default to Xet backend for deduplication
            cache_dir: std::env::var("HF_HOME")
                .or_else(|_| {
                    std::env::var("HOME").map(|home| format!("{}/.cache/huggingface", home))
                })
                .unwrap_or_else(|_| "/tmp/huggingface".to_string())
                .into(),
            show_progress: false, // Progress handled via channels
            use_cache: true,      // Enable caching for performance
        }
    }
}
