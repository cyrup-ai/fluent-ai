//! Fluent API extensions and utilities
//!
//! Provides additional fluent interface components including download
//! functionality and progress tracking for enhanced user experience.

use futures_util::StreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use crate::{DownloadStream, HttpError};

/// Builder for download-specific operations
///
/// Provides a fluent interface for configuring and executing file downloads
/// with progress tracking and error handling.
pub struct DownloadBuilder {
    stream: DownloadStream,
}

impl DownloadBuilder {
    /// Create a new download builder with the provided stream
    ///
    /// # Arguments
    /// * `stream` - Download stream from HTTP client
    pub(crate) fn new(stream: DownloadStream) -> Self {
        Self { stream }
    }

    /// Save the downloaded file to a local path
    ///
    /// Downloads the file from the stream and saves it to the specified
    /// local filesystem path with progress tracking.
    ///
    /// # Arguments
    /// * `local_path` - Local filesystem path where the file should be saved
    ///
    /// # Returns
    /// `Result<DownloadProgress, HttpError>` - Download progress on success
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// # tokio_test::block_on(async {
    /// let progress = Http3Builder::new(&client)
    ///     .download_file("https://example.com/large-file.zip")
    ///     .save("/tmp/downloaded-file.zip")
    ///     .await?;
    ///
    /// println!("Downloaded {} bytes to {}",
    ///     progress.bytes_written,
    ///     progress.local_path);
    ///
    /// if let Some(percentage) = progress.progress_percentage() {
    ///     println!("Download completed: {:.1}%", percentage);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn save(self, local_path: &str) -> Result<DownloadProgress, HttpError> {
        let mut file = File::create(local_path)
            .await
            .map_err::<HttpError, _>(|e| e.into())?;
        let mut stream = self.stream;
        let mut total_written = 0;
        let mut chunk_count = 0;
        let mut total_size = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(download_chunk) => {
                    total_size = download_chunk.total_size;
                    let bytes_written = file
                        .write(&download_chunk.data)
                        .await
                        .map_err::<HttpError, _>(|e| e.into())?;
                    total_written += bytes_written as u64;
                    chunk_count += 1;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(DownloadProgress {
            chunk_count,
            bytes_written: total_written,
            total_size,
            local_path: local_path.to_string(),
            is_complete: true,
        })
    }

    /// Set a custom destination path (alternative to save)
    ///
    /// Provides a fluent alternative to the save method for setting
    /// the download destination.
    ///
    /// # Arguments
    /// * `path` - Local filesystem path for the download
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// # tokio_test::block_on(async {
    /// let progress = Http3Builder::new(&client)
    ///     .download_file("https://example.com/file.zip")
    ///     .destination("/downloads/file.zip")
    ///     .await?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn destination(self, path: &str) -> Result<DownloadProgress, HttpError> {
        self.save(path).await
    }

    /// Start the download with progress monitoring
    ///
    /// Alias for save() that emphasizes the streaming nature of the download.
    ///
    /// # Arguments
    /// * `local_path` - Local filesystem path for the download
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// # tokio_test::block_on(async {
    /// let progress = Http3Builder::new(&client)
    ///     .download_file("https://example.com/file.zip")
    ///     .start("/downloads/file.zip")
    ///     .await?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn start(self, local_path: &str) -> Result<DownloadProgress, HttpError> {
        self.save(local_path).await
    }
}

/// Download progress information for saved files
///
/// Contains detailed information about a completed or in-progress download
/// including byte counts, progress percentage, and completion status.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Number of chunks received during download
    pub chunk_count: u32,
    /// Total bytes written to local file
    pub bytes_written: u64,
    /// Total expected file size if known from headers
    pub total_size: Option<u64>,
    /// Local filesystem path where file was saved
    pub local_path: String,
    /// Whether the download completed successfully
    pub is_complete: bool,
}

impl DownloadProgress {
    /// Calculate progress percentage if total size is known
    ///
    /// Returns the download progress as a percentage (0.0 to 100.0)
    /// if the total file size was provided in HTTP headers.
    ///
    /// # Returns
    /// `Option<f64>` - Progress percentage, or None if total size unknown
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::builder::fluent::DownloadProgress;
    ///
    /// let progress = DownloadProgress {
    ///     chunk_count: 42,
    ///     bytes_written: 1024000,
    ///     total_size: Some(2048000),
    ///     local_path: "/tmp/file.zip".to_string(),
    ///     is_complete: false,
    /// };
    ///
    /// if let Some(percentage) = progress.progress_percentage() {
    ///     println!("Download progress: {:.1}%", percentage);
    /// }
    /// ```
    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_written as f64 / total as f64) * 100.0
            }
        })
    }

    /// Check if download is complete
    ///
    /// Returns true if the download has finished successfully.
    ///
    /// # Examples
    /// ```no_run
    /// # use fluent_ai_http3::builder::fluent::DownloadProgress;
    /// # let progress = DownloadProgress {
    /// #     chunk_count: 42,
    /// #     bytes_written: 2048000,
    /// #     total_size: Some(2048000),
    /// #     local_path: "/tmp/file.zip".to_string(),
    /// #     is_complete: true,
    /// # };
    /// if progress.is_finished() {
    ///     println!("Download completed: {}", progress.local_path);
    /// }
    /// ```
    pub fn is_finished(&self) -> bool {
        self.is_complete
    }

    /// Get a human-readable status string
    ///
    /// Returns a formatted string describing the current download status.
    ///
    /// # Examples
    /// ```no_run
    /// # use fluent_ai_http3::builder::fluent::DownloadProgress;
    /// # let progress = DownloadProgress {
    /// #     chunk_count: 42,
    /// #     bytes_written: 1024000,
    /// #     total_size: Some(2048000),
    /// #     local_path: "/tmp/file.zip".to_string(),
    /// #     is_complete: false,
    /// # };
    /// println!("Status: {}", progress.status_string());
    /// ```
    pub fn status_string(&self) -> String {
        if self.is_complete {
            format!(
                "Completed: {} bytes saved to {}",
                self.bytes_written, self.local_path
            )
        } else if let Some(percentage) = self.progress_percentage() {
            format!(
                "In progress: {:.1}% ({} / {} bytes)",
                percentage,
                self.bytes_written,
                self.total_size.unwrap_or(0)
            )
        } else {
            format!("In progress: {} bytes downloaded", self.bytes_written)
        }
    }
}
