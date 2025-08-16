//! Configuration methods for Http3Builder
//!
//! Provides methods for configuring request behavior including timeouts,
//! retry attempts, and debug logging.

use super::builder_core::Http3Builder;

impl<S> Http3Builder<S> {
    /// Enable debug logging for this request
    ///
    /// When enabled, detailed request and response information will be logged
    /// to help with debugging and development.
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn debug(mut self) -> Self {
        self.debug_enabled = true;
        self
    }

    /// Set request timeout in seconds
    ///
    /// # Arguments  
    /// * `seconds` - Timeout duration in seconds
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .timeout_seconds(30)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        let timeout = std::time::Duration::from_secs(seconds);
        // Store timeout in the request configuration with zero allocation
        self.request = self.request.with_timeout(timeout);
        self
    }

    /// Set retry attempts for failed requests
    ///
    /// # Arguments
    /// * `attempts` - Number of retry attempts (0 disables retries)
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .retry_attempts(3)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn retry_attempts(mut self, attempts: u32) -> Self {
        // Store retry attempts in the request
        self.request = self.request.with_retry_attempts(attempts);
        self
    }
}
