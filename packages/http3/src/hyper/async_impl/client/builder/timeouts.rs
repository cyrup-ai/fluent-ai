//! Timeout and connection configuration methods for ClientBuilder
//!
//! Contains methods for configuring various timeout settings,
//! connection pooling, and related network parameters.

use std::time::Duration;

use super::types::ClientBuilder;

impl ClientBuilder {
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

    /// Set the pool idle timeout.
    pub fn pool_idle_timeout(mut self, timeout: Option<Duration>) -> ClientBuilder {
        self.config.pool_idle_timeout = timeout;
        self
    }
}
