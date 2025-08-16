//! H3Client types and constructors
//!
//! Core H3Client struct with connection pool management and cookie store support.

use std::sync::Arc;

#[cfg(feature = "cookies")]
use crate::common::cookie;
use crate::hyper::async_impl::h3_client::connect::H3Connector;
use crate::hyper::async_impl::h3_client::pool::Pool;

/// HTTP/3 client with connection pooling and optional cookie support
#[derive(Clone)]
pub struct H3Client {
    pub(crate) pool: Pool,
    pub(crate) connector: H3Connector,
    #[cfg(feature = "cookies")]
    pub(crate) cookie_store: Option<Arc<dyn cookie::CookieStore>>,
}

impl H3Client {
    /// Create new H3Client without cookie support
    pub fn new() -> Option<Self> {
        let connector = H3Connector::new()?;
        Some(Self {
            pool: Pool::new(),
            connector,
            #[cfg(feature = "cookies")]
            cookie_store: None,
        })
    }

    /// Create new H3Client with cookie store
    #[cfg(feature = "cookies")]
    pub fn with_cookie_store(cookie_store: Arc<dyn cookie::CookieStore>) -> Option<Self> {
        let connector = H3Connector::new()?;
        Some(Self {
            pool: Pool::new(),
            connector,
            cookie_store: Some(cookie_store),
        })
    }

    /// Get reference to connection pool
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Get reference to connector
    pub fn connector(&self) -> &H3Connector {
        &self.connector
    }

    /// Check if cookie support is enabled
    #[cfg(feature = "cookies")]
    pub fn has_cookies(&self) -> bool {
        self.cookie_store.is_some()
    }
}
