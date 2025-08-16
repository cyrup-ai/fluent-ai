//! Proxy constructor methods
//!
//! Static factory methods for creating different types of proxy configurations
//! including all-traffic, HTTP-only, HTTPS-only, and custom proxy routing.

use std::sync::Arc;

use crate::hyper::into_url::{IntoUrl, IntoUrlSealed};
use super::super::url_handling::Custom;
use super::types::{Extra, Intercept, Proxy};
use crate::Url;

impl Proxy {
    /// Proxy **all** traffic to the passed URL.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::all("http://my.prox")?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn all<U: IntoUrlSealed>(proxy_url: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(Intercept::All(proxy_url.into_url()?)))
    }

    /// Proxy **HTTP** traffic to the passed URL.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::http("http://my.prox")?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn http<U: IntoUrlSealed>(proxy_url: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(Intercept::Http(proxy_url.into_url()?)))
    }

    /// Proxy **HTTPS** traffic to the passed URL.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::https("http://my.prox")?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn https<U: IntoUrlSealed>(proxy_url: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(Intercept::Https(proxy_url.into_url()?)))
    }

    /// Provide a custom function to determine what traffic to proxy to where.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let target = "https://my.prox";
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::custom(move |url| {
    ///         if url.host_str() == Some("hyper.rs") {
    ///             target.parse().ok()
    ///         } else {
    ///             None
    ///         }
    ///     }));
    /// # Ok(())
    /// # }
    /// ```
    pub fn custom<F>(f: F) -> Proxy
    where
        F: Fn(&Url) -> Option<Url> + Send + Sync + 'static,
    {
        Proxy::new(Intercept::Custom(Custom {
            func: Arc::new(move |url| Some(Ok(f(url)?))),
            no_proxy: None,
        }))
    }

    /// Internal constructor for creating a new Proxy with the given intercept configuration
    pub(crate) fn new(intercept: Intercept) -> Proxy {
        Proxy {
            extra: Extra {
                auth: None,
                misc: None,
            },
            intercept,
            no_proxy: None,
        }
    }
}
