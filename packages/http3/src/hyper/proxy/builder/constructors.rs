//! Proxy constructor methods
//!
//! Static methods for creating different types of proxy configurations
//! including HTTP, HTTPS, all-traffic, and custom proxy interceptors.

use super::types::{Proxy, ProxyIntercept};
use super::super::into_proxy::{IntoProxy, IntoProxySealed};
use super::super::matcher::Custom;

impl Proxy {
    /// Proxy all HTTP traffic to the passed URL.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::http("http://my.prox")?)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn http<U: IntoProxy>(proxy_scheme: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(ProxyIntercept::Http(proxy_scheme.into_proxy()?)))
    }

    /// Proxy all HTTPS traffic to the passed URL.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::https("https://example.prox:4545")?)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn https<U: IntoProxy>(proxy_scheme: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(ProxyIntercept::Https(proxy_scheme.into_proxy()?)))
    }

    /// Proxy **all** traffic to the passed URL.
    ///
    /// "All" refers to `https` and `http` URLs. Other schemes are not
    /// recognized by http3.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::all("http://pro.xy")?)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn all<U: IntoProxy>(proxy_scheme: U) -> crate::Result<Proxy> {
        Ok(Proxy::new(ProxyIntercept::All(proxy_scheme.into_proxy()?)))
    }

    /// Provide a custom function to determine what traffic to proxy to where.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let target = crate::hyper::Url::parse("https://my.prox")?;
    /// let client = crate::hyper::Client::builder()
    ///     .proxy(crate::hyper::Proxy::custom(move |url| {
    ///         if url.host_str() == Some("hyper.rs") {
    ///             Some(target.clone())
    ///         } else {
    ///             None
    ///         }
    ///     }))
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn custom<F, U: IntoProxy>(fun: F) -> Proxy
    where
        F: Fn(&crate::Url) -> Option<U> + Send + Sync + 'static,
    {
        Proxy::new(ProxyIntercept::Custom(Custom::new(move |url| {
            fun(url).map(IntoProxy::into_proxy)
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_http_creation() {
        let proxy = Proxy::http("http://proxy.example.com:8080")
            .expect("Failed to create HTTP proxy");
        match proxy.intercept() {
            ProxyIntercept::Http(url) => {
                assert_eq!(url.as_str(), "http://proxy.example.com:8080/");
            }
            _ => panic!("Expected Http intercept"),
        }
    }

    #[test]
    fn test_proxy_https_creation() {
        let proxy = Proxy::https("https://proxy.example.com:8080")
            .expect("Failed to create HTTPS proxy");
        match proxy.intercept() {
            ProxyIntercept::Https(url) => {
                assert_eq!(url.as_str(), "https://proxy.example.com:8080/");
            }
            _ => panic!("Expected Https intercept"),
        }
    }

    #[test]
    fn test_proxy_all_creation() {
        let proxy = Proxy::all("http://proxy.example.com:8080")
            .expect("Failed to create All proxy");
        match proxy.intercept() {
            ProxyIntercept::All(url) => {
                assert_eq!(url.as_str(), "http://proxy.example.com:8080/");
            }
            _ => panic!("Expected All intercept"),
        }
    }

    #[test]
    fn test_proxy_custom_creation() {
        let proxy = Proxy::custom(|_url| Some("http://custom.proxy"));
        match proxy.intercept() {
            ProxyIntercept::Custom(_) => {
                // Custom proxy created successfully
            }
            _ => panic!("Expected Custom intercept"),
        }
    }
}