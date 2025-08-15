//! Proxy builder API and public interface
//!
//! This module provides the main Proxy struct and its builder methods
//! for configuring HTTP, HTTPS, and custom proxy settings.

use std::sync::Arc;
use http::{header::HeaderValue, HeaderMap, Uri};
use super::types::{Extra, Intercept, Via};
use super::matcher::{NoProxy, Custom};
use super::into_proxy::{IntoProxy, IntoProxySealed};

/// Configuration of a proxy that a `Client` should pass requests to.
///
/// A `Proxy` has a couple pieces to it:
///
/// - a URL of how to talk to the proxy
/// - rules on what `Client` requests should be directed to the proxy
///
/// For instance, let's look at `Proxy::http`:
///
/// ```rust
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let proxy = crate::hyper::Proxy::http("https://secure.example")?;
/// # Ok(())
/// # }
/// ```
///
/// This proxy will intercept all HTTP requests, and make use of the proxy
/// at `https://secure.example`. A request to `http://hyper.rs` will talk
/// to your proxy. A request to `https://hyper.rs` will not.
///
/// Multiple `Proxy` rules can be configured for a `Client`. The `Client` will
/// check each `Proxy` in the order it was added. This could mean that a
/// `Proxy` added first with eager intercept rules, such as `Proxy::all`,
/// would prevent a `Proxy` later in the list from ever working, so take care.
///
/// By enabling the `"socks"` feature it is possible to use a socks proxy:
/// ```rust
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let proxy = crate::hyper::Proxy::http("socks5://192.168.1.1:9000")?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Proxy {
    pub(crate) extra: Extra,
    pub(crate) intercept: ProxyIntercept,
    pub(crate) no_proxy: Option<NoProxy>,
}

#[derive(Clone)]
pub(crate) enum ProxyIntercept {
    Http(crate::Url),
    Https(crate::Url),
    All(crate::Url),
    Custom(Custom),
}

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

    fn new(intercept: ProxyIntercept) -> Proxy {
        Proxy {
            extra: Extra::default(),
            intercept,
            no_proxy: None,
        }
    }

    /// Set the `Proxy-Authorization` header using Basic auth.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .basic_auth("Aladdin", "open sesame");
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn basic_auth(mut self, username: &str, password: &str) -> Proxy {
        self.extra = self.extra.with_auth(encode_basic_auth(username, password));
        self
    }

    /// Set the `Proxy-Authorization` header to a custom value.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .custom_http_auth(http::header::HeaderValue::from_static("Bearer token123"));
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn custom_http_auth(mut self, header_value: HeaderValue) -> Proxy {
        self.extra = self.extra.with_auth(header_value);
        self
    }

    /// Set custom headers to include with proxy requests.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut headers = http::HeaderMap::new();
    /// headers.insert("X-Custom", http::header::HeaderValue::from_static("value"));
    /// 
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .custom_headers(headers);
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn custom_headers(mut self, headers: HeaderMap) -> Proxy {
        self.extra = self.extra.with_headers(headers);
        self
    }

    /// Set a no-proxy rule to bypass the proxy for certain requests.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .no_proxy("localhost,*.internal");
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn no_proxy<S: Into<String>>(mut self, no_proxy: S) -> Proxy {
        self.no_proxy = Some(NoProxy::new(no_proxy.into()));
        self
    }

    /// Get the proxy intercept configuration
    pub(crate) fn intercept(&self) -> &ProxyIntercept {
        &self.intercept
    }

    /// Get the extra configuration (auth, headers)
    pub(crate) fn extra(&self) -> &Extra {
        &self.extra
    }

    /// Get the no-proxy configuration
    pub(crate) fn no_proxy(&self) -> Option<&NoProxy> {
        self.no_proxy.as_ref()
    }
}

impl std::fmt::Debug for Proxy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Proxy")
            .field("intercept", &"<intercept>")
            .field("extra", &self.extra)
            .field("no_proxy", &self.no_proxy)
            .finish()
    }
}

impl std::fmt::Debug for ProxyIntercept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyIntercept::Http(url) => f.debug_tuple("Http").field(url).finish(),
            ProxyIntercept::Https(url) => f.debug_tuple("Https").field(url).finish(),
            ProxyIntercept::All(url) => f.debug_tuple("All").field(url).finish(),
            ProxyIntercept::Custom(_) => f.debug_tuple("Custom").field(&"<function>").finish(),
        }
    }
}

/// Encode basic authentication credentials
fn encode_basic_auth(username: &str, password: &str) -> HeaderValue {
    use base64::Engine;
    let credentials = format!("{}:{}", username, password);
    let encoded = base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes());
    let auth_value = format!("Basic {}", encoded);
    
    HeaderValue::from_str(&auth_value)
        .unwrap_or_else(|_| HeaderValue::from_static("Basic invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_http_creation() {
        let proxy = Proxy::http("http://proxy.example.com:8080").unwrap();
        match proxy.intercept() {
            ProxyIntercept::Http(url) => {
                assert_eq!(url.as_str(), "http://proxy.example.com:8080/");
            }
            _ => panic!("Expected Http intercept"),
        }
    }

    #[test]
    fn test_proxy_basic_auth() {
        let proxy = Proxy::http("http://proxy.example.com:8080")
            .unwrap()
            .basic_auth("user", "pass");
        
        assert!(proxy.extra().auth().is_some());
        let auth_header = proxy.extra().auth().unwrap();
        assert!(auth_header.to_str().unwrap().starts_with("Basic "));
    }

    #[test]
    fn test_proxy_custom_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Custom", HeaderValue::from_static("test"));
        
        let proxy = Proxy::http("http://proxy.example.com:8080")
            .unwrap()
            .custom_headers(headers);
        
        assert!(proxy.extra().headers().is_some());
        assert_eq!(proxy.extra().headers().unwrap().len(), 1);
    }

    #[test]
    fn test_proxy_no_proxy() {
        let proxy = Proxy::http("http://proxy.example.com:8080")
            .unwrap()
            .no_proxy("localhost,*.internal");
        
        assert!(proxy.no_proxy().is_some());
    }

    #[test]
    fn test_encode_basic_auth() {
        let header = encode_basic_auth("user", "pass");
        let expected = "Basic dXNlcjpwYXNz"; // base64 of "user:pass"
        assert_eq!(header.to_str().unwrap(), expected);
    }
}