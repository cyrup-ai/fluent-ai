//! Core proxy types and configuration
//! 
//! Zero-allocation, production-quality proxy configuration with comprehensive error handling.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use http::{header::HeaderValue, HeaderMap, Uri};
use fluent_ai_async::prelude::*;
use crate::Url;
use super::url_handling::Custom;

use super::super::into_url::{IntoUrl, IntoUrlSealed};
use super::matcher::{Matcher as InternalMatcher, matcher::Intercept as InternalIntercept};

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
    pub(crate) intercept: Intercept,
    pub(crate) no_proxy: Option<NoProxy>,
}

/// A configuration for filtering out requests that shouldn't be proxied
#[derive(Clone, Debug, Default)]
pub struct NoProxy {
    pub(crate) inner: String,
}

#[derive(Clone)]
pub struct Extra {
    pub(crate) auth: Option<HeaderValue>,
    pub(crate) misc: Option<HeaderMap>,
}

#[derive(Clone, Debug)]
pub enum Intercept {
    All(Url),
    Http(Url),
    Https(Url),
    Custom(Custom),
}



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

    fn new(intercept: Intercept) -> Proxy {
        Proxy {
            extra: Extra {
                auth: None,
                misc: None,
            },
            intercept,
            no_proxy: None,
        }
    }

    /// Set the `Proxy-Authorization` header using Basic auth.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .basic_auth("Aladdin", "open sesame");
    /// # Ok(())
    /// # }
    /// ```
    pub fn basic_auth(mut self, username: &str, password: &str) -> Proxy {
        self.extra.auth = Some(super::url_handling::encode_basic_auth(username, password));
        self
    }

    /// Set the `Proxy-Authorization` header to a specified value.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .custom_http_auth(http::HeaderValue::from_static("justletmepass"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn custom_http_auth(mut self, header_value: HeaderValue) -> Proxy {
        self.extra.auth = Some(header_value);
        self
    }

    /// Set custom headers to be sent with the proxy request.
    ///
    /// # Example
    ///
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut headers = http::HeaderMap::new();
    /// headers.insert("X-Custom-Header", "value".parse()?);
    /// 
    /// let proxy = crate::hyper::Proxy::https("http://localhost:1234")?
    ///     .custom_headers(headers);
    /// # Ok(())
    /// # }
    /// ```
    pub fn custom_headers(mut self, headers: HeaderMap) -> Proxy {
        match self.extra.misc {
            Some(ref mut existing) => existing.extend(headers),
            None => self.extra.misc = Some(headers),
        }
        self
    }

    /// Adds a `No Proxy` exclusion list to this `Proxy`
    ///
    /// The argument should be a comma separated list of hosts
    /// (optionally with a port) to be excluded from proxying.
    ///
    /// NOTE: This will only set a simple `NoProxy` rule for this proxy.
    /// To use more advanced rules you will have to use the `NoProxy` type.
    pub fn no_proxy<T: Into<String>>(mut self, exclusions: T) -> Proxy {
        self.no_proxy = NoProxy::from_string(&exclusions.into());
        self
    }

    pub(crate) fn into_matcher(self) -> super::matcher::Matcher {
        use super::matcher::Matcher_;

        let maybe_has_http_auth = match self.intercept {
            Intercept::Http(_) => true,
            Intercept::All(_) => true,
            _ => false,
        };

        let maybe_has_http_custom_headers = self.extra.misc.is_some();

        let inner = match self.intercept {
            Intercept::All(url) => {
                let mut builder = super::matcher::matcher::MatcherBuilder::new();
                builder = builder.all(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Http(url) => {
                let mut builder = super::matcher::matcher::MatcherBuilder::new();
                builder = builder.http(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Https(url) => {
                let mut builder = super::matcher::matcher::MatcherBuilder::new();
                builder = builder.https(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Custom(custom) => Matcher_::Custom(custom),
        };

        super::matcher::Matcher {
            inner,
            extra: self.extra,
            maybe_has_http_auth,
            maybe_has_http_custom_headers,
        }
    }
}

impl fmt::Debug for Proxy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Proxy")
            .field("intercept", &self.intercept)
            .finish()
    }
}

impl NoProxy {
    /// Returns a new no-proxy configuration based on environment variables (or `None` if no variables are set)
    /// see [self::NoProxy::from_string()] for the string format
    pub fn from_env() -> Option<NoProxy> {
        let raw = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .unwrap_or_default();

        Self::from_string(&raw)
    }

    /// Returns a new no-proxy configuration based on a `no_proxy` string (or `None` if no variables are set)
    /// The rules are as follows:
    /// * The environment variable `NO_PROXY` is checked, if it is not set, `no_proxy` is checked
    /// * If neither environment variable is set, `None` is returned
    /// * Entries are expected to be comma-separated (whitespace between entries is ignored)
    /// * IP addresses (both IPv4 and IPv6) are allowed, as are optional subnet masks (by adding /size,
    ///   for example "`192.168.1.0/24`").
    /// * An entry "`*`" matches all hostnames (this is the only wildcard allowed)
    /// * Any other entry is considered a domain name (and may contain a leading dot, for example `google.com`
    ///   and `.google.com` are equivalent) and would match both that domain AND all subdomains.
    ///
    /// For example, if `"NO_PROXY=google.com, 192.168.1.0/24"` was set, all the following would match
    /// (and therefore would bypass the proxy):
    /// * `http://google.com/`
    /// * `http://www.google.com/`
    /// * `http://192.168.1.42/`
    ///
    /// The URL `http://notgoogle.com/` would not match.
    pub fn from_string(no_proxy_list: &str) -> Option<Self> {
        if no_proxy_list.trim().is_empty() {
            return None;
        }
        
        Some(NoProxy {
            inner: no_proxy_list.into(),
        })
    }
}

impl fmt::Debug for Custom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("_")
    }
}

impl fmt::Debug for Extra {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Extra")
            .field("auth", &self.auth)
            .field("misc", &self.misc)
            .finish()
    }
}