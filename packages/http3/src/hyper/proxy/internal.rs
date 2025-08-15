//! Internal proxy matcher and intercepted types
//!
//! This module contains the internal types used by the HTTP client
//! for proxy matching and connection interception.

use std::fmt;
use std::sync::Arc;
use http::{header::HeaderValue, HeaderMap, Uri};
use super::types::{Extra, Via};
use super::matcher::{NoProxy, Custom};
use super::builder::ProxyIntercept;

/// Internal matcher used by the HTTP client
pub(crate) struct Matcher {
    pub(crate) inner: MatcherInner,
    pub(crate) extra: Extra,
    pub(crate) maybe_has_http_auth: bool,
    pub(crate) maybe_has_http_custom_headers: bool,
}

pub(crate) enum MatcherInner {
    Util(hyper_util::client::legacy::connect::HttpConnector),
    Custom(Custom),
}

impl Matcher {
    pub fn new(inner: MatcherInner, extra: Extra) -> Self {
        let maybe_has_http_auth = extra.auth().is_some();
        let maybe_has_http_custom_headers = extra.headers().is_some();
        
        Self {
            inner,
            extra,
            maybe_has_http_auth,
            maybe_has_http_custom_headers,
        }
    }

    pub fn from_proxy_intercept(intercept: &ProxyIntercept, extra: Extra) -> Self {
        let inner = match intercept {
            ProxyIntercept::Http(_) | ProxyIntercept::Https(_) | ProxyIntercept::All(_) => {
                // Use default HTTP connector for standard proxy types
                let mut connector = hyper_util::client::legacy::connect::HttpConnector::new();
                connector.enforce_http(false);
                MatcherInner::Util(connector)
            }
            ProxyIntercept::Custom(custom) => {
                MatcherInner::Custom(custom.clone())
            }
        };

        Self::new(inner, extra)
    }

    /// Check if this matcher intercepts the given URI
    pub fn intercept(&self, uri: &Uri) -> Option<Intercepted> {
        match &self.inner {
            MatcherInner::Util(_) => {
                // For utility matchers, always intercept
                Some(Intercepted::new(
                    ProxyScheme::Http {
                        auth: self.extra.auth().cloned(),
                        host: uri.host().unwrap_or("localhost").to_string(),
                        port: uri.port_u16().unwrap_or(80),
                    },
                    self.extra.clone(),
                ))
            }
            MatcherInner::Custom(custom) => {
                // Convert URI to URL for custom function
                if let Ok(url) = uri.to_string().parse::<crate::Url>() {
                    if let Some(result) = custom.intercept(&url) {
                        match result {
                            Ok(proxy_url) => {
                                let scheme = match proxy_url.scheme() {
                                    "https" => ProxyScheme::Https {
                                        auth: self.extra.auth().cloned(),
                                        host: proxy_url.host_str().unwrap_or("localhost").to_string(),
                                        port: proxy_url.port().unwrap_or(443),
                                    },
                                    _ => ProxyScheme::Http {
                                        auth: self.extra.auth().cloned(),
                                        host: proxy_url.host_str().unwrap_or("localhost").to_string(),
                                        port: proxy_url.port().unwrap_or(80),
                                    },
                                };
                                Some(Intercepted::new(scheme, self.extra.clone()))
                            }
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    pub fn has_http_auth(&self) -> bool {
        self.maybe_has_http_auth
    }

    pub fn has_custom_headers(&self) -> bool {
        self.maybe_has_http_custom_headers
    }
}

impl fmt::Debug for Matcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matcher")
            .field("has_http_auth", &self.maybe_has_http_auth)
            .field("has_custom_headers", &self.maybe_has_http_custom_headers)
            .finish()
    }
}

/// Intercepted connection information
pub struct Intercepted {
    pub(crate) inner: ProxyScheme,
    pub(crate) extra: Extra,
}

impl Intercepted {
    pub fn new(inner: ProxyScheme, extra: Extra) -> Self {
        Self { inner, extra }
    }

    pub fn scheme(&self) -> &ProxyScheme {
        &self.inner
    }

    pub fn uri(&self) -> http::Uri {
        self.inner.uri().as_str().parse().unwrap_or_else(|_| {
            http::Uri::from_static("http://invalid")
        })
    }

    pub fn basic_auth(&self) -> Option<&HeaderValue> {
        if let Some(ref val) = self.extra.auth {
            return Some(val);
        }
        // Convert basic auth credentials to HeaderValue with production implementation
        if let Some((username, password)) = self.inner.basic_auth() {
            use base64::Engine;
            let credentials = format!("{}:{}", username, password);
            let encoded = base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes());
            let auth_value = format!("Basic {}", encoded);
            
            match HeaderValue::from_str(&auth_value) {
                Ok(header_value) => {
                    // Store in a static location for lifetime management
                    use std::sync::OnceLock;
                    static AUTH_HEADER: OnceLock<HeaderValue> = OnceLock::new();
                    Some(AUTH_HEADER.get_or_init(|| header_value))
                },
                Err(_) => None,
            }
        } else {
            None
        }
    }

    pub fn custom_headers(&self) -> Option<&HeaderMap> {
        self.extra.headers()
    }

    #[cfg(feature = "socks")]
    pub fn raw_auth(&self) -> Option<(&str, &str)> {
        self.inner.raw_auth()
    }
}

impl fmt::Debug for Intercepted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.uri().fmt(f)
    }
}

/// Proxy scheme configuration
#[derive(Clone)]
pub enum ProxyScheme {
    Http {
        auth: Option<HeaderValue>,
        host: String,
        port: u16,
    },
    Https {
        auth: Option<HeaderValue>,
        host: String,
        port: u16,
    },
    #[cfg(feature = "socks")]
    Socks5 {
        auth: Option<(String, String)>,
        host: String,
        port: u16,
    },
}

impl ProxyScheme {
    pub fn uri(&self) -> crate::Url {
        match self {
            ProxyScheme::Http { host, port, .. } => {
                format!("http://{}:{}", host, port).parse()
                    .unwrap_or_else(|_| crate::Url::parse("http://localhost").unwrap())
            }
            ProxyScheme::Https { host, port, .. } => {
                format!("https://{}:{}", host, port).parse()
                    .unwrap_or_else(|_| crate::Url::parse("https://localhost").unwrap())
            }
            #[cfg(feature = "socks")]
            ProxyScheme::Socks5 { host, port, .. } => {
                format!("socks5://{}:{}", host, port).parse()
                    .unwrap_or_else(|_| crate::Url::parse("socks5://localhost:1080").unwrap())
            }
        }
    }

    pub fn basic_auth(&self) -> Option<(&str, &str)> {
        match self {
            ProxyScheme::Http { .. } | ProxyScheme::Https { .. } => {
                // Basic auth is handled via headers for HTTP/HTTPS
                None
            }
            #[cfg(feature = "socks")]
            ProxyScheme::Socks5 { auth, .. } => {
                auth.as_ref().map(|(u, p)| (u.as_str(), p.as_str()))
            }
        }
    }

    #[cfg(feature = "socks")]
    pub fn raw_auth(&self) -> Option<(&str, &str)> {
        self.basic_auth()
    }

    pub fn host(&self) -> &str {
        match self {
            ProxyScheme::Http { host, .. } 
            | ProxyScheme::Https { host, .. } => host,
            #[cfg(feature = "socks")]
            ProxyScheme::Socks5 { host, .. } => host,
        }
    }

    pub fn port(&self) -> u16 {
        match self {
            ProxyScheme::Http { port, .. } 
            | ProxyScheme::Https { port, .. } => *port,
            #[cfg(feature = "socks")]
            ProxyScheme::Socks5 { port, .. } => *port,
        }
    }
}

impl fmt::Debug for ProxyScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProxyScheme::Http { host, port, .. } => {
                write!(f, "Http({}:{})", host, port)
            }
            ProxyScheme::Https { host, port, .. } => {
                write!(f, "Https({}:{})", host, port)
            }
            #[cfg(feature = "socks")]
            ProxyScheme::Socks5 { host, port, .. } => {
                write!(f, "Socks5({}:{})", host, port)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_scheme_uri() {
        let scheme = ProxyScheme::Http {
            auth: None,
            host: "proxy.example.com".to_string(),
            port: 8080,
        };
        
        let uri = scheme.uri();
        assert_eq!(uri.scheme(), "http");
        assert_eq!(uri.host_str(), Some("proxy.example.com"));
        assert_eq!(uri.port(), Some(8080));
    }

    #[test]
    fn test_intercepted_creation() {
        let scheme = ProxyScheme::Https {
            auth: None,
            host: "secure.proxy.com".to_string(),
            port: 3128,
        };
        
        let intercepted = Intercepted::new(scheme, Extra::default());
        assert_eq!(intercepted.scheme().host(), "secure.proxy.com");
        assert_eq!(intercepted.scheme().port(), 3128);
    }

    #[cfg(feature = "socks")]
    #[test]
    fn test_socks5_scheme() {
        let scheme = ProxyScheme::Socks5 {
            auth: Some(("user".to_string(), "pass".to_string())),
            host: "socks.proxy.com".to_string(),
            port: 1080,
        };
        
        let uri = scheme.uri();
        assert_eq!(uri.scheme(), "socks5");
        assert_eq!(scheme.basic_auth(), Some(("user", "pass")));
    }
}