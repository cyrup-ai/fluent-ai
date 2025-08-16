//! Core proxy types and MessageChunk implementations
//!
//! This module contains the fundamental proxy types including ProxyUrl wrapper,
//! Via enum, and core proxy configuration structures.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use http::{header::HeaderValue, HeaderMap, Uri};
use fluent_ai_async::prelude::*;
use crate::Url;

/// MessageChunk wrapper for URL parsing with proper error handling
#[derive(Debug, Clone)]
pub struct ProxyUrl {
    pub(crate) url: crate::Url,
    pub(crate) error_message: Option<String>,
}

impl ProxyUrl {
    pub fn new(url: crate::Url) -> Self {
        Self {
            url,
            error_message: None,
        }
    }
    
    pub fn into_url(self) -> crate::Url {
        self.url
    }
}

impl MessageChunk for ProxyUrl {
    fn bad_chunk(error: String) -> Self {
        // Simplified fallback URL creation
        let fallback_url = crate::Url::parse("http://localhost")
            .or_else(|_| crate::Url::parse("http://127.0.0.1"))
            .or_else(|_| crate::Url::parse("http://0.0.0.0"))
            .unwrap_or_else(|_| {
                // Last resort - create a basic URL that should always work
                // This should never fail as it's a valid URL
                crate::Url::parse("http://invalid")
                    .expect("Basic fallback URL should always parse")
            });

        ProxyUrl {
            url: fallback_url,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<String> {
        self.error_message.clone()
    }
}

/// Proxy connection method
#[derive(Debug, Clone)]
pub enum Via {
    Http,
    Https,
    Socks5,
}

/// Proxy intercept configuration
#[derive(Debug, Clone)]
pub struct Intercept {
    pub(crate) proxy_uri: crate::Url,
    pub(crate) via: Via,
}

impl Intercept {
    pub fn new(proxy_uri: crate::Url, via: Via) -> Self {
        Self { proxy_uri, via }
    }

    /// Get the proxy URI
    pub fn proxy_uri(&self) -> &crate::Url {
        &self.proxy_uri
    }

    /// Get the connection method
    pub fn via(&self) -> &Via {
        &self.via
    }

    /// Extract basic auth credentials from proxy URI
    pub fn basic_auth(&self) -> Option<(&str, &str)> {
        let auth = self.proxy_uri.username();
        if !auth.is_empty() {
            return Some((auth, self.proxy_uri.password().unwrap_or("")));
        }
        None
    }
}

/// Extra configuration for proxy connections
#[derive(Clone)]
pub struct Extra {
    pub(crate) auth: Option<HeaderValue>,
    pub(crate) misc: Option<HeaderMap>,
}

impl Extra {
    pub fn new() -> Self {
        Self {
            auth: None,
            misc: None,
        }
    }

    pub fn with_auth(mut self, auth: HeaderValue) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.misc = Some(headers);
        self
    }

    pub fn auth(&self) -> Option<&HeaderValue> {
        self.auth.as_ref()
    }

    pub fn headers(&self) -> Option<&HeaderMap> {
        self.misc.as_ref()
    }
}

impl Default for Extra {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Extra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Extra")
            .field("auth", &self.auth.is_some())
            .field("misc", &self.misc.as_ref().map(|h| h.len()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_url_creation() {
        let url = crate::Url::parse("http://proxy.example.com:8080")
            .expect("Test URL should parse correctly");
        let proxy_url = ProxyUrl::new(url.clone());
        assert_eq!(proxy_url.url, url);
        assert!(!proxy_url.is_error());
    }

    #[test]
    fn test_proxy_url_bad_chunk() {
        let bad_proxy = ProxyUrl::bad_chunk("Test error".to_string());
        assert!(bad_proxy.is_error());
        assert_eq!(bad_proxy.error_message(), Some("Test error"));
    }

    #[test]
    fn test_intercept_basic_auth() {
        let url = crate::Url::parse("http://user:pass@proxy.example.com:8080")
            .expect("Test URL with auth should parse correctly");
        let intercept = Intercept::new(url, Via::Http);
        
        let auth = intercept.basic_auth();
        assert_eq!(auth, Some(("user", "pass")));
    }

    #[test]
    fn test_extra_configuration() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Custom", HeaderValue::from_static("value"));
        
        let extra = Extra::new()
            .with_headers(headers);
            
        assert!(extra.headers().is_some());
        assert_eq!(extra.headers().expect("Headers should be present").len(), 1);
    }
}