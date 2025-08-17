//! IntoProxy trait and implementations for URL conversion
//!
//! This module provides the IntoProxy trait for converting various types
//! into proxy URLs with proper error handling.

use std::error::Error;
use std::fmt;

/// A trait for converting types into proxy URLs
pub trait IntoProxy: IntoProxySealed {
    fn into_proxy(self) -> Result<crate::Url, Box<dyn Error + Send + Sync>>;
}

/// Sealed trait to prevent external implementations
pub trait IntoProxySealed {}

impl IntoProxySealed for &str {}
impl IntoProxySealed for String {}
impl IntoProxySealed for crate::Url {}

impl IntoProxy for &str {
    fn into_proxy(self) -> Result<crate::Url, Box<dyn Error + Send + Sync>> {
        self.parse()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
    }
}

impl IntoProxy for String {
    fn into_proxy(self) -> Result<crate::Url, Box<dyn Error + Send + Sync>> {
        self.parse()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
    }
}

impl IntoProxy for crate::Url {
    fn into_proxy(self) -> Result<crate::Url, Box<dyn Error + Send + Sync>> {
        Ok(self)
    }
}

/// Error type for proxy URL parsing failures
#[derive(Debug)]
pub struct ProxyParseError {
    message: String,
}

impl ProxyParseError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for ProxyParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Proxy parse error: {}", self.message)
    }
}

impl Error for ProxyParseError {}

/// Validate a proxy URL for common issues
pub fn validate_proxy_url(url: &crate::Url) -> Result<(), ProxyParseError> {
    // Check scheme
    match url.scheme() {
        "http" | "https" | "socks5" => {}
        scheme => {
            return Err(ProxyParseError::new(format!(
                "Unsupported proxy scheme: {}. Supported schemes are http, https, socks5",
                scheme
            )));
        }
    }

    // Check host
    if url.host_str().is_none() {
        return Err(ProxyParseError::new(
            "Proxy URL must have a host".to_string()
        ));
    }

    // Check port for common issues
    if let Some(port) = url.port() {
        if port == 0 {
            return Err(ProxyParseError::new(
                "Proxy port cannot be 0".to_string()
            ));
        }
    }

    Ok(())
}

/// Parse a proxy URL with validation
pub fn parse_proxy_url(input: &str) -> Result<crate::Url, ProxyParseError> {
    let url = input.parse::<crate::Url>()
        .map_err(|e| ProxyParseError::new(format!("Failed to parse URL: {}", e)))?;
    
    validate_proxy_url(&url)?;
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_into_proxy() {
        let proxy_url = "http://proxy.example.com:8080".into_proxy().unwrap();
        assert_eq!(proxy_url.scheme(), "http");
        assert_eq!(proxy_url.host_str(), Some("proxy.example.com"));
        assert_eq!(proxy_url.port(), Some(8080));
    }

    #[test]
    fn test_string_into_proxy() {
        let proxy_str = "https://secure.proxy.com:3128".to_string();
        let proxy_url = proxy_str.into_proxy().unwrap();
        assert_eq!(proxy_url.scheme(), "https");
        assert_eq!(proxy_url.host_str(), Some("secure.proxy.com"));
        assert_eq!(proxy_url.port(), Some(3128));
    }

    #[test]
    fn test_url_into_proxy() {
        let original_url = crate::Url::parse("http://proxy.test").unwrap();
        let proxy_url = original_url.clone().into_proxy().unwrap();
        assert_eq!(proxy_url, original_url);
    }

    #[test]
    fn test_validate_proxy_url_valid() {
        let url = crate::Url::parse("http://proxy.example.com:8080").unwrap();
        assert!(validate_proxy_url(&url).is_ok());
    }

    #[test]
    fn test_validate_proxy_url_invalid_scheme() {
        let url = crate::Url::parse("ftp://proxy.example.com").unwrap();
        let result = validate_proxy_url(&url);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported proxy scheme"));
    }

    #[test]
    fn test_validate_proxy_url_no_host() {
        let url = crate::Url::parse("http:///path").unwrap();
        let result = validate_proxy_url(&url);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must have a host"));
    }

    #[test]
    fn test_parse_proxy_url_valid() {
        let result = parse_proxy_url("http://proxy.example.com:8080");
        assert!(result.is_ok());
        let url = result.unwrap();
        assert_eq!(url.host_str(), Some("proxy.example.com"));
        assert_eq!(url.port(), Some(8080));
    }

    #[test]
    fn test_parse_proxy_url_invalid() {
        let result = parse_proxy_url("not-a-url");
        assert!(result.is_err());
    }

    #[test]
    fn test_socks5_proxy() {
        let proxy_url = "socks5://127.0.0.1:1080".into_proxy().unwrap();
        assert_eq!(proxy_url.scheme(), "socks5");
        assert!(validate_proxy_url(&proxy_url).is_ok());
    }
}