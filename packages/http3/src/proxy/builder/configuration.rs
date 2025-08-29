//! Proxy configuration methods
//!
//! Instance methods for configuring proxy authentication, custom headers,
//! and no-proxy rules with production-quality error handling.

use http::{header::HeaderValue, HeaderMap};
use super::types::Proxy;
use super::super::matcher::NoProxy;

impl Proxy {
    /// Set the `Proxy-Authorization` header using Basic auth.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate http3;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let proxy = crate::proxy::Proxy::https("http://localhost:1234")?
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
    /// let proxy = crate::proxy::Proxy::https("http://localhost:1234")?
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
    /// let proxy = crate::proxy::Proxy::https("http://localhost:1234")?
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
    /// let proxy = crate::proxy::Proxy::https("http://localhost:1234")?
    ///     .no_proxy("localhost,*.internal");
    /// # Ok(())
    /// # }
    /// # fn main() {}
    /// ```
    pub fn no_proxy<S: Into<String>>(mut self, no_proxy: S) -> Proxy {
        self.no_proxy = Some(NoProxy::new(no_proxy.into()));
        self
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
    use super::super::types::ProxyIntercept;

    fn create_test_proxy() -> Proxy {
        Proxy::new(ProxyIntercept::Http(
            crate::Url::parse("http://proxy.example.com:8080")
                .expect("Failed to parse test proxy URL")
        ))
    }

    #[test]
    fn test_proxy_basic_auth() {
        let proxy = create_test_proxy().basic_auth("user", "pass");
        
        assert!(proxy.extra().auth().is_some());
        let auth_header = proxy.extra().auth().expect("Auth header should be present");
        assert!(auth_header.to_str().expect("Auth header should be valid UTF-8").starts_with("Basic "));
    }

    #[test]
    fn test_proxy_custom_auth() {
        let auth_value = HeaderValue::from_static("Bearer token123");
        let proxy = create_test_proxy().custom_http_auth(auth_value.clone());
        
        assert!(proxy.extra().auth().is_some());
        assert_eq!(proxy.extra().auth().expect("Auth header should be present"), &auth_value);
    }

    #[test]
    fn test_proxy_custom_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Custom", HeaderValue::from_static("test"));
        
        let proxy = create_test_proxy().custom_headers(headers);
        
        assert!(proxy.extra().headers().is_some());
        assert_eq!(proxy.extra().headers().expect("Headers should be present").len(), 1);
    }

    #[test]
    fn test_proxy_no_proxy() {
        let proxy = create_test_proxy().no_proxy("localhost,*.internal");
        
        assert!(proxy.no_proxy().is_some());
    }

    #[test]
    fn test_encode_basic_auth() {
        let header = encode_basic_auth("user", "pass");
        let expected = "Basic dXNlcjpwYXNz"; // base64 of "user:pass"
        assert_eq!(header.to_str().expect("Header should be valid UTF-8"), expected);
    }

    #[test]
    fn test_encode_basic_auth_invalid_chars() {
        // Test with characters that might cause issues
        let header = encode_basic_auth("user\n", "pass\r");
        assert!(header.to_str().expect("Header should be valid UTF-8").starts_with("Basic "));
    }
}