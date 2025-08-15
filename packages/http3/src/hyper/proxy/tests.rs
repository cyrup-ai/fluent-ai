//! Comprehensive test suite for proxy functionality
//! 
//! Production-quality tests covering all proxy types, authentication, and edge cases.

#[cfg(test)]
mod tests {
    use super::*;
    use http::Uri;

    fn url(s: &str) -> http::Uri {
        s.parse().expect("test URI should parse")
    }

    fn intercepted_uri(p: &super::Matcher, s: &str) -> Uri {
        p.intercept(&s.parse().expect("test URI should parse"))
            .expect("should intercept")
            .uri()
            .clone()
    }

    #[test]
    fn test_http() {
        let target = "http://example.domain/";
        let p = super::Proxy::http(target)
            .expect("test proxy should create")
            .into_matcher();

        let http = "http://hyper.rs";
        let other = "https://hyper.rs";

        assert_eq!(intercepted_uri(&p, http), target);
        assert!(p.intercept(&url(other)).is_none());
    }

    #[test]
    fn test_https() {
        let target = "http://example.domain/";
        let p = super::Proxy::https(target)
            .expect("test proxy should create")
            .into_matcher();

        let http = "http://hyper.rs";
        let other = "https://hyper.rs";

        assert!(p.intercept(&url(http)).is_none());
        assert_eq!(intercepted_uri(&p, other), target);
    }

    #[test]
    fn test_all() {
        let target = "http://example.domain/";
        let p = super::Proxy::all(target)
            .expect("test proxy should create")
            .into_matcher();

        let http = "http://hyper.rs";
        let https = "https://hyper.rs";

        assert_eq!(intercepted_uri(&p, http), target);
        assert_eq!(intercepted_uri(&p, https), target);
    }

    #[test]
    fn test_custom() {
        let target1 = "http://example.domain/";
        let target2 = "https://example.domain/";
        let p = super::Proxy::custom(move |url| {
            if url.host_str() == Some("hyper.rs") {
                target1.parse().ok()
            } else if url.scheme() == "http" {
                target2.parse().ok()
            } else {
                None::<crate::Url>
            }
        })
        .into_matcher();

        let http = "http://seanmonstar.com";
        let https = "https://hyper.rs";
        let other = "x-youve-never-heard-of-me-mr-proxy://seanmonstar.com";

        assert_eq!(intercepted_uri(&p, http), target2);
        assert_eq!(intercepted_uri(&p, https), target1);
        assert!(p.intercept(&url(other)).is_none());
    }

    #[test]
    fn test_standard_with_custom_auth_header() {
        let target = "http://example.domain/";
        let p = super::Proxy::all(target)
            .expect("test proxy should create")
            .custom_http_auth(http::HeaderValue::from_static("testme"))
            .into_matcher();

        let got = p.intercept(&url("http://anywhere.local"))
            .expect("should intercept");
        let auth = got.basic_auth().expect("should have auth");
        assert_eq!(auth, "testme");
    }

    #[test]
    fn test_custom_with_custom_auth_header() {
        let target = "http://example.domain/";
        let p = super::Proxy::custom(move |_| target.parse::<crate::Url>().ok())
            .custom_http_auth(http::HeaderValue::from_static("testme"))
            .into_matcher();

        let got = p.intercept(&url("http://anywhere.local"))
            .expect("should intercept");
        let auth = got.basic_auth().expect("should have auth");
        assert_eq!(auth, "testme");
    }

    #[test]
    fn test_maybe_has_http_auth() {
        let m = super::Proxy::all("https://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(!m.maybe_has_http_auth(), "https always tunnels");

        let m = super::Proxy::all("http://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(m.maybe_has_http_auth(), "http forwards");
    }

    #[test]
    fn test_socks_proxy_default_port() {
        let m = super::Proxy::all("socks5://example.com")
            .expect("test proxy should create")
            .into_matcher();

        let http = "http://hyper.rs";
        let https = "https://hyper.rs";

        assert_eq!(intercepted_uri(&m, http).port_u16(), Some(1080));
        assert_eq!(intercepted_uri(&m, https).port_u16(), Some(1080));

        // custom port
        let m = super::Proxy::all("socks5://example.com:1234")
            .expect("test proxy should create")
            .into_matcher();

        assert_eq!(intercepted_uri(&m, http).port_u16(), Some(1234));
        assert_eq!(intercepted_uri(&m, https).port_u16(), Some(1234));
    }

    #[test]
    fn test_no_proxy_from_env() {
        // Test environment variable parsing
        std::env::set_var("NO_PROXY", "localhost,127.0.0.1,*.example.com");
        
        let no_proxy = super::NoProxy::from_env();
        assert!(no_proxy.is_some());
        
        let no_proxy = no_proxy.unwrap();
        assert_eq!(no_proxy.inner, "localhost,127.0.0.1,*.example.com");
        
        // Clean up
        std::env::remove_var("NO_PROXY");
    }

    #[test]
    fn test_no_proxy_from_string() {
        let no_proxy = super::NoProxy::from_string("google.com,192.168.1.0/24");
        assert!(no_proxy.is_some());
        
        let no_proxy = no_proxy.unwrap();
        assert_eq!(no_proxy.inner, "google.com,192.168.1.0/24");
        
        // Test empty string
        let no_proxy = super::NoProxy::from_string("");
        assert!(no_proxy.is_none());
        
        // Test whitespace only
        let no_proxy = super::NoProxy::from_string("   ");
        assert!(no_proxy.is_none());
    }

    #[test]
    fn test_proxy_with_no_proxy() {
        let proxy = super::Proxy::all("http://proxy.example.com")
            .expect("proxy should create")
            .no_proxy("localhost,127.0.0.1");
        
        assert!(proxy.no_proxy.is_some());
        assert_eq!(proxy.no_proxy.unwrap().inner, "localhost,127.0.0.1");
    }

    #[test]
    fn test_proxy_basic_auth() {
        let proxy = super::Proxy::http("http://proxy.example.com")
            .expect("proxy should create")
            .basic_auth("user", "pass");
        
        assert!(proxy.extra.auth.is_some());
    }

    #[test]
    fn test_proxy_custom_headers() {
        let mut headers = http::HeaderMap::new();
        headers.insert("X-Custom", "value".parse().unwrap());
        
        let proxy = super::Proxy::https("https://proxy.example.com")
            .expect("proxy should create")
            .custom_headers(headers);
        
        assert!(proxy.extra.misc.is_some());
        assert_eq!(
            proxy.extra.misc.unwrap().get("X-Custom").unwrap(),
            "value"
        );
    }

    #[test]
    fn test_matcher_system() {
        let matcher = super::Matcher::system();
        
        // System matcher should be created successfully
        assert!(matcher.maybe_has_http_auth);
        assert!(matcher.maybe_has_http_custom_headers);
    }

    #[test]
    fn test_url_auth() {
        let mut url = crate::Url::parse("http://example.com").unwrap();
        super::url_auth(&mut url, "testuser", "testpass");
        
        assert_eq!(url.username(), "testuser");
        assert_eq!(url.password(), Some("testpass"));
    }

    #[test]
    fn test_encode_basic_auth() {
        let header = super::encode_basic_auth("user", "pass");
        
        // Should create a valid header value
        assert!(!header.is_empty());
        
        // Should contain Basic auth format
        let header_str = header.to_str().unwrap();
        assert!(header_str.starts_with("Basic "));
    }

    #[test]
    fn test_proxy_url_message_chunk() {
        use super::url_handling::ProxyUrl;
        use fluent_ai_async::prelude::MessageChunk;
        
        // Test successful URL creation
        let url = crate::Url::parse("http://example.com").unwrap();
        let proxy_url = ProxyUrl::new(url.clone());
        assert_eq!(proxy_url.into_url(), url);
        
        // Test bad chunk creation
        let bad_chunk = ProxyUrl::bad_chunk("Test error".to_string());
        assert!(bad_chunk.error().is_some());
        assert_eq!(bad_chunk.error().unwrap(), "Test error");
    }

    #[test]
    fn test_matcher_patterns() {
        use super::matcher::matcher::Matcher;
        
        let patterns = vec!["*.example.com".to_string(), "localhost".to_string()];
        let matcher = Matcher::new(patterns);
        
        // Test wildcard matching
        let uri: http::Uri = "http://sub.example.com".parse().unwrap();
        assert!(matcher.matches(&uri));
        
        // Test exact matching
        let uri: http::Uri = "http://localhost".parse().unwrap();
        assert!(matcher.matches(&uri));
        
        // Test non-matching
        let uri: http::Uri = "http://other.com".parse().unwrap();
        assert!(!matcher.matches(&uri));
    }

    #[test]
    fn test_matcher_builder() {
        use super::matcher::matcher::MatcherBuilder;
        
        let builder = MatcherBuilder::new()
            .all("http://proxy1.com".to_string())
            .no("localhost")
            .http("http://proxy2.com".to_string())
            .https("https://proxy3.com".to_string());
        
        let matcher = builder.build();
        
        // Builder should create matcher successfully
        assert_eq!(matcher.patterns.len(), 1); // Only "localhost" in no_patterns
        assert_eq!(matcher.patterns[0], "localhost");
    }
}