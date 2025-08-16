//! Authentication and authorization tests for proxy functionality

use fluent_ai_http3::hyper::proxy::Proxy;
use fluent_ai_http3::Url;

// Test utilities
fn url(s: &str) -> Url {
    s.parse().expect("test url should parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_with_custom_auth_header() {
        let target = "http://example.domain/";
        let p = Proxy::all(target)
            .expect("test proxy should create")
            .custom_http_auth(http::HeaderValue::from_static("testme"))
            .into_matcher();

        let got = p
            .intercept(&url("http://anywhere.local"))
            .expect("should intercept");
        let auth = got.basic_auth().expect("should have auth");
        assert_eq!(auth, "testme");
    }

    #[test]
    fn test_custom_with_custom_auth_header() {
        let target = "http://example.domain/";
        let p = Proxy::custom(move |_| target.parse::<Url>().ok())
            .custom_http_auth(http::HeaderValue::from_static("testme"))
            .into_matcher();

        let got = p
            .intercept(&url("http://anywhere.local"))
            .expect("should intercept");
        let auth = got.basic_auth().expect("should have auth");
        assert_eq!(auth, "testme");
    }

    #[test]
    fn test_maybe_has_http_auth() {
        let m = Proxy::all("https://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(!m.maybe_has_http_auth(), "https always tunnels");

        let m = Proxy::all("http://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(m.maybe_has_http_auth(), "http forwards");
    }

    #[test]
    fn test_proxy_basic_auth() {
        let proxy = Proxy::http("http://proxy.example.com")
            .expect("proxy should create")
            .basic_auth("user", "pass");

        assert!(proxy.extra.auth.is_some());
    }

    #[test]
    fn test_proxy_custom_headers() {
        let mut headers = http::HeaderMap::new();
        headers.insert("X-Custom", "value".parse().unwrap());

        let proxy = Proxy::https("https://proxy.example.com")
            .expect("proxy should create")
            .custom_headers(headers);

        assert!(proxy.extra.misc.is_some());
        assert_eq!(proxy.extra.misc.unwrap().get("X-Custom").unwrap(), "value");
    }

    #[test]
    fn test_url_auth() {
        use fluent_ai_http3::hyper::proxy::url_auth;
        let mut url = Url::parse("http://example.com").unwrap();
        url_auth(&mut url, "testuser", "testpass");

        assert_eq!(url.username(), "testuser");
        assert_eq!(url.password(), Some("testpass"));
    }

    #[test]
    fn test_encode_basic_auth() {
        use fluent_ai_http3::hyper::proxy::encode_basic_auth;
        let header = encode_basic_auth("user", "pass");

        // Should create a valid header value
        assert!(!header.is_empty());

        // Should contain Basic auth format
        let header_str = header.to_str().unwrap();
        assert!(header_str.starts_with("Basic "));
    }
}