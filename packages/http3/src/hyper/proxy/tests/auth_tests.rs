//! Authentication and authorization tests for proxy functionality

use super::utilities::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_with_custom_auth_header() {
        let target = "http://example.domain/";
        let p = super::super::super::Proxy::all(target)
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
        let p = super::super::super::Proxy::custom(move |_| target.parse::<crate::Url>().ok())
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
        let m = super::super::super::Proxy::all("https://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(!m.maybe_has_http_auth(), "https always tunnels");

        let m = super::super::super::Proxy::all("http://letme:in@yo.local")
            .expect("test proxy should create")
            .into_matcher();
        assert!(m.maybe_has_http_auth(), "http forwards");
    }

    #[test]
    fn test_proxy_basic_auth() {
        let proxy = super::super::super::Proxy::http("http://proxy.example.com")
            .expect("proxy should create")
            .basic_auth("user", "pass");

        assert!(proxy.extra.auth.is_some());
    }

    #[test]
    fn test_proxy_custom_headers() {
        let mut headers = http::HeaderMap::new();
        headers.insert("X-Custom", "value".parse().unwrap());

        let proxy = super::super::super::Proxy::https("https://proxy.example.com")
            .expect("proxy should create")
            .custom_headers(headers);

        assert!(proxy.extra.misc.is_some());
        assert_eq!(proxy.extra.misc.unwrap().get("X-Custom").unwrap(), "value");
    }

    #[test]
    fn test_url_auth() {
        let mut url = crate::Url::parse("http://example.com").unwrap();
        super::super::super::url_auth(&mut url, "testuser", "testpass");

        assert_eq!(url.username(), "testuser");
        assert_eq!(url.password(), Some("testpass"));
    }

    #[test]
    fn test_encode_basic_auth() {
        let header = super::super::super::encode_basic_auth("user", "pass");

        // Should create a valid header value
        assert!(!header.is_empty());

        // Should contain Basic auth format
        let header_str = header.to_str().unwrap();
        assert!(header_str.starts_with("Basic "));
    }
}