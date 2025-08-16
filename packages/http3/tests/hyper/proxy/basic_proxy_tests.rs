//! Basic proxy functionality tests

use fluent_ai_http3::hyper::proxy::Proxy;
use fluent_ai_http3::Url;

// Test utilities
fn url(s: &str) -> Url {
    s.parse().expect("test url should parse")
}

fn intercepted_uri(matcher: &fluent_ai_http3::hyper::proxy::Matcher, uri: &str) -> String {
    matcher
        .intercept(&url(uri))
        .expect("should intercept")
        .uri()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http() {
        let target = "http://example.domain/";
        let p = Proxy::http(target)
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
        let p = Proxy::https(target)
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
        let p = Proxy::all(target)
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

        let p = Proxy::custom(move |url| {
            if url.scheme() == "http" {
                target1.parse().ok()
            } else {
                target2.parse().ok()
            }
        })
        .into_matcher();

        let http = "http://hyper.rs";
        let https = "https://hyper.rs";

        assert_eq!(intercepted_uri(&p, http), target1);
        assert_eq!(intercepted_uri(&p, https), target2);
    }

    #[test]
    fn test_proxy_no_proxy() {
        let target = "http://example.domain/";
        let p = Proxy::http(target)
            .expect("test proxy should create")
            .no_proxy("hyper.rs")
            .into_matcher();

        let http = "http://hyper.rs";
        let other = "http://other.rs";

        assert!(p.intercept(&url(http)).is_none());
        assert_eq!(intercepted_uri(&p, other), target);
    }
}