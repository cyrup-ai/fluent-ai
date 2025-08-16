//! Basic proxy functionality tests

use super::utilities::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http() {
        let target = "http://example.domain/";
        let p = super::super::super::Proxy::http(target)
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
        let p = super::super::super::Proxy::https(target)
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
        let p = super::super::super::Proxy::all(target)
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
        let p = super::super::super::Proxy::custom(move |url| {
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
    fn test_socks_proxy_default_port() {
        let m = super::super::super::Proxy::all("socks5://example.com")
            .expect("test proxy should create")
            .into_matcher();

        let http = "http://hyper.rs";
        let https = "https://hyper.rs";

        assert_eq!(intercepted_uri(&m, http).port_u16(), Some(1080));
        assert_eq!(intercepted_uri(&m, https).port_u16(), Some(1080));

        // custom port
        let m = super::super::super::Proxy::all("socks5://example.com:1234")
            .expect("test proxy should create")
            .into_matcher();

        assert_eq!(intercepted_uri(&m, http).port_u16(), Some(1234));
        assert_eq!(intercepted_uri(&m, https).port_u16(), Some(1234));
    }
}