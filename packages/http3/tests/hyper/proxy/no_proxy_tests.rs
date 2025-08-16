//! No-proxy configuration and environment variable tests

use fluent_ai_http3::hyper::proxy::{NoProxy, Proxy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_proxy_from_env() {
        // Test environment variable parsing
        std::env::set_var("NO_PROXY", "localhost,127.0.0.1,*.example.com");

        let no_proxy = NoProxy::from_env();
        assert!(no_proxy.is_some());

        let no_proxy = no_proxy.unwrap();
        assert_eq!(no_proxy.inner, "localhost,127.0.0.1,*.example.com");

        // Clean up
        std::env::remove_var("NO_PROXY");
    }

    #[test]
    fn test_no_proxy_from_string() {
        let no_proxy = NoProxy::from_string("google.com,192.168.1.0/24");
        assert!(no_proxy.is_some());

        let no_proxy = no_proxy.unwrap();
        assert_eq!(no_proxy.inner, "google.com,192.168.1.0/24");

        // Test empty string
        let no_proxy = NoProxy::from_string("");
        assert!(no_proxy.is_none());

        // Test whitespace only
        let no_proxy = NoProxy::from_string("   ");
        assert!(no_proxy.is_none());
    }

    #[test]
    fn test_proxy_with_no_proxy() {
        let proxy = Proxy::all("http://proxy.example.com")
            .expect("proxy should create")
            .no_proxy("localhost,127.0.0.1");

        assert!(proxy.no_proxy.is_some());
        assert_eq!(proxy.no_proxy.unwrap().inner, "localhost,127.0.0.1");
    }
}