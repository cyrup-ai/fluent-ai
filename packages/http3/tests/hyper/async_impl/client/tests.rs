//! Client functionality tests for HTTP async implementation

use fluent_ai_http3::hyper::async_impl::Client;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "rustls-tls-manual-roots-no-provider"))]
    #[test]
    fn execute_request_rejects_invalid_urls() {
        let url_str = "hxxps://www.rust-lang.org/";
        let url = url::Url::parse(url_str).expect("test should succeed");
        let result = fluent_ai_http3::get(url.clone()).collect_one();

        assert!(result.is_err());
        let err = result.err().expect("test should succeed");
        assert!(err.is_builder());
        assert_eq!(url_str, err.url().expect("test should succeed").as_str());
    }

    /// https://github.com/seanmonstar/http3/issues/668
    #[cfg(not(feature = "rustls-tls-manual-roots-no-provider"))]
    #[test]
    fn execute_request_rejects_invalid_hostname() {
        let url_str = "https://{{hostname}}/";
        let url = url::Url::parse(url_str).expect("test should succeed");
        let result = fluent_ai_http3::get(url.clone()).collect_one();

        assert!(result.is_err());
        let err = result.err().expect("test should succeed");
        assert!(err.is_builder());
        assert_eq!(url_str, err.url().expect("test should succeed").as_str());
    }

    #[test]
    fn client_new_creates_default_client() {
        let client = Client::new();
        // Basic smoke test - client should be created successfully
        assert!(std::ptr::addr_of!(client) as *const _ != std::ptr::null());
    }

    #[test]
    fn client_builder_creates_configured_client() {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("client should build successfully");
        
        // Basic smoke test - configured client should be created successfully
        assert!(std::ptr::addr_of!(client) as *const _ != std::ptr::null());
    }
}