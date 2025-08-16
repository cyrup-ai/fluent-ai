//! URL conversion and validation tests

use fluent_ai_http3::hyper::into_url::IntoUrl;
use std::error::Error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_url_file_scheme() {
        let err = "file:///etc/hosts".into_url().unwrap_err();
        if let Some(source) = err.source() {
            assert_eq!(source.to_string(), "URL scheme is not allowed");
        } else {
            panic!("error should have source");
        }
    }

    #[test]
    fn into_url_blob_scheme() {
        let err = "blob:https://example.com".into_url().unwrap_err();
        if let Some(source) = err.source() {
            assert_eq!(source.to_string(), "URL scheme is not allowed");
        } else {
            panic!("error should have source");
        }
    }

    #[test]
    fn into_url_valid_http() {
        let url = "http://example.com".into_url().unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("example.com"));
    }

    #[test]
    fn into_url_valid_https() {
        let url = "https://example.com/path".into_url().unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("example.com"));
        assert_eq!(url.path(), "/path");
    }

    #[test]
    fn into_url_invalid_scheme() {
        let schemes = ["ftp://example.com", "data:text/plain,hello", "javascript:alert('hi')"];
        
        for scheme_url in &schemes {
            let result = scheme_url.into_url();
            assert!(result.is_err(), "Should reject scheme: {}", scheme_url);
        }
    }

    #[test]
    fn into_url_malformed() {
        let malformed_urls = ["not-a-url", "://missing-scheme", "http://"];
        
        for malformed in &malformed_urls {
            let result = malformed.into_url();
            assert!(result.is_err(), "Should reject malformed URL: {}", malformed);
        }
    }

    #[cfg(target_arch = "wasm32")]
    mod wasm_tests {
        use super::*;
        use wasm_bindgen_test::*;

        #[wasm_bindgen_test]
        fn wasm_into_url_valid() {
            let url = "https://example.com".into_url().unwrap();
            assert_eq!(url.scheme(), "https");
        }
    }
}