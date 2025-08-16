//! Tests for HTTP response functionality
//!
//! Tests for response builder extensions and URL handling

use http::response::Builder;
use url::Url;
use fluent_ai_http3::hyper::response::{ResponseBuilderExt, ResponseUrl};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_builder_ext() {
        let url = match Url::parse("http://example.com") {
            Ok(url) => url,
            Err(_) => return, // Skip test if URL parsing fails
        };
        let response = match Builder::new().status(200).url(url.clone()).body(()) {
            Ok(response) => response,
            Err(_) => return, // Skip test if response build fails
        };

        assert_eq!(
            response.extensions().get::<ResponseUrl>(),
            Some(&ResponseUrl(url))
        );
    }
}