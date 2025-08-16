//! Body handling tests for HTTP async implementation

use fluent_ai_http3::hyper::async_impl::Body;
use http_body::Body as _;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_bytes() {
        let test_data = b"Test body";
        let body = Body::from(&test_data[..]);
        assert_eq!(body.as_bytes(), Some(&test_data[..]));
    }

    #[test]
    fn body_exact_length() {
        let empty_body = Body::empty();
        assert!(empty_body.is_end_stream());
        assert_eq!(empty_body.size_hint().exact(), Some(0));

        let bytes_body = Body::reusable("abc".into());
        assert!(!bytes_body.is_end_stream());
        assert_eq!(bytes_body.size_hint().exact(), Some(3));

        // can delegate even when wrapped
        let stream_body = Body::wrap(empty_body);
        assert!(stream_body.is_end_stream());
        assert_eq!(stream_body.size_hint().exact(), Some(0));
    }

    #[test]
    fn test_body_from_string() {
        let content = "Hello, World!";
        let body = Body::from(content);
        assert_eq!(body.as_bytes(), Some(content.as_bytes()));
    }

    #[test]
    fn test_body_from_vec() {
        let data = vec![1, 2, 3, 4, 5];
        let body = Body::from(data.clone());
        assert_eq!(body.as_bytes(), Some(data.as_slice()));
    }

    #[test]
    fn test_empty_body() {
        let body = Body::empty();
        assert!(body.is_end_stream());
        assert_eq!(body.size_hint().exact(), Some(0));
        assert_eq!(body.as_bytes(), Some(&[]));
    }
}