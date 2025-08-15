#[cfg(test)]
mod tests {
    #![cfg(not(feature = "rustls-tls-manual-roots-no-provider"))]

    #[test]
    fn execute_request_rejects_invalid_urls() {
        let url_str = "hxxps://www.rust-lang.org/";
        let url = url::Url::parse(url_str).expect("test should succeed");
        let result = crate::get(url.clone()).collect_one();

        assert!(result.is_err());
        let err = result.err().expect("test should succeed");
        assert!(err.is_builder());
        assert_eq!(url_str, err.url().expect("test should succeed").as_str());
    }

    /// https://github.com/seanmonstar/http3/issues/668
    #[test]
    fn execute_request_rejects_invalid_hostname() {
        let url_str = "https://{{hostname}}/";
        let url = url::Url::parse(url_str).expect("test should succeed");
        let result = crate::get(url.clone()).collect_one();

        assert!(result.is_err());
        let err = result.err().expect("test should succeed");
        assert!(err.is_builder());
        assert_eq!(url_str, err.url().expect("test should succeed").as_str());
    }

    #[test]
    fn test_future_size() {
        let s = std::mem::size_of::<super::Pending>();
        assert!(s < 128, "size_of::<Pending>() == {s}, too big");
    }
}