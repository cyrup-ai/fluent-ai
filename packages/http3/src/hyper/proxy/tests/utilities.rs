//! Test utilities and helper functions

use http::Uri;

pub fn url(s: &str) -> http::Uri {
    s.parse().expect("test URI should parse")
}

pub fn intercepted_uri(p: &super::super::Matcher, s: &str) -> Uri {
    p.intercept(&s.parse().expect("test URI should parse"))
        .expect("should intercept")
        .uri()
        .clone()
}