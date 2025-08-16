//! HttpResponse struct and basic constructors
//!
//! This module contains the core Response type definition and its basic constructor methods.

use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use hyper::rt::Sleep;
use hyper::{HeaderMap, StatusCode, Version};
use hyper_util::client::legacy::connect::HttpInfo;
use url::Url;

use super::super::super::body::{ResponseBody, response};
use super::super::super::decoder::{Accepts, Decoder};
#[cfg(feature = "cookies")]
use crate::cookie;

/// An HTTP Response
pub struct Response {
    pub(super) res: hyper::Response<Decoder>,
    pub(super) url: Box<Url>,
    #[cfg(feature = "cookies")]
    pub(super) cookie_jar: Option<crate::cookie::CookieJar>,
}

impl Response {
    pub(super) fn new(
        res: hyper::Response<ResponseBody>,
        url: Url,
        accepts: Accepts,
        total_timeout: Option<Pin<Box<dyn Sleep>>>,
        read_timeout: Option<Duration>,
    ) -> Response {
        let (mut parts, body) = res.into_parts();
        let deadline =
            total_timeout.map(|_| std::time::Instant::now() + std::time::Duration::from_secs(30));
        let content_encoding = parts
            .headers
            .get("content-encoding")
            .and_then(|h| h.to_str().ok());
        let decoder = Decoder::detect(content_encoding);
        let response_body = response(body, total_timeout, read_timeout);
        let res = hyper::Response::from_parts(parts, decoder);

        Response {
            res,
            url: Box::new(url),
            #[cfg(feature = "cookies")]
            cookie_jar: None,
        }
    }
}
