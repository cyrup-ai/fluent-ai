use super::types::{Error, Kind};

pub(crate) type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Creates an `Error` for a builder error.
pub fn builder<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Builder).with(e.into())
}

/// Creates an `Error` for a request error.
pub fn request<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Request).with(e.into())
}

/// Creates an `Error` for a redirect error.
pub fn redirect<E: Into<BoxError>>(e: E, url: crate::Url) -> Error {
    Error::new(Kind::Redirect).with(e.into()).with_url(url)
}

/// Creates an `Error` for a body error.
pub fn body<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Body).with(e.into())
}

/// Creates an `Error` for a decode error.
pub fn decode<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Decode).with(e.into())
}

/// Creates an `Error` for an upgrade error.
pub fn upgrade<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Upgrade).with(e.into())
}

// Additional constructors needed by other modules
pub fn url_invalid_uri(url: crate::Url) -> Error {
    Error::new(Kind::Builder).with(super::helpers::BadScheme).with_url(url)
}

pub fn url_bad_scheme(url: crate::Url) -> Error {
    Error::new(Kind::Builder).with(super::helpers::BadScheme).with_url(url)
}

pub fn status_code(
    url: crate::Url,
    status: crate::StatusCode,
    #[cfg(not(target_arch = "wasm32"))] reason: Option<hyper::ext::ReasonPhrase>,
) -> Error {
    Error::new(
        Kind::Status(
            status,
            #[cfg(not(target_arch = "wasm32"))]
            reason,
        )
    ).with_url(url)
}
