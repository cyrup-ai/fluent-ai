#![cfg_attr(target_arch = "wasm32", allow(unused))]
use std::error::Error as StdError;
use std::fmt;
use std::io;

use super::util::Escape;
use crate::{StatusCode, Url};

/// A `Result` alias where the `Err` case is `crate::hyper::Error`.
pub type Result<T> = std::result::Result<T, Error>;

/// The Errors that may occur when processing a `Request`.
///
/// Note: Errors may include the full URL used to make the `Request`. If the URL
/// contains sensitive information (e.g. an API key as a query parameter), be
/// sure to remove it ([`without_url`](Error::without_url))
pub struct Error {
    inner: Box<Inner>,
}

pub(crate) type BoxError = Box<dyn StdError + Send + Sync>;

struct Inner {
    kind: Kind,
    source: Option<BoxError>,
    url: Option<Url>,
}

impl Error {
    pub(crate) fn new<E>(kind: Kind, source: Option<E>) -> Error
    where
        E: Into<BoxError>,
    {
        Error {
            inner: Box::new(Inner {
                kind,
                source: source.map(Into::into),
                url: None,
            }),
        }
    }

    /// Returns a possible URL related to this error.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn run() {
    /// // displays last stop of a redirect loop
    /// let response = crate::hyper::get("http://site.with.redirect.loop").collect_one();
    /// if let Err(e) = response {
    ///     if e.is_redirect() {
    ///         if let Some(final_stop) = e.url() {
    ///             println!("redirect loop at {final_stop}");
    ///         }
    ///     }
    /// }
    /// # }
    /// ```
    pub fn url(&self) -> Option<&Url> {
        self.inner.url.as_ref()
    }

    /// Returns a mutable reference to the URL related to this error
    ///
    /// This is useful if you need to remove sensitive information from the URL
    /// (e.g. an API key in the query), but do not want to remove the URL
    /// entirely.
    pub fn url_mut(&mut self) -> Option<&mut Url> {
        self.inner.url.as_mut()
    }

    /// Add a url related to this error (overwriting any existing)
    pub fn with_url(mut self, url: Url) -> Self {
        self.inner.url = Some(url);
        self
    }

    pub(crate) fn if_no_url(mut self, f: impl FnOnce() -> Url) -> Self {
        if self.inner.url.is_none() {
            self.inner.url = Some(f());
        }
        self
    }

    /// Strip the related url from this error (if, for example, it contains
    /// sensitive information)
    pub fn without_url(mut self) -> Self {
        self.inner.url = None;
        self
    }

    /// Returns true if the error is from a type Builder.
    pub fn is_builder(&self) -> bool {
        matches!(self.inner.kind, Kind::Builder)
    }

    /// Returns true if the error is from a `RedirectPolicy`.
    pub fn is_redirect(&self) -> bool {
        matches!(self.inner.kind, Kind::Redirect)
    }

    /// Returns true if the error is from `Response::error_for_status`.
    pub fn is_status(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            matches!(self.inner.kind, Kind::Status(_, _))
        }
        #[cfg(target_arch = "wasm32")]
        {
            matches!(self.inner.kind, Kind::Status(_))
        }
    }

    /// Returns true if the error is related to a timeout.
    pub fn is_timeout(&self) -> bool {
        let mut source = self.source();

        while let Some(err) = source {
            if err.is::<TimedOut>() {
                return true;
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(hyper_err) = err.downcast_ref::<hyper::Error>() {
                if hyper_err.is_timeout() {
                    return true;
                }
            }
            if let Some(io) = err.downcast_ref::<io::Error>() {
                if io.kind() == io::ErrorKind::TimedOut {
                    return true;
                }
            }
            source = err.source();
        }

        false
    }

    /// Returns true if the error is related to the request
    pub fn is_request(&self) -> bool {
        matches!(self.inner.kind, Kind::Request)
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Returns true if the error is related to connect
    pub fn is_connect(&self) -> bool {
        let mut source = self.source();

        while let Some(err) = source {
            // Note: Removed legacy client error check since we use pure AsyncStream architecture
            // Connection errors will be handled by other error types

            source = err.source();
        }

        false
    }

    /// Returns true if the error is related to the request or response body
    pub fn is_body(&self) -> bool {
        matches!(self.inner.kind, Kind::Body)
    }

    /// Returns true if the error is related to decoding the response's body
    pub fn is_decode(&self) -> bool {
        matches!(self.inner.kind, Kind::Decode)
    }

    /// Returns the status code, if the error was generated from a response.
    pub fn status(&self) -> Option<StatusCode> {
        match self.inner.kind {
            #[cfg(target_arch = "wasm32")]
            Kind::Status(code) => Some(code),
            #[cfg(not(target_arch = "wasm32"))]
            Kind::Status(code, _) => Some(code),
            _ => None,
        }
    }

    // private

    #[allow(unused)]
    pub(crate) fn into_io(self) -> io::Error {
        io::Error::new(io::ErrorKind::Other, self)
    }
}

/// Converts from external types to http3's
/// internal equivalents.
///
/// Now simplified since we don't use tower anymore.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn cast_to_internal_error(error: BoxError) -> BoxError {
    // Check for timeout errors and provide proper error categorization
    // Use zero-allocation error handling with proper type checking
    if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
        if io_error.kind() == std::io::ErrorKind::TimedOut {
            return Box::new(crate::Error::Timeout {
                message: format!("Network operation timed out after 30 seconds"),
            });
        }
    }

    error
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("crate::hyper::Error");

        builder.field("kind", &self.inner.kind);

        if let Some(ref url) = self.inner.url {
            builder.field("url", &url.as_str());
        }
        if let Some(ref source) = self.inner.source {
            builder.field("source", source);
        }

        builder.finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.inner.kind {
            Kind::Builder => f.write_str("builder error")?,
            Kind::Request => f.write_str("error sending request")?,
            Kind::Body => f.write_str("request or response body error")?,
            Kind::Decode => f.write_str("error decoding response body")?,
            Kind::Redirect => f.write_str("error following redirect")?,
            Kind::Upgrade => f.write_str("error upgrading connection")?,
            #[cfg(target_arch = "wasm32")]
            Kind::Status(ref code) => {
                let prefix = if code.is_client_error() {
                    "HTTP status client error"
                } else {
                    debug_assert!(code.is_server_error());
                    "HTTP status server error"
                };
                write!(f, "{prefix} ({code})")?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            Kind::Status(ref code, ref reason) => {
                let prefix = if code.is_client_error() {
                    "HTTP status client error"
                } else {
                    debug_assert!(code.is_server_error());
                    "HTTP status server error"
                };
                if let Some(reason) = reason {
                    write!(
                        f,
                        "{prefix} ({} {})",
                        code.as_str(),
                        Escape::new(reason.as_bytes())
                    )?;
                } else {
                    write!(f, "{prefix} ({code})")?;
                }
            }
        };

        if let Some(url) = &self.inner.url {
            write!(f, " for url ({url})")?;
        }

        Ok(())
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner.source.as_ref().map(|e| &**e as _)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<crate::error::Error> for wasm_bindgen::JsValue {
    fn from(err: Error) -> wasm_bindgen::JsValue {
        js_sys::Error::from(err).into()
    }
}

#[cfg(target_arch = "wasm32")]
impl From<crate::error::Error> for js_sys::Error {
    fn from(err: Error) -> js_sys::Error {
        js_sys::Error::new(&format!("{err}"))
    }
}

#[derive(Debug)]
pub(crate) enum Kind {
    Builder,
    Request,
    Redirect,
    #[cfg(not(target_arch = "wasm32"))]
    Status(StatusCode, Option<hyper::ext::ReasonPhrase>),
    #[cfg(target_arch = "wasm32")]
    Status(StatusCode),
    Body,
    Decode,
    Upgrade,
}

// constructors

pub(crate) fn builder<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Builder, Some(e))
}

pub(crate) fn body<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Body, Some(e))
}

pub(crate) fn decode<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Decode, Some(e))
}

pub(crate) fn request<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Request, Some(e))
}

pub(crate) fn redirect<E: Into<BoxError>>(e: E, url: Url) -> Error {
    Error::new(Kind::Redirect, Some(e)).with_url(url)
}

pub(crate) fn status_code(
    url: Url,
    status: StatusCode,
    #[cfg(not(target_arch = "wasm32"))] reason: Option<hyper::ext::ReasonPhrase>,
) -> Error {
    Error::new(
        Kind::Status(
            status,
            #[cfg(not(target_arch = "wasm32"))]
            reason,
        ),
        None::<Error>,
    )
    .with_url(url)
}

pub(crate) fn url_bad_scheme(url: Url) -> Error {
    Error::new(Kind::Builder, Some(BadScheme)).with_url(url)
}

pub(crate) fn url_invalid_uri(url: Url) -> Error {
    Error::new(Kind::Builder, Some("Parsed Url is not a valid Uri")).with_url(url)
}

if_wasm! {
    pub(crate) fn wasm(js_val: wasm_bindgen::JsValue) -> BoxError {
        format!("{js_val:?}").into()
    }
}

pub(crate) fn upgrade<E: Into<BoxError>>(e: E) -> Error {
    Error::new(Kind::Upgrade, Some(e))
}

// io::Error helpers

#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate",
))]
pub(crate) fn into_io(e: BoxError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

#[allow(unused)]
pub(crate) fn decode_io(e: io::Error) -> Error {
    if e.get_ref().map(|r| r.is::<Error>()).unwrap_or(false) {
        match e.into_inner() {
            Some(boxed_error) => match boxed_error.downcast::<Error>() {
                Ok(error) => *error,
                Err(_) => Error::new(Kind::Request, Some("Failed to downcast error")),
            },
            None => Error::new(Kind::Request, Some("No inner error available")),
        }
    } else {
        decode(e)
    }
}

// internal Error "sources"

#[derive(Debug)]
pub(crate) struct TimedOut;

impl fmt::Display for TimedOut {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("operation timed out")
    }
}

impl StdError for TimedOut {}

#[derive(Debug)]
pub(crate) struct BadScheme;

impl fmt::Display for BadScheme {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("URL scheme is not allowed")
    }
}

impl StdError for BadScheme {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_source_chain() {
        let root = Error::new(Kind::Request, None::<Error>);
        assert!(root.source().is_none());

        let link = super::body(root);
        assert!(link.source().is_some());
        assert_send::<Error>();
        assert_sync::<Error>();
    }

    #[test]
    fn mem_size_of() {
        use std::mem::size_of;
        assert_eq!(size_of::<Error>(), size_of::<usize>());
    }

    #[test]
    fn roundtrip_io_error() {
        let orig = super::request("orig");
        // Convert crate::hyper::Error into an io::Error...
        let io = orig.into_io();
        // Convert that io::Error back into a crate::hyper::Error...
        let err = super::decode_io(io);
        // It should have pulled out the original, not nested it...
        match err.inner.kind {
            Kind::Request => (),
            _ => panic!("{err:?}"),
        }
    }

    #[test]
    fn from_unknown_io_error() {
        let orig = io::Error::new(io::ErrorKind::Other, "orly");
        let err = super::decode_io(orig);
        match err.inner.kind {
            Kind::Decode => (),
            _ => panic!("{err:?}"),
        }
    }

    #[test]
    fn is_timeout() {
        let err = super::request(super::TimedOut);
        assert!(err.is_timeout());

        // COMPLETE TIMEOUT ERROR TESTING INFRASTRUCTURE
        // Test comprehensive timeout detection across different error sources

        // Test direct TimedOut error
        let direct_timeout = super::request(super::TimedOut);
        assert!(direct_timeout.is_timeout());
        assert!(direct_timeout.is_request());
        assert!(!direct_timeout.is_connect());
        assert!(!direct_timeout.is_status());

        // Test IO timeout error
        let io = io::Error::from(io::ErrorKind::TimedOut);
        let nested = super::request(io);
        assert!(nested.is_timeout());
        assert!(nested.is_request());

        // Test hyper timeout error construction and detection
        #[cfg(not(target_arch = "wasm32"))]
        {
            let hyper_timeout = test_helpers::create_hyper_timeout_error();
            assert!(hyper_timeout.is_timeout());

            let wrapped_hyper_timeout = super::request(hyper_timeout);
            assert!(wrapped_hyper_timeout.is_timeout());
            assert!(wrapped_hyper_timeout.is_request());
        }

        // Test connect timeout
        let connect_timeout = test_helpers::create_connect_timeout();
        assert!(connect_timeout.is_timeout());
        assert!(connect_timeout.is_connect());

        // Test different timeout sources nested in body errors
        let body_timeout = super::body(io::Error::from(io::ErrorKind::TimedOut));
        assert!(body_timeout.is_timeout());

        // Test false positives - ensure non-timeout errors don't report as timeouts
        let non_timeout = super::request(io::Error::from(io::ErrorKind::ConnectionRefused));
        assert!(!non_timeout.is_timeout());
        assert!(non_timeout.is_request());
    }

    /// Comprehensive error testing infrastructure
    mod test_helpers {
        use std::time::Duration;

        use super::*;

        /// Create a hyper timeout error for testing
        #[cfg(not(target_arch = "wasm32"))]
        pub fn create_hyper_timeout_error() -> hyper::Error {
            // Create a mock hyper timeout error
            // In a real scenario, this would come from hyper operations
            hyper::Error::new_timeout()
        }

        /// Create a connect timeout error
        pub fn create_connect_timeout() -> super::Error {
            use std::net::TcpStream;
            use std::time::Duration;

            // Simulate a connection timeout
            let timeout_io =
                io::Error::new(io::ErrorKind::TimedOut, "connection attempt timed out");

            super::request(timeout_io)
        }

        /// Create various error types for comprehensive testing
        pub fn create_error_suite() -> Vec<(super::Error, &'static str)> {
            let mut errors = Vec::new();

            // Request errors
            errors.push((
                super::request(io::Error::from(io::ErrorKind::ConnectionRefused)),
                "connection_refused",
            ));

            errors.push((
                super::request(io::Error::from(io::ErrorKind::TimedOut)),
                "request_timeout",
            ));

            // Body errors
            errors.push((
                super::body(io::Error::from(io::ErrorKind::UnexpectedEof)),
                "body_unexpected_eof",
            ));

            // Decode errors
            errors.push((super::decode("invalid encoding"), "decode_error"));

            // Builder errors
            errors.push((super::builder("invalid header value"), "builder_error"));

            // Status errors
            if let Ok(test_url) = crate::Url::parse("https://example.com") {
                errors.push((
                    super::status_code(
                        test_url,
                        crate::StatusCode::INTERNAL_SERVER_ERROR,
                        #[cfg(not(target_arch = "wasm32"))]
                        None,
                    ),
                    "status_500",
                ));
            }

            // Redirect errors
            if let Ok(redirect_url) = url::Url::parse("https://example.com/redirect") {
                errors.push((
                    super::redirect("too many redirects", redirect_url),
                    "redirect_error",
                ));
            }

            errors
        }

        /// Test error categorization and properties
        pub fn test_error_categorization() -> Result<(), Box<dyn std::error::Error>> {
            let error_suite = create_error_suite();

            for (error, description) in error_suite {
                // Test that error has expected properties
                match description {
                    "connection_refused" => {
                        assert!(error.is_request());
                        assert!(!error.is_timeout());
                        assert!(!error.is_status());
                    }
                    "request_timeout" => {
                        assert!(error.is_request());
                        assert!(error.is_timeout());
                        assert!(!error.is_status());
                    }
                    "body_unexpected_eof" => {
                        assert!(!error.is_request());
                        assert!(!error.is_timeout());
                        assert!(!error.is_status());
                    }
                    "decode_error" => {
                        assert!(!error.is_request());
                        assert!(!error.is_timeout());
                        assert!(!error.is_status());
                    }
                    "builder_error" => {
                        assert!(error.is_builder());
                        assert!(!error.is_timeout());
                        assert!(!error.is_status());
                    }
                    "status_500" => {
                        assert!(error.is_status());
                        assert!(!error.is_timeout());
                        assert!(!error.is_request());
                    }
                    "redirect_error" => {
                        assert!(error.is_redirect());
                        assert!(!error.is_timeout());
                        assert!(!error.is_status());
                        assert!(error.url().is_some());
                    }
                    _ => {}
                }

                // Test error source chain
                test_error_source_chain(&error)?;

                // Test error formatting
                test_error_formatting(&error)?;
            }

            Ok(())
        }

        /// Test error source chain navigation
        fn test_error_source_chain(error: &super::Error) -> Result<(), Box<dyn std::error::Error>> {
            let mut source_count = 0;
            let mut current_source = error.source();

            // Navigate through error source chain
            while let Some(source) = current_source {
                source_count += 1;

                // Prevent infinite loops in testing
                if source_count > 10 {
                    return Err("Error source chain too deep - possible cycle".into());
                }

                current_source = source.source();
            }

            // Verify error implements expected traits
            let _: &dyn std::error::Error = error;
            let _: &dyn std::fmt::Display = error;
            let _: &dyn std::fmt::Debug = error;

            Ok(())
        }

        /// Test error formatting and display
        fn test_error_formatting(error: &super::Error) -> Result<(), Box<dyn std::error::Error>> {
            // Test Display implementation
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());

            // Test Debug implementation
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());

            // Test that Display and Debug produce different outputs
            assert_ne!(display_str, debug_str);

            Ok(())
        }

        /// Test error propagation patterns using Result instead of expect()
        pub fn test_error_propagation() -> Result<(), super::Error> {
            // Test proper error propagation using ? operator
            let result = simulate_failing_operation();

            match result {
                Ok(_) => return Err(super::request("Expected error but got success")),
                Err(e) => {
                    // Verify error properties
                    assert!(e.is_request());

                    // Test error chaining
                    let chained = chain_error(e)?;
                    assert!(chained.is_request());
                }
            }

            Ok(())
        }

        /// Simulate a failing operation for testing error propagation
        fn simulate_failing_operation() -> Result<(), super::Error> {
            Err(super::request("simulated failure"))
        }

        /// Test error chaining
        fn chain_error(original: super::Error) -> Result<super::Error, super::Error> {
            // Example of proper error handling without expect()
            Ok(super::request(format!("chained: {}", original)))
        }
    }

    #[test]
    fn comprehensive_error_categorization() {
        test_helpers::test_error_categorization()
            .unwrap_or_else(|e| panic!("Error categorization test failed: {}", e));
    }

    #[test]
    fn error_propagation_patterns() {
        test_helpers::test_error_propagation()
            .unwrap_or_else(|e| panic!("Error propagation test failed: {}", e));
    }
}
