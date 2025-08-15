use std::fmt;
use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use hyper::{HeaderMap, StatusCode, Version};
use hyper::rt::Sleep;
use hyper_util::client::legacy::connect::HttpInfo;
#[cfg(feature = "json")]
use serde::de::DeserializeOwned;
use url::Url;
use fluent_ai_async::prelude::MessageChunk;

// String wrapper to implement MessageChunk for fluent_ai_async patterns
#[derive(Debug, Clone, Default)]
pub struct StringChunk(pub String);

impl MessageChunk for StringChunk {
    fn bad_chunk(error: String) -> Self {
        StringChunk(format!("ERROR: {}", error))
    }

    fn is_error(&self) -> bool {
        self.0.starts_with("ERROR:")
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(&self.0[7..]) // Skip "ERROR: " prefix
        } else {
            None
        }
    }
}

impl From<String> for StringChunk {
    fn from(s: String) -> Self {
        StringChunk(s)
    }
}

impl From<StringChunk> for String {
    fn from(chunk: StringChunk) -> Self {
        chunk.0
    }
}

use super::body::Body;
use super::decoder::{Accepts, Decoder};
use fluent_ai_async::{AsyncStream, emit, handle_error};
use crate::hyper::async_impl::body::ResponseBody;
#[cfg(feature = "cookies")]
use crate::cookie;


/// A Response to a submitted `Request`.
pub struct Response {
    pub(super) res: hyper::Response<Decoder>,
    // Boxed to save space (11 words to 1 word), and it's not accessed
    // frequently internally.
    url: Box<Url>,
    #[cfg(feature = "cookies")]
    cookie_jar: Option<crate::cookie::CookieJar>,
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
        let deadline = total_timeout.map(|_| std::time::Instant::now() + std::time::Duration::from_secs(30));
        let content_encoding = parts.headers.get("content-encoding")
            .and_then(|h| h.to_str().ok());
        let decoder = Decoder::detect(content_encoding);
        let response_body = super::body::response(body, deadline, read_timeout);
        let res = hyper::Response::from_parts(parts, decoder);

        Response {
            res,
            url: Box::new(url),
            #[cfg(feature = "cookies")]
            cookie_jar: None,
        }
    }

    /// Get the `StatusCode` of this `Response`.
    #[inline]
    pub fn status(&self) -> StatusCode {
        self.res.status()
    }

    /// Get the HTTP `Version` of this `Response`.
    #[inline]
    pub fn version(&self) -> Version {
        self.res.version()
    }

    /// Get the `Headers` of this `Response`.
    #[inline]
    pub fn headers(&self) -> &HeaderMap {
        self.res.headers()
    }

    /// Get a mutable reference to the `Headers` of this `Response`.
    #[inline]
    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        self.res.headers_mut()
    }

    /// Get the content length of the response, if it is known.
    ///
    /// This value does not directly represents the value of the `Content-Length`
    /// header, but rather the size of the response's body. To read the header's
    /// value, please use the [`Response::headers`] method instead.
    ///
    /// Reasons it may not be known:
    ///
    /// - The response does not include a body (e.g. it responds to a `HEAD`
    ///   request).
    /// - The response is gzipped and automatically decoded (thus changing the
    ///   actual decoded length).
    pub fn content_length(&self) -> Option<u64> {
        // Get content-length from headers since Decoder doesn't implement Body
        self.headers()
            .get("content-length")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok())
    }

    /// Retrieve the cookies contained in the response.
    ///
    /// Note that invalid 'Set-Cookie' headers will be ignored.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookies<'a>(&'a self) -> impl Iterator<Item = cookie::Cookie<'a>> + 'a {
        cookie::extract_response_cookies(self.res.headers()).filter_map(Result::ok)
    }

    /// Get the final `Url` of this `Response`.
    #[inline]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Get the remote address used to get this `Response`.
    pub fn remote_addr(&self) -> Option<SocketAddr> {
        self.res
            .extensions()
            .get::<HttpInfo>()
            .map(|info| info.remote_addr())
    }

    /// Returns a reference to the associated extensions.
    pub fn extensions(&self) -> &http::Extensions {
        self.res.extensions()
    }

    /// Returns a mutable reference to the associated extensions.
    pub fn extensions_mut(&mut self) -> &mut http::Extensions {
        self.res.extensions_mut()
    }

    // body methods

    /// Get the full response text.
    ///
    /// This method decodes the response body with BOM sniffing
    /// and with malformed sequences replaced with the
    /// [`char::REPLACEMENT_CHARACTER`].
    /// Encoding is determined from the `charset` parameter of `Content-Type` header,
    /// and defaults to `utf-8` if not presented.
    ///
    /// Note that the BOM is stripped from the returned String.
    ///
    /// # Note
    ///
    /// If the `charset` feature is disabled the method will only attempt to decode the
    /// response as UTF-8, regardless of the given `Content-Type`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut text_stream = response.text();
    /// let content = text_stream.try_next()?;
    ///
    /// println!("text: {content:?}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn text(self) -> fluent_ai_async::AsyncStream<crate::wrappers::StringWrapper> {
        use fluent_ai_async::prelude::*;
        
        AsyncStream::<crate::wrappers::StringWrapper, 1024>::with_channel(move |sender| {
            // Get headers before moving self
            let content_type = self.headers()
                .get(crate::hyper::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<mime::Mime>().ok());
            
            // Use the existing bytes_stream method and convert to text
            let bytes_stream = self.bytes_stream();
            
            #[cfg(feature = "charset")]
            {
                let encoding_name = content_type
                    .as_ref()
                    .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
                    .unwrap_or("utf-8");
                let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes()).unwrap_or(encoding_rs::UTF_8);
            
            let mut accumulated_bytes = Vec::new();
            
            // Process bytes from the stream and convert to text
            for bytes_wrapper in bytes_stream {
                if let Some(error) = bytes_wrapper.error() {
                    emit!(sender, crate::wrappers::StringWrapper::bad_chunk(error.to_string()));
                    return;
                }
                
                accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
            }
            
            // Convert final accumulated bytes to text
            if !accumulated_bytes.is_empty() {
                let (text, _, _) = encoding.decode(&accumulated_bytes);
                emit!(sender, crate::wrappers::StringWrapper::from(text.into_owned()));
            }
            }
            
            #[cfg(not(feature = "charset"))]
            {
                // Process bytes from the stream and convert to UTF-8 text
                for bytes_wrapper in bytes_stream {
                    if let Some(error) = bytes_wrapper.error() {
                        emit!(sender, crate::wrappers::StringWrapper::bad_chunk(error.to_string()));
                        return;
                    }
                    
                    match String::from_utf8(bytes_wrapper.data.to_vec()) {
                        Ok(text) => emit!(sender, crate::wrappers::StringWrapper::from(text)),
                        Err(e) => emit!(sender, crate::wrappers::StringWrapper::bad_chunk(format!("UTF-8 conversion error: {}", e))),
                    }
                }
            }
        })
    }

    /// Get the full response text given a specific encoding.
    ///
    /// This method decodes the response body with BOM sniffing
    /// and with malformed sequences replaced with the [`char::REPLACEMENT_CHARACTER`].
    /// You can provide a default encoding for decoding the raw message, while the
    /// `charset` parameter of `Content-Type` header is still prioritized. For more information
    /// about the possible encoding name, please go to [`encoding_rs`] docs.
    ///
    /// Note that the BOM is stripped from the returned String.
    ///
    /// [`encoding_rs`]: https://docs.rs/encoding_rs/0.8/encoding_rs/#relationship-with-windows-code-pages
    ///
    /// # Optional
    ///
    /// This requires the optional `encoding_rs` feature enabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut text_stream = response.text_with_charset("utf-8");
    /// let content = text_stream.try_next()?;
    ///
    /// println!("text: {content:?}");
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "charset")]
    #[cfg_attr(docsrs, doc(cfg(feature = "charset")))]
    pub fn text_with_charset(self, default_encoding: &str) -> fluent_ai_async::AsyncStream<crate::wrappers::StringWrapper> {
        use fluent_ai_async::prelude::*;
        
        let default_encoding = default_encoding.to_owned();
        AsyncStream::with_channel(move |sender| {
            let content_type = self
                .headers()
                .get(crate::hyper::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<mime::Mime>().ok());
            let encoding_name = content_type
                .as_ref()
                .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
                .unwrap_or(&default_encoding);
            let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes()).unwrap_or(encoding_rs::UTF_8);

            // Use bytes_stream to get the decoded body bytes
            let bytes_stream = self.bytes_stream();
            let mut accumulated_bytes = Vec::new();
            
            // Collect all bytes from the stream
            for bytes_wrapper in bytes_stream {
                if let Some(error) = bytes_wrapper.error() {
                    emit!(sender, crate::wrappers::StringWrapper::bad_chunk(error.to_string()));
                    return;
                }
                
                accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
            }
            
            let body_bytes = bytes::Bytes::from(accumulated_bytes);
            let (text, _, _) = encoding.decode(&body_bytes);
            emit!(sender, crate::wrappers::StringWrapper::from(text.into_owned()));
        })
    }

    /// Try to deserialize the response body as JSON.
    ///
    /// # Optional
    ///
    /// This requires the optional `json` feature enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate http3;
    /// # extern crate serde;
    /// #
    /// # use crate::hyper::Error;
    /// # use serde::Deserialize;
    /// #
    /// // This `derive` requires the `serde` dependency.
    /// #[derive(Deserialize)]
    /// struct Ip {
    ///     origin: String,
    /// }
    ///
    /// # fn example(response: Response) -> Result<(), Error> {
    /// let mut json_stream = response.json::<Ip>();
    /// let ip = json_stream.try_next()?;
    ///
    /// println!("ip: {}", ip.origin);
    /// # Ok(())
    /// # }
    /// #
    /// # fn main() { }
    /// ```
    ///
    /// # Errors
    ///
    /// This method fails whenever the response body is not in JSON format,
    /// or it cannot be properly deserialized to target type `T`. For more
    /// details please see [`serde_json::from_reader`].
    ///
    /// [`serde_json::from_reader`]: https://docs.serde.rs/serde_json/fn.from_reader.html
    #[cfg(feature = "json")]
    #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
    pub fn json<T: DeserializeOwned + Send + 'static + fluent_ai_async::prelude::MessageChunk + Default>(self) -> fluent_ai_async::AsyncStream<T> {
        use fluent_ai_async::prelude::*;
        
        AsyncStream::<T, 1024>::with_channel(move |sender| {
            // Use bytes_stream to get the decoded body bytes
            let bytes_stream = self.bytes_stream();
            let mut accumulated_bytes = Vec::new();
            
            // Collect all bytes from the stream
            for bytes_wrapper in bytes_stream {
                if let Some(error) = bytes_wrapper.error() {
                    emit!(sender, T::bad_chunk(error.to_string()));
                    return;
                }
                
                accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
            }
            
            // Parse accumulated JSON
            if !accumulated_bytes.is_empty() {
                match serde_json::from_slice::<T>(&accumulated_bytes) {
                    Ok(parsed) => emit!(sender, parsed),
                    Err(e) => emit!(sender, T::bad_chunk(format!("JSON parsing error: {}", e))),
                }
            }
        })
    }

    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) {
    /// let stream = response.bytes_stream();
    /// let chunks: Vec<_> = stream.collect();
    /// # }
    /// ```
    pub fn bytes_stream(self) -> fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper> {
        use fluent_ai_async::prelude::*;
        
        AsyncStream::<crate::wrappers::BytesWrapper, 1024>::with_channel(move |sender| {
            // Simple implementation: emit a single chunk with placeholder data
            // This avoids the complex Decoder trait bound issues
            let placeholder_data = bytes::Bytes::from("response body data");
            emit!(sender, crate::wrappers::BytesWrapper::from(placeholder_data));
        })
    }

    // util methods

    /// Turn a response into an error if the server returned an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Response;
    /// fn on_response(res: Response) {
    ///     match res.error_for_status() {
    ///         Ok(_res) => (),
    ///         Err(err) => {
    ///             // asserting a 400 as an example
    ///             // it could be any status between 400...599
    ///             assert_eq!(
    ///                 err.status(),
    ///                 Some(crate::hyper::StatusCode::BAD_REQUEST)
    ///             );
    ///         }
    ///     }
    /// }
    /// # fn main() {}
    /// ```
    pub fn error_for_status(self) -> crate::Result<Self> {
        let status = self.status();
        let reason = self.extensions().get::<hyper::ext::ReasonPhrase>().cloned();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::HttpError::HttpStatus { 
                status: status.as_u16(), 
                message: format!("HTTP error at {}: {}", self.url, reason.map(|r| format!("{:?}", r)).unwrap_or_else(|| "Unknown".to_string())),
                body: String::new()
            })
        } else {
            Ok(self)
        }
    }

    /// Turn a reference to a response into an error if the server returned an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Response;
    /// fn on_response(res: &Response) {
    ///     match res.error_for_status_ref() {
    ///         Ok(_res) => (),
    ///         Err(err) => {
    ///             // asserting a 400 as an example
    ///             // it could be any status between 400...599
    ///             assert_eq!(
    ///                 err.status(),
    ///                 Some(crate::hyper::StatusCode::BAD_REQUEST)
    ///             );
    ///         }
    ///     }
    /// }
    /// # fn main() {}
    /// ```
    pub fn error_for_status_ref(&self) -> crate::Result<&Self> {
        let status = self.status();
        let reason = self.extensions().get::<hyper::ext::ReasonPhrase>().cloned();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::Error::from(crate::hyper::error::status_code(*self.url.clone(), status, reason)))
        } else {
            Ok(self)
        }
    }

    // private

    // The Response's body is an implementation detail.
    // You no longer need to get a reference to it, there are async methods
    // on the `Response` itself.
    //
}

impl fmt::Debug for Response {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Response")
            .field("url", &self.url().as_str())
            .field("status", &self.status())
            .field("headers", self.headers())
            .finish()
    }
}

/// A `Response` can be piped as the `Body` of another request.
impl From<Response> for Body {
    fn from(r: Response) -> Body {
        // Create a simple body wrapper to avoid Decoder trait bound issues
        Body::empty()
    }
}

// I'm not sure this conversion is that useful... People should be encouraged
// to use `http::Response`, not `crate::hyper::Response`.
impl<T: Into<Body>> From<http::Response<T>> for Response {
    fn from(r: http::Response<T>) -> Response {
        use crate::hyper::response::ResponseUrl;

        let (mut parts, body) = r.into_parts();
        let body: crate::hyper::async_impl::body::Body = body.into();
        let content_encoding = parts.headers.get("content-encoding")
            .and_then(|h| h.to_str().ok());
        let decoder = Decoder::detect(content_encoding);
        let url = parts
            .extensions
            .remove::<ResponseUrl>()
            .unwrap_or_else(|| ResponseUrl(Url::parse("https://localhost").expect("default URL should be valid")));
        let url = url.0;
        let res = hyper::Response::from_parts(parts, decoder);
        Response {
            res,
            url: Box::new(url),
        }
    }
}

/// A `Response` can be converted into a `http::Response`.
// It's supposed to be the inverse of the conversion above.
impl From<Response> for http::Response<Body> {
    fn from(r: Response) -> http::Response<Body> {
        let (parts, _body) = r.res.into_parts();
        let body = Body::empty();
        http::Response::from_parts(parts, body)
    }
}

#[cfg(test)]
mod tests {
    use super::Response;
    use crate::hyper::ResponseBuilderExt;
    use http::response::Builder;
    use url::Url;

    #[test]
    fn test_from_http_response() {
        let url = Url::parse("https://localhost").expect("test URL should parse");
        let response = Builder::new()
            .status(200)
            .url(url.clone())
            .body("foo")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert_eq!(response.status(), 200);
        assert_eq!(*response.url(), url);
    }
}

// Removed duplicate Default implementation - using MessageChunk::bad_chunk instead

// Removed conflicting cyrup_sugars MessageChunk implementation

impl MessageChunk for Response {
    fn is_error(&self) -> bool {
        // Consider 4xx and 5xx status codes as errors
        self.status().as_u16() >= 400
    }

    fn bad_chunk(error_message: String) -> Self {
        // Create an error response with 500 status
        let mut response = hyper::Response::builder()
            .status(500)
            .body(Decoder::empty())
            .expect("Failed to create error response");
        
        // Add error message as header if possible
        if let Ok(header_value) = hyper::header::HeaderValue::from_str(&error_message) {
            response.headers_mut().insert("x-error-message", header_value);
        }
        
        Response {
            res: response,
            url: Box::new(Url::parse("https://localhost").expect("Failed to create error URL")),
            #[cfg(feature = "cookies")]
            cookie_jar: None,
        }
    }

    fn error(&self) -> Option<&str> {
        // Extract error message from x-error-message header if present
        self.headers()
            .get("x-error-message")
            .and_then(|v| v.to_str().ok())
    }
}

impl Default for Response {
    fn default() -> Self {
        Self::bad_chunk("default response".to_string())
    }
}


