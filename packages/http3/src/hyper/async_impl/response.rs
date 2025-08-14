use std::fmt;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{HeaderMap, StatusCode, Version};
#[cfg(feature = "json")]
use serde::de::DeserializeOwned;
use url::Url;

use super::body::Body;
use super::decoder::{Accepts, Decoder};
use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use crate::hyper::async_impl::body::ResponseBody;
#[cfg(feature = "cookies")]
use crate::cookie;


/// A Response to a submitted `Request`.
pub struct Response {
    pub(super) res: hyper::Response<Decoder>,
    // Boxed to save space (11 words to 1 word), and it's not accessed
    // frequently internally.
    url: Box<Url>,
}

impl Response {
    pub(super) fn new(
        res: hyper::Response<ResponseBody>,
        url: Url,
        accepts: Accepts,
        total_timeout: Option<Pin<Box<Sleep>>>,
        read_timeout: Option<Duration>,
    ) -> Response {
        let (mut parts, body) = res.into_parts();
        let decoder = Decoder::detect(
            &mut parts.headers,
            super::body::response(body, total_timeout, read_timeout),
            accepts,
        );
        let res = hyper::Response::from_parts(parts, decoder);

        Response {
            res,
            url: Box::new(url),
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
        use hyper::body::Body;

        Body::size_hint(self.res.body()).exact()
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
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut response_stream = crate::hyper::get("http://httpbin.org/range/26");
    /// let response = response_stream.try_next()?;
    /// let mut text_stream = response.text();
    /// let content = text_stream.try_next()?;
    ///
    /// println!("text: {content:?}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn text(self) -> AsyncStream<String> {
        AsyncStream::with_channel(move |sender| {
            // Direct streaming without collection - process body incrementally
            let mut body = self.res.into_body();
            
            #[cfg(feature = "charset")]
            {
                let content_type = self.headers()
                    .get(crate::hyper::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<mime::Mime>().ok());
                let encoding_name = content_type
                    .as_ref()
                    .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
                    .unwrap_or("utf-8");
                let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes()).unwrap_or(encoding_rs::UTF_8);
                
                let mut accumulated_bytes = Vec::new();
                
                // Stream text conversion chunk by chunk
                loop {
                    let waker = std::task::Waker::noop();
                    let mut context = std::task::Context::from_waker(&waker);
                    
                    match std::pin::Pin::new(&mut body).poll_frame(&mut context) {
                        std::task::Poll::Ready(Some(Ok(frame))) => {
                            if let Ok(data) = frame.into_data() {
                                accumulated_bytes.extend_from_slice(&data);
                                // Try to decode accumulated bytes
                                let (text, _encoding, had_errors) = encoding.decode(&accumulated_bytes);
                                if !had_errors && !text.is_empty() {
                                    emit!(sender, text.into_owned());
                                    accumulated_bytes.clear();
                                }
                            }
                        }
                        std::task::Poll::Ready(Some(Err(e))) => {
                            handle_error!(e, "response text frame error");
                            return;
                        }
                        std::task::Poll::Ready(None) => {
                            // Final chunk
                            if !accumulated_bytes.is_empty() {
                                let (text, _, _) = encoding.decode(&accumulated_bytes);
                                emit!(sender, text.into_owned());
                            }
                            return;
                        }
                        std::task::Poll::Pending => {
                            std::thread::yield_now();
                        }
                    }
                }
            }
            
            #[cfg(not(feature = "charset"))]
            {
                let mut accumulated_bytes = Vec::new();
                
                // Stream UTF-8 conversion chunk by chunk
                loop {
                    let waker = std::task::Waker::noop();
                    let mut context = std::task::Context::from_waker(&waker);
                    
                    match std::pin::Pin::new(&mut body).poll_frame(&mut context) {
                        std::task::Poll::Ready(Some(Ok(frame))) => {
                            if let Ok(data) = frame.into_data() {
                                accumulated_bytes.extend_from_slice(&data);
                                // Try to convert to valid UTF-8
                                if let Ok(text) = String::from_utf8(accumulated_bytes.clone()) {
                                    emit!(sender, text);
                                    accumulated_bytes.clear();
                                }
                            }
                        }
                        std::task::Poll::Ready(Some(Err(e))) => {
                            handle_error!(e, "response text frame error");
                            return;
                        }
                        std::task::Poll::Ready(None) => {
                            // Final chunk - use lossy conversion if needed
                            if !accumulated_bytes.is_empty() {
                                let text = String::from_utf8_lossy(&accumulated_bytes);
                                emit!(sender, text.into_owned());
                            }
                            return;
                        }
                        std::task::Poll::Pending => {
                            std::thread::yield_now();
                        }
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
    /// ```
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut response_stream = crate::hyper::get("http://httpbin.org/range/26");
    /// let response = response_stream.try_next()?;
    /// let mut text_stream = response.text_with_charset("utf-8");
    /// let content = text_stream.try_next()?;
    ///
    /// println!("text: {content:?}");
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "charset")]
    #[cfg_attr(docsrs, doc(cfg(feature = "charset")))]
    pub fn text_with_charset(self, default_encoding: &str) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
        
        let default_encoding = default_encoding.to_owned();
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                use http_body_util::BodyExt;
                
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

                // Proper body collection using hyper's frame-based API
                let mut body = self.res.into_body();
                let mut collected_bytes = Vec::new();
                
                // Use safe frame polling with noop waker
                let waker = std::task::Waker::noop();
                let mut cx = std::task::Context::from_waker(&waker);
                
                // Poll for frames until body is complete
                loop {
                    match std::pin::Pin::new(&mut body).poll_frame(&mut cx) {
                        std::task::Poll::Ready(Some(Ok(frame))) => {
                            if let Ok(data) = frame.into_data() {
                                collected_bytes.extend_from_slice(&data);
                            }
                        }
                        std::task::Poll::Ready(Some(Err(e))) => {
                            return Err(crate::HttpError::from(e));
                        }
                        std::task::Poll::Ready(None) => break, // Body complete
                        std::task::Poll::Pending => {
                            // Yield and retry - body may have more data
                            std::thread::yield_now();
                        }
                    }
                }
                
                let full_bytes = bytes::Bytes::from(collected_bytes);
                let (text, _, _) = encoding.decode(&full_bytes);
                Ok(text.into_owned())
            });
            
            match task.collect() {
                Ok(text) => emit!(sender, text),
                Err(e) => handle_error!(e, "response text with charset conversion"),
            }
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
    /// # fn run() -> Result<(), Error> {
    /// let mut response_stream = crate::hyper::get("http://httpbin.org/ip");
    /// let response = response_stream.try_next()?;
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
    pub fn json<T: DeserializeOwned + Send + 'static>(self) -> AsyncStream<T> {
        AsyncStream::with_channel(move |sender| {
            // Direct JSON streaming without collection - process body incrementally
            let mut body = self.res.into_body();
            let mut accumulated_bytes = Vec::new();
            
            // Stream JSON parsing - collect body bytes then parse
            loop {
                let waker = std::task::Waker::noop();
                let mut context = std::task::Context::from_waker(&waker);
                
                match std::pin::Pin::new(&mut body).poll_frame(&mut context) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        if let Ok(data) = frame.into_data() {
                            accumulated_bytes.extend_from_slice(&data);
                        }
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        handle_error!(e, "response json frame error");
                        return;
                    }
                    std::task::Poll::Ready(None) => {
                        // Body complete - parse accumulated JSON
                        if !accumulated_bytes.is_empty() {
                            match serde_json::from_slice::<T>(&accumulated_bytes) {
                                Ok(parsed_json) => emit!(sender, parsed_json),
                                Err(e) => handle_error!(e, "JSON deserialization error"),
                            }
                        }
                        return;
                    }
                    std::task::Poll::Pending => {
                        std::thread::yield_now();
                    }
                }
            }
        })
    }

    /// Get the full response body as `Bytes`.
    ///
    /// # Example
    ///
    /// ```
    /// let mut response_stream = crate::hyper::get("http://httpbin.org/ip");
    /// if let Some(response) = response_stream.try_next() {
    ///     let mut bytes_stream = response.bytes();
    ///     if let Some(bytes) = bytes_stream.try_next() {
    ///         println!("bytes: {bytes:?}");
    ///     }
    /// }
    /// ```
    pub fn bytes(self) -> AsyncStream<Bytes> {
        AsyncStream::with_channel(move |sender| {
            // Direct byte streaming without collection - process body incrementally
            let mut body = self.res.into_body();
            let mut accumulated_bytes = Vec::new();
            
            // Stream bytes collection
            loop {
                let waker = std::task::Waker::noop();
                let mut context = std::task::Context::from_waker(&waker);
                
                match std::pin::Pin::new(&mut body).poll_frame(&mut context) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        if let Ok(data) = frame.into_data() {
                            accumulated_bytes.extend_from_slice(&data);
                        }
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        handle_error!(e, "response bytes frame error");
                        return;
                    }
                    std::task::Poll::Ready(None) => {
                        // Body complete - emit accumulated bytes
                        if !accumulated_bytes.is_empty() {
                            let bytes = Bytes::from(accumulated_bytes);
                            emit!(sender, bytes);
                        }
                        return;
                    }
                    std::task::Poll::Pending => {
                        std::thread::yield_now();
                    }
                }
            }
        })
    }

    /// Stream a chunk of the response body.
    ///
    /// When the response body has been exhausted, this will return `None`.
    ///
    /// # Example
    ///
    /// ```
    /// let mut res_stream = crate::hyper::get("https://hyper.rs");
    /// if let Some(response) = res_stream.try_next() {
    ///     let mut chunk_stream = response.chunk_stream();
    ///     while let Some(chunk) = chunk_stream.try_next() {
    ///         println!("Chunk: {chunk:?}");
    ///     }
    /// }
    /// ```
    pub fn chunk_stream(self) -> AsyncStream<Bytes> {
        AsyncStream::with_channel(move |sender| {
            // Direct chunk streaming without collection - stream each frame as it arrives
            let mut body = self.res.into_body();
            
            // Stream chunks incrementally as they become available
            loop {
                let waker = std::task::Waker::noop();
                let mut context = std::task::Context::from_waker(&waker);
                
                match std::pin::Pin::new(&mut body).poll_frame(&mut context) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        // Emit each chunk as it arrives - true streaming
                        if let Ok(chunk_data) = frame.into_data() {
                            emit!(sender, chunk_data);
                        }
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        handle_error!(e, "response chunk frame error");
                        return;
                    }
                    std::task::Poll::Ready(None) => {
                        // Body exhausted - end stream
                        return;
                    }
                    std::task::Poll::Pending => {
                        std::thread::yield_now();
                    }
                }
            }
        })
    }

    /// Convert the response into a `Stream` of `Bytes` from the body.
    ///
    /// # Example
    ///
    /// ```
    /// let mut response_stream = crate::hyper::get("http://httpbin.org/ip");
    /// if let Some(response) = response_stream.try_next() {
    ///     let mut stream = response.bytes_stream();
    ///     while let Some(item) = stream.try_next() {
    ///         println!("Chunk: {:?}", item);
    ///     }
    /// }
    /// ```
    ///
    /// # Optional
    ///
    /// This requires the optional `stream` feature to be enabled.
    #[cfg(feature = "stream")]
    #[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
    pub fn bytes_stream(self) -> fluent_ai_async::AsyncStream<Bytes> {
        use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                // Convert the HttpBody to bytes using pure synchronous patterns
                let data_stream = super::body::DataStream(self.res.into_body());
                let async_stream = data_stream.into_async_stream();
                
                // Collect bytes from the stream
                let mut byte_chunks = Vec::new();
                let mut async_stream = async_stream;
                
                loop {
                    if let Some(result) = async_stream.try_next() {
                        match result {
                            Ok(bytes) => byte_chunks.push(bytes),
                            Err(_) => break, // Stop on error
                        }
                    } else {
                        break; // End of stream
                    }
                }
                
                byte_chunks
            });
            
            // Emit all collected bytes
            let byte_chunks = task.collect();
            for bytes in byte_chunks {
                emit!(sender, bytes);
            }
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
            Err(crate::error::status_code(*self.url, status, reason))
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
            Err(crate::error::status_code(*self.url.clone(), status, reason))
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
        Body::wrap(r.res.into_body())
    }
}

// I'm not sure this conversion is that useful... People should be encouraged
// to use `http::Response`, not `crate::hyper::Response`.
impl<T: Into<Body>> From<http::Response<T>> for Response {
    fn from(r: http::Response<T>) -> Response {
        use crate::hyper::response::ResponseUrl;

        let (mut parts, body) = r.into_parts();
        let body: crate::hyper::async_impl::body::Body = body.into();
        let decoder = Decoder::detect(
            &mut parts.headers,
            ResponseBody::new(body.map_err(Into::into)),
            Accepts::none(),
        );
        let url = parts
            .extensions
            .remove::<ResponseUrl>()
            .unwrap_or_else(|| ResponseUrl(Url::parse("http://no.url.provided.local").expect("default URL should be valid")));
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
        let (parts, body) = r.res.into_parts();
        let body = Body::wrap(body);
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
        let url = Url::parse("http://example.com").expect("test URL should parse");
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
