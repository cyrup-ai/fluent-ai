use std::fmt;

use bytes::Bytes;
use http::{HeaderMap, StatusCode};
use js_sys::Uint8Array;
use url::Url;
use fluent_ai_async::AsyncStream;
use wasm_bindgen_futures;

use crate::wasm::AbortGuard;

#[cfg(feature = "stream")]
use wasm_bindgen::JsCast;

#[cfg(feature = "stream")]
use futures_util::stream::{self, StreamExt};

#[cfg(feature = "json")]
use serde::de::DeserializeOwned;

/// A Response to a submitted `Request`.
pub struct Response {
    http: http::Response<web_sys::Response>,
    _abort: AbortGuard,
    // Boxed to save space (11 words to 1 word), and it's not accessed
    // frequently internally.
    url: Box<Url>,
}

impl Response {
    pub(super) fn new(
        res: http::Response<web_sys::Response>,
        url: Url,
        abort: AbortGuard,
    ) -> Response {
        Response {
            http: res,
            url: Box::new(url),
            _abort: abort,
        }
    }

    /// Get the `StatusCode` of this `Response`.
    #[inline]
    pub fn status(&self) -> StatusCode {
        self.http.status()
    }

    /// Get the `Headers` of this `Response`.
    #[inline]
    pub fn headers(&self) -> &HeaderMap {
        self.http.headers()
    }

    /// Get a mutable reference to the `Headers` of this `Response`.
    #[inline]
    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        self.http.headers_mut()
    }

    /// Get the content-length of this response, if known.
    ///
    /// Reasons it may not be known:
    ///
    /// - The server didn't send a `content-length` header.
    /// - The response is compressed and automatically decoded (thus changing
    ///   the actual decoded length).
    pub fn content_length(&self) -> Option<u64> {
        self.headers()
            .get(http::header::CONTENT_LENGTH)?
            .to_str()
            .ok()?
            .parse()
            .ok()
    }

    /// Get the final `Url` of this `Response`.
    #[inline]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /* It might not be possible to detect this in JS?
    /// Get the HTTP `Version` of this `Response`.
    #[inline]
    pub fn version(&self) -> Version {
        self.http.version()
    }
    */

    /// Try to deserialize the response body as JSON.
    #[cfg(feature = "json")]
    #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
    pub fn json<T: DeserializeOwned>(self) -> AsyncStream<Result<T, crate::Error>> {
        AsyncStream::with_channel(move |sender| {
            wasm_bindgen_futures::spawn_local(async move {
                let web_response = self.http.into_body();
                let result = match web_response.array_buffer() {
                    Ok(promise) => {
                        match wasm_bindgen_futures::JsFuture::from(promise).await {
                            Ok(js_value) => {
                                let array_buffer = js_sys::ArrayBuffer::from(js_value);
                                let uint8_array = js_sys::Uint8Array::new(&array_buffer);
                                let mut bytes_vec = vec![0; uint8_array.length() as usize];
                                uint8_array.copy_to(&mut bytes_vec);
                                match serde_json::from_slice(&bytes_vec) {
                                    Ok(parsed) => Ok(parsed),
                                    Err(e) => Err(crate::error::decode(e)),
                                }
                            },
                            Err(js_error) => Err(crate::error::wasm(js_error)),
                        }
                    },
                    Err(js_error) => Err(crate::error::wasm(js_error)),
                };
                fluent_ai_async::emit!(sender, result);
            });
        })
    }

    /// Get the response text.
    pub fn text(self) -> AsyncStream<Result<String, crate::Error>> {
        AsyncStream::with_channel(move |sender| {
            wasm_bindgen_futures::spawn_local(async move {
                let web_response = self.http.into_body();
                let result = match web_response.text() {
                    Ok(promise) => {
                        match wasm_bindgen_futures::JsFuture::from(promise).await {
                            Ok(js_value) => {
                                match js_value.as_string() {
                                    Some(text) => Ok(text),
                                    None => Err(crate::error::decode("Response text is not a string")),
                                }
                            },
                            Err(js_error) => Err(crate::error::wasm(js_error)),
                        }
                    },
                    Err(js_error) => Err(crate::error::wasm(js_error)),
                };
                fluent_ai_async::emit!(sender, result);
            });
        })
    }

    /// Get the response as bytes
    pub fn bytes(self) -> AsyncStream<Result<Bytes, crate::Error>> {
        AsyncStream::with_channel(move |sender| {
            wasm_bindgen_futures::spawn_local(async move {
                let web_response = self.http.into_body();
                let result = match web_response.array_buffer() {
                    Ok(promise) => {
                        match wasm_bindgen_futures::JsFuture::from(promise).await {
                            Ok(js_value) => {
                                let array_buffer = js_sys::ArrayBuffer::from(js_value);
                                let uint8_array = js_sys::Uint8Array::new(&array_buffer);
                                let mut bytes_vec = vec![0; uint8_array.length() as usize];
                                uint8_array.copy_to(&mut bytes_vec);
                                Ok(Bytes::from(bytes_vec))
                            },
                            Err(js_error) => Err(crate::error::wasm(js_error)),
                        }
                    },
                    Err(js_error) => Err(crate::error::wasm(js_error)),
                };
                fluent_ai_async::emit!(sender, result);
            });
        })
    }

    /// Convert the response into a `Stream` of `Bytes` from the body.
    #[cfg(feature = "stream")]
    pub fn bytes_stream(self) -> fluent_ai_async::AsyncStream<Bytes> {
        use fluent_ai_async::{AsyncStream, handle_error};

        let web_response = self.http.into_body();
        let _abort = self._abort;

        AsyncStream::with_channel(move |sender| {
            if let Some(body) = web_response.body() {
                let body = wasm_streams::ReadableStream::from_raw(body.unchecked_into());
                
                // Production WASM stream handling using fluent_ai_async patterns
                spawn_task(move || async move {
                    use futures_util::StreamExt;
                    let mut stream = body.into_stream();
                    
                    while let Some(buf_js) = stream.next().await {
                        match buf_js {
                            Ok(js_value) => {
                                let buffer = Uint8Array::new(&js_value);
                                let mut bytes = vec![0; buffer.length() as usize];
                                buffer.copy_to(&mut bytes);
                                let chunk = Bytes::from(bytes);
                                emit!(sender, crate::response::HttpResponseChunk::from_bytes(chunk));
                            }
                            Err(js_error) => {
                                let error_msg = format!("WASM stream read error: {:?}", js_error);
                                emit!(sender, crate::response::HttpResponseChunk::bad_chunk(error_msg));
                                return;
                            }
                });
            } else {
                // No body available - emit empty response chunk
                emit!(sender, crate::response::HttpResponseChunk::empty());
            }
        })
    }

    // util methods

    /// Turn a response into an error if the server returned an error.
    pub fn error_for_status(self) -> crate::Result<Self> {
        let status = self.status();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::error::status_code(*self.url, status))
        } else {
            Ok(self)
        }
    }

    /// Turn a reference to a response into an error if the server returned an error.
    pub fn error_for_status_ref(&self) -> crate::Result<&Self> {
        let status = self.status();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::error::status_code(*self.url.clone(), status))
        } else {
            Ok(self)
        }
    }
}

impl fmt::Debug for Response {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Response")
            //.field("url", self.url())
            .field("status", &self.status())
            .field("headers", self.headers())
            .finish()
    }
}
