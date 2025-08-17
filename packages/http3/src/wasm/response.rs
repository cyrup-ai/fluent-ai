//! WASM-specific response types
//!
//! This module provides the CANONICAL WasmResponse implementation that consolidates
//! all WASM-specific response handling for browser environments.

use std::fmt;

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use http::{HeaderMap, StatusCode, Version};
use serde::de::DeserializeOwned;
use url::Url;

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, ArrayBuffer, Uint8Array};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys;

use crate::streaming::pipeline::StreamingResponse;
use crate::streaming::chunks::HttpChunk;

/// WASM-specific response wrapper that extends StreamingResponse with browser capabilities
/// 
/// This is the CANONICAL WasmResponse implementation that consolidates all
/// WASM-specific response handling into a single, comprehensive type.
pub struct WasmResponse {
    /// Core streaming response
    pub inner: StreamingResponse,
    
    /// WASM-specific data
    #[cfg(target_arch = "wasm32")]
    pub web_response: Option<web_sys::Response>,
    #[cfg(target_arch = "wasm32")]
    pub abort_controller: Option<web_sys::AbortController>,
    
    /// Response URL
    pub url: Url,
    
    /// WASM-specific metadata
    pub redirected: bool,
    pub response_type: WasmResponseType,
    
    /// Non-WASM fallback
    #[cfg(not(target_arch = "wasm32"))]
    pub web_response: Option<String>,
}

/// WASM response types
#[derive(Debug, Clone, Copy)]
pub enum WasmResponseType {
    Basic,
    Cors,
    Error,
    Opaque,
    OpaqueRedirect,
}

impl fmt::Debug for WasmResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmResponse")
            .field("status", &self.inner.status)
            .field("url", &self.url)
            .field("redirected", &self.redirected)
            .field("response_type", &self.response_type)
            .finish()
    }
}

impl WasmResponse {
    /// Create new WASM response from StreamingResponse
    #[inline]
    pub fn new(inner: StreamingResponse, url: Url) -> Self {
        Self {
            inner,
            #[cfg(target_arch = "wasm32")]
            web_response: None,
            #[cfg(target_arch = "wasm32")]
            abort_controller: None,
            url,
            redirected: false,
            response_type: WasmResponseType::Basic,
            #[cfg(not(target_arch = "wasm32"))]
            web_response: None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    /// Create WASM response from web_sys::Response
    pub fn from_web_response(
        web_response: web_sys::Response,
        url: Url,
        abort_controller: Option<web_sys::AbortController>,
    ) -> Result<Self, JsValue> {
        let status = StatusCode::from_u16(web_response.status())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        
        let mut headers = HeaderMap::new();
        
        // Extract headers from web response
        if let Ok(headers_iter) = web_response.headers().entries() {
            let mut iter = js_sys::try_iter(&headers_iter)?
                .ok_or_else(|| JsValue::from_str("Headers iterator failed"))?;
            
            while let Some(entry) = iter.next()? {
                if let Ok(array) = entry.dyn_into::<js_sys::Array>() {
                    if array.length() >= 2 {
                        if let (Some(key), Some(value)) = (array.get(0).as_string(), array.get(1).as_string()) {
                            if let (Ok(header_name), Ok(header_value)) = 
                                (key.parse::<http::HeaderName>(), value.parse::<http::HeaderValue>()) {
                                headers.insert(header_name, header_value);
                            }
                        }
                    }
                }
            }
        }

        // Create response chunks from web response body
        let response_chunks = Self::create_chunks_from_web_response(&web_response)?;
        
        let streaming_response = StreamingResponse::with_chunks(
            status,
            headers,
            Version::HTTP_11, // Default for WASM
            response_chunks,
        );

        let response_type = match web_response.type_() {
            web_sys::ResponseType::Basic => WasmResponseType::Basic,
            web_sys::ResponseType::Cors => WasmResponseType::Cors,
            web_sys::ResponseType::Error => WasmResponseType::Error,
            web_sys::ResponseType::Opaque => WasmResponseType::Opaque,
            web_sys::ResponseType::Opaqueredirect => WasmResponseType::OpaqueRedirect,
            _ => WasmResponseType::Basic,
        };

        Ok(Self {
            inner: streaming_response,
            web_response: Some(web_response),
            abort_controller,
            url,
            redirected: false, // TODO: Extract from web response
            response_type,
        })
    }

    #[cfg(target_arch = "wasm32")]
    /// Create response chunks from web_sys::Response
    fn create_chunks_from_web_response(
        web_response: &web_sys::Response,
    ) -> Result<AsyncStream<HttpResponseChunk, 1024>, JsValue> {
        let body = web_response.body();
        
        if let Some(readable_stream) = body {
            let reader = readable_stream.get_reader();
            
            Ok(AsyncStream::with_channel(move |sender| {
                wasm_bindgen_futures::spawn_local(async move {
                    loop {
                        match JsFuture::from(reader.read()).await {
                            Ok(chunk) => {
                                let chunk_obj = chunk.dyn_into::<js_sys::Object>().unwrap();
                                
                                // Check if done
                                let done = js_sys::Reflect::get(&chunk_obj, &JsValue::from_str("done"))
                                    .unwrap_or(JsValue::FALSE)
                                    .as_bool()
                                    .unwrap_or(false);
                                
                                if done {
                                    emit!(sender, HttpResponseChunk::Complete);
                                    break;
                                }
                                
                                // Get value
                                if let Ok(value) = js_sys::Reflect::get(&chunk_obj, &JsValue::from_str("value")) {
                                    if let Ok(uint8_array) = value.dyn_into::<Uint8Array>() {
                                        let mut bytes = vec![0; uint8_array.length() as usize];
                                        uint8_array.copy_to(&mut bytes);
                                        
                                        emit!(sender, HttpResponseChunk::Body {
                                            data: Bytes::from(bytes),
                                            final_chunk: false,
                                        });
                                    }
                                }
                            }
                            Err(_) => {
                                emit!(sender, HttpResponseChunk::Error {
                                    message: std::sync::Arc::from("Stream read error"),
                                });
                                break;
                            }
                        }
                    }
                });
            }))
        } else {
            // No body
            Ok(AsyncStream::with_channel(|sender| {
                emit!(sender, HttpResponseChunk::Complete);
            }))
        }
    }

    // Delegate core methods to inner StreamingResponse

    /// Get the status code
    #[inline]
    pub fn status(&self) -> StatusCode {
        self.inner.status
    }

    /// Get the headers
    #[inline]
    pub fn headers(&self) -> &HeaderMap {
        &self.inner.headers
    }

    /// Get the HTTP version
    #[inline]
    pub fn version(&self) -> Version {
        self.inner.version
    }

    /// Get the URL
    #[inline]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Check if response was redirected
    #[inline]
    pub fn redirected(&self) -> bool {
        self.redirected
    }

    /// Get response type
    #[inline]
    pub fn response_type(&self) -> WasmResponseType {
        self.response_type
    }

    /// Check if response is successful
    #[inline]
    pub fn is_success(&self) -> bool {
        self.inner.is_success()
    }

    /// Check if response is redirect
    #[inline]
    pub fn is_redirect(&self) -> bool {
        self.inner.is_redirect()
    }

    /// Check if response is client error
    #[inline]
    pub fn is_client_error(&self) -> bool {
        self.inner.is_client_error()
    }

    /// Check if response is server error
    #[inline]
    pub fn is_server_error(&self) -> bool {
        self.inner.is_server_error()
    }

    /// Get next response chunk
    #[inline]
    pub fn try_next(&mut self) -> Option<HttpResponseChunk> {
        self.inner.try_next()
    }

    /// Collect response body as bytes
    pub fn collect_bytes(self) -> Vec<u8> {
        self.inner.collect_bytes()
    }

    /// Collect response body as string
    pub fn collect_string(self) -> Result<String, std::string::FromUtf8Error> {
        self.inner.collect_string()
    }

    /// Collect and deserialize JSON
    pub fn collect_json<T>(self) -> Result<T, serde_json::Error>
    where
        T: DeserializeOwned,
    {
        self.inner.collect_json()
    }

    /// Get response statistics
    #[inline]
    pub fn stats(&self) -> crate::streaming::pipeline::ResponseStats {
        self.inner.stats()
    }

    #[cfg(target_arch = "wasm32")]
    /// Get the underlying web_sys::Response
    #[inline]
    pub fn web_response(&self) -> Option<&web_sys::Response> {
        self.web_response.as_ref()
    }

    #[cfg(target_arch = "wasm32")]
    /// Clone the web response (if available)
    pub fn clone_web_response(&self) -> Option<web_sys::Response> {
        self.web_response.as_ref().and_then(|resp| resp.clone().ok())
    }

    #[cfg(target_arch = "wasm32")]
    /// Abort the response (if abort controller is available)
    pub fn abort(&self) {
        if let Some(controller) = &self.abort_controller {
            controller.abort();
        }
    }

    /// Convert from StreamingResponse
    #[inline]
    pub fn from_streaming_response(response: StreamingResponse, url: Url) -> Self {
        Self::new(response, url)
    }

    /// Convert to StreamingResponse
    #[inline]
    pub fn into_streaming_response(self) -> StreamingResponse {
        self.inner
    }

    /// Get reference to inner StreamingResponse
    #[inline]
    pub fn as_streaming_response(&self) -> &StreamingResponse {
        &self.inner
    }

    /// Get mutable reference to inner StreamingResponse
    #[inline]
    pub fn as_streaming_response_mut(&mut self) -> &mut StreamingResponse {
        &mut self.inner
    }

    /// Transform response chunks with a mapping function
    pub fn map_chunks<F, T>(self, mapper: F) -> AsyncStream<T, 1024>
    where
        F: Fn(HttpResponseChunk) -> T + Send + 'static,
        T: Send + 'static,
    {
        self.inner.map_chunks(mapper)
    }

    /// Filter response chunks based on predicate
    pub fn filter_chunks<F>(self, predicate: F) -> AsyncStream<HttpResponseChunk, 1024>
    where
        F: Fn(&HttpResponseChunk) -> bool + Send + 'static,
    {
        self.inner.filter_chunks(predicate)
    }
}

/// WASM response builder for ergonomic construction
#[derive(Debug)]
pub struct WasmResponseBuilder {
    status: StatusCode,
    headers: HeaderMap,
    version: Version,
    url: Url,
    response_type: WasmResponseType,
    redirected: bool,
}

impl WasmResponseBuilder {
    /// Create new builder
    #[inline]
    pub fn new(url: Url) -> Self {
        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            version: Version::HTTP_11,
            url,
            response_type: WasmResponseType::Basic,
            redirected: false,
        }
    }

    /// Set status code
    #[inline]
    pub fn status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }

    /// Add header
    #[inline]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: TryInto<http::HeaderName>,
        V: TryInto<http::HeaderValue>,
    {
        if let (Ok(name), Ok(val)) = (key.try_into(), value.try_into()) {
            self.headers.insert(name, val);
        }
        self
    }

    /// Set HTTP version
    #[inline]
    pub fn version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    /// Set response type
    #[inline]
    pub fn response_type(mut self, response_type: WasmResponseType) -> Self {
        self.response_type = response_type;
        self
    }

    /// Set redirected flag
    #[inline]
    pub fn redirected(mut self, redirected: bool) -> Self {
        self.redirected = redirected;
        self
    }

    /// Build WasmResponse with chunk stream
    pub fn build_with_chunks(self, chunks: AsyncStream<HttpResponseChunk, 1024>) -> WasmResponse {
        let streaming_response = StreamingResponse::with_chunks(
            self.status,
            self.headers,
            self.version,
            chunks,
        );

        let mut wasm_response = WasmResponse::new(streaming_response, self.url);
        wasm_response.response_type = self.response_type;
        wasm_response.redirected = self.redirected;
        wasm_response
    }

    /// Build WasmResponse with body bytes
    pub fn build_with_bytes(self, body: Vec<u8>) -> WasmResponse {
        let chunks = AsyncStream::with_channel(move |sender| {
            if !body.is_empty() {
                emit!(sender, HttpResponseChunk::Body {
                    data: Bytes::from(body),
                    final_chunk: true,
                });
            }
            emit!(sender, HttpResponseChunk::Complete);
        });

        self.build_with_chunks(chunks)
    }

    /// Build WasmResponse with JSON body
    pub fn build_with_json<T: serde::Serialize>(
        self,
        json: &T,
    ) -> Result<WasmResponse, serde_json::Error> {
        let body = serde_json::to_vec(json)?;
        Ok(self.build_with_bytes(body))
    }
}