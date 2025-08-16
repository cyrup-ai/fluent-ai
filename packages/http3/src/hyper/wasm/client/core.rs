//! Core client implementation for WASM HTTP client
//!
//! Contains the main Client struct and convenience methods for making
//! HTTP requests in WebAssembly environments.

use std::{fmt, sync::Arc};

use fluent_ai_async::AsyncStream;
use http::{HeaderMap, Method, header::Entry};
use wasm_bindgen::prelude::UnwrapThrowExt as _;

use super::config::Config;
use super::fetch::fetch;
use super::{Request, RequestBuilder, Response};
use crate::IntoUrl;

/// dox
#[derive(Clone)]
pub struct Client {
    config: Arc<Config>,
}

impl Client {
    /// dox
    pub fn new() -> Self {
        Client::builder().build().unwrap_throw()
    }

    /// Create client with existing config
    pub(super) fn new_with_config(config: Config) -> Self {
        Client {
            config: Arc::new(config),
        }
    }

    /// dox
    pub fn builder() -> super::builder::ClientBuilder {
        super::builder::ClientBuilder::new()
    }

    /// Convenience method to make a `GET` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn get<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::GET, url)
    }

    /// Convenience method to make a `POST` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn post<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::POST, url)
    }

    /// Convenience method to make a `PUT` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn put<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::PUT, url)
    }

    /// Convenience method to make a `PATCH` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn patch<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::PATCH, url)
    }

    /// Convenience method to make a `DELETE` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn delete<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::DELETE, url)
    }

    /// Convenience method to make a `HEAD` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn head<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::HEAD, url)
    }

    /// Start building a `Request` with the `Method` and `Url`.
    ///
    /// Returns a `RequestBuilder`, which will allow setting headers and
    /// request body before sending.
    ///
    /// # Errors
    ///
    /// This method fails whenever supplied `Url` cannot be parsed.
    pub fn request<U: IntoUrl>(&self, method: Method, url: U) -> RequestBuilder {
        let req = url.into_url().map(move |url| Request::new(method, url));
        RequestBuilder::new(self.clone(), req)
    }

    /// Executes a `Request`.
    ///
    /// A `Request` can be built manually with `Request::new()` or obtained
    /// from a RequestBuilder with `RequestBuilder::build()`.
    ///
    /// You should prefer to use the `RequestBuilder` and
    /// `RequestBuilder::send()`.
    ///
    /// # Errors
    ///
    /// This method fails if there was an error while sending request,
    /// redirect loop was detected or redirect limit was exhausted.
    pub fn execute(&self, request: Request) -> fluent_ai_async::AsyncStream<Response> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};

        let future = self.execute_request(request);
        AsyncStream::with_channel(move |sender| {
            wasm_bindgen_futures::spawn_local(async move {
                match future.await {
                    Ok(response) => emit!(sender, response),
                    Err(e) => handle_error!(crate::HttpError::from(e), "wasm client execute"),
                }
            });
        })
    }

    // merge request headers with Client default_headers, prior to external http fetch
    pub(super) fn merge_headers(&self, req: &mut Request) {
        let headers: &mut HeaderMap = req.headers_mut();
        // insert default headers in the request headers
        // without overwriting already appended headers.
        for (key, value) in self.config.headers.iter() {
            if let Entry::Vacant(entry) = headers.entry(key) {
                entry.insert(value.clone());
            }
        }
    }

    pub(super) fn execute_request(
        &self,
        mut req: Request,
    ) -> AsyncStream<Result<Response, crate::Error>> {
        self.merge_headers(&mut req);
        fetch(req)
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Client");
        self.config.fmt_fields(&mut builder);
        builder.finish()
    }
}
