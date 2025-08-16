use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use http::Method;
use http::Uri;
use http::header::{HeaderMap, PROXY_AUTHORIZATION};
use http::uri::Scheme;

use super::config::HttpVersionPref;
use crate::hyper::IntoUrl;
use crate::hyper::async_impl::decoder::Accepts;
#[cfg(feature = "http3")]
use crate::hyper::async_impl::h3_client::H3Client;
use crate::hyper::async_impl::request::{Request, RequestBuilder};
use crate::hyper::config::{RequestConfig, RequestTimeout};
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
use crate::hyper::proxy::Matcher as ProxyMatcher;
use crate::hyper::redirect;

/// An asynchronous `Client` to make Requests with.
///
/// The Client has various configuration values to tweak, but the defaults
/// are set to what is usually the most commonly desired value. To configure a
/// `Client`, use `Client::builder()`.
///
/// The `Client` holds a connection pool internally, so it is advised that
/// you create one and **reuse** it.
///
/// You do **not** have to wrap the `Client` in an [`Rc`] or [`Arc`] to **reuse** it,
/// because it already uses an [`Arc`] internally.
///
/// [`Rc`]: std::rc::Rc
#[derive(Clone)]
pub struct Client {
    pub(super) inner: Arc<ClientRef>,
}

// Simple hyper service wrapper for AsyncStream compatibility
pub struct SimpleHyperService {
    #[cfg(feature = "cookies")]
    pub(super) cookie_store: Option<Arc<dyn cookie::CookieStore>>,
    pub(super) hyper: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        hyper::body::Incoming,
    >,
}

pub struct ClientRef {
    pub(super) accepts: Accepts,
    #[cfg(feature = "cookies")]
    pub(super) cookie_store: Option<Arc<dyn cookie::CookieStore>>,
    pub(super) headers: HeaderMap,
    pub(super) redirect_policy: redirect::Policy,
    #[cfg(feature = "http3")]
    pub(super) h3_client: Option<H3Client>,
    pub(super) hyper: SimpleHyperService,
    pub(super) referer: bool,
    pub(super) request_timeout: RequestConfig<RequestTimeout>,
    pub(super) read_timeout: Option<Duration>,
    pub(super) proxies: Arc<Vec<ProxyMatcher>>,
    pub(super) proxies_maybe_http_auth: bool,
    pub(super) proxies_maybe_http_custom_headers: bool,
    pub(super) https_only: bool,
    pub(super) redirect_policy_desc: Option<String>,
}

impl Client {
    /// Constructs a new `Client`.
    ///
    /// # Panics
    ///
    /// This method panics if a TLS backend cannot be initialized, or the resolver
    /// cannot load the system configuration.
    ///
    /// Use `Client::builder()` if you wish to handle the failure as an `Error`
    /// instead of panicking.
    pub fn new() -> Client {
        ClientBuilder::new().build().expect("Client::new()")
    }

    /// Creates a `ClientBuilder` to configure a `Client`.
    ///
    /// This is the same as `ClientBuilder::new()`.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Convenience method to make a `GET` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn get<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::GET, url)
    }

    /// Convenience method to make a `POST` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn post<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::POST, url)
    }

    /// Convenience method to make a `PUT` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn put<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::PUT, url)
    }

    /// Convenience method to make a `PATCH` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn patch<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::PATCH, url)
    }

    /// Convenience method to make a `DELETE` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn delete<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::DELETE, url)
    }

    /// Convenience method to make a `HEAD` request to a URL.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
    pub fn head<U: IntoUrl>(&self, url: U) -> RequestBuilder {
        self.request(Method::HEAD, url)
    }

    /// Start building a `Request` with the `Method` and `Url`.
    ///
    /// Returns a `RequestBuilder`, which will allow setting headers and
    /// the request body before sending.
    ///
    /// # Errors
    ///
    /// This method fails whenever the supplied `Url` cannot be parsed.
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
    pub fn execute(
        &self,
        request: Request,
    ) -> fluent_ai_async::AsyncStream<crate::response::HttpResponseChunk> {
        // execute_request now returns AsyncStream directly - no conversion needed
        self.execute_request(request)
    }

    pub(super) fn proxy_auth(&self, dst: &Uri, headers: &mut HeaderMap) {
        if !self.inner.proxies_maybe_http_auth {
            return;
        }

        // Only set the header here if the destination scheme is 'http',
        // since otherwise, the header will be included in the CONNECT tunnel
        // request instead.
        if dst.scheme() != Some(&Scheme::HTTP) {
            return;
        }

        if headers.contains_key(PROXY_AUTHORIZATION) {
            return;
        }

        for proxy in self.inner.proxies.iter() {
            if let Some(header) = proxy.http_non_tunnel_basic_auth(dst) {
                headers.insert(PROXY_AUTHORIZATION, header);
                break;
            }
        }
    }

    pub(super) fn proxy_custom_headers(&self, dst: &Uri, headers: &mut HeaderMap) {
        if !self.inner.proxies_maybe_http_custom_headers {
            return;
        }

        if dst.scheme() != Some(&Scheme::HTTP) {
            return;
        }

        for proxy in self.inner.proxies.iter() {
            if let Some(iter) = proxy.http_non_tunnel_custom_headers(dst) {
                iter.iter().for_each(|(key, value)| {
                    headers.insert(key, value.clone());
                });
                break;
            }
        }
    }
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Client");
        self.inner.fmt_fields(&mut builder);
        builder.finish()
    }
}

impl ClientRef {
    pub(super) fn fmt_fields(&self, f: &mut fmt::DebugStruct<'_, '_>) {
        // Instead of deriving Debug, only print fields when their output
        // would provide relevant or interesting data.

        #[cfg(feature = "cookies")]
        {
            if let Some(_) = self.cookie_store {
                f.field("cookie_store", &true);
            }
        }

        f.field("accepts", &self.accepts);

        if !self.proxies.is_empty() {
            f.field("proxies", &self.proxies);
        }

        if let Some(s) = &self.redirect_policy_desc {
            f.field("redirect_policy", s);
        }

        if self.referer {
            f.field("referer", &true);
        }

        f.field("default_headers", &self.headers);

        self.request_timeout.fmt_as_field(f);

        if let Some(ref d) = self.read_timeout {
            f.field("read_timeout", d);
        }
    }
}

// Re-export ClientBuilder for convenience
pub use super::builder::ClientBuilder;
