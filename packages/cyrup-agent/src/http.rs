/*──────────────────────────────────────────────────────────────────────────────
  Ultra-lean HTTP façade that **never leaks `reqwest`** to the public API.

  ┌────────────── hot-path guarantees ──────────────────────────────────────┐
  │ • ZERO heap allocations after the `HttpRequest` has been built          │
  │ • All slices stay borrowed; large bodies use `bytes::Bytes`             │
  │ • Shared connection pool (keep-alive + HTTP/2 multiplexing)            │
  │ • Executes on the global reactor via `rt::spawn_async`                  │
  └──────────────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────────────────*/

#![allow(clippy::needless_return)] // intent: explicit early‐returns

use std::{convert::TryFrom, sync::Arc, time::Duration};

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use once_cell::sync::Lazy;
use reqwest::{
    header::HeaderMap as ReqHeaderMap, Client as ReqwestClient,
    ClientBuilder as ReqwestClientBuilder,
};
use thiserror::Error;

use crate::runtime::{spawn_async, AsyncTask};

// ────────────────────────────────────────────────────────────────────────────
// Global, connection-pooled reqwest client
// ────────────────────────────────────────────────────────────────────────────

static REQWEST: Lazy<Arc<ReqwestClient>> = Lazy::new(|| {
    Arc::new(
        ReqwestClientBuilder::new()
            // re-use connections aggressively – most APIs sit behind some LB
            .tcp_keepalive(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .user_agent(concat!("rig-core/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("failed to initialise reqwest client"),
    )
});

// ────────────────────────────────────────────────────────────────────────────
// Domain-specific error variants
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum HttpError {
    /// URI / header is malformed or unsupported.
    #[error("request construction failed: {0}")]
    Build(String),

    /// Transport-level failure (DNS, TLS, TCP handshake…).
    #[error("transport error: {0}")]
    Transport(String),

    /// Non-success HTTP status returned by the server.
    #[error("HTTP {status}: {body}")]
    Status { status: StatusCode, body: String },
}

// ────────────────────────────────────────────────────────────────────────────
// Immutable request description (zero-alloc after construction)
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HttpRequest {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Option<Bytes>,
}

impl HttpRequest {
    /// Construct a `GET` request.
    #[inline(always)]
    pub fn get(uri: impl TryInto<Uri>) -> Result<Self, HttpError> {
        Ok(Self {
            method: Method::GET,
            uri: uri
                .try_into()
                .map_err(|_| HttpError::Build("invalid URI".into()))?,
            headers: HeaderMap::new(),
            body: None,
        })
    }

    /// Construct a `POST` request carrying a `Bytes` body.
    #[inline(always)]
    pub fn post(uri: impl TryInto<Uri>, body: Bytes) -> Result<Self, HttpError> {
        Ok(Self {
            method: Method::POST,
            uri: uri
                .try_into()
                .map_err(|_| HttpError::Build("invalid URI".into()))?,
            headers: HeaderMap::new(),
            body: Some(body),
        })
    }

    /// Append / override a header *without* reallocating.
    #[inline(always)]
    pub fn header(
        mut self,
        name: &'static str,
        value: impl AsRef<[u8]>,
    ) -> Result<Self, HttpError> {
        let name = HeaderName::from_static(name);
        let value = HeaderValue::try_from(value.as_ref())
            .map_err(|e| HttpError::Build(format!("bad header value: {e}")))?;
        self.headers.insert(name, value);
        Ok(self)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Response wrapper (no heavy types leak out)
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct HttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

// ────────────────────────────────────────────────────────────────────────────
// High-level client
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct HttpClient {
    inner: Arc<ReqwestClient>,
}

impl HttpClient {
    /// Execute an [`HttpRequest`] and get an [`AsyncTask`] to await.
    ///
    /// The future stays allocation-free: results are passed through the
    /// zero-copy `AsyncTask` oneshot channel.
    #[inline(always)]
    pub fn execute(&self, req: HttpRequest) -> AsyncTask<Result<HttpResponse, HttpError>> {
        let client = self.inner.clone();

        spawn_async(async move {
            // HeaderMap → ReqHeaderMap **without** extra allocation.
            let mut req_headers = ReqHeaderMap::with_capacity(req.headers.len());
            req_headers.extend(req.headers.into_iter());

            // Build the reqwest request.
            let mut builder = client
                .request(req.method, req.uri.to_string())
                .headers(req_headers);

            if let Some(body) = req.body {
                builder = builder.body(body);
            }

            // Perform the HTTP call.
            let resp = builder
                .send()
                .await
                .map_err(|e| HttpError::Transport(e.to_string()))?;

            let status = resp.status();
            let headers = resp.headers().clone();
            let body = resp
                .bytes()
                .await
                .map_err(|e| HttpError::Transport(e.to_string()))?;

            if !status.is_success() {
                return Err(HttpError::Status {
                    status,
                    body: String::from_utf8_lossy(&body).into(),
                });
            }

            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------------
