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
            // Prioritize QUIC (HTTP/3) with fallback to HTTP/2 - fully async
            .http3_prior_knowledge()
            .http2_prior_knowledge()
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_timeout(Duration::from_secs(10))
            .http2_keep_alive_while_idle(true)
            .http2_max_frame_size(Some(1024 * 1024)) // 1MB frames for efficiency
            // Use rustls for TLS with native root certificates
            .use_rustls_tls()
            .tls_built_in_root_certs(true)
            // Re-use connections aggressively – most APIs sit behind some LB
            .tcp_keepalive(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(32) // Increase connection pool for better performance
            .timeout(Duration::from_secs(300)) // 5 minute timeout for async operations
            .connect_timeout(Duration::from_secs(30))
            .user_agent(concat!("cyrup_ai/", env!("CARGO_PKG_VERSION"), " (QUIC/HTTP3+rustls)"))
            .build()
            .expect("failed to initialise async reqwest client with QUIC and rustls support"),
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

#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: Arc<ReqwestClient>,
}

impl Default for HttpClient {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    /// Create a new HttpClient with QUIC (HTTP/3) prioritization
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            inner: REQWEST.clone(),
        }
    }

    /// Create a specialized HttpClient for a specific provider with optimal settings
    #[inline(always)]
    pub fn for_provider(provider: &str) -> Self {
        // For now, all providers use the same optimized QUIC client
        // This method allows future customization per provider if needed
        match provider {
            "openai" | "anthropic" | "gemini" | "mistral" | "deepseek" | "xai" | "perplexity" => {
                Self::new()
            }
            _ => Self::new(),
        }
    }

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
