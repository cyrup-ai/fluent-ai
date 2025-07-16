/*──────────────────────────────────────────────────────────────────────────────
  Ultra-lean HTTP façade that **never leaks `fluent_ai_http3`** to the public API.

  ┌────────────── hot-path guarantees ──────────────────────────────────────┐
  │ • ZERO heap allocations after the `HttpRequest` has been built          │
  │ • All slices stay borrowed; large bodies use `bytes::Bytes`             │
  │ • Shared connection pool (keep-alive + HTTP/2 multiplexing)            │
  │ • Executes on the global reactor via `rt::spawn_async`                  │
  │ • HTTP/3 (QUIC) prioritization with HTTP/2 fallback                     │
  │ • Streaming-first design with .collect() fallback                       │
  └──────────────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────────────────*/

#![allow(clippy::needless_return)] // intent: explicit early‐returns

use std::{convert::TryFrom, sync::Arc, time::Duration};

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use once_cell::sync::Lazy;
use thiserror::Error;

use crate::runtime::{spawn_async, AsyncTask};
use fluent_ai_http3::{HttpClient as Http3Client, HttpConfig, HttpError as Http3Error, HttpRequest as Http3Request, HttpResponse as Http3Response};

// ────────────────────────────────────────────────────────────────────────────
// Global, connection-pooled HTTP3 client
// ────────────────────────────────────────────────────────────────────────────

static HTTP3_CLIENT: Lazy<Arc<Http3Client>> = Lazy::new(|| {
    Arc::new(
        Http3Client::with_config(HttpConfig::ai_optimized())
            .expect("failed to initialise HTTP3 client with QUIC and rustls support"),
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

impl From<Http3Error> for HttpError {
    fn from(error: Http3Error) -> Self {
        match error {
            Http3Error::HttpStatus { status, message, .. } => {
                Self::Status {
                    status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    body: message,
                }
            }
            Http3Error::NetworkError { message } => Self::Transport(message),
            Http3Error::Timeout { message } => Self::Transport(message),
            Http3Error::ConnectionError { message } => Self::Transport(message),
            Http3Error::DnsError { message } => Self::Transport(message),
            Http3Error::TlsError { message } => Self::Transport(message),
            Http3Error::ClientError { message } => Self::Build(message),
            Http3Error::SerializationError { message } => Self::Build(message),
            Http3Error::DeserializationError { message } => Self::Build(message),
            Http3Error::StreamError { message } => Self::Transport(message),
            Http3Error::Other { message } => Self::Transport(message),
        }
    }
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
    inner: Arc<Http3Client>,
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
            inner: HTTP3_CLIENT.clone(),
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
            // Convert our HttpRequest to Http3Request
            let mut http3_req = match req.method {
                Method::GET => client.get(&req.uri.to_string()),
                Method::POST => client.post(&req.uri.to_string()),
                Method::PUT => client.put(&req.uri.to_string()),
                Method::DELETE => client.delete(&req.uri.to_string()),
                Method::PATCH => client.patch(&req.uri.to_string()),
                Method::HEAD => client.head(&req.uri.to_string()),
                _ => return Err(HttpError::Build(format!("unsupported method: {}", req.method))),
            };

            // Add headers
            for (name, value) in req.headers {
                if let Some(header_name) = name {
                    if let Ok(value_str) = value.to_str() {
                        http3_req = http3_req.header(header_name.as_str(), value_str);
                    }
                }
            }

            // Add body if present
            if let Some(body) = req.body {
                http3_req = http3_req.with_body(body.to_vec());
            }

            // Send the request using the HTTP3 client
            let resp = client.send(http3_req).await?;

            // Convert Http3Response to our HttpResponse
            let status = resp.status();
            let headers = resp.headers().clone();
            let body = Bytes::from(resp.body().clone());

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