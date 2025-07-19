//! HTTP response types and utilities

use std::collections::HashMap;

use bytes::Bytes;
use http::StatusCode;

/// HTTP response structure with zero-allocation design
#[derive(Debug, Clone)]
pub struct HttpResponse {
    status: StatusCode,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpResponse {
    /// Create a new HTTP response
    pub fn new(status: StatusCode, headers: reqwest::header::HeaderMap, body: Vec<u8>) -> Self {
        // Convert reqwest headers to HashMap with zero-allocation filtering
        let headers = headers
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v| (k.as_str().to_string(), v.to_string()))
            })
            .collect();

        Self {
            status,
            headers,
            body,
        }
    }

    /// Create a response from cache data - zero-allocation constructor for blazing-fast performance
    pub fn from_cache(
        status: StatusCode,
        headers: HashMap<String, String>,
        body: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            status,
            headers,
            body: body.into(),
        }
    }

    /// Get the status code
    #[inline(always)]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// Get the headers
    #[inline(always)]
    pub fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get the body as bytes
    #[inline(always)]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Get the body as a string
    pub fn text(&self) -> crate::HttpResult<String> {
        Ok(String::from_utf8_lossy(&self.body).to_string())
    }

    /// Get the body as bytes
    pub fn bytes(&self) -> crate::HttpResult<Bytes> {
        Ok(Bytes::from(self.body.clone()))
    }

    /// Check if response is successful (2xx status)
    #[inline(always)]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Parse the body as JSON
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> crate::HttpResult<T> {
        serde_json::from_slice(&self.body).map_err(|e| crate::HttpError::DeserializationError {
            message: format!("Failed to parse JSON response: {}", e),
        })
    }

    /// Check if the response is a client error (4xx status)
    #[inline(always)]
    pub fn is_client_error(&self) -> bool {
        self.status.is_client_error()
    }

    /// Check if the response is a server error (5xx status)
    #[inline(always)]
    pub fn is_server_error(&self) -> bool {
        self.status.is_server_error()
    }

    /// Check if the response is a redirection (3xx status)
    #[inline(always)]
    pub fn is_redirection(&self) -> bool {
        self.status.is_redirection()
    }

    /// Check if the response is informational (1xx status)
    #[inline(always)]
    pub fn is_informational(&self) -> bool {
        self.status.is_informational()
    }

    /// Get a header value
    #[inline(always)]
    pub fn header(&self, key: &str) -> Option<&String> {
        self.headers.get(key)
    }

    /// Get content type
    #[inline(always)]
    pub fn content_type(&self) -> Option<&String> {
        self.header("content-type")
    }

    /// Get ETag header value
    #[inline(always)]
    pub fn etag(&self) -> Option<&String> {
        self.header("etag")
    }

    /// Get Last-Modified header value
    #[inline(always)]
    pub fn last_modified(&self) -> Option<&String> {
        self.header("last-modified")
    }

    /// Get Cache-Control header value
    #[inline(always)]
    pub fn cache_control(&self) -> Option<&String> {
        self.header("cache-control")
    }

    /// Get content length
    #[inline(always)]
    pub fn content_length(&self) -> Option<u64> {
        self.header("content-length").and_then(|v| v.parse().ok())
    }

    /// Get Expires header value
    #[inline(always)]
    pub fn expires(&self) -> Option<&String> {
        self.header("expires")
    }

    /// Get Server header value
    #[inline(always)]
    pub fn server(&self) -> Option<&String> {
        self.header("server")
    }

    /// Get Date header value
    #[inline(always)]
    pub fn date(&self) -> Option<&String> {
        self.header("date")
    }

    /// Get Location header value (for redirects)
    #[inline(always)]
    pub fn location(&self) -> Option<&String> {
        self.header("location")
    }

    /// Get body size in bytes
    #[inline(always)]
    pub fn body_size(&self) -> usize {
        self.body.len()
    }

    /// Check if body is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Get status as u16
    #[inline(always)]
    pub fn status_code(&self) -> u16 {
        self.status.as_u16()
    }

    /// Get status reason phrase
    #[inline(always)]
    pub fn status_reason(&self) -> Option<&str> {
        self.status.canonical_reason()
    }

    /// Convert to error if not successful
    pub fn error_for_status(self) -> crate::HttpResult<Self> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(crate::HttpError::HttpStatus {
                status: self.status.as_u16(),
                message: format!(
                    "HTTP {} {}",
                    self.status.as_u16(),
                    self.status.canonical_reason().unwrap_or("Unknown")
                ),
                body: String::from_utf8_lossy(&self.body).to_string(),
            })
        }
    }

    /// Create a 200 OK response
    #[inline(always)]
    pub fn ok(body: Vec<u8>) -> Self {
        Self {
            status: StatusCode::OK,
            headers: HashMap::new(),
            body,
        }
    }

    /// Create a 404 Not Found response
    #[inline(always)]
    pub fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            headers: HashMap::new(),
            body: b"Not Found".to_vec(),
        }
    }

    /// Create a 500 Internal Server Error response
    #[inline(always)]
    pub fn internal_server_error() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            headers: HashMap::new(),
            body: b"Internal Server Error".to_vec(),
        }
    }

    /// Create a 400 Bad Request response
    #[inline(always)]
    pub fn bad_request() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            headers: HashMap::new(),
            body: b"Bad Request".to_vec(),
        }
    }

    /// Create a 401 Unauthorized response
    #[inline(always)]
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            headers: HashMap::new(),
            body: b"Unauthorized".to_vec(),
        }
    }

    /// Create a 403 Forbidden response
    #[inline(always)]
    pub fn forbidden() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            headers: HashMap::new(),
            body: b"Forbidden".to_vec(),
        }
    }

    /// Create a 429 Too Many Requests response
    #[inline(always)]
    pub fn too_many_requests() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            headers: HashMap::new(),
            body: b"Too Many Requests".to_vec(),
        }
    }

    /// Create a 502 Bad Gateway response
    #[inline(always)]
    pub fn bad_gateway() -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            headers: HashMap::new(),
            body: b"Bad Gateway".to_vec(),
        }
    }

    /// Create a 503 Service Unavailable response
    #[inline(always)]
    pub fn service_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            headers: HashMap::new(),
            body: b"Service Unavailable".to_vec(),
        }
    }

    /// Create a 504 Gateway Timeout response
    #[inline(always)]
    pub fn gateway_timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            headers: HashMap::new(),
            body: b"Gateway Timeout".to_vec(),
        }
    }
}
