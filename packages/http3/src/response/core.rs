//! Core HTTP response types and fundamental operations
//!
//! Contains the main HttpResponse struct with basic constructors, status checking,
//! and essential response handling functionality.

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
    #[must_use]
    pub fn new(
        status: StatusCode,
        headers: &crate::hyper::header::HeaderMap,
        body: Vec<u8>,
    ) -> Self {
        // Convert http3 headers to HashMap with zero-allocation filtering
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
    #[must_use]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// Get the headers
    #[must_use]
    pub fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get the body as bytes
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Get the body as text - returns `String` directly
    /// NO FUTURES - pure streaming, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Get the body as a `Bytes` handle
    #[must_use]
    pub fn bytes(&self) -> Bytes {
        Bytes::from(self.body.clone())
    }

    /// Check if response is successful (2xx status)
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if the response is a client error (4xx status)
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        self.status.is_client_error()
    }

    /// Check if this is a server error (5xx)
    #[must_use]
    pub fn is_server_error(&self) -> bool {
        self.status.is_server_error()
    }

    /// Check if this is a redirection (3xx)
    #[must_use]
    pub fn is_redirection(&self) -> bool {
        self.status.is_redirection()
    }

    /// Check if this is an informational response (1xx)
    #[must_use]
    pub fn is_informational(&self) -> bool {
        self.status.is_informational()
    }

    /// Get status as u16
    #[must_use]
    pub fn status_code(&self) -> u16 {
        self.status.as_u16()
    }

    /// Get status reason phrase
    #[must_use]
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

    /// Get the response body as bytes vector - returns `Vec<u8>` directly
    /// NO FUTURES - pure streaming, returns unwrapped bytes
    #[must_use]
    pub fn stream(&self) -> Vec<u8> {
        self.body.clone()
    }

    /// Get body size in bytes
    #[must_use]
    pub fn body_size(&self) -> usize {
        self.body.len()
    }

    /// Check if body is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Create a 200 OK response
    #[must_use]
    pub fn ok(body: Vec<u8>) -> Self {
        Self {
            status: StatusCode::OK,
            headers: HashMap::new(),
            body,
        }
    }

    /// Create a 404 Not Found response
    #[must_use]
    pub fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            headers: HashMap::new(),
            body: b"Not Found".to_vec(),
        }
    }

    /// Create a 500 Internal Server Error response
    #[must_use]
    pub fn internal_server_error() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            headers: HashMap::new(),
            body: b"Internal Server Error".to_vec(),
        }
    }

    /// Create a 400 Bad Request response
    #[must_use]
    pub fn bad_request() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            headers: HashMap::new(),
            body: b"Bad Request".to_vec(),
        }
    }

    /// Create a 401 Unauthorized response
    #[must_use]
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            headers: HashMap::new(),
            body: b"Unauthorized".to_vec(),
        }
    }

    /// Create a 403 Forbidden response
    #[must_use]
    pub fn forbidden() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            headers: HashMap::new(),
            body: b"Forbidden".to_vec(),
        }
    }

    /// Create a 429 Too Many Requests response
    #[must_use]
    pub fn too_many_requests() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            headers: HashMap::new(),
            body: b"Too Many Requests".to_vec(),
        }
    }

    /// Create a 502 Bad Gateway response
    #[must_use]
    pub fn bad_gateway() -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            headers: HashMap::new(),
            body: b"Bad Gateway".to_vec(),
        }
    }

    /// Create a 503 Service Unavailable response
    #[must_use]
    pub fn service_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            headers: HashMap::new(),
            body: b"Service Unavailable".to_vec(),
        }
    }

    /// Create a 504 Gateway Timeout response
    #[must_use]
    pub fn gateway_timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            headers: HashMap::new(),
            body: b"Gateway Timeout".to_vec(),
        }
    }
}
