//! Authentication Module - Bearer tokens, API keys, Basic auth, OAuth 2.0, JWT

use base64::{Engine as _, engine::general_purpose};
use http::{HeaderMap, HeaderName, HeaderValue};

use crate::HttpResult;

/// Authentication provider trait for different auth types
pub trait AuthProvider {
    /// Apply authentication to headers
    fn apply_auth(&self, headers: &mut HeaderMap) -> HttpResult<()>;

    /// Get authentication method name
    fn auth_type(&self) -> &'static str;
}

/// Bearer token authentication
pub struct BearerToken {
    token: String,
}

impl BearerToken {
    /// Create new bearer token auth
    #[inline(always)]
    pub fn new(token: String) -> Self {
        Self { token }
    }
}

impl AuthProvider for BearerToken {
    fn apply_auth(&self, headers: &mut HeaderMap) -> HttpResult<()> {
        if !self.token.is_empty() {
            let auth_header = format!("Bearer {}", self.token);
            headers.insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_str(&auth_header)?,
            );
        }
        Ok(())
    }

    #[inline(always)]
    fn auth_type(&self) -> &'static str {
        "Bearer"
    }
}

/// API key authentication (header or query parameter)
pub struct ApiKey {
    key: String,
    placement: ApiKeyPlacement,
}

/// Where to place the API key
pub enum ApiKeyPlacement {
    Header(String),
    Query(String),
}

impl ApiKey {
    /// Create new API key auth
    #[inline(always)]
    pub fn new(key: String, placement: ApiKeyPlacement) -> Self {
        Self { key, placement }
    }
}

impl AuthProvider for ApiKey {
    fn apply_auth(&self, headers: &mut HeaderMap) -> HttpResult<()> {
        if let ApiKeyPlacement::Header(header_name) = &self.placement {
            if !self.key.is_empty() {
                let header_name = HeaderName::from_bytes(header_name.as_bytes())?;
                headers.insert(header_name, HeaderValue::from_str(&self.key)?);
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn auth_type(&self) -> &'static str {
        "ApiKey"
    }
}

/// Basic authentication
pub struct BasicAuth {
    username: String,
    password: Option<String>,
}

impl BasicAuth {
    /// Create new basic auth
    #[inline(always)]
    pub fn new(username: String, password: Option<String>) -> Self {
        Self { username, password }
    }
}

impl AuthProvider for BasicAuth {
    fn apply_auth(&self, headers: &mut HeaderMap) -> HttpResult<()> {
        if !self.username.is_empty() {
            let credentials = format!(
                "{}:{}",
                self.username,
                self.password.as_deref().unwrap_or_default()
            );
            let encoded = general_purpose::STANDARD.encode(credentials.as_bytes());
            let auth_header = format!("Basic {}", encoded);
            headers.insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_str(&auth_header)?,
            );
        }
        Ok(())
    }

    #[inline(always)]
    fn auth_type(&self) -> &'static str {
        "Basic"
    }
}

/// Authentication errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum AuthError {
    #[error("Invalid authentication token")]
    InvalidToken,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Authentication token expired")]
    TokenExpired,
    #[error("Authentication encoding error")]
    EncodingError,
    #[error("Authentication required")]
    AuthRequired,
}
