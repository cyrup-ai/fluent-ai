//! Middleware for the memory API
//! This module contains middleware functions for authentication, logging, etc.

use std::time::Instant;
use std::collections::HashMap;

use axum::{
    body::Body, 
    http::{Request, StatusCode, HeaderMap, HeaderValue}, 
    middleware::Next, 
    response::{IntoResponse, Response}
};
use tower_http::cors::{Any, CorsLayer};
use serde::{Deserialize, Serialize};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;

/// JWT secret key (in production, this should come from environment variables)
static JWT_SECRET: Lazy<String> = Lazy::new(|| {
    std::env::var("JWT_SECRET").unwrap_or_else(|_| "default-secret-key".to_string())
});

/// Valid API keys (in production, this should come from a database)
static VALID_API_KEYS: Lazy<HashMap<String, UserContext>> = Lazy::new(|| {
    let mut keys = HashMap::new();
    // Add default API keys - in production, load from secure storage
    keys.insert("admin-key-123".to_string(), UserContext {
        user_id: "admin".to_string(),
        email: "admin@example.com".to_string(),
        roles: vec!["admin".to_string()],
        permissions: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
        expires_at: None,
    });
    keys.insert("user-key-456".to_string(), UserContext {
        user_id: "user1".to_string(),
        email: "user1@example.com".to_string(),
        roles: vec!["user".to_string()],
        permissions: vec!["read".to_string(), "write".to_string()],
        expires_at: None,
    });
    keys
});

/// User context extracted from authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub email: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
struct JwtClaims {
    sub: String,  // Subject (user ID)
    email: String,
    roles: Vec<String>,
    permissions: Vec<String>,
    exp: i64,     // Expiration time
    iat: i64,     // Issued at
}

/// Authentication errors
#[derive(Debug, Clone)]
pub enum AuthError {
    InvalidToken,
    ExpiredToken,
    InvalidApiKey,
    MissingCredentials,
    InsufficientPermissions,
}

/// Add CORS middleware
pub fn cors_middleware() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Request logging middleware
pub async fn logging_middleware(request: Request<Body>, next: Next) -> impl IntoResponse {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    println!("{} {} - {:?}", method, uri, duration);

    response
}

/// Production authentication middleware with JWT and API key support
pub async fn auth_middleware(mut request: Request<Body>, next: Next) -> impl IntoResponse {
    // Extract authorization header
    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    // Check for API key in header or query parameter
    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|h| h.to_str().ok())
        .or_else(|| {
            request.uri()
                .query()
                .and_then(|q| {
                    q.split('&')
                        .find(|param| param.starts_with("api_key="))
                        .and_then(|param| param.split('=').nth(1))
                })
        });

    // Validate authentication
    let auth_result = if let Some(auth_header) = auth_header {
        validate_jwt_token(auth_header).await
    } else if let Some(api_key) = api_key {
        validate_api_key(api_key).await
    } else {
        Err(AuthError::MissingCredentials)
    };

    match auth_result {
        Ok(user_context) => {
            // Add user context to request extensions
            request.extensions_mut().insert(user_context);
            
            // Add security headers to response
            let response = next.run(request).await;
            add_security_headers(response)
        }
        Err(auth_error) => {
            tracing::warn!("Authentication failed: {:?}", auth_error);
            
            let error_response = match auth_error {
                AuthError::InvalidToken | AuthError::ExpiredToken => {
                    (StatusCode::UNAUTHORIZED, "Invalid or expired token")
                }
                AuthError::InvalidApiKey => {
                    (StatusCode::UNAUTHORIZED, "Invalid API key")
                }
                AuthError::MissingCredentials => {
                    (StatusCode::UNAUTHORIZED, "Missing authentication credentials")
                }
                AuthError::InsufficientPermissions => {
                    (StatusCode::FORBIDDEN, "Insufficient permissions")
                }
            };
            
            error_response.into_response()
        }
    }
}

/// Validate JWT token and extract user context
async fn validate_jwt_token(auth_header: &str) -> Result<UserContext, AuthError> {
    // Extract token from "Bearer <token>" format
    let token = if auth_header.starts_with("Bearer ") {
        &auth_header[7..]
    } else {
        return Err(AuthError::InvalidToken);
    };

    // Decode and validate JWT
    let decoding_key = DecodingKey::from_secret(JWT_SECRET.as_bytes());
    let validation = Validation::new(Algorithm::HS256);
    
    match decode::<JwtClaims>(token, &decoding_key, &validation) {
        Ok(token_data) => {
            let claims = token_data.claims;
            
            // Check if token is expired
            let now = Utc::now().timestamp();
            if claims.exp < now {
                return Err(AuthError::ExpiredToken);
            }
            
            Ok(UserContext {
                user_id: claims.sub,
                email: claims.email,
                roles: claims.roles,
                permissions: claims.permissions,
                expires_at: Some(DateTime::from_timestamp(claims.exp, 0).unwrap_or_else(Utc::now)),
            })
        }
        Err(_) => Err(AuthError::InvalidToken),
    }
}

/// Validate API key and return associated user context
async fn validate_api_key(api_key: &str) -> Result<UserContext, AuthError> {
    VALID_API_KEYS
        .get(api_key)
        .cloned()
        .ok_or(AuthError::InvalidApiKey)
}

/// Add security headers to response
fn add_security_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    
    // Add security headers
    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        "X-Frame-Options",
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        "X-XSS-Protection",
        HeaderValue::from_static("1; mode=block"),
    );
    headers.insert(
        "Strict-Transport-Security",
        HeaderValue::from_static("max-age=31536000; includeSubDomains"),
    );
    headers.insert(
        "Referrer-Policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert(
        "Content-Security-Policy",
        HeaderValue::from_static("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"),
    );
    
    response
}
