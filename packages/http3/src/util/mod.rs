//! Utility functions and types for HTTP3 client
//!
//! Common utilities used across the HTTP3 client implementation
//! including URL handling, header processing, and type conversions.

pub mod basic_auth;
pub mod cookies;
pub mod escape;
pub mod header_utils;
pub mod random;
pub mod type_conversions;
pub mod url_helpers;

// Re-export common utilities
pub use basic_auth::{decode_basic_auth, encode_basic_auth};
pub use cookies::{format_cookie, parse_cookie, validate_cookie};
pub use escape::{html_escape, url_decode, url_encode};
pub use header_utils::{format_headers, parse_headers, validate_header};
pub use random::{generate_boundary, generate_nonce};
pub use type_conversions::{from_bytes, to_bytes, to_string};
pub use url_helpers::{normalize_url, parse_url, validate_url};

/// Replace headers function for compatibility
pub fn replace_headers(headers: &mut http::HeaderMap, new_headers: http::HeaderMap) {
    headers.clear();
    headers.extend(new_headers);
}

/// Merge headers function
pub fn merge_headers(target: &mut http::HeaderMap, source: http::HeaderMap) {
    target.extend(source);
}
