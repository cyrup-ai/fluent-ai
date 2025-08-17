//! Header parsing, validation, and formatting utilities with zero-allocation optimizations

use http::{HeaderMap, HeaderName, HeaderValue};

use crate::error::HttpResult;

/// Parse headers from string format
#[inline]
pub fn parse_headers(header_str: &str) -> HttpResult<HeaderMap> {
    let mut headers = HeaderMap::new();

    for line in header_str.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim();
            let value = value.trim();

            let header_name = create_header_name(name)?;
            let header_value = create_header_value(value)?;

            headers.insert(header_name, header_value);
        }
    }

    Ok(headers)
}

/// Format headers as string representation
#[inline]
pub fn format_headers(headers: &HeaderMap) -> String {
    let mut result = String::new();

    for (name, value) in headers {
        result.push_str(name.as_str());
        result.push_str(": ");
        if let Ok(value_str) = value.to_str() {
            result.push_str(value_str);
        }
        result.push('\n');
    }

    result
}

/// Validate header name and value combination
#[inline]
pub fn validate_header(name: &str, value: &str) -> HttpResult<()> {
    create_header_name(name)?;
    create_header_value(value)?;
    Ok(())
}

/// Create header value from string
#[inline]
pub fn create_header_value(value: &str) -> HttpResult<HeaderValue> {
    HeaderValue::from_str(value).map_err(|e| crate::error::HttpError::InvalidHeader {
        message: e.to_string(),
        name: "unknown".to_string(),
        value: Some(value.to_string()),
        error_source: None,
    })
}

/// Create header name from string
#[inline]
pub fn create_header_name(name: &str) -> HttpResult<HeaderName> {
    HeaderName::from_bytes(name.as_bytes()).map_err(|e| crate::error::HttpError::InvalidHeader {
        message: e.to_string(),
        name: name.to_string(),
        value: None,
        error_source: None,
    })
}

/// Merge header maps with conflict resolution
#[inline]
pub fn merge_headers(base: &mut HeaderMap, additional: HeaderMap) {
    for (name, value) in additional {
        if let Some(name) = name {
            base.insert(name, value);
        }
    }
}

/// Extract content type from headers
#[inline]
pub fn extract_content_type(headers: &HeaderMap) -> Option<&str> {
    headers.get("content-type").and_then(|v| v.to_str().ok())
}

/// Check if headers indicate compressed content
#[inline]
pub fn is_compressed(headers: &HeaderMap) -> bool {
    headers
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
        .map(|encoding| !matches!(encoding, "identity" | ""))
        .unwrap_or(false)
}

/// Get content length from headers
#[inline]
pub fn get_content_length(headers: &HeaderMap) -> Option<u64> {
    headers
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
}
