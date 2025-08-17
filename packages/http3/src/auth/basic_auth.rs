//! Basic authentication utilities

use std::io::Write;

use base64::prelude::BASE64_STANDARD;
use base64::{Engine, write::EncoderWriter};
use http::HeaderValue;

pub fn basic_auth<U, P>(
    username: U,
    password: Option<P>,
) -> Result<HeaderValue, crate::error::HttpError>
where
    U: std::fmt::Display,
    P: std::fmt::Display,
{
    let mut buf = b"Basic ".to_vec();
    {
        let mut encoder = EncoderWriter::new(&mut buf, &BASE64_STANDARD);
        let _ = write!(encoder, "{username}:");
        if let Some(password) = password {
            let _ = write!(encoder, "{password}");
        }
    }
    let mut header =
        HeaderValue::from_bytes(&buf).map_err(|_e| crate::error::HttpError::InvalidHeader {
            message: format!(
                "Invalid authorization header: {}",
                String::from_utf8_lossy(&buf)
            ),
            name: "authorization".to_string(),
            value: Some(String::from_utf8_lossy(&buf).to_string()),
            error_source: None,
        })?;
    header.set_sensitive(true);
    Ok(header)
}

/// Encode basic authentication credentials for compatibility
pub fn encode_basic_auth(username: &str, password: &str) -> String {
    let credentials = format!("{}:{}", username, password);
    BASE64_STANDARD.encode(credentials.as_bytes())
}

/// Decode basic authentication credentials
pub fn decode_basic_auth(encoded: &str) -> Result<(String, String), crate::error::HttpError> {
    let decoded =
        BASE64_STANDARD
            .decode(encoded)
            .map_err(|_| crate::error::HttpError::InvalidHeader {
                message: "Invalid base64 encoding in authorization header".to_string(),
                name: "authorization".to_string(),
                value: Some(encoded.to_string()),
                error_source: None,
            })?;

    let credentials =
        String::from_utf8(decoded).map_err(|_| crate::error::HttpError::InvalidHeader {
            message: "Invalid UTF-8 in authorization header".to_string(),
            name: "authorization".to_string(),
            value: Some(encoded.to_string()),
            error_source: None,
        })?;

    let parts: Vec<&str> = credentials.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(crate::error::HttpError::InvalidHeader {
            message: "Invalid format in authorization header".to_string(),
            name: "authorization".to_string(),
            value: Some(encoded.to_string()),
            error_source: None,
        });
    }

    Ok((parts[0].to_string(), parts[1].to_string()))
}
