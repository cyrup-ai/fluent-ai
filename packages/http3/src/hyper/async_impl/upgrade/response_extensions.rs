//! Response extensions for HTTP upgrade detection
//!
//! Utilities for detecting upgrade protocols from HTTP responses
//! and handling protocol negotiation with proper error handling.

use http::{HeaderMap, HeaderValue, Response};

use super::types::UpgradeProtocol;

/// Detect upgrade protocol from HTTP response headers
pub fn detect_upgrade_protocol(headers: &HeaderMap<HeaderValue>) -> Option<UpgradeProtocol> {
    // Check for Upgrade header
    let upgrade_header = headers.get("upgrade")?;
    let upgrade_value = upgrade_header.to_str().ok()?.to_lowercase();

    match upgrade_value.as_str() {
        "websocket" => Some(UpgradeProtocol::WebSocket),
        "h2c" => Some(UpgradeProtocol::Http2ServerPush),
        other => Some(UpgradeProtocol::Custom(other.to_string())),
    }
}

/// Check if response indicates a successful upgrade
pub fn is_upgrade_response<T>(response: &Response<T>) -> bool {
    response.status() == http::StatusCode::SWITCHING_PROTOCOLS
        && response.headers().contains_key("upgrade")
}

/// Extract connection upgrade information from response
pub fn extract_upgrade_info<T>(response: &Response<T>) -> Result<UpgradeProtocol, String> {
    if !is_upgrade_response(response) {
        return Err("Response does not indicate a successful upgrade".to_string());
    }

    detect_upgrade_protocol(response.headers())
        .ok_or_else(|| "Could not determine upgrade protocol".to_string())
}

/// Validate upgrade headers for protocol compliance
pub fn validate_upgrade_headers(
    headers: &HeaderMap<HeaderValue>,
    expected_protocol: &UpgradeProtocol,
) -> Result<(), String> {
    let detected = detect_upgrade_protocol(headers).ok_or("No upgrade protocol detected")?;

    match (expected_protocol, &detected) {
        (UpgradeProtocol::WebSocket, UpgradeProtocol::WebSocket) => Ok(()),
        (UpgradeProtocol::Http2ServerPush, UpgradeProtocol::Http2ServerPush) => Ok(()),
        (UpgradeProtocol::Custom(expected), UpgradeProtocol::Custom(actual))
            if expected == actual =>
        {
            Ok(())
        }
        _ => Err(format!(
            "Protocol mismatch: expected {:?}, got {:?}",
            expected_protocol, detected
        )),
    }
}
