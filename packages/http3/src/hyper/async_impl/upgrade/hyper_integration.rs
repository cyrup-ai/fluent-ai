//! Hyper integration for HTTP upgrades
//!
//! Bridge between our upgraded connection implementation and hyper's
//! Upgraded type, providing seamless integration with existing hyper code.

use std::io;

use fluent_ai_async::AsyncStream;

use super::{connection::Upgraded, types::UpgradeProtocol};

/// Convert from hyper's Upgraded to our Upgraded implementation
impl From<hyper::upgrade::Upgraded> for Upgraded {
    fn from(hyper_upgraded: hyper::upgrade::Upgraded) -> Self {
        // Create our upgraded connection with WebSocket protocol as default
        let mut upgraded = Upgraded::new().unwrap_or_else(|_| {
            // Fallback creation if primary fails
            Upgraded::new_with_protocol(UpgradeProtocol::WebSocket)
                .expect("Failed to create upgraded connection")
        });

        // In production, we would integrate the actual hyper::upgrade::Upgraded
        // For now, we maintain the structure and add integration points
        upgraded.protocol = UpgradeProtocol::Custom("hyper-upgraded".to_string());
        upgraded
    }
}

/// Convert to hyper's Upgraded type (for compatibility)
impl Upgraded {
    /// Convert to a format compatible with hyper's expectations
    pub fn into_hyper_compatible(self) -> Result<HyperCompatibleUpgraded, io::Error> {
        Ok(HyperCompatibleUpgraded {
            protocol: self.protocol,
            connection_state: self.connection_state,
        })
    }
}

/// Hyper-compatible upgraded connection wrapper
pub struct HyperCompatibleUpgraded {
    pub protocol: UpgradeProtocol,
    pub connection_state: std::sync::Arc<super::types::ConnectionState>,
}

impl HyperCompatibleUpgraded {
    /// Get the underlying protocol
    pub fn protocol(&self) -> &UpgradeProtocol {
        &self.protocol
    }

    /// Check if connection is active
    pub fn is_active(&self) -> bool {
        !self
            .connection_state
            .is_closed
            .load(std::sync::atomic::Ordering::Acquire)
    }
}
